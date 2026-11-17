"""Per-camera processing pipeline.

Each camera gets two threads: the reader (network I/O, newest frame only) and the pipeline
(inference, tracking, rules, persistence). Inference never runs on the asyncio loop, so the
API and live view stay responsive while the model works.
"""
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.types import EventCandidate, Track
from app.pipeline.annotate import crop, draw_highlight, draw_tracks
from app.pipeline.describer import DescriptionRequest
from app.pipeline.detectors import ObjectTracker, make_tracker
from app.pipeline.rules import RuleEngine, ZoneRule
from app.services.camera import CameraReader
from app.services.descriptions import DescriptionService
from app.services.notifier import WebhookNotifier
from app.services.store import Store

log = get_logger(__name__)

TrackerFactory = Callable[[], ObjectTracker]
JPEG_QUALITY = [cv2.IMWRITE_JPEG_QUALITY, 80]
# Image size drives VLM token cost: both images plus the prompt must fit the model's context window.
CONTEXT_MAX_SIDE = 768  # full-frame context image; detail comes from the crop
CROP_MAX_SIDE = 512


def fit_within(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1:
        return image
    return cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)


def encode_jpeg(image: np.ndarray) -> bytes | None:
    ok, data = cv2.imencode(".jpg", image, JPEG_QUALITY)
    if not ok:
        return None
    return data.tobytes()


class CameraPipeline:
    def __init__(self, camera: dict[str, Any], reader: CameraReader, tracker: ObjectTracker, store: Store,
                 notifier: WebhookNotifier, media_dir: Path, fps: float, descriptions: DescriptionService | None = None) -> None:
        self.camera_id = camera["id"]
        self.descriptions = descriptions
        self.reader = reader
        self.tracker = tracker
        self.store = store
        self.notifier = notifier
        self.media_dir = media_dir
        self.interval = 1 / max(fps, 0.1)
        self.rule_engine = RuleEngine()
        self.live_jpeg: bytes | None = None
        self.live_seq = 0
        self.tracks: list[Track] = []
        self.last_error: str | None = None
        self._latencies: deque[float] = deque(maxlen=30)
        self._processed_at: deque[float] = deque(maxlen=30)
        self._rules: list[ZoneRule] = []
        self._rules_dirty = True
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.reader.start()
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"pipeline-{self.camera_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.reader.stop()

    def reload_rules(self) -> None:
        self._rules_dirty = True

    def status(self) -> dict[str, Any]:
        fps = 0.0
        if len(self._processed_at) > 1:
            span = self._processed_at[-1] - self._processed_at[0]
            fps = (len(self._processed_at) - 1) / span if span > 0 else 0.0
        latency = sum(self._latencies) / len(self._latencies) if self._latencies else None
        return {
            "processing_fps": round(fps, 1),
            "inference_ms": round(latency * 1000) if latency is not None else None,
            "active_tracks": len(self.tracks),
            "pipeline_error": self.last_error,
            "zone_occupancy": {rule.id: self.rule_engine.occupants(rule.id) for rule in self._rules},
        }

    def _run(self) -> None:
        last_seq = 0
        while not self._stop.is_set():
            started = time.monotonic()
            frame = self.reader.latest()
            if frame is None or frame.seq == last_seq:
                self._stop.wait(0.02)
                continue
            last_seq = frame.seq
            self._process_safely(frame.image, frame.captured_at)
            self._stop.wait(max(0.0, self.interval - (time.monotonic() - started)))

    def _process_safely(self, image: np.ndarray, captured_at: float) -> None:
        try:
            self._process(image, captured_at)
            self.last_error = None
        except Exception as exc:  # one bad frame or a model hiccup must not kill the camera
            if self.last_error != str(exc):
                log.exception(f"Pipeline error on camera '{self.camera_id}'", camera_id=self.camera_id)
            self.last_error = str(exc)
            self._stop.wait(1.0)

    def _process(self, image: np.ndarray, captured_at: float) -> None:
        inference_started = time.monotonic()
        tracks = self.tracker.track(image)
        self._latencies.append(time.monotonic() - inference_started)
        self._processed_at.append(time.monotonic())
        self.tracks = tracks
        height, width = image.shape[:2]
        candidates = self.rule_engine.evaluate(self._active_rules(), tracks, width, height, captured_at)
        annotated = draw_tracks(image.copy(), tracks)
        self._publish_live(annotated)
        for candidate in candidates:
            self._record(candidate, image, annotated, captured_at)

    def _active_rules(self) -> list[ZoneRule]:
        if self._rules_dirty:
            self._rules_dirty = False
            self._rules = [ZoneRule.from_record(r) for r in self.store.rules(self.camera_id) if r["enabled"] and r["kind"] == "zone"]
        return self._rules

    def _publish_live(self, annotated: np.ndarray) -> None:
        jpeg = encode_jpeg(annotated)
        if jpeg is None:
            return
        self.live_jpeg = jpeg
        self.live_seq += 1

    def _record(self, candidate: EventCandidate, raw: np.ndarray, annotated: np.ndarray, captured_at: float) -> None:
        describing = self.descriptions is not None and self.descriptions.enabled
        if candidate.event_type == "activity" and not describing:
            return  # activity events exist only to carry a description
        # On exit the object has already left the frame: its last box would crop empty background.
        present = candidate.event_type != "zone_exit"
        event_id = str(uuid.uuid4())
        snapshot = annotated.copy()
        if present:
            snapshot = draw_highlight(snapshot, candidate.box, f"{candidate.label} #{candidate.track_id}")
        snapshot_path = self._write_media(f"{event_id}.jpg", snapshot)
        thumbnail = crop(raw, candidate.box) if present else None
        if thumbnail is not None:
            thumbnail = fit_within(thumbnail, CROP_MAX_SIDE)
        thumbnail_path = self._write_media(f"{event_id}_thumb.jpg", thumbnail) if thumbnail is not None else None
        describe = describing and thumbnail_path is not None
        event = {
            "id": event_id,
            "camera_id": self.camera_id,
            "rule_id": candidate.rule_id,
            "event_type": candidate.event_type,
            "severity": candidate.severity,
            "track_id": candidate.track_id,
            "label": candidate.label,
            "details": candidate.details | {"rule_name": candidate.rule_name, "box": candidate.box},
            "snapshot_path": snapshot_path,
            "thumbnail_path": thumbnail_path,
            "occurred_at": datetime.fromtimestamp(captured_at, timezone.utc).isoformat(),
            "description_status": "pending" if describe else None,
        }
        self.store.add_event(event)
        log.info(
            f"{candidate.label} #{candidate.track_id} {candidate.event_type} '{candidate.rule_name}'",
            camera_id=self.camera_id,
            event_context={"uuid": event_id, "type": candidate.event_type, "rule_uuid": candidate.rule_id},
        )
        if not describe:
            self.notifier.enqueue(event)
            return
        self._request_description(event, candidate, thumbnail, raw)

    def _request_description(self, event: dict[str, Any], candidate: EventCandidate, thumbnail: np.ndarray, raw: np.ndarray) -> None:
        crop_jpeg = encode_jpeg(thumbnail)
        frame_jpeg = encode_jpeg(fit_within(raw, CONTEXT_MAX_SIDE))
        if crop_jpeg is None or frame_jpeg is None:
            self.store.set_description(event["id"], "failed", {"error": "Could not encode images"})
            self.notifier.enqueue(event)
            return
        track_key = (self.camera_id, candidate.rule_id, candidate.track_id)
        request = DescriptionRequest(
            crop_jpeg=crop_jpeg,
            frame_jpeg=frame_jpeg,
            label=candidate.label,
            zone_name=candidate.rule_name,
            event_type=candidate.event_type,
            dwell_seconds=candidate.details.get("dwell_seconds"),
            previous_summary=self.descriptions.previous_summary(track_key),
        )
        self.descriptions.submit(event, request, track_key)

    def _write_media(self, filename: str, image: np.ndarray) -> str | None:
        if not cv2.imwrite(str(self.media_dir / filename), image, JPEG_QUALITY):
            log.error(f"Failed to write media file '{filename}'", camera_id=self.camera_id)
            return None
        return filename


class SurveillanceEngine:
    def __init__(self, store: Store, notifier: WebhookNotifier, settings: Settings, tracker_factory: TrackerFactory | None = None,
                 descriptions: DescriptionService | None = None) -> None:
        self.store = store
        self.notifier = notifier
        self.descriptions = descriptions
        self.settings = settings
        self.media_dir = settings.data_dir / "media"
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.pipelines: dict[str, CameraPipeline] = {}
        self._tracker_factory = tracker_factory or self._default_tracker
        self._detector_status = {"name": settings.detector, "ready": False, "detail": "Not loaded yet"}

    def _default_tracker(self) -> ObjectTracker:
        s = self.settings
        return make_tracker(s.detector, s.yolo_model, s.detect_classes, s.device, s.min_confidence)

    def start_camera(self, camera_id: str) -> CameraPipeline | None:
        camera = self.store.camera(camera_id)
        if camera is None:
            return None
        pipeline = self.pipelines.get(camera_id)
        if pipeline is None:
            tracker = self._tracker_factory()
            self._detector_status = {"name": tracker.name, "ready": tracker.ready, "detail": tracker.detail}
            reader = CameraReader(camera_id, camera["rtsp_url"], self.settings.stream_timeout_seconds)
            pipeline = CameraPipeline(camera, reader, tracker, self.store, self.notifier, self.media_dir,
                                      self.settings.inference_fps, self.descriptions)
            self.pipelines[camera_id] = pipeline
        pipeline.start()
        return pipeline

    def stop(self) -> None:
        for pipeline in self.pipelines.values():
            pipeline.stop()

    def rules_changed(self, camera_id: str) -> None:
        pipeline = self.pipelines.get(camera_id)
        if pipeline is None:
            return
        pipeline.reload_rules()

    def detector_status(self) -> dict[str, Any]:
        return self._detector_status

    def camera_status(self, camera: dict[str, Any]) -> dict[str, Any]:
        base = {"id": camera["id"], "name": camera["name"], "enabled": camera["enabled"]}
        pipeline = self.pipelines.get(camera["id"])
        if pipeline is None:
            return base | {"running": False, "online": False, "error": "Camera has not been started", "frame_size": None}
        reader = pipeline.reader
        return base | {
            "running": reader.running,
            "online": reader.online,
            "error": reader.last_error,
            "last_frame_at": reader.last_frame_at,
            "frame_size": reader.frame_size,
        } | pipeline.status()

    def redescribe(self, event: dict[str, Any]) -> str | None:
        """Queue a fresh description for a stored event. Returns an error message, or None when queued."""
        if self.descriptions is None or not self.descriptions.enabled:
            return "Descriptions are disabled"
        if not event.get("thumbnail_path") or not event.get("snapshot_path"):
            return "Event has no object image to describe"
        crop_path, frame_path = self.media_dir / event["thumbnail_path"], self.media_dir / event["snapshot_path"]
        if not crop_path.exists() or not frame_path.exists():
            return "Event media has expired"
        request = DescriptionRequest(
            crop_jpeg=crop_path.read_bytes(),
            frame_jpeg=frame_path.read_bytes(),
            label=event["label"],
            zone_name=event["details"].get("rule_name", "zone"),
            event_type=event["event_type"],
            dwell_seconds=event["details"].get("dwell_seconds"),
        )
        self.store.set_description(event["id"], "pending", None)
        self.descriptions.submit(event, request, (event["camera_id"], event["rule_id"], event["track_id"]), notify=False)
        return None

    def purge_old_media(self) -> int:
        files = self.store.purge_events_older_than(self.settings.event_retention_days)
        for name in files:
            (self.media_dir / name).unlink(missing_ok=True)
        return len(files)
