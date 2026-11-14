"""Object detection + multi-object tracking behind one small interface.

Detection and tracking are coupled on purpose: ByteTrack needs every detection (including
low-confidence ones) from the same model pass, and Ultralytics runs both in one call.
"""
from pathlib import Path
from typing import Protocol

import numpy as np

from app.core.logging import get_logger
from app.core.types import Track

log = get_logger(__name__)


class ObjectTracker(Protocol):
    name: str
    ready: bool
    detail: str

    def track(self, frame: np.ndarray) -> list[Track]: ...


class NullTracker:
    """Keeps the video path alive when inference is disabled or failed to load."""

    name = "null"

    def __init__(self, detail: str = "Detection disabled (DETECTOR=null)", ready: bool = True) -> None:
        self.ready = ready
        self.detail = detail

    def track(self, frame: np.ndarray) -> list[Track]:
        return []


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


class UltralyticsTracker:
    """YOLO detector with ByteTrack. Keep one instance per camera: track state lives in the model."""

    name = "ultralytics"

    def __init__(self, model_path: str, classes: tuple[str, ...], device: str, min_confidence: float) -> None:
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self._device = resolve_device(device)
        self._min_confidence = min_confidence
        wanted = set(classes)
        self._class_ids = [cid for cid, label in self._model.names.items() if label in wanted] or None
        self.ready = True
        self.detail = f"{Path(model_path).name} on {self._device} · {len(self._class_ids or self._model.names)} classes"

    def track(self, frame: np.ndarray) -> list[Track]:
        result = self._model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=self._class_ids,
            device=self._device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        tracks = []
        for xyxy, cls, conf, tid in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist(), boxes.id.tolist()):
            if conf < self._min_confidence:
                continue
            tracks.append(Track(int(tid), result.names[int(cls)], float(conf), tuple(map(float, xyxy))))
        return tracks


def make_tracker(kind: str, model_path: str, classes: tuple[str, ...], device: str, min_confidence: float) -> ObjectTracker:
    if kind.lower() != "ultralytics":
        return NullTracker()
    try:
        return UltralyticsTracker(model_path, classes, device, min_confidence)
    except Exception as exc:  # model download / torch import failures must not take the video path down
        log.error(f"Detector '{model_path}' failed to load", detector=kind, error=str(exc))
        return NullTracker(detail=f"Detector failed to load: {exc}", ready=False)
