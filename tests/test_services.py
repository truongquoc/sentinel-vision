import json
import time
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np
import pytest

from app.core.types import Track
from app.services.camera import redact
from app.services.engine import CameraPipeline
from app.services.notifier import WebhookNotifier, sign
from app.services.store import Store
from tests.factories import TrackFactory, ZoneRecordFactory

T0 = 1_700_000_000.0


@pytest.fixture
def store(tmp_path) -> Store:
    s = Store(tmp_path / "test.db")
    s.upsert_camera("cam-1", "Gate", "rtsp://u:p@10.0.0.2/stream")
    yield s
    s.close()


def event_record(**overrides) -> dict:
    base = {"id": "e1", "camera_id": "cam-1", "rule_id": "r1", "event_type": "zone_enter", "severity": "medium",
            "track_id": 1, "label": "person", "details": {}, "snapshot_path": "e1.jpg", "thumbnail_path": None,
            "occurred_at": datetime.now(timezone.utc).isoformat()}
    return base | overrides


def test_upsert_camera_updates_url(store):
    store.upsert_camera("cam-1", "Gate", "rtsp://new")
    assert store.camera("cam-1")["rtsp_url"] == "rtsp://new"


def test_rule_crud(store):
    rule = ZoneRecordFactory(id="r1")
    store.add_rule(rule)
    assert store.rule("r1")["config"]["polygon"] == rule["config"]["polygon"]
    assert store.update_rule("r1", "Renamed", rule["config"], False)
    assert store.rule("r1")["name"] == "Renamed" and store.rule("r1")["enabled"] is False
    assert store.delete_rule("r1") and store.rule("r1") is None


def test_event_filters_and_ack(store):
    store.add_event(event_record(id="a", label="person"))
    store.add_event(event_record(id="b", label="car", event_type="loitering"))
    assert [e["id"] for e in store.events(label="car")] == ["b"]
    assert [e["id"] for e in store.events(event_type="zone_enter")] == ["a"]
    assert store.acknowledge("a") and not store.acknowledge("a")
    assert [e["id"] for e in store.events(unacknowledged=True)] == ["b"]


def test_purge_returns_media_of_old_events(store):
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    store.add_event(event_record(id="old", occurred_at=old, snapshot_path="old.jpg", thumbnail_path="old_t.jpg"))
    store.add_event(event_record(id="new"))
    assert sorted(store.purge_events_older_than(14)) == ["old.jpg", "old_t.jpg"]
    assert [e["id"] for e in store.events()] == ["new"]


def test_redact_hides_credentials():
    assert redact("rtsp://admin:secret@192.168.1.5:554/cam") == "rtsp://***@192.168.1.5:554/cam"
    assert redact("/tmp/video.mp4") == "/tmp/video.mp4"


def test_webhook_retries_then_delivers_signed_body():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(503 if len(calls) < 3 else 200)

    notifier = WebhookNotifier("http://hook.test/x", "test-secret", base_delay=0.01, transport=httpx.MockTransport(handler))
    notifier.start()
    notifier.enqueue(event_record())
    deadline = time.monotonic() + 3
    while notifier.delivered == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    notifier.stop()
    assert notifier.delivered == 1 and len(calls) == 3
    assert calls[-1].headers["X-Sentinel-Signature"] == sign("test-secret", calls[-1].content)
    assert json.loads(calls[-1].content)["id"] == "e1"


class ScriptedTracker:
    name, ready, detail = "scripted", True, "test"

    def __init__(self, script: list[list[Track]]) -> None:
        self.script = list(script)

    def track(self, frame: np.ndarray) -> list[Track]:
        return self.script.pop(0) if self.script else []


class RecordingNotifier:
    enabled = True

    def __init__(self) -> None:
        self.sent: list[dict] = []

    def enqueue(self, event: dict) -> None:
        self.sent.append(event)


def test_pipeline_records_event_with_snapshot_and_thumbnail(store, tmp_path):
    store.add_rule(ZoneRecordFactory(id="r1", name="Driveway"))
    person = TrackFactory(track_id=9)
    notifier = RecordingNotifier()
    pipeline = CameraPipeline(store.camera("cam-1"), reader=None, tracker=ScriptedTracker([[person], [person]]),
                              store=store, notifier=notifier, media_dir=tmp_path, fps=10)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    pipeline._process(frame, T0)
    pipeline._process(frame, T0 + 0.1)

    [event] = store.events()
    assert (event["event_type"], event["track_id"], event["details"]["rule_name"]) == ("zone_enter", 9, "Driveway")
    assert (tmp_path / event["snapshot_path"]).exists() and (tmp_path / event["thumbnail_path"]).exists()
    assert [e["id"] for e in notifier.sent] == [event["id"]]
    assert pipeline.live_jpeg is not None and pipeline.status()["active_tracks"] == 1
