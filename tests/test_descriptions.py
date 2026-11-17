import base64
import json

import cv2
import httpx
import numpy as np
import pytest

from app.pipeline.describer import (
    OUTPUT_SCHEMA,
    DescriberError,
    Description,
    DescriptionRequest,
    OllamaDescriber,
    build_prompt,
    parse_description,
)
from app.pipeline.rules import RuleEngine, ZoneRule
from app.services.descriptions import DescriptionService
from app.services.engine import CameraPipeline
from app.services.store import Store
from tests.factories import DescribedZoneRecordFactory, TrackFactory, ZoneRecordFactory

T0 = 1_700_000_000.0
MODEL_REPLY = {
    "summary": "A man in a grey hoodie is standing at the gate holding a phone.",
    "appearance": "grey hoodie, dark jeans",
    "action": "standing still, looking into the yard",
    "carrying": ["phone"],
    "concern": "medium",
    "concern_reason": "lingering at the gate",
}


def make_request(**overrides) -> DescriptionRequest:
    base = dict(crop_jpeg=b"crop-bytes", frame_jpeg=b"frame-bytes", label="person", zone_name="Front gate", event_type="zone_enter")
    return DescriptionRequest(**(base | overrides))


# ---------------------------------------------------------------- parsing & prompt
def test_parse_description_normalises_fields():
    d = parse_description(json.dumps(MODEL_REPLY | {"concern": "HIGH", "carrying": "bag"}), "m", 12)
    assert (d.concern, d.carrying, d.model, d.latency_ms) == ("high", ["bag"], "m", 12)


def test_parse_description_unknown_concern_becomes_low():
    assert parse_description(json.dumps(MODEL_REPLY | {"concern": "spooky"}), "m", 1).concern == "low"


@pytest.mark.parametrize("content", ["not json", "[]", json.dumps(MODEL_REPLY | {"summary": " "})])
def test_parse_description_rejects_bad_output(content):
    with pytest.raises(DescriberError):
        parse_description(content, "m", 1)


def test_prompt_includes_situation_and_previous_summary():
    prompt = build_prompt(make_request(event_type="loitering", dwell_seconds=42, previous_summary="Man at gate."))
    assert "42 seconds" in prompt and "Front gate" in prompt and "Man at gate." in prompt


# ---------------------------------------------------------------- Ollama client
def ollama(handler) -> OllamaDescriber:
    return OllamaDescriber("http://ollama.test", "qwen3-vl:8b", transport=httpx.MockTransport(handler))


def test_ollama_sends_both_images_and_schema():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": json.dumps(MODEL_REPLY)}})

    d = ollama(handler).describe(make_request())
    assert d.summary == MODEL_REPLY["summary"] and d.model == "qwen3-vl:8b"
    user = seen["messages"][1]
    assert user["images"] == [base64.b64encode(b"crop-bytes").decode(), base64.b64encode(b"frame-bytes").decode()]
    assert seen["format"] == OUTPUT_SCHEMA and seen["stream"] is False


def test_ollama_requests_context_large_enough_for_two_images():
    # Ollama's default 4096-token context overflows with two images + prompt ("request (4241 tokens) exceeds...").
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": json.dumps(MODEL_REPLY)}})

    OllamaDescriber("http://ollama.test", "m", num_ctx=12288, transport=httpx.MockTransport(handler)).describe(make_request())
    assert seen["options"]["num_ctx"] == 12288
    ollama(handler).describe(make_request())
    assert seen["options"]["num_ctx"] == 8192


def test_ollama_missing_model_explains_fix():
    describer = ollama(lambda r: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(DescriberError, match="ollama pull qwen3-vl:8b"):
        describer.describe(make_request())


def test_ollama_unreachable_is_describer_error():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(DescriberError, match="ConnectError"):
        ollama(handler).describe(make_request())


@pytest.mark.parametrize("models, ready", [([], False), ([{"name": "qwen3-vl:8b", "model": "qwen3-vl:8b"}], True)])
def test_ollama_status_checks_model_installed(models, ready):
    status = ollama(lambda r: httpx.Response(200, json={"models": models})).status()
    assert status["ready"] is ready


# ---------------------------------------------------------------- service
class FakeDescriber:
    name = "fake"
    enabled = True

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.requests: list[DescriptionRequest] = []

    def status(self) -> dict:
        return {"name": self.name, "enabled": True, "ready": True, "detail": "fake"}

    def describe(self, request: DescriptionRequest) -> Description:
        self.requests.append(request)
        if self.fail:
            raise DescriberError("model exploded")
        return Description(**MODEL_REPLY, model="fake", latency_ms=5)


class RecordingNotifier:
    enabled = True

    def __init__(self) -> None:
        self.sent: list[dict] = []

    def enqueue(self, event: dict) -> None:
        self.sent.append(event)


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db")
    s.upsert_camera("cam-1", "Gate", "rtsp://x")
    yield s
    s.close()


def stored_event(store: Store, event_id: str, event_type: str = "zone_enter") -> dict:
    event = {"id": event_id, "camera_id": "cam-1", "rule_id": "r1", "event_type": event_type, "severity": "medium",
             "track_id": 7, "label": "person", "details": {}, "snapshot_path": None, "thumbnail_path": None,
             "occurred_at": "2026-11-14T10:00:00+00:00", "description_status": "pending"}
    store.add_event(event)
    return event


def service(store, describer=None, clock=None) -> tuple[DescriptionService, RecordingNotifier]:
    notifier = RecordingNotifier()
    svc = DescriptionService(describer or FakeDescriber(), store, notifier, notify_wait_seconds=20, clock=clock or FakeClock())
    return svc, notifier


KEY = ("cam-1", "r1", 7)


def test_success_stores_description_and_notifies_with_it(store):
    svc, notifier = service(store)
    svc.submit(stored_event(store, "e1"), make_request(), KEY)
    assert svc.process_next(wait_seconds=0)
    saved = store.event("e1")
    assert saved["description_status"] == "done" and saved["description"]["concern"] == "medium"
    assert [e["description"]["summary"] for e in notifier.sent] == [MODEL_REPLY["summary"]]
    assert svc.previous_summary(KEY) == MODEL_REPLY["summary"]


def test_failure_still_sends_alert_without_description(store):
    svc, notifier = service(store, FakeDescriber(fail=True))
    svc.submit(stored_event(store, "e1"), make_request(), KEY)
    svc.process_next(wait_seconds=0)
    assert store.event("e1")["description_status"] == "failed"
    assert store.event("e1")["description"] == {"error": "model exploded"}
    assert [e["description"] for e in notifier.sent] == [None]


def test_slow_description_releases_alert_after_deadline_and_only_once(store):
    clock = FakeClock()
    svc, notifier = service(store, clock=clock)
    svc.submit(stored_event(store, "e1"), make_request(), KEY)
    svc.flush_overdue()
    assert notifier.sent == []
    clock.now += 21
    svc.flush_overdue()
    assert [e["description"] for e in notifier.sent] == [None]
    svc.process_next(wait_seconds=0)  # description lands later: stored, but no second alert
    assert len(notifier.sent) == 1 and store.event("e1")["description_status"] == "done"


def test_newer_activity_supersedes_queued_one(store):
    svc, notifier = service(store)
    svc.submit(stored_event(store, "a1", "activity"), make_request(event_type="activity"), KEY)
    svc.submit(stored_event(store, "a2", "activity"), make_request(event_type="activity"), KEY)
    assert store.event("a1")["description_status"] == "skipped"
    assert "a1" not in [e["id"] for e in store.events()]  # hidden from the timeline
    svc.process_next(wait_seconds=0)
    assert not svc.process_next(wait_seconds=0)
    assert [e["id"] for e in notifier.sent] == ["a2"]


def test_failed_activity_update_sends_nothing(store):
    svc, notifier = service(store, FakeDescriber(fail=True))
    svc.submit(stored_event(store, "a1", "activity"), make_request(event_type="activity"), KEY)
    svc.process_next(wait_seconds=0)
    assert notifier.sent == []


def test_redescribe_does_not_resend_alert(store):
    svc, notifier = service(store)
    svc.submit(stored_event(store, "e1"), make_request(), KEY, notify=False)
    svc.process_next(wait_seconds=0)
    assert notifier.sent == [] and store.event("e1")["description_status"] == "done"


# ---------------------------------------------------------------- rules: activity cadence
def test_activity_emitted_every_interval_while_inside():
    rule = ZoneRule.from_record(DescribedZoneRecordFactory())
    engine, person = RuleEngine(), TrackFactory(track_id=4)
    types = []
    for t in (0, 0.1, 3, 5.2, 8, 10.3, 12.2, 15, 17.3):
        types += [e.event_type for e in engine.evaluate([rule], [person], 640, 480, T0 + t)]
    # enter at 0.1; activity at 5.2 and 10.3; loiter at 12.2 restarts the clock; next activity at 17.3
    assert types == ["zone_enter", "activity", "activity", "loitering", "activity"]


def test_activity_off_when_interval_zero():
    rule = ZoneRule.from_record(ZoneRecordFactory())
    engine, person = RuleEngine(), TrackFactory()
    types = [e.event_type for t in (0, 0.1, 30, 60) for e in engine.evaluate([rule], [person], 640, 480, T0 + t)]
    assert "activity" not in types


# ---------------------------------------------------------------- pipeline integration
class ScriptedTracker:
    name, ready, detail = "scripted", True, "test"

    def __init__(self, script):
        self.script = list(script)

    def track(self, frame):
        return self.script.pop(0) if self.script else []


def make_pipeline(store, tmp_path, tracker, descriptions):
    notifier = descriptions.notifier if descriptions else RecordingNotifier()
    return CameraPipeline(store.camera("cam-1"), reader=None, tracker=tracker, store=store, notifier=notifier,
                          media_dir=tmp_path, fps=10, descriptions=descriptions), notifier


def test_pipeline_queues_description_with_crop_and_context(store, tmp_path):
    store.add_rule(DescribedZoneRecordFactory(id="r1", name="Front gate"))
    person = TrackFactory(track_id=9)
    describer = FakeDescriber()
    svc, notifier = service(store, describer)
    pipeline, _ = make_pipeline(store, tmp_path, ScriptedTracker([[person], [person]]), svc)
    frame = np.full((480, 640, 3), 90, dtype=np.uint8)
    pipeline._process(frame, T0)
    pipeline._process(frame, T0 + 0.1)

    [event] = store.events()
    assert event["description_status"] == "pending" and notifier.sent == []  # alert waits for the description
    svc.process_next(wait_seconds=0)
    [request] = describer.requests
    assert request.zone_name == "Front gate" and request.crop_jpeg[:2] == b"\xff\xd8" and request.frame_jpeg[:2] == b"\xff\xd8"
    assert notifier.sent[0]["description"]["summary"] == MODEL_REPLY["summary"]


def test_pipeline_caps_crop_size_so_image_tokens_stay_bounded(store, tmp_path):
    store.add_rule(DescribedZoneRecordFactory(id="r1"))
    person = TrackFactory(track_id=3, box=(100.0, 0.0, 1700.0, 2100.0))  # huge box on a 4K frame
    describer = FakeDescriber()
    svc, _ = service(store, describer)
    pipeline, _ = make_pipeline(store, tmp_path, ScriptedTracker([[person], [person]]), svc)
    frame = np.full((2160, 3840, 3), 90, dtype=np.uint8)
    pipeline._process(frame, T0)
    pipeline._process(frame, T0 + 0.1)
    svc.process_next(wait_seconds=0)
    crop_img = cv2.imdecode(np.frombuffer(describer.requests[0].crop_jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert max(crop_img.shape[:2]) <= 512


def test_pipeline_exit_event_has_no_crop_and_is_not_described(store, tmp_path):
    store.add_rule(ZoneRecordFactory(id="r1", config=ZoneRecordFactory().get("config") | {"events": ["exit"]}))
    person = TrackFactory()
    svc, notifier = service(store)
    pipeline, _ = make_pipeline(store, tmp_path, ScriptedTracker([[person], [person], [], []]), svc)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for t in (0, 0.1, 0.5, 3.0):
        pipeline._process(frame, T0 + t)
    [event] = store.events()
    assert event["event_type"] == "zone_exit" and event["thumbnail_path"] is None and event["description_status"] is None
    assert [e["id"] for e in notifier.sent] == [event["id"]]


def test_pipeline_drops_activity_events_when_descriptions_disabled(store, tmp_path):
    store.add_rule(DescribedZoneRecordFactory(id="r1", config=DescribedZoneRecordFactory().get("config") | {"events": ["enter"]}))
    person = TrackFactory()
    pipeline, _ = make_pipeline(store, tmp_path, ScriptedTracker([[person]] * 4), None)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for t in (0, 0.1, 6, 12):
        pipeline._process(frame, T0 + t)
    assert [e["event_type"] for e in store.events()] == ["zone_enter"]
