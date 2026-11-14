import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.pipeline.describer import NullDescriber
from app.pipeline.detectors import NullTracker

POLYGON = [[0.1, 0.5], [0.9, 0.5], [0.9, 1.0], [0.1, 1.0]]


@pytest.fixture
def client(tmp_path):
    config = Settings(data_dir=tmp_path, camera_id="cam-1", camera_rtsp_url="", detector="null", webhook_url="")
    with TestClient(create_app(config, tracker_factory=NullTracker, autostart=False, describer=NullDescriber())) as c:
        yield c


def test_health_reports_unstarted_camera(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["cameras"][0]["id"] == "cam-1" and body["cameras"][0]["running"] is False
    assert "rtsp_url" not in body["cameras"][0]


def test_rule_lifecycle(client):
    created = client.post("/api/rules", json={"camera_id": "cam-1", "name": "Gate", "config": {"polygon": POLYGON, "loiter_seconds": 20}})
    assert created.status_code == 201
    rule = created.json()
    assert rule["config"]["events"] == ["enter", "loiter"]

    updated = client.put(f"/api/rules/{rule['id']}", json={"name": "Gate", "config": rule["config"], "enabled": False})
    assert updated.json()["enabled"] is False

    assert client.delete(f"/api/rules/{rule['id']}").status_code == 204
    assert client.get("/api/rules").json() == []


@pytest.mark.parametrize("config", [
    {"polygon": [[0, 0], [1, 1]]},                       # not a polygon
    {"polygon": [[0, 0], [1.5, 0], [1, 1]]},             # outside normalised range
    {"polygon": POLYGON, "events": []},                   # nothing to emit
    {"polygon": POLYGON, "loiter_seconds": -1},
])
def test_rule_validation(client, config):
    assert client.post("/api/rules", json={"camera_id": "cam-1", "name": "x", "config": config}).status_code == 422


def test_rule_for_unknown_camera(client):
    response = client.post("/api/rules", json={"camera_id": "nope", "name": "x", "config": {"polygon": POLYGON}})
    assert response.status_code == 404


def test_missing_event_media_is_404(client):
    assert client.get("/api/events/does-not-exist/snapshot").status_code == 404


def test_live_endpoints_require_started_camera(client):
    assert client.get("/api/cameras/cam-1/snapshot").status_code == 409


def test_new_zone_describes_activity_every_20s_by_default(client):
    rule = client.post("/api/rules", json={"camera_id": "cam-1", "name": "Gate", "config": {"polygon": POLYGON}}).json()
    assert rule["config"]["describe_every_seconds"] == 20


def test_redescribe_unknown_event_is_404(client):
    assert client.post("/api/events/nope/describe").status_code == 404


def test_health_reports_describer(client):
    assert client.get("/api/health").json()["describer"]["enabled"] is False
