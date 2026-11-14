import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from app.services.engine import CameraPipeline, SurveillanceEngine
from app.services.store import Store

router = APIRouter(prefix="/api")

Coordinate = Annotated[float, Field(ge=0, le=1)]
EventKind = Literal["enter", "loiter", "exit"]


class ZoneConfig(BaseModel):
    polygon: list[tuple[Coordinate, Coordinate]] = Field(min_length=3, max_length=64)
    classes: list[str] = Field(default_factory=lambda: ["person"])
    events: list[EventKind] = Field(default_factory=lambda: ["enter", "loiter"], min_length=1)
    loiter_seconds: float | None = Field(default=None, gt=0, le=3600)
    severity: Literal["low", "medium", "high"] = "medium"
    min_frames: int = Field(default=2, ge=1, le=30)
    exit_grace_seconds: float = Field(default=2.0, ge=0, le=60)


class RuleIn(BaseModel):
    camera_id: str
    name: str = Field(min_length=1, max_length=80)
    kind: Literal["zone"] = "zone"
    config: ZoneConfig
    enabled: bool = True


class RuleUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    config: ZoneConfig
    enabled: bool = True


def _store(request: Request) -> Store:
    return request.app.state.store


def _engine(request: Request) -> SurveillanceEngine:
    return request.app.state.engine


def _media_dir(request: Request) -> Path:
    return request.app.state.settings.data_dir / "media"


def _require_pipeline(request: Request, camera_id: str) -> CameraPipeline:
    pipeline = _engine(request).pipelines.get(camera_id)
    if pipeline is None:
        raise HTTPException(409, "Camera has not been started")
    return pipeline


# health & cameras ------------------------------------------------------------
@router.get("/health")
def health(request: Request) -> dict:
    store, engine = _store(request), _engine(request)
    midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    notifier = engine.notifier
    return {
        "status": "ok",
        "detector": engine.detector_status(),
        "cameras": [engine.camera_status(c) for c in store.cameras()],
        "events_today": store.count_events_since(midnight),
        "notifier": {"enabled": notifier.enabled, "delivered": notifier.delivered, "failed": notifier.failed},
    }


@router.get("/cameras")
def cameras(request: Request) -> list[dict]:
    engine = _engine(request)
    return [engine.camera_status(c) for c in _store(request).cameras()]


@router.post("/cameras/{camera_id}/start")
def start_camera(camera_id: str, request: Request) -> dict:
    engine = _engine(request)
    if engine.start_camera(camera_id) is None:
        raise HTTPException(404, "Camera not found")
    return engine.camera_status(_store(request).camera(camera_id))


@router.get("/cameras/{camera_id}/snapshot")
def camera_snapshot(camera_id: str, request: Request) -> Response:
    pipeline = _require_pipeline(request, camera_id)
    if pipeline.live_jpeg is None:
        raise HTTPException(503, "No frame available yet")
    return Response(pipeline.live_jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/cameras/{camera_id}/mjpeg")
async def camera_mjpeg(camera_id: str, request: Request) -> StreamingResponse:
    pipeline = _require_pipeline(request, camera_id)

    async def frames():
        sent = -1
        while not await request.is_disconnected():
            if pipeline.live_jpeg is None or pipeline.live_seq == sent:
                await asyncio.sleep(0.03)
                continue
            sent = pipeline.live_seq
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + pipeline.live_jpeg + b"\r\n"

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


# rules -----------------------------------------------------------------------
@router.get("/rules")
def list_rules(request: Request, camera_id: str | None = None) -> list[dict]:
    return _store(request).rules(camera_id)


@router.post("/rules", status_code=201)
def create_rule(payload: RuleIn, request: Request) -> dict:
    store = _store(request)
    if store.camera(payload.camera_id) is None:
        raise HTTPException(404, "Camera not found")
    rule = payload.model_dump() | {"id": str(uuid.uuid4())}
    store.add_rule(rule)
    _engine(request).rules_changed(payload.camera_id)
    return store.rule(rule["id"])


@router.put("/rules/{rule_id}")
def update_rule(rule_id: str, payload: RuleUpdate, request: Request) -> dict:
    store = _store(request)
    existing = store.rule(rule_id)
    if existing is None:
        raise HTTPException(404, "Rule not found")
    store.update_rule(rule_id, payload.name, payload.config.model_dump(), payload.enabled)
    _engine(request).rules_changed(existing["camera_id"])
    return store.rule(rule_id)


@router.delete("/rules/{rule_id}", status_code=204)
def delete_rule(rule_id: str, request: Request) -> None:
    store = _store(request)
    existing = store.rule(rule_id)
    if existing is None:
        raise HTTPException(404, "Rule not found")
    store.delete_rule(rule_id)
    _engine(request).rules_changed(existing["camera_id"])


# events ----------------------------------------------------------------------
@router.get("/events")
def list_events(
    request: Request,
    camera_id: str | None = None,
    event_type: str | None = None,
    label: str | None = None,
    unacknowledged: bool = False,
    limit: int = Query(100, ge=1, le=500),
) -> list[dict]:
    return _store(request).events(camera_id, event_type, label, unacknowledged, limit)


@router.get("/events/{event_id}")
def get_event(event_id: str, request: Request) -> dict:
    event = _store(request).event(event_id)
    if event is None:
        raise HTTPException(404, "Event not found")
    return event


@router.post("/events/{event_id}/ack")
def acknowledge_event(event_id: str, request: Request) -> dict:
    store = _store(request)
    if store.event(event_id) is None:
        raise HTTPException(404, "Event not found")
    store.acknowledge(event_id)
    return {"ok": True}


@router.get("/events/{event_id}/snapshot")
def event_snapshot(event_id: str, request: Request) -> FileResponse:
    return _event_media(request, event_id, "snapshot_path")


@router.get("/events/{event_id}/thumbnail")
def event_thumbnail(event_id: str, request: Request) -> FileResponse:
    return _event_media(request, event_id, "thumbnail_path")


def _event_media(request: Request, event_id: str, column: str) -> FileResponse:
    event = _store(request).event(event_id)
    if event is None or not event.get(column):
        raise HTTPException(404, "Media not found")
    path = _media_dir(request) / Path(event[column]).name  # name only: never follow a stored path outside media/
    if not path.exists():
        raise HTTPException(404, "Media expired")
    return FileResponse(path, media_type="image/jpeg")
