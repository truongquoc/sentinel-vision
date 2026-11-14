"""Natural-language descriptions of tracked objects from a vision-language model.

The model sees two images: a tight crop of the object (detail) and the whole frame (context),
and must answer in a fixed JSON schema so results can be stored and rendered reliably.
"""
import base64
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import httpx

from app.core.logging import get_logger

log = get_logger(__name__)

CONCERN_LEVELS = ("none", "low", "medium", "high")

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "appearance": {"type": "string"},
        "action": {"type": "string"},
        "carrying": {"type": "array", "items": {"type": "string"}},
        "concern": {"type": "string", "enum": list(CONCERN_LEVELS)},
        "concern_reason": {"type": "string"},
    },
    "required": ["summary", "appearance", "action", "carrying", "concern", "concern_reason"],
}

SYSTEM_PROMPT = """You describe security camera footage for a home owner.
You receive two images: (1) a close crop of one tracked object, (2) the full camera frame for context.
Describe ONLY the object in image 1, using image 2 to understand where it is and what it interacts with.

Rules:
- Be factual and specific about what is visible: clothing, colours, posture, hands, objects held, direction of movement.
- Never guess identity, name, age, ethnicity, or intent you cannot see. Say "unclear" when you cannot tell.
- summary: one short sentence a person would read on a phone notification.
- action: what the object is doing right now (e.g. "walking toward the gate", "standing still, looking into the yard").
- carrying: list of items held or carried; empty list if none.
- concern: none = ordinary passer-by or resident activity; low = unusual but benign;
  medium = lingering, peering, trying handles, or approaching doors/vehicles; high = climbing, forcing entry, weapons, damage.
- concern_reason: a few words justifying the concern level."""


@dataclass(frozen=True)
class DescriptionRequest:
    crop_jpeg: bytes
    frame_jpeg: bytes
    label: str
    zone_name: str
    event_type: str
    dwell_seconds: float | None = None
    previous_summary: str | None = None


@dataclass(frozen=True)
class Description:
    summary: str
    appearance: str
    action: str
    carrying: list[str] = field(default_factory=list)
    concern: str = "none"
    concern_reason: str = ""
    model: str = ""
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DescriberError(RuntimeError):
    pass


class Describer(Protocol):
    name: str

    @property
    def enabled(self) -> bool: ...

    def status(self) -> dict[str, Any]: ...

    def describe(self, request: DescriptionRequest) -> Description: ...


class NullDescriber:
    name = "null"
    enabled = False

    def status(self) -> dict[str, Any]:
        return {"name": self.name, "enabled": False, "ready": False, "detail": "Descriptions disabled (DESCRIBER=null)"}

    def describe(self, request: DescriptionRequest) -> Description:
        raise DescriberError("Descriptions are disabled")


def build_prompt(request: DescriptionRequest) -> str:
    situation = {
        "zone_enter": f"The {request.label} has just entered the zone '{request.zone_name}'.",
        "loitering": f"The {request.label} has stayed in the zone '{request.zone_name}' for {request.dwell_seconds or 0:.0f} seconds.",
        "activity": f"The {request.label} is still in the zone '{request.zone_name}' ({request.dwell_seconds or 0:.0f} seconds so far).",
        "zone_exit": f"The {request.label} is leaving the zone '{request.zone_name}'.",
    }.get(request.event_type, f"A {request.label} was detected in the zone '{request.zone_name}'.")
    lines = [situation, f"The detector labelled image 1 as: {request.label}."]
    if request.previous_summary:
        lines.append(f"Earlier description of the same {request.label}: \"{request.previous_summary}\". Focus on what they are doing now and what changed.")
    lines.append("Answer in the required JSON format.")
    return "\n".join(lines)


def parse_description(content: str, model: str, latency_ms: int) -> Description:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise DescriberError(f"Model returned invalid JSON: {content[:120]!r}") from exc
    if not isinstance(data, dict) or not str(data.get("summary", "")).strip():
        raise DescriberError("Model response has no summary")
    concern = str(data.get("concern", "none")).lower()
    carrying = data.get("carrying") or []
    return Description(
        summary=str(data["summary"]).strip(),
        appearance=str(data.get("appearance", "")).strip(),
        action=str(data.get("action", "")).strip(),
        carrying=[str(item) for item in carrying] if isinstance(carrying, list) else [str(carrying)],
        concern=concern if concern in CONCERN_LEVELS else "low",
        concern_reason=str(data.get("concern_reason", "")).strip(),
        model=model,
        latency_ms=latency_ms,
    )


class OllamaDescriber:
    """Talks to a local Ollama server. The model must be pulled separately (`ollama pull <model>`)."""

    name = "ollama"
    enabled = True
    STATUS_TTL_SECONDS = 30.0

    def __init__(self, base_url: str, model: str, timeout_seconds: float = 60.0, transport: httpx.BaseTransport | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout_seconds, transport=transport)
        self._status: dict[str, Any] | None = None
        self._status_at = 0.0

    def status(self) -> dict[str, Any]:
        if self._status is not None and time.monotonic() - self._status_at < self.STATUS_TTL_SECONDS:
            return self._status
        self._status = {"name": self.name, "enabled": True, "model": self.model} | self._probe()
        self._status_at = time.monotonic()
        return self._status

    def _probe(self) -> dict[str, Any]:
        try:
            response = self._client.get("/api/tags", timeout=3)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            return {"ready": False, "detail": f"Ollama unreachable at {self.base_url} ({type(exc).__name__})"}
        installed = {m.get("name") for m in response.json().get("models", [])} | {m.get("model") for m in response.json().get("models", [])}
        wanted = {self.model, f"{self.model}:latest"}
        if not installed & wanted:
            return {"ready": False, "detail": f"Model '{self.model}' not installed — run: ollama pull {self.model}"}
        return {"ready": True, "detail": f"{self.model} via Ollama"}

    def describe(self, request: DescriptionRequest) -> Description:
        payload = {
            "model": self.model,
            "stream": False,
            "format": OUTPUT_SCHEMA,
            "keep_alive": "30m",
            "options": {"temperature": 0.2},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_prompt(request),
                    "images": [base64.b64encode(request.crop_jpeg).decode(), base64.b64encode(request.frame_jpeg).decode()],
                },
            ],
        }
        started = time.monotonic()
        content = self._chat(payload)
        return parse_description(content, self.model, int((time.monotonic() - started) * 1000))

    def _chat(self, payload: dict[str, Any]) -> str:
        try:
            response = self._client.post("/api/chat", json=payload)
        except httpx.HTTPError as exc:
            self._status = None  # force a fresh probe so the dashboard shows the real state
            raise DescriberError(f"Ollama request failed: {type(exc).__name__}: {exc}") from exc
        if response.status_code == 404:
            self._status = None
            raise DescriberError(f"Model '{self.model}' not found in Ollama — run: ollama pull {self.model}")
        if response.status_code >= 400:
            raise DescriberError(f"Ollama HTTP {response.status_code}: {response.text[:200]}")
        return response.json().get("message", {}).get("content", "")


def make_describer(kind: str, base_url: str, model: str, timeout_seconds: float) -> Describer:
    if kind.lower() != "ollama":
        return NullDescriber()
    return OllamaDescriber(base_url, model, timeout_seconds)
