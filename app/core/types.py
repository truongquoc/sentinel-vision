from dataclasses import dataclass, field
from typing import Any

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


@dataclass(frozen=True)
class Track:
    """An object the tracker has confirmed in the current frame."""

    track_id: int
    label: str
    confidence: float
    box: Box


@dataclass(frozen=True)
class EventCandidate:
    rule_id: str
    rule_name: str
    event_type: str  # zone_enter | loitering | zone_exit
    severity: str
    track_id: int
    label: str
    box: Box | None
    details: dict[str, Any] = field(default_factory=dict)
