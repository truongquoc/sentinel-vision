from dataclasses import dataclass
from datetime import datetime
from typing import Any

@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 pixels

@dataclass
class TrackedObject:
    track_id: int
    label: str
    confidence: float
    box: tuple[float, float, float, float]
    age: int
    first_seen: datetime
    last_seen: datetime

@dataclass
class EventCandidate:
    rule_id: str
    event_type: str
    severity: str
    track_id: int
    label: str
    details: dict[str, Any]
