"""Zone rules as a per-track state machine.

For every (rule, track) pair the engine remembers whether the object is inside the zone:

    outside --(inside for min_frames)--> inside   emits zone_enter
    inside  --(dwell >= loiter_seconds)--> inside  emits loitering (once per visit)
    inside  --(outside/lost for exit_grace)--> gone emits zone_exit

Requiring a few consecutive frames to enter, and a grace period to leave, stops a box that
jitters across the zone edge from producing a stream of enter/exit pairs.
"""
from dataclasses import dataclass, field
from typing import Any

from app.core.types import EventCandidate, Track
from app.pipeline.geometry import Point, ground_point, point_in_polygon

EVENT_TYPES = {"enter": "zone_enter", "loiter": "loitering", "exit": "zone_exit"}


@dataclass(frozen=True)
class ZoneRule:
    id: str
    name: str
    polygon: list[Point]
    classes: frozenset[str]
    emit: frozenset[str]  # subset of EVENT_TYPES keys
    severity: str = "medium"
    loiter_seconds: float | None = None
    min_frames: int = 2
    exit_grace_seconds: float = 2.0

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "ZoneRule":
        cfg = record["config"]
        return cls(
            id=record["id"],
            name=record["name"],
            polygon=[(float(x), float(y)) for x, y in cfg.get("polygon", [])],
            classes=frozenset(cfg.get("classes") or []),
            emit=frozenset(cfg.get("events") or ["enter", "loiter"]),
            severity=cfg.get("severity", "medium"),
            loiter_seconds=cfg.get("loiter_seconds"),
            min_frames=int(cfg.get("min_frames", 2)),
            exit_grace_seconds=float(cfg.get("exit_grace_seconds", 2.0)),
        )

    def watches(self, label: str) -> bool:
        return not self.classes or label in self.classes


@dataclass
class _Occupancy:
    label: str
    frames_inside: int = 0
    entered_at: float | None = None
    last_inside_at: float = 0.0
    loiter_emitted: bool = False
    last_box: tuple | None = None


@dataclass
class RuleEngine:
    """Holds occupancy state for one camera. Not thread-safe; owned by that camera's pipeline."""

    _state: dict[tuple[str, int], _Occupancy] = field(default_factory=dict)

    def evaluate(self, rules: list[ZoneRule], tracks: list[Track], width: int, height: int, now: float) -> list[EventCandidate]:
        self._forget_removed_rules({rule.id for rule in rules})
        events: list[EventCandidate] = []
        for rule in rules:
            events.extend(self._evaluate_rule(rule, tracks, width, height, now))
        return events

    def occupants(self, rule_id: str) -> int:
        return sum(1 for (rid, _), occ in self._state.items() if rid == rule_id and occ.entered_at is not None)

    def _evaluate_rule(self, rule: ZoneRule, tracks: list[Track], width: int, height: int, now: float) -> list[EventCandidate]:
        events: list[EventCandidate] = []
        seen_inside: set[int] = set()
        for track in tracks:
            if not rule.watches(track.label):
                continue
            if not point_in_polygon(ground_point(track.box, width, height), rule.polygon):
                continue
            seen_inside.add(track.track_id)
            events.extend(self._observe_inside(rule, track, now))
        for key in [k for k in self._state if k[0] == rule.id and k[1] not in seen_inside]:
            events.extend(self._observe_absent(rule, key, now))
        return events

    def _observe_inside(self, rule: ZoneRule, track: Track, now: float) -> list[EventCandidate]:
        occ = self._state.setdefault((rule.id, track.track_id), _Occupancy(label=track.label))
        occ.frames_inside += 1
        occ.last_inside_at = now
        occ.last_box = track.box
        if occ.entered_at is None and occ.frames_inside >= rule.min_frames:
            occ.entered_at = now
            return self._emit(rule, "enter", track.track_id, occ, {"confidence": round(track.confidence, 3)})
        if occ.entered_at is None or occ.loiter_emitted or rule.loiter_seconds is None:
            return []
        dwell = now - occ.entered_at
        if dwell < rule.loiter_seconds:
            return []
        occ.loiter_emitted = True
        return self._emit(rule, "loiter", track.track_id, occ, {"dwell_seconds": round(dwell, 1)})

    def _observe_absent(self, rule: ZoneRule, key: tuple[str, int], now: float) -> list[EventCandidate]:
        occ = self._state[key]
        if occ.entered_at is None:
            del self._state[key]  # never confirmed inside; just jitter on the edge
            return []
        if now - occ.last_inside_at < rule.exit_grace_seconds:
            return []
        del self._state[key]
        dwell = occ.last_inside_at - occ.entered_at
        return self._emit(rule, "exit", key[1], occ, {"dwell_seconds": round(dwell, 1)})

    @staticmethod
    def _emit(rule: ZoneRule, kind: str, track_id: int, occ: _Occupancy, details: dict[str, Any]) -> list[EventCandidate]:
        if kind not in rule.emit:
            return []
        return [EventCandidate(rule.id, rule.name, EVENT_TYPES[kind], rule.severity, track_id, occ.label, occ.last_box, details)]

    def _forget_removed_rules(self, live_rule_ids: set[str]) -> None:
        for key in [k for k in self._state if k[0] not in live_rule_ids]:
            del self._state[key]
