import pytest

from app.pipeline.geometry import ground_point, point_in_polygon
from app.pipeline.rules import RuleEngine, ZoneRule
from tests.factories import TrackFactory, VehicleTrackFactory, ZoneRecordFactory

W, H = 640, 480
T0 = 1_700_000_000.0  # fixed clock at the test boundary
INSIDE = (300.0, 200.0, 340.0, 400.0)   # feet at y=400 -> 0.83, inside bottom-half zone
OUTSIDE = (300.0, 20.0, 340.0, 200.0)   # feet at y=200 -> 0.42, above the zone


@pytest.fixture
def rule() -> ZoneRule:
    return ZoneRule.from_record(ZoneRecordFactory())


def run(engine: RuleEngine, rule: ZoneRule, frames: list[tuple[float, list]]) -> list[str]:
    out = []
    for t, tracks in frames:
        out += [e.event_type for e in engine.evaluate([rule], tracks, W, H, T0 + t)]
    return out


def test_point_in_polygon_square():
    square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert point_in_polygon((0.5, 0.5), square)
    assert not point_in_polygon((1.5, 0.5), square)
    assert not point_in_polygon((0.5, 0.5), [(0, 0), (1, 1)])


def test_ground_point_uses_feet_not_centre():
    assert ground_point((100, 100, 200, 300), 400, 400) == (0.375, 0.75)


def test_enter_fires_once_after_min_frames(rule):
    person = TrackFactory(track_id=7, box=INSIDE)
    events = run(RuleEngine(), rule, [(0, [person]), (0.1, [person]), (0.2, [person]), (0.3, [person])])
    assert events == ["zone_enter"]


def test_single_frame_blip_does_not_enter(rule):
    person = TrackFactory(track_id=7)
    events = run(RuleEngine(), rule, [(0, [person]), (0.1, []), (5, [])])
    assert events == []


def test_torso_in_zone_but_feet_outside_is_ignored(rule):
    person = TrackFactory(box=OUTSIDE)
    assert run(RuleEngine(), rule, [(0, [person]), (0.1, [person])]) == []


def test_loiter_fires_once_after_dwell(rule):
    person = TrackFactory(track_id=3)
    frames = [(t, [person]) for t in (0, 0.1, 5, 10.1, 15, 30)]
    assert run(RuleEngine(), rule, frames) == ["zone_enter", "loitering"]


def test_exit_waits_for_grace_period_then_reports_dwell(rule):
    engine = RuleEngine()
    person = TrackFactory(track_id=3)
    assert run(engine, rule, [(0, [person]), (0.1, [person]), (4, [person])]) == ["zone_enter"]
    assert run(engine, rule, [(4.5, [])]) == []  # within grace: occlusion, not an exit
    exits = engine.evaluate([rule], [], W, H, T0 + 6.5)
    assert [e.event_type for e in exits] == ["zone_exit"]
    assert exits[0].details["dwell_seconds"] == pytest.approx(3.9)


def test_brief_occlusion_does_not_create_second_entry(rule):
    person = TrackFactory(track_id=3)
    frames = [(0, [person]), (0.1, [person]), (1, []), (1.5, [person]), (1.6, [person])]
    assert run(RuleEngine(), rule, frames) == ["zone_enter"]


def test_class_filter(rule):
    car = VehicleTrackFactory()
    assert run(RuleEngine(), rule, [(0, [car]), (0.1, [car])]) == []


def test_tracks_are_independent(rule):
    a, b = TrackFactory(track_id=1), TrackFactory(track_id=2)
    assert run(RuleEngine(), rule, [(0, [a, b]), (0.1, [a, b])]) == ["zone_enter", "zone_enter"]


def test_unselected_event_kinds_are_not_emitted():
    rule = ZoneRule.from_record(ZoneRecordFactory(config={"polygon": [[0, 0.5], [1, 0.5], [1, 1], [0, 1]], "events": ["exit"]}))
    person = TrackFactory()
    engine = RuleEngine()
    assert run(engine, rule, [(0, [person]), (0.1, [person])]) == []
    assert run(engine, rule, [(5, [])]) == ["zone_exit"]


def test_deleted_rule_state_is_dropped(rule):
    engine = RuleEngine()
    person = TrackFactory()
    run(engine, rule, [(0, [person]), (0.1, [person])])
    assert engine.occupants(rule.id) == 1
    engine.evaluate([], [person], W, H, T0 + 1)
    assert engine.occupants(rule.id) == 0
