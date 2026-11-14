import factory

from app.core.types import Track


class TrackFactory(factory.Factory):
    """A confirmed track: a person standing mid-frame in a 640x480 image."""

    class Meta:
        model = Track

    track_id = factory.Sequence(lambda n: n + 1)
    label = "person"
    confidence = factory.Faker("pyfloat", min_value=0.5, max_value=0.99)
    box = (300.0, 200.0, 340.0, 400.0)


class VehicleTrackFactory(TrackFactory):
    label = "car"
    box = (100.0, 300.0, 260.0, 420.0)


class ZoneRecordFactory(factory.DictFactory):
    """A stored zone rule covering the bottom half of the frame."""

    id = factory.Faker("uuid4")
    camera_id = "cam-1"
    name = factory.Faker("word")
    kind = "zone"
    enabled = True
    config = factory.LazyFunction(lambda: {
        "polygon": [[0, 0.5], [1, 0.5], [1, 1], [0, 1]],
        "classes": ["person"],
        "events": ["enter", "loiter", "exit"],
        "loiter_seconds": 10,
        "severity": "medium",
        "min_frames": 2,
        "exit_grace_seconds": 2.0,
        "describe_every_seconds": 0,
    })


class DescribedZoneRecordFactory(ZoneRecordFactory):
    """A zone that asks the AI describer for an activity update every 5 seconds."""

    config = factory.LazyFunction(lambda: {
        "polygon": [[0, 0.5], [1, 0.5], [1, 1], [0, 1]],
        "classes": ["person"],
        "events": ["enter", "loiter"],
        "loiter_seconds": 12,
        "describe_every_seconds": 5,
    })
