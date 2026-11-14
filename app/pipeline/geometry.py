from app.core.types import Box

Point = tuple[float, float]


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    """Even-odd ray casting; polygon vertices and point share one coordinate space."""
    if len(polygon) < 3:
        return False
    x, y = point
    hit = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        crosses = (y1 > y) != (y2 > y)
        if crosses and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


def ground_point(box: Box, width: int, height: int) -> Point:
    """Bottom-centre of the box, normalised to 0..1.

    People and vehicles 'stand' on the ground, so their feet decide which zone they are in.
    Using the box centre would let a tall person's torso enter a gate zone while they are still
    on the pavement.
    """
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2 / width, y2 / height)
