import cv2
import numpy as np

from app.core.types import Box, Track

TRACK_COLOR = (115, 210, 75)  # BGR
HIGHLIGHT_COLOR = (60, 80, 235)


def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    for track in tracks:
        _draw_box(frame, track.box, f"{track.label} #{track.track_id} {track.confidence:.0%}", TRACK_COLOR)
    return frame


def draw_highlight(frame: np.ndarray, box: Box | None, caption: str) -> np.ndarray:
    if box is None:
        return frame
    _draw_box(frame, box, caption, HIGHLIGHT_COLOR, thickness=3)
    return frame


def crop(frame: np.ndarray, box: Box | None, pad: float = 0.15) -> np.ndarray | None:
    """Crop around a box with padding: the thumbnail later fed to a vision-language model."""
    if box is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    left, top = max(0, int(x1 - px)), max(0, int(y1 - py))
    right, bottom = min(w, int(x2 + px)), min(h, int(y2 + py))
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right].copy()


def _draw_box(frame: np.ndarray, box: Box, caption: str, color: tuple[int, int, int], thickness: int = 2) -> None:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(frame, caption, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
