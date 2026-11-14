"""RTSP ingestion. One thread per camera keeps only the newest frame.

Surveillance prefers a fresh frame over a complete one, so frames are overwritten, never queued.
Both opening and reading have timeouts: the most common camera failure is a stream that stops
sending without closing the socket, and a blocking read would hang forever on it.
"""
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

import cv2
import numpy as np

from app.core.logging import get_logger

log = get_logger(__name__)

MAX_BACKOFF_SECONDS = 30.0


@dataclass(frozen=True)
class Frame:
    seq: int
    image: np.ndarray
    captured_at: float


def redact(url: str) -> str:
    """Strip credentials so a camera URL can be logged or returned by the API."""
    parts = urlsplit(url)
    if not parts.password and not parts.username:
        return url
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    return parts._replace(netloc=f"***@{host}{port}").geturl()


def is_network_source(url: str) -> bool:
    return url.split("://", 1)[0].lower() in {"rtsp", "rtsps", "http", "https", "rtmp"}


class CameraReader:
    def __init__(self, camera_id: str, url: str, timeout_seconds: float = 8.0) -> None:
        self.camera_id = camera_id
        self.url = url
        self.timeout_ms = int(timeout_seconds * 1000)
        self.online = False
        self.last_error: str | None = None
        self.last_frame_at: float | None = None
        self.frame_size: tuple[int, int] | None = None  # width, height
        self._latest: Frame | None = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"camera-{self.camera_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def latest(self) -> Frame | None:
        return self._latest

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            if not self.url:
                self._mark_offline("No camera URL configured")
                self._stop.wait(10)
                continue
            capture = self._open()
            if capture is None:
                self._stop.wait(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
                continue
            backoff = 1.0
            self._read_until_failure(capture)
            capture.release()

    def _open(self) -> cv2.VideoCapture | None:
        params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.timeout_ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.timeout_ms]
        capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, params)
        if capture.isOpened():
            log.info(f"Camera '{self.camera_id}' stream opened", camera_id=self.camera_id, source=redact(self.url))
            return capture
        capture.release()
        self._mark_offline("Unable to open stream (check reachability, credentials, RTSP enabled, and path)")
        return None

    def _read_until_failure(self, capture: cv2.VideoCapture) -> None:
        # Local files are for testing: replay them at their native rate instead of as fast as possible.
        file_delay = 0.0 if is_network_source(self.url) else 1 / (capture.get(cv2.CAP_PROP_FPS) or 15)
        while not self._stop.is_set():
            ok, image = capture.read()
            if not ok:
                self._mark_offline("Stream stalled or closed; reconnecting")
                return
            self._seq += 1
            now = time.time()
            self._latest = Frame(self._seq, image, now)
            self.frame_size = (image.shape[1], image.shape[0])
            self.last_frame_at = now
            if not self.online:
                log.info(f"Camera '{self.camera_id}' receiving frames", camera_id=self.camera_id, frame_size=self.frame_size)
            self.online = True
            self.last_error = None
            if file_delay:
                time.sleep(file_delay)

    def _mark_offline(self, reason: str) -> None:
        if self.online or self.last_error != reason:
            log.warning(f"Camera '{self.camera_id}' offline: {reason}", camera_id=self.camera_id, source=redact(self.url))
        self.online = False
        self.last_error = reason
