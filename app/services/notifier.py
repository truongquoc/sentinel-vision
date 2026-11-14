"""Webhook delivery off the video path, with bounded retries.

Delivery runs on its own thread so a slow or dead receiver never delays detection.
The queue is bounded: if the receiver is down for a long time, the oldest pending
notifications are dropped (and logged) rather than growing memory without limit.
"""
import hashlib
import hmac
import json
import queue
import threading
from typing import Any

import httpx

from app.core.logging import get_logger

log = get_logger(__name__)


def sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


class WebhookNotifier:
    def __init__(self, url: str, secret: str, max_attempts: int = 4, base_delay: float = 1.0, queue_size: int = 500,
                 transport: httpx.BaseTransport | None = None) -> None:
        self.url = url
        self.secret = secret
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self._transport = transport
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.delivered = 0
        self.failed = 0

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def start(self) -> None:
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._run, name="webhook-notifier", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def enqueue(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            dropped = self._queue.get_nowait()
            self.failed += 1
            log.error(f"Webhook queue full, dropped event '{dropped['id']}'", event_context={"uuid": dropped["id"], "type": dropped["event_type"]})
            self._queue.put_nowait(event)

    def _run(self) -> None:
        with httpx.Client(timeout=5, transport=self._transport) as client:
            while not self._stop.is_set():
                try:
                    event = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                self._deliver(client, event)

    def _deliver(self, client: httpx.Client, event: dict[str, Any]) -> None:
        body = json.dumps(event, default=str).encode()
        headers = {"Content-Type": "application/json"}
        if self.secret:
            headers["X-Sentinel-Signature"] = sign(self.secret, body)
        context = {"uuid": event["id"], "type": event["event_type"]}
        for attempt in range(1, self.max_attempts + 1):
            error = self._post(client, body, headers)
            if error is None:
                self.delivered += 1
                return
            log.warning(f"Webhook attempt {attempt}/{self.max_attempts} failed for event '{event['id']}'", event_context=context, error=error)
            if self._stop.wait(self.base_delay * 2 ** (attempt - 1)):
                return
        self.failed += 1
        log.error(f"Webhook gave up on event '{event['id']}'", event_context=context)

    def _post(self, client: httpx.Client, body: bytes, headers: dict[str, str]) -> str | None:
        try:
            response = client.post(self.url, content=body, headers=headers)
        except httpx.HTTPError as exc:
            return f"{type(exc).__name__}: {exc}"
        if response.status_code >= 400:
            return f"HTTP {response.status_code}"
        return None
