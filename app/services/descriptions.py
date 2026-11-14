"""Runs the vision-language model off the video path and attaches results to events.

A VLM takes seconds per image, far slower than detection, so jobs go through a small queue
served by one worker thread:

- Events (enter / loiter / exit) are saved immediately with status "pending" and updated
  when the description arrives.
- Their webhook waits for the description, but never longer than `notify_wait_seconds`;
  after that it is sent without one, so a slow or missing model never silences an alert.
- Periodic "activity" jobs for the same object coalesce: if a newer one arrives before the
  older ran, the older is dropped. Only the latest view of what someone is doing matters.
"""
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

from app.core.logging import get_logger
from app.pipeline.describer import Describer, DescriberError, DescriptionRequest
from app.services.notifier import WebhookNotifier
from app.services.store import Store

log = get_logger(__name__)

MAX_REMEMBERED_TRACKS = 1000


@dataclass
class _Job:
    event: dict[str, Any]
    request: DescriptionRequest
    submitted_at: float
    track_key: tuple[str, str, int]
    notified: bool = False

    @property
    def periodic(self) -> bool:
        return self.event["event_type"] == "activity"


class DescriptionService:
    def __init__(self, describer: Describer, store: Store, notifier: WebhookNotifier, notify_wait_seconds: float = 20.0,
                 max_pending: int = 32, clock: Callable[[], float] = time.monotonic) -> None:
        self.describer = describer
        self.store = store
        self.notifier = notifier
        self.notify_wait_seconds = notify_wait_seconds
        self.max_pending = max_pending
        self._clock = clock
        self._pending: OrderedDict[str, _Job] = OrderedDict()
        self._in_flight: _Job | None = None
        self._last_summary: OrderedDict[tuple[str, str, int], str] = OrderedDict()
        self._cond = threading.Condition()
        self._notify_lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.completed = 0
        self.failed = 0
        self.dropped = 0

    @property
    def enabled(self) -> bool:
        return self.describer.enabled

    def start(self) -> None:
        if not self.enabled:
            return
        for target, name in ((self._work_loop, "describer"), (self._deadline_loop, "describer-deadline")):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()

    def status(self) -> dict[str, Any]:
        with self._cond:
            pending = len(self._pending) + (1 if self._in_flight else 0)
        return self.describer.status() | {"pending": pending, "completed": self.completed, "failed": self.failed, "dropped": self.dropped}

    def previous_summary(self, track_key: tuple[str, str, int]) -> str | None:
        return self._last_summary.get(track_key)

    def _remember_summary(self, track_key: tuple[str, str, int], summary: str) -> None:
        self._last_summary[track_key] = summary
        self._last_summary.move_to_end(track_key)
        while len(self._last_summary) > MAX_REMEMBERED_TRACKS:
            self._last_summary.popitem(last=False)

    # submission ---------------------------------------------------------------
    def submit(self, event: dict[str, Any], request: DescriptionRequest, track_key: tuple[str, str, int], notify: bool = True) -> None:
        """Queue a description. The caller must already have stored the event with status 'pending'.

        notify=False is for re-describing an old event: the alert for it went out long ago.
        """
        job = _Job(event, request, self._clock(), track_key, notified=not notify)
        queue_key = f"activity:{track_key}" if job.periodic else event["id"]
        with self._cond:
            superseded = self._pending.pop(queue_key, None)
            self._pending[queue_key] = job
            overflow = self._take_overflow()
            self._cond.notify_all()
        if superseded is not None:
            self._mark_skipped(superseded, "superseded by a newer activity description")
        for dropped in overflow:
            self._mark_skipped(dropped, "description queue full")

    def _take_overflow(self) -> list[_Job]:
        dropped = []
        while len(self._pending) > self.max_pending:
            key = next((k for k, j in self._pending.items() if j.periodic), next(iter(self._pending)))
            dropped.append(self._pending.pop(key))
        return dropped

    def _mark_skipped(self, job: _Job, reason: str) -> None:
        self.dropped += 1
        self.store.set_description(job.event["id"], "skipped", {"error": reason})
        self._notify(job, None)

    # worker -------------------------------------------------------------------
    def _work_loop(self) -> None:
        while not self._stop.is_set():
            self.process_next()

    def process_next(self, wait_seconds: float = 1.0) -> bool:
        """Describe the oldest queued job. Returns False when there was nothing to do."""
        job = self._next_job(wait_seconds)
        if job is None:
            return False
        self._run_job(job)
        with self._cond:
            self._in_flight = None
        return True

    def _next_job(self, wait_seconds: float) -> _Job | None:
        with self._cond:
            if not self._pending and wait_seconds > 0:
                self._cond.wait(timeout=wait_seconds)
            if not self._pending or self._stop.is_set():
                return None
            _, job = self._pending.popitem(last=False)
            self._in_flight = job
            return job

    def _run_job(self, job: _Job) -> None:
        event_id = job.event["id"]
        try:
            description = self.describer.describe(job.request)
        except DescriberError as exc:
            self._record_failure(job, str(exc))
            return
        except Exception as exc:  # never let one bad response kill the worker
            log.exception(f"Unexpected describer failure for event '{event_id}'", event_context={"uuid": event_id})
            self._record_failure(job, f"{type(exc).__name__}: {exc}")
            return
        payload = description.to_dict()
        self.store.set_description(event_id, "done", payload)
        self._remember_summary(job.track_key, description.summary)
        self.completed += 1
        log.info(
            f"Described {job.event['label']} #{job.event['track_id']}: {description.summary}",
            event_context={"uuid": event_id, "type": job.event["event_type"]},
            concern=description.concern,
            latency_ms=description.latency_ms,
        )
        self._notify(job, payload)

    def _record_failure(self, job: _Job, error: str) -> None:
        self.failed += 1
        self.store.set_description(job.event["id"], "failed", {"error": error})
        log.warning(f"Description failed for event '{job.event['id']}': {error}", event_context={"uuid": job.event["id"]})
        self._notify(job, None)

    # notification gating ------------------------------------------------------
    def _deadline_loop(self) -> None:
        while not self._stop.wait(0.5):
            self.flush_overdue()

    def flush_overdue(self) -> None:
        """Send webhooks for events whose description is taking too long."""
        now = self._clock()
        with self._cond:
            jobs = list(self._pending.values()) + ([self._in_flight] if self._in_flight else [])
        for job in jobs:
            if job.periodic or job.notified or now - job.submitted_at < self.notify_wait_seconds:
                continue
            self._notify(job, None)

    def _notify(self, job: _Job, description: dict[str, Any] | None) -> None:
        with self._notify_lock:  # worker and deadline threads may race for the same job
            if job.notified:
                return
            job.notified = True
        if job.periodic and description is None:
            return  # an activity update without a description carries no news
        self.notifier.enqueue(job.event | {"description": description})
