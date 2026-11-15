"""The only place requests wait.

Callers `await acquire(job)` to get a dispatch slot and must call `release(job, ...)`
when the backend work ends. Each model has an in-flight limit (default 1, matching
Ollama 0.40.14 serving Qwen3.5 one request at a time), so the backend's own queue stays
empty and the order chosen here is the order that actually runs.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from .events import EventLog
from .job import Job, JobState
from .policies import Policy, fcfs


class Scheduler:
    def __init__(self, events: EventLog, *, default_limit: int = 1,
                 model_limits: dict[str, int] | None = None, policy: Policy = fcfs,
                 now: Callable[[], float] = time.monotonic):
        self.events = events
        self.default_limit = default_limit
        self.model_limits = dict(model_limits or {})
        self.policy = policy
        self.now = now
        self._queue: list[Job] = []
        self._waiters: dict[str, asyncio.Future[None]] = {}
        self._running: dict[str, list[Job]] = {}

    # ------------------------------------------------------------------ public
    def new_job(self, kind: str, model: str, priority) -> Job:
        return Job(kind=kind, model=model, priority=priority, arrival_t=self.now())

    def limit(self, key: str) -> int:
        return self.model_limits.get(key, self.default_limit)

    async def acquire(self, job: Job, keep_seniority: bool = False) -> None:
        """Wait for a dispatch slot. `keep_seniority=True` re-queues the job with its
        previous enqueue time (used after preemption)."""
        if not keep_seniority or job.enqueue_t is None:
            job.enqueue_t = self.now()
        job.state = JobState.QUEUED
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters[job.id] = fut
        self._queue.append(job)
        running = self._running.get(job.key, [])
        job.blocked_by = [r.describe() for r in running]
        self.events.emit("job.arrived", job.id, kind=job.kind, model=job.key,
                         priority=job.priority.value, label=job.label or None,
                         queued_ahead=len(self._queue) - 1, running_on_model=[j.id for j in running])
        self._tick()
        try:
            await fut
        except asyncio.CancelledError:
            # Two cases. (1) Still queued: just leave the queue. (2) The slot was granted
            # in the same loop iteration the caller got cancelled, so `await fut` raised
            # even though we hold a slot. Forgetting case 2 leaks the slot forever.
            if job.state is JobState.QUEUED:
                self._remove(job)
                job.state = JobState.CANCELLED
                job.end_t = self.now()
                self.events.emit("job.cancelled", job.id, stage="queued",
                                 waited_s=round(job.end_t - job.arrival_t, 4))
                self._tick()
            else:
                self.release(job, JobState.CANCELLED, stage="dispatched")
            raise
        finally:
            self._waiters.pop(job.id, None)

    def release(self, job: Job, outcome: JobState, **info: Any) -> None:
        """Idempotent: only the first call for a running job has an effect."""
        if job.state is not JobState.RUNNING:
            return
        self._running[job.key].remove(job)
        job.state = outcome
        job.end_t = self.now()
        fields = {"run_s": round(job.end_t - job.dispatch_t, 4)}
        if job.first_token_t is not None:
            fields["ttft_s"] = round(job.first_token_t - job.arrival_t, 4)
        fields.update(info)
        self.events.emit(f"job.{outcome.value}", job.id, **fields)
        self._tick()

    def mark_first_token(self, job: Job) -> None:
        if job.first_token_t is None:
            job.first_token_t = self.now()
            self.events.emit("job.first_token", job.id,
                             ttft_s=round(job.first_token_t - job.arrival_t, 4))

    def status(self) -> dict[str, Any]:
        now = self.now()
        return {
            "policy": getattr(self.policy, "__name__", "custom"),
            "queued": [{"id": j.id, "model": j.key, "priority": j.priority.value,
                        "kind": j.kind, "waiting_s": round(now - j.enqueue_t, 3)} for j in self._queue],
            "running": [{"id": j.id, "model": j.key, "priority": j.priority.value, "kind": j.kind,
                         "running_s": round(now - j.dispatch_t, 3)}
                        for jobs in self._running.values() for j in jobs],
            "limits": {"default": self.default_limit, **self.model_limits},
        }

    # ------------------------------------------------------------------ internals
    def _can_run(self, job: Job) -> bool:
        return len(self._running.get(job.key, [])) < self.limit(job.key)

    def _remove(self, job: Job) -> None:
        try:
            self._queue.remove(job)
        except ValueError:
            pass

    def _tick(self) -> None:
        """Start every job the policy allows. Called on arrival, release and cancel."""
        while True:
            job = self.policy(self._queue, self._can_run, self.now())
            if job is None:
                return
            self._remove(job)
            self._running.setdefault(job.key, []).append(job)
            job.state = JobState.RUNNING
            job.dispatch_t = self.now()
            wait = job.dispatch_t - job.enqueue_t
            for q in self._queue:  # everyone still waiting for this model now waits behind `job`
                if q.key == job.key and job.describe() not in q.blocked_by:
                    q.blocked_by.append(job.describe())
            if not job.blocked_by:
                reason = "slot free on arrival"
            else:
                held = ", ".join(job.blocked_by) or "other jobs"
                reason = (f"waited {wait:.2f}s for {job.key} (limit {self.limit(job.key)}); "
                          f"slot held by {held}")
            self.events.emit("job.dispatched", job.id, wait_s=round(wait, 4),
                             policy=getattr(self.policy, "__name__", "custom"), reason=reason)
            fut = self._waiters.get(job.id)
            if fut is not None and not fut.done():
                fut.set_result(None)
