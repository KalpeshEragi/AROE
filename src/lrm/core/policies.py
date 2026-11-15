"""Scheduling policies: pure functions, no I/O. The simulator will call these too.

A policy gets the queue, a predicate telling whether a job could start right now (its
model has a free slot) and the current time, and returns the job to start, or None.
Every policy orders by `enqueue_t` within a level, so ties go to whoever waited longest.
"""
from __future__ import annotations

from typing import Callable, Sequence

from .job import Job, Priority

Policy = Callable[[Sequence[Job], Callable[[Job], bool], float], "Job | None"]

LEVEL = {Priority.INTERACTIVE: 0, Priority.NORMAL: 1, Priority.BACKGROUND: 2}


def _first_runnable(jobs, can_run):
    for job in jobs:
        if can_run(job):
            return job
    return None


def fcfs(queue: Sequence[Job], can_run: Callable[[Job], bool], now: float) -> Job | None:
    """First come, first served *per model*.

    The oldest job whose model has a free slot goes first. A busy model does not block
    jobs for other models; that matches how Ollama behaves on its own (separate runner
    per model), so this is the baseline the priority policies are compared against.
    """
    return _first_runnable(sorted(queue, key=lambda j: j.enqueue_t), can_run)


def strict_priority(queue: Sequence[Job], can_run: Callable[[Job], bool], now: float) -> Job | None:
    """Lowest level first, oldest first within a level. Background can starve."""
    return _first_runnable(sorted(queue, key=lambda j: (LEVEL[j.priority], j.enqueue_t)), can_run)


def effective_level(job: Job, now: float, promote_after_s: float) -> int:
    """Class level minus one for every `promote_after_s` waited, floored at 0."""
    waited = max(0.0, now - job.enqueue_t)
    return max(0, LEVEL[job.priority] - int(waited // promote_after_s))


def make_aging(promote_after_s: float = 15.0) -> Policy:
    """Priority with aging.

    A waiting job moves up one level per `promote_after_s`. Once a background job reaches
    level 0 it competes with interactive jobs by enqueue time, so it beats every
    interactive job that arrived after it. Its wait is therefore bounded by about
    2 * promote_after_s plus the work queued ahead of it at level 0 (no starvation).
    """
    if promote_after_s <= 0:
        raise ValueError("promote_after_s must be positive")

    def priority_aging(queue, can_run, now):
        return _first_runnable(
            sorted(queue, key=lambda j: (effective_level(j, now, promote_after_s), j.enqueue_t)), can_run)

    priority_aging.promote_after_s = promote_after_s
    return priority_aging


def by_name(name: str, promote_after_s: float = 15.0) -> Policy:
    if name == "fcfs":
        return fcfs
    if name == "priority":
        return strict_priority
    if name == "aging":
        return make_aging(promote_after_s)
    raise ValueError(f"unknown policy {name!r}; use fcfs, priority or aging")
