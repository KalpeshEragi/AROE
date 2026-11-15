"""Scheduling policies: pure functions, no I/O, no clock. The simulator will call these too.

A policy gets the queue (in arrival order) and a predicate telling whether a job could
start right now (its model has a free slot), and returns the job to start, or None.
"""
from __future__ import annotations

from typing import Callable, Sequence

from .job import Job

Policy = Callable[[Sequence[Job], Callable[[Job], bool]], "Job | None"]


def fcfs(queue: Sequence[Job], can_run: Callable[[Job], bool]) -> Job | None:
    """First come, first served *per model*.

    The oldest job whose model has a free slot goes first. A busy model does not block
    jobs for other models; that matches how Ollama behaves on its own (separate runner
    per model), so this is the fair baseline to compare M2's priority policies against.
    """
    for job in queue:
        if can_run(job):
            return job
    return None
