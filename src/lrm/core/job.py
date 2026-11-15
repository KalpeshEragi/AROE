from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class Priority(str, Enum):
    INTERACTIVE = "interactive"
    NORMAL = "normal"
    BACKGROUND = "background"

    @classmethod
    def parse(cls, value: str | None) -> "Priority":
        if not value:
            return cls.NORMAL
        try:
            return cls(value.strip().lower())
        except ValueError:
            raise ValueError(f"unknown priority {value!r}; use interactive, normal or background")


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def model_key(name: str) -> str:
    """Ollama treats 'bge-m3' and 'bge-m3:latest' as the same model."""
    return name if ":" in name else f"{name}:latest"


_ids = itertools.count(1)


@dataclass
class Job:
    kind: str                 # "chat" or "embed"
    model: str
    priority: Priority
    arrival_t: float          # scheduler clock (monotonic seconds); never changes
    id: str = field(default_factory=lambda: f"j{next(_ids)}")
    # When this job last joined the queue. Policies order and age by this. Normally equal
    # to the latest acquire() time; a job that was preempted keeps its old value so it
    # does not lose its place.
    enqueue_t: float | None = None
    label: str = ""                       # e.g. "chunk 3/60", shown in explanations
    blocked_by: list[str] = field(default_factory=list)  # jobs that held the slot while this one waited
    # Preemption (cancel and requeue). The gateway sets `preemptible` and, while the job
    # runs, `on_preempt` (cancels its backend call). The scheduler only decides.
    preemptible: bool = False
    preemptions: int = 0
    preempt_requested: bool = False
    on_preempt: Callable[[], object] | None = field(default=None, repr=False)

    def describe(self) -> str:
        extra = f", {self.label}" if self.label else ""
        return f"{self.id} ({self.priority.value} {self.kind}{extra})"
    state: JobState = JobState.QUEUED
    dispatch_t: float | None = None
    first_token_t: float | None = None
    end_t: float | None = None

    @property
    def key(self) -> str:
        return model_key(self.model)

    @property
    def wait_s(self) -> float | None:
        return None if self.dispatch_t is None else self.dispatch_t - self.arrival_t
