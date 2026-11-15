from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum


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
    arrival_t: float          # scheduler clock (monotonic seconds)
    id: str = field(default_factory=lambda: f"j{next(_ids)}")
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
