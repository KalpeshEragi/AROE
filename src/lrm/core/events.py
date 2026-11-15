"""Append-only event log. One record of what happened serves explanations, status and replay."""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable


class EventLog:
    def __init__(self, path: str | Path | None = None, keep: int = 2000,
                 now: Callable[[], float] = time.monotonic):
        self._now = now
        self._recent: deque[dict[str, Any]] = deque(maxlen=keep)
        self._file = open(path, "a", encoding="utf-8") if path else None

    def emit(self, type_: str, job_id: str | None = None, **fields: Any) -> dict[str, Any]:
        ev = {"t": round(self._now(), 6), "wall": round(time.time(), 3), "type": type_}
        if job_id is not None:
            ev["job"] = job_id
        ev.update(fields)
        self._recent.append(ev)
        if self._file:
            # Synchronous write + flush: a few microseconds on a local disk. Fine at this
            # request rate; revisit if the log ever shows up in the overhead benchmark.
            self._file.write(json.dumps(ev) + "\n")
            self._file.flush()
        return ev

    def recent(self, n: int = 200, job_id: str | None = None) -> list[dict[str, Any]]:
        evs = [e for e in self._recent if job_id is None or e.get("job") == job_id]
        return evs[-n:]

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
