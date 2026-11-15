"""Start the gateway:  python -m lrm --port 8080 --events events.jsonl"""
from __future__ import annotations

import argparse

import uvicorn

from .adapters.ollama import OllamaAdapter
from .api.app import create_app
from .core.events import EventLog
from .core.job import model_key
from .core.scheduler import Scheduler


def build(ollama_url: str, events_path: str | None, default_limit: int, model_limits: dict[str, int]):
    events = EventLog(events_path)
    sched = Scheduler(events, default_limit=default_limit, model_limits=model_limits)
    return create_app(OllamaAdapter(ollama_url), sched, events)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="lrm", description="Local AI Resource Manager gateway")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--ollama", default="http://127.0.0.1:11434", help="Ollama base URL")
    p.add_argument("--events", default="lrm-events.jsonl", help="event log file ('' to disable)")
    p.add_argument("--default-limit", type=int, default=1,
                   help="max in-flight requests per model (default 1)")
    p.add_argument("--model-limit", action="append", default=[], metavar="MODEL=N",
                   help="per-model in-flight limit, e.g. bge-m3=2 (repeatable)")
    a = p.parse_args(argv)
    limits = {}
    for spec in a.model_limit:
        name, _, n = spec.partition("=")
        limits[model_key(name)] = int(n)
    app = build(a.ollama, a.events or None, a.default_limit, limits)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
