"""Start the gateway:  python -m lrm --port 8080 --events events.jsonl"""
from __future__ import annotations

import argparse

import uvicorn

from .adapters.ollama import OllamaAdapter
from .api.app import create_app
from .core.events import EventLog
from .core.job import model_key
from .core.policies import by_name
from .core.scheduler import Scheduler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="lrm", description="Local AI Resource Manager gateway")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--ollama", default="http://127.0.0.1:11434", help="Ollama base URL")
    p.add_argument("--events", default="lrm-events.jsonl", help="event log file ('' to disable)")
    p.add_argument("--default-limit", type=int, default=1,
                   help="max in-flight requests per model (default 1)")
    p.add_argument("--model-limit", action="append", default=[], metavar="MODEL=N",
                   help="per-model in-flight limit, e.g. bge-m3=2 (repeatable)")
    p.add_argument("--policy", choices=["fcfs", "priority", "aging"], default="aging",
                   help="scheduling policy (default aging = priority with aging)")
    p.add_argument("--promote-after", type=float, default=15.0, metavar="SECONDS",
                   help="aging: a waiting job moves up one class per this many seconds (default 15)")
    p.add_argument("--no-preempt", dest="preempt", action="store_false",
                   help="never cancel-and-requeue background non-streaming chat")
    p.add_argument("--max-preemptions", type=int, default=2,
                   help="max times one job can be preempted (default 2)")
    p.add_argument("--embed-chunk", type=int, default=32,
                   help="max inputs per backend embedding call; 0 = never split (default 32)")
    a = p.parse_args(argv)
    limits = {}
    for spec in a.model_limit:
        name, _, n = spec.partition("=")
        limits[model_key(name)] = int(n)
    a.model_limits = limits
    return a


def build(a: argparse.Namespace):
    events = EventLog(a.events or None)
    sched = Scheduler(events, default_limit=a.default_limit, model_limits=a.model_limits,
                      policy=by_name(a.policy, a.promote_after), preempt=a.preempt,
                      max_preemptions=a.max_preemptions)
    return create_app(OllamaAdapter(a.ollama), sched, events, embed_chunk=a.embed_chunk)


def main(argv: list[str] | None = None) -> None:
    a = parse_args(argv)
    uvicorn.run(build(a), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
