# M2: Priority scheduling

Status 2026-11-15: built, tested (88 tests), and benchmarked on real Ollama. In Phase 1's head-of-line scenario, interactive time-to-first-token went from **39.5 s (Ollama direct) to 0.49 s**, with background work taking ≈3% longer.

## Purpose

Phase 1 (E1b) showed an interactive chat request waiting **31–35 s** for its first token behind four background requests to the same model, because Ollama's queue is first-come-first-served. M2 lets interactive work go first without letting background work starve.

## What was built

| Piece | Where | What it does |
|---|---|---|
| Strict priority policy | `core/policies.py` `strict_priority` | Interactive → normal → background; oldest first within a class |
| Priority with aging (default) | `core/policies.py` `make_aging` | A waiting job moves up one class per `promote_after_s` (default 15 s) |
| Preemption (cancel and requeue) | `core/scheduler.py` `_maybe_preempt`, `requeue`; `api/app.py` `run_preemptible_chat` | An interactive arrival cancels one background non-streaming chat job on the same model; that job reruns later with its original place in line |
| Embedding chunking | `api/app.py` `/v1/embeddings` | Inputs above 32 run as separate backend calls; the job yields its slot between chunks |
| Wait explanations | `core/scheduler.py` `_tick` | Every dispatch reason names the jobs that held the slot, with labels such as "chunk 3/60" and "[preempted]" |
| CLI flags | `cli.py` | `--policy {fcfs,priority,aging}`, `--promote-after`, `--no-preempt`, `--max-preemptions`, `--embed-chunk` |

Defaults are aging + preemption + chunks of 32. `--policy fcfs --no-preempt` reproduces M1.

## The computer science

**Starvation and aging.** Strict priority is optimal for interactive latency and unbounded for everyone else: under sustained interactive load, background never runs. Aging is the classic fix (Unix schedulers raise a waiting process's priority over time). Here, a job's *effective level* is its class level minus one per `promote_after_s` waited, floored at 0. A background job reaches level 0 after 2 × 15 s = 30 s. From then on it is ordered by enqueue time among level-0 jobs, so it beats every interactive job that arrived after it.

**The bound.** Its wait is at most about 30 s, plus the level-0 work that was queued ahead of it, plus the request in progress. `tests/test_policies.py` checks this on a deterministic overload simulation: one slot, 4 s jobs, interactive arrivals every 3 s (more than the slot can serve). Strict priority **never** starts the background job within 600 s. Aging starts it within 30 + 4 s.

**Preemption without kernel preemption.** GPU work can't be paused safely from outside, so preemption is *abort and restart* at the request level. Ollama stops generating when its connection closes (Phase 1, E5). Restarting throws away the background job's prefill and the tokens it had generated. That cost is logged per preemption (`job.preempted`: `wasted_tokens`, `wasted_s`), so the latency gain and the wasted work can be compared.

**Why only background non-streaming chat is preemptible.** A non-streaming client sees nothing until the end, so a rerun is invisible apart from the delay. A streaming client has already received tokens. Restarting would send a second, possibly different, answer. Streaming background requests therefore finish normally, and an interactive request waits at most one request for them.

**Preserving seniority vs. fresh enqueue.** A preempted job re-queues with its **original** enqueue time: it was interrupted, so it shouldn't lose its place or its aging. An embedding chunk re-queues with a **fresh** enqueue time. Otherwise a long indexing job would age up to level 0 across all its chunks and block search queries again. Each chunk ages on its own.

**Victim choice.** Among eligible jobs, preempt the one that started most recently: least work thrown away. Each job can be preempted at most `max_preemptions` times (default 2). After that it runs to completion, which prevents a background job from being restarted forever (livelock).

## Failure cases handled

- **Preemption requested just as the victim finishes.** `task.cancel()` on a finished task does nothing; the job completes normally.
- **The victim's client disconnects while it is re-queued.** It leaves the queue like any other queued job.
- **The handler itself is cancelled during a preempted call.** This is told apart from preemption with `asyncio.current_task().cancelling()`, so it is recorded as a cancellation, not a requeue.
- **A backend returns the wrong number of embeddings for a chunk.** The request fails with 502 instead of returning misaligned vectors.

## Tests

`pytest -q`: 88 passed. The M2-specific ones:
- **`test_policies.py`:** ordering for each policy, the effective-level table, aged background vs. newer and older interactive jobs, and the starvation and aging-bound simulation.
- **`test_scheduler.py`:**
  - preemption happens and the victim keeps its place in line;
  - four cases where it must *not* happen: not preemptible, normal priority, preemption off, limit reached;
  - only interactive arrivals preempt;
  - aging inside the live scheduler;
  - explanations name every blocker, including when the wait was under 1 ms (a bug found while building preemption and fixed in its own commit).
- **`test_gateway.py`** (fake Ollama on a real HTTP server):
  - an interactive request preempts a background non-streaming chat: queue wait **1 ms**; the background job still returns its complete answer from the rerun; the backend saw exactly 3 requests;
  - a streaming background request is *not* preempted;
  - a 70-input embedding request is split 32/32/6 and reassembled in order;
  - an interactive embedding waits **63 ms** behind a 10-chunk background job, instead of the ~850 ms that job had left.
- **`test_cli.py`:** defaults and flags.

## Benchmark on real Ollama

`bench/m2_queue.py`, run 2026-11-15. RTX 2050 4 GB, Ollama 0.40.15, qwen3.5:4b already loaded, GPU otherwise idle. Phase 1's E1b scenario:
- Per trial: 4 background **non-streaming** chat requests (64 tokens each) are sent at once.
- 0.5 s later, an interactive **streaming** request (8 tokens) arrives.
- 3 trials per configuration.

Raw data: `bench/results/m2_queue.jsonl`, the gateway event logs `m2_events_*.jsonl`, and `m2_stdout.txt`.

| Configuration | Interactive time to first token (3 trials) | Median | Background batch, median | Preemptions / tokens thrown away (all trials) |
|---|---|---|---|---|
| Ollama direct | 41.87, 39.53, 39.34 s | 39.5 s | 41.3 s | — |
| Gateway, FCFS, no preemption | 39.01, 39.68, 40.94 s | 39.7 s | 41.5 s | 0 / 0 |
| Gateway, aging, no preemption | 9.47, 9.63, 9.58 s | 9.6 s | 41.6 s | 0 / 0 |
| **Gateway, aging + preemption** (default) | **0.50, 0.48, 0.49 s** | **0.49 s** | 42.4 s | 3 / 12 |

**Reading it:**
1. **The gateway in FCFS mode behaves like Ollama** (39.7 vs 39.5 s). The gateway itself isn't the effect; the policy is.
2. **Priority alone cuts the wait about 4×** (39.5 → 9.6 s). The interactive request still waits for the one background request already running, about 9.5 s here. That matches the bound: one request in progress.
3. **Priority plus preemption cuts it about 80×** (39.5 → 0.49 s). The interactive request alone takes ~0.24 s to first token (Phase 1, E1b), so preemption adds ~0.25 s for cancelling the background generation and starting the new prefill.
4. **The cost to background work was small in this scenario:** batch time 41.3 → 42.4 s (+1.1 s, ≈3%), with 4 tokens thrown away per preemption. All 12 background requests completed with HTTP 200.

**Limits of this result.**
- **The interactive request always arrived ~0.65 s after the batch started**, when the job to preempt had generated only ~4 tokens. An interactive request arriving later in a long background generation would throw away more work. Wasted work grows with how far along the victim is; this benchmark shows only the cheap end.
- **n = 3 per configuration.** The spread is small (≤ 2.6 s for the slow rows, ≤ 0.02 s for preemption), but these aren't confidence intervals.
- **Ollama direct took longer today** (39.5 s) than in Phase 1 (31–35 s), with ~10 s per background request instead of ~8 s. Same model and prompt; the Ollama version changed from 0.40.14 to 0.40.15, and laptop thermal state varies. The comparison here holds because all four configurations ran back to back in one session.
- **Only Qwen chat was benchmarked.** The embedding-chunking gain (63 ms vs ~850 ms) comes from the fake-backend test, not from real bge-m3.

## Not done in M2

- Preempting streaming requests.
- Priority from API keys: the `X-LRM-Priority` header is trusted from any local program, so priority is cooperative, not enforced.
- Weighted fair queuing (simulator only).
- Memory awareness (M3).
- Tuning `promote_after_s` from data: 15 s is a starting value.
