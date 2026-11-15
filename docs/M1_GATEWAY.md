# M1: Gateway skeleton

Status 2026-11-08: built and tested; **both exit criteria met.**
- Streaming and disconnect handling are tested end to end over real sockets.
- Proxy overhead ≤5 ms (S4): **1.0 ms** non-streaming and **1.3 ms** streaming TTFT against an instant backend over keep-alive connections; 2.7–2.8 ms with fresh connections. Against real Ollama, the overhead can't be distinguished from Ollama's own variation (details below).

The first version (httpx client) missed S4 at 8–14 ms. Switching the backend client to aiohttp fixed it; see "Overhead".

## What it is

An OpenAI-compatible HTTP gateway in front of Ollama. It is **the only place requests wait**. Each model gets an in-flight limit (default 1), so Ollama's own queue stays empty and the order chosen here is the order that runs. M1 uses first-come-first-served per model. M2 adds priority classes behind the same interface.

```
client --(OpenAI API)--> gateway: parse -> Job -> Scheduler.acquire --(slot)--> OllamaAdapter --(native API, aiohttp)--> Ollama
                                   \______________ EventLog (every step) ______________/
```

### Run it

```bash
.venv\Scripts\python -m lrm --port 8080 --events lrm-events.jsonl
```

Point any OpenAI client at `http://127.0.0.1:8080/v1`. Optional header: `X-LRM-Priority: interactive|normal|background`. M1 records it but does not act on it yet.

| Route | Purpose |
|---|---|
| `POST /v1/chat/completions` | Chat, streaming (SSE) or not. Text only. Extensions: `think`, `keep_alive`, `options` (raw Ollama options) |
| `POST /v1/embeddings` | Embeddings, float output |
| `GET /v1/models` | Ollama's installed models |
| `GET /lrm/status` | Queued and running jobs with wait times; per-model limits |
| `GET /lrm/events?job=j12&n=200` | Recent events, optionally for one job |

Response headers: `X-LRM-Job-Id`, `X-LRM-Queue-Wait-Ms`.

Rejected with a clear 400 (not supported yet): images or multi-part content, tools, `n > 1`, base64 embeddings, token-array inputs.

**Dependencies:** fastapi, uvicorn, aiohttp. Dev only: pytest, httpx (test client).

## Design decisions

1. **Ollama's native API instead of its OpenAI endpoint.** Only the native API accepts `keep_alive` and `num_ctx` and returns load, prefill and decode timings. The scheduler (M2) and memory planner (M3) need all three. The cost is translation code (`api/openai_compat.py`, pure functions, unit-tested).
2. **aiohttp for backend calls**, chosen by measurement (see "Overhead"). The gateway sees only the adapter's own `BackendError` and `ChatStream` (`lines()`, `aclose()`), never aiohttp types, so swapping the library again would touch one file.
3. **FCFS *per model*, not one global line.** If Qwen is busy, an embedding request for bge-m3 starts right away. That's how Ollama behaves on its own (one runner per model), so it's the fair baseline for M2's policies.
4. **Policies are pure functions** (`core/policies.py`): queue plus a "can this job start?" check in, chosen job out. No I/O, no clock. The Phase 4 simulator will call the same functions.
5. **The scheduler takes an injected clock.** Tests use a fake clock, so wait-time assertions are exact, not timing-dependent.
6. **One event log** (`core/events.py`), append-only JSONL plus an in-memory buffer. It is both the explanation record and, later, the replay trace.

## Concurrency details (the parts most likely to break)

**1. A cancellation arriving at the same moment as the slot.** `acquire()` waits on a future. If the scheduler grants the slot (sets the future's result) and the waiting task is cancelled before it resumes, `await fut` raises `CancelledError` even though the job now holds a slot. If that case is ignored, the slot leaks and the model is blocked forever. `acquire()` checks the job's state in its `except` block and releases the slot. Test: `test_cancel_in_same_instant_as_grant_does_not_leak_slot`.

**2. Disconnects while waiting.** Starlette doesn't cancel a request handler when the client disconnects; it only notices when it tries to send. A request waiting in the queue, or during a 12 s model load, would otherwise keep its place and keep Ollama working for nobody. `until_done_or_disconnect()` waits on the real work and checks `request.is_disconnected()` every 250 ms, cancelling the work if the client is gone. Cost: none for fast requests (`asyncio.wait` returns as soon as the work finishes), and at most 250 ms of extra delay in noticing.

**3. Cleanup that survives cancellation.** Stream cleanup (close the upstream connection, release the slot) lives in `ManagedStream.__call__`'s `finally`, not inside the body generator, because the generator's `finally` never runs if the client leaves before the first chunk. It runs inside `anyio.CancelScope(shield=True)`. anyio cancellation is *level-triggered*: once a scope is cancelled, *every* later `await` inside it raises immediately. Without the shield, the upstream close would itself be cancelled and the Ollama connection would stay open.

**4. Really closing the backend connection.** With aiohttp, `response.release()` on a half-read response would not reliably close the socket. The adapter calls `close()` unless the body was fully read. Ollama only stops generating when its TCP connection closes (Phase 1, E5).

**5. Idempotent release.** Several paths can end a job (normal end, error, disconnect). `release()` acts only on the first call, so double-release bugs cannot corrupt the slot count.

## Tests (run 2026-11-08: 57 passed in 7.1 s)

```bash
.venv\Scripts\python -m pytest -q
```

- `test_scheduler.py`: FCFS order; other models not blocked; `bge-m3` and `bge-m3:latest` treated as one model; cancel while queued; the cancel/grant race; idempotent release; wait time and reason; and a **randomized invariant test** over 30 seeds × 120 random steps. Invariants checked: per-model limit never exceeded, no job dispatched twice, dispatch order matches arrival order within each model.
- `test_openai_compat.py`: option mapping, rejection of unsupported input, chunk, finish-reason and usage conversion, ns → s timings, embeddings.
- `test_gateway.py`: against a **fake Ollama running as a real HTTP server** in a thread. Covers non-streaming and streaming output in OpenAI format, a backend HTTP 500 → 502 with the backend's message and the slot freed (the E8 CUDA-crash case), backend unreachable → 502, an error inside the stream, 400s that create no job (and never reach the backend), and embeddings and models. The gateway also runs as a real uvicorn server for the disconnect tests:
  - client leaves mid-stream → the fake backend **observes its connection closing 3 ms later**, the stream does not run to the end, the slot is freed, and `job.cancelled` is logged with `stage=client_disconnected`;
  - client leaves while queued → removed from the queue, never sent to the backend, first request unaffected.

Not covered yet: tests against real Ollama in the suite (the overhead benchmark below exercised it), many concurrent clients, and gateway restart behavior.

## Overhead (exit criterion S4)

`bench/m1_overhead.py`: the same client code alternates DIRECT (client → backend) and GATEWAY (client → lrm → backend) request by request. The gateway and the fake backend run as separate processes. "Overhead" = median of per-pair differences [bootstrap 95% CI]. Raw data: `bench/results/m1_overhead_*.jsonl`.

| Backend | Client connections | Gateway client | Non-streaming | Streaming TTFT | n |
|---|---|---|---|---|---|
| Instant fake | keep-alive | httpx (first version) | 2.6 ms [2.58, 2.64] | 11.7 ms [11.3, 12.0] | 300 |
| Instant fake | keep-alive | **aiohttp** | **1.04 ms [1.01, 1.06]** | **1.32 ms [1.24, 1.41]** | 300 |
| Instant fake | fresh per request | httpx | 7.8 ms [7.3, 8.6] | 10.7 ms [9.6, 12.5] | 300 |
| Instant fake | fresh per request | **aiohttp** | **2.69 ms [2.44, 2.94]** | **2.76 ms [2.30, 3.36]** | 300 |
| Real Ollama, qwen3.5:4b, 1 token | keep-alive | httpx | 8.3 ms [6.4, 10.8] | 13.6 ms [10.6, 18.4] | 40 |
| Real Ollama | keep-alive | **aiohttp** | 1.6 ms [−0.7, 6.0] | 1.1 ms [−2.5, 4.7] | 40 |
| Real Ollama | fresh per request | **aiohttp** | −1.3 ms [−8.0, 8.1] | 4.7 ms [−1.5, 9.8] | 40 |

Direct Ollama baseline: ~244 ms total and ~244–255 ms TTFT.

**How to read the real-Ollama rows.** With aiohttp, the intervals include zero: at n=40, the gateway's cost is smaller than Ollama's own run-to-run variation, so these rows cannot confirm "≤5 ms" on their own. The fake-backend rows measure the gateway's own cost precisely, and they are well under 5 ms.

**How the cause was found:**
1. The streaming delay happened *before* the gateway's response headers went out (16 ms vs 1.4 ms direct).
2. Timing each step put ~7 ms in httpx opening the upstream stream; scheduling took 0.04 ms.
3. It was **not** TCP Nagle: `TCP_NODELAY` on the httpx transport changed nothing.
4. A client micro-benchmark (`bench/client_compare.py`, `results/client_compare.jsonl`, n=300 each) measured per-request cost to an instant local server. POST: httpx 2.32 ms, aiohttp 0.70 ms, stdlib `http.client` 0.61 ms. Stream open: httpx 1.71 ms, aiohttp 0.76 ms.
5. Inside the gateway, the difference was much larger than the micro-benchmark suggested (streaming TTFT 11.7 → 1.3 ms). The exact reason httpx's streaming path is slow in this setup was not identified.

**Also tried:** uvicorn's C HTTP parser (`httptools`). It failed to load with "DLL load failed: The filename or extension is too long", because this working folder's path exceeds Windows' path-length limit. With aiohttp it isn't needed, so it was removed and remains untested.

**Environment note:** localhost requests on this laptop are bimodal. About 10% of fresh-connection requests stall ~20 ms (p90 24 ms vs median 3.7 ms), cause unknown. That is why fresh-connection rows have wider intervals.

## Known limitations

- Priority is recorded, not used (M2).
- No memory awareness (M3). Ollama still makes all load and evict decisions.
- Disconnect detection while waiting has up to 250 ms latency (polling).
- Event log writes are synchronous. They didn't show up in the overhead measurement.
- `X-LRM-Priority` is trusted from any client; there is no auth. The gateway binds to 127.0.0.1 by default.
