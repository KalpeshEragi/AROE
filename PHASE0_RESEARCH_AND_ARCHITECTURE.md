# Local AI Resource Manager — Phase 0: Research and Architecture

Date: 2026-11-07. Status: proposal, awaiting approval. No code written.

Labels used throughout:
- **[MEASURED]** observed on this machine today (sample sizes noted; n=1 means a single observation, not a benchmark).
- **[DOCUMENTED]** from official docs or source, with a link.
- **[HYPOTHESIS]** unverified; each one has an experiment in section L.

---

## A. Executive technical summary

**Verdict: the project is worth building, but not as it was framed.** About half of the proposed feature list already exists in Ollama, llama-swap, and llama.cpp's server. Building another "model swapper with a dashboard" would duplicate them.

What already exists:
- Ollama already queues requests, keeps several models loaded together, estimates fit before loading, evicts idle models, applies keep-alive timeouts, and shrinks context on OOM ([sched.go](https://github.com/ollama/ollama/blob/main/server/sched.go), [FAQ](https://docs.ollama.com/faq)). On this machine it kept Qwen3.5-4B (partly on GPU) and bge-m3 (fully on GPU) loaded together on 4 GB with no eviction [MEASURED].
- llama-swap and llama-server's router mode already do on-demand model swapping, TTL unload, LRU unload with a memory margin, metrics, and a web UI ([llama-swap](https://github.com/mostlygeek/llama-swap), [llama-server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)).

What none of them do (the gap):
1. **Priority / quality of service.** Ollama's pending queue is FIFO with no priority field. llama-swap and llama-server document no priority either. An interactive chat waits behind a batch of background embeddings.
2. **Coordination across backends.** Each backend decides "does it fit?" alone, against device-wide free VRAM. Ollama cannot know that whisper.cpp or an image generator is about to allocate 1.5 GB. Two backends making independent decisions against the same shrinking pool is a race.
3. **Compute contention, not just memory.** Two models can both *fit* and still slow each other down, because they time-share one GPU and the same CPU cores. Every existing tool treats "fits in memory" as the only constraint.
4. **Explainable decisions and reproducible evaluation.** No tool records why a request waited, and none lets you replay a workload against different policies.

**Proposed contribution:** a thin, backend-agnostic *control plane* (Python) that sits in front of existing backends and owns admission and dispatch order. It does not replace their model loading. It adds priority scheduling with bounded starvation, cross-backend memory admission using a *calibrated* estimator, explanations for every decision, and a trace-driven simulator that is itself validated against measured runs.

**Feasibility on 4 GB / 16 GB:** feasible, with a reframing. On 4 GB, "multiple concurrent GPU workloads" mostly means *time-multiplexing*: Qwen3.5-4B alone does not fit on the GPU. That makes scheduling *more* important, not less, because every decision trades switching cost against latency. Image generation will almost certainly need exclusive GPU access; it should be a later, exclusive-mode workload class.

**Biggest risk:** the core value claim ("background work hurts interactive latency, and we can fix it") has not been measured yet. Experiment E1 must run before anything else. If the interference is small, the project shrinks to cross-backend admission plus explainability.

---

## Environment findings [MEASURED]

| Item | Value | Notes |
|---|---|---|
| GPU | NVIDIA GeForce RTX 2050 (laptop), Ampere, compute capability 8.6 | 55 W cap, WDDM driver model |
| VRAM | 4096 MiB total, 132 MiB reserved, 3964 MiB free at idle | Display seems to run on the integrated GPU (0 MiB used at idle) |
| Driver / CUDA | 577.00 / CUDA 12.9 | |
| CPU | Intel Core i5-12450H, 8 cores (4P+4E), 12 threads | Hybrid P/E cores: CPU-offloaded layers run on mixed-speed cores |
| RAM | 15.7 GiB usable | |
| Ollama | 0.40.14 | 0.40.15 adds parallel requests for recurrent models; 0.40.14 serves Qwen3.5 one request at a time |
| Models | qwen3.5:4b (Q4_K_M, 4.66 B params, vision), bge-m3, nomic-embed-text | |

**Qwen3.5-4B is a hybrid architecture**, not a standard transformer. From GGUF metadata: 32 blocks, `full_attention_interval = 4`, so only **8 layers have a KV cache**; the other 24 use a recurrent (gated-delta / SSM) state of fixed size. `head_count_kv = 4`, key/value length 256.

**Cold load of qwen3.5:4b (default settings, 4096 context):**

| Measurement | Value |
|---|---|
| First attempt | **Crashed**: `CUDA error: shared object initialization failed` in `ggml_cuda_kernel_can_use_pdl`, runner exit `0xc0000409`, after ~31 s |
| Second attempt | Succeeded. `load_duration` 11.9 s, wall 15.3 s (n=1) |
| Layer placement | 16 / 34 layers on GPU ("50%/50% CPU/GPU" in `ollama ps`) |
| Backend-reported buffers | GPU weights 1384 MiB, CPU weights 1627 MiB, KV 128 MiB (64 GPU + 64 CPU), recurrent state 50.25 MiB, GPU compute buffer 99.7 MiB |
| Device VRAM used after load | 2617 MiB (nvidia-smi) |
| Decode speed | ~11 tokens/s (8 tokens in 703 ms; n=1, too small to trust) |
| Ollama's fit inputs (log) | `available=2.8 GiB free=3.2 GiB minimum=457 MiB` |

**Co-residency test:** with Qwen loaded, a bge-m3 embedding request loaded bge-m3 fully on GPU (25/25 layers, 577 MiB weights) in 4.0 s. VRAM went 2617 → 3358 MiB. Qwen was **not** evicted; the next Qwen request had a 15 ms load time.

**Per-process VRAM on Windows:** `nvidia-smi --query-compute-apps` lists both `llama-server.exe` processes with `used_memory = [N/A]`. Per-process attribution is not available under WDDM. Only device totals are.

Three things these numbers show:
1. **Backends fail in practice.** A crash on the first load is a real failure mode the manager must handle (retry with backoff, mark backend unhealthy).
2. **Backend accounting ≠ device usage.** Ollama's buffers for Qwen add up to ~1570 MiB of GPU memory, but the device shows 2617 MiB used. Roughly 1 GiB is unexplained: probably the per-process CUDA context, cuBLAS workspace, and possibly the vision projector [HYPOTHESIS]. bge-m3 added 741 MiB on the device for 577 MiB of weights, about 160 MiB of extra overhead per process. **Any estimator built only from model metadata will be badly wrong.**
3. **Ollama leaves VRAM unused.** After loading Qwen, 1348 MiB stayed free while 18 layers ran on the CPU. Ollama's placement is conservative and fixed at load time [MEASURED]; whether more layers would fit safely is an open question (E4).

---

## B. Existing solution comparison

| Capability | Ollama 0.40 | llama-swap | llama-server (router mode) | vLLM | **Proposed** |
|---|---|---|---|---|---|
| OpenAI-compatible API | Yes | Yes (proxy) | Yes | Yes | Proxy |
| Runs inference | Yes | No (starts servers) | Yes | Yes | **No** |
| Several models loaded at once | Yes, if they fit (default max 3 per GPU) | Via `matrix` DSL | Yes, `--models-max` (default 4) | One model per server | Delegates to backend |
| Automatic eviction | Yes (idle/expiry; victim logic in `findRunnerToUnload`) | TTL | LRU + `--models-memory-margin` | n/a | Cost-aware, cross-backend |
| Fit estimate before load | Yes (`PredictServerVRAM`, 80% of free) | No | Memory margin only | Pre-allocates fraction of VRAM | Calibrated from measured loads |
| Request queue | FIFO, `OLLAMA_MAX_QUEUE` 512, 503 when full | Not documented | Slots; 503 option when none free | Yes | **Priority classes + aging** |
| Request priority | **No** (none in scheduler source) | **No** (not documented) | **No** (not documented) | **Yes** (`priority` scheduling policy, [PR #5958](https://github.com/vllm-project/vllm/pull/5958)), preempts running requests | Yes, at request boundaries |
| Coordination across different backends | No | Swaps between the servers it launches; no shared memory budget | No | No | **Yes, core goal** |
| Compute-interference awareness | No | No | No | Within one model's batch | **Yes (dispatch gating)** |
| Explanations for waits | Logs only | Request inspection in UI | Logs | Metrics | **Per-decision reason records** |
| Trace replay / policy comparison | No | No | No | Benchmark scripts | **Yes, core goal** |
| Native Windows | Yes | Yes | Yes | No (Linux; WSL only) | Yes |
| Fits 4 GB VRAM use case | Yes | Yes | Yes | Poor fit (needs more VRAM, Linux) | Yes |

Not researched in depth (would need checking before any claim): LocalAI, LM Studio's server, GPUStack, Text Generation Inference. None of them should be described as lacking a feature until checked.

**Honest reading of this table:** the novel part is *not* model lifecycle. It is (1) priority across workloads, (2) a single memory budget shared by several backends, (3) treating compute interference as a scheduling constraint, and (4) a validated simulator. vLLM proves priority scheduling is a recognized need, but it targets datacenter GPUs and Linux, and it schedules within one model.

---

## C. Problem statement and the engineering gap

**Problem.** On one consumer GPU, local AI workloads from independent applications compete for VRAM, GPU compute time, and CPU cores. Each backend manages its own models with FIFO ordering and a local view of memory. The results: interactive requests queue behind batch work; backends race for the same free VRAM; and when something is slow, nobody can say why.

**Gap, stated precisely enough to test:**

1. *Priority inversion through FIFO queues* — [DOCUMENTED] Ollama's queue has no priority. To validate: E1 shows interactive TTFT getting worse under background load.
2. *Cross-backend admission races* — [HYPOTHESIS] Two backends that each check free VRAM independently can both decide "it fits" and together exceed it. Test with E6 (Ollama + whisper-server loading at the same time).
3. *Interference between co-resident models* — [HYPOTHESIS] An embedding batch running on the GPU slows Qwen's decoding even though both fit. E1 measures this directly.
4. *Silent memory oversubscription on Windows* — [HYPOTHESIS] The Windows NVIDIA driver can spill VRAM into shared system memory ("CUDA Sysmem Fallback Policy"). There, "OOM" may show up as a sudden severe slowdown instead of a crash. If true, the manager must watch latency, not only allocation errors. E7 tests this.

**What is *not* a gap and should not be built:** model swapping, TTL unloading, basic LRU eviction, an OpenAI proxy by itself, a model download manager.

---

## D. Proposed architecture

### Main design decision: one dispatcher

If our gateway reorders requests but forwards all of them to Ollama straight away, Ollama's own FIFO queue decides the real order, and our scheduler is just decoration. **Rule: the manager is the only place requests wait.** It sends a backend only as many requests as that backend can run at once (for Ollama 0.40.14 with Qwen3.5: one per model). Everything else waits in our queue, where priority applies.

That is what makes the design simpler than the proposed eight-component control plane. Several proposed components merge:

```
Applications (OpenAI SDKs, scripts)
        │  HTTP: /v1/chat/completions, /v1/embeddings, (later) /v1/audio/transcriptions
        ▼
┌──────────────────────── CONTROL PLANE (one asyncio process) ─────────────────────────┐
│ Gateway ── parses request, assigns class (header / API key / route), creates Job      │
│    │                                                                                  │
│    ▼                                                                                  │
│ Scheduler ── per-class queues, aging, picks next Job when a dispatch slot is free     │
│    │   asks ▼                                                                         │
│ Resource Planner ── memory ledger + calibrated estimator + eviction choice            │
│    │   (admit / evict-then-admit / wait / reject, each with a Reason)                 │
│    ▼                                                                                  │
│ Dispatcher ── concurrency gates per backend and per GPU; streams responses back       │
│                                                                                       │
│ Event Log ── every state change and decision as an append-only event (JSONL)          │
│    ├─ feeds: status API, WebSocket, CLI, later the dashboard                          │
│    └─ is the trace the simulator replays                                              │
│ Telemetry Sampler ── NVML device memory/util, psutil RAM, backend /api/ps every ~500ms│
└───────────────────────────────────────────────────────────────────────────────────────┘
        │  Backend Adapter interface (load, unload, run, cancel, status, capabilities)
        ▼
   OllamaAdapter  (MVP)    WhisperServerAdapter (later)    LlamaServerAdapter (later)
        ▼
   Ollama  ───────────── whisper.cpp server ──────────── llama-server
```

Simplifications compared with the original proposal, and why:
- **Request Classifier is part of the Gateway.** In the MVP, classification is a lookup (header or key), not a component.
- **Admission Controller, Memory Planner, and Model Lifecycle Manager become one Resource Planner.** They share one data structure (the memory ledger) and must decide atomically. Splitting them invites the same race we want to remove.
- **Policy Engine = the Scheduler's pluggable policy.** Policies are functions with one interface, the same ones the simulator uses.
- **Trace Recorder = the Event Log.** One record of what happened serves both explanations and replay. Telemetry and traces never drift apart, because they are the same data.
- **Process Supervisor is deferred.** Ollama manages its own runners. A supervisor is needed only when we launch whisper-server or llama-server ourselves.

### Component responsibilities

| Component | Owns | Does not do |
|---|---|---|
| Gateway | HTTP API, request validation, job creation, client disconnect → cancel | Any scheduling decision |
| Scheduler | Queues, ordering policy, aging, queue limits | Memory math |
| Resource Planner | Memory ledger (estimated + measured), estimator, calibration table, eviction choice | Ordering between classes |
| Dispatcher | Concurrency gates, calling adapters, streaming back, timeouts | Choosing what runs next |
| Adapters | Backend-specific translation and capability flags | Policy |
| Event Log | Append-only events, explanation text | Interpretation |
| Simulator | Discrete-event replay using the same Scheduler and Planner code with a fake clock and fake backends | Real I/O |

**The Scheduler and Planner use no I/O and take time from an injected clock.** That one choice is what makes deterministic simulation and tests possible, because the simulator runs the *production* policy code, not a copy.

---

## E. Request lifecycle

Example: an interactive chat request arrives while a background embedding batch is running.

1. **Arrive.** App sends `POST /v1/chat/completions` with `stream: true` and header `X-LRM-Priority: interactive` (or an API key mapped to a class). The Gateway validates, creates `Job{id, class, model, backend, est_tokens, arrival_t}`, emits `job.arrived`.
2. **Enqueue.** Scheduler puts it in the INTERACTIVE queue. If that queue is full, it rejects with 429 and a reason.
3. **Scheduling tick.** Triggered on any arrival, completion, or memory change, not on a timer. The policy picks the highest effective-priority job. Effective priority = class base + aging term.
4. **Plan resources.** The Planner looks up the model's estimated peak footprint for this context size and compares it with the ledger:
   - *Already resident* → admit.
   - *Fits in free memory minus the safety margin* → admit with a load (switch cost charged).
   - *Fits only after evicting* idle, unpinned models → choose a victim by cost (section F), emit `model.evict_planned`, ask the adapter to unload (`keep_alive: 0`), wait for device memory to actually drop, then admit.
   - *Fits only after a running job finishes* → stay queued with reason `waiting_for_release(job=…)`.
   - *Can never fit*, even on an empty GPU → reject, or suggest a smaller context or more CPU offload.
5. **Gate.** The Dispatcher checks concurrency gates: per-backend in-flight limit, plus the GPU interference gate (default: while an INTERACTIVE job is running, no new BACKGROUND job is dispatched to the GPU). The running embedding batch is *not* interrupted; it finishes its current chunk.
6. **Dispatch.** The adapter forwards the request to Ollama (`/api/chat` or Ollama's OpenAI endpoint), passing `keep_alive` and `num_ctx` explicitly so Ollama never makes a surprise reload decision. Emits `job.dispatched` with wait time and reason.
7. **Stream.** Tokens are relayed to the client. The first token's time is recorded (TTFT).
8. **Complete or cancel.** On finish: record Ollama's `load_duration`, `prompt_eval_duration`, `eval_count`, `eval_duration`. On client disconnect: close the upstream stream; whether Ollama stops generating then is verified by E5, not assumed.
9. **Release.** The Planner updates the ledger. A measured device-memory delta is stored if this job caused a load (section F). The scheduler ticks; background work resumes.
10. **Explain.** Every job ends with a summary built from its events, for example: *"Waited 3.8 s: qwen3.5:4b needed ~2.6 GiB of VRAM, 1.4 GiB was free; waited for background job #812 (bge-m3, chunk 14/40) to finish, then evicted bge-m3 (reload cost ~4 s, idle 9 s)."*

**Background yielding** works through *chunking*. The manager splits a large embedding job (for example 10,000 documents) into small sub-requests, such as 32 inputs each. The chunk size sets the worst-case delay an interactive request sees from background work. This is the only "preemption" available, and it is the honest one: yielding at request boundaries, with the boundary spacing chosen by us.

---

## F. Memory estimation and admission control

### Principle: estimate, then calibrate with measurement

From the findings above, metadata-only estimates were off by ~1 GiB for Qwen. Design:

```
predicted_peak(model, ctx, n_seq, backend_version, gpu_layers)
    = calibration[model, ctx_bucket, backend_version]      if a measured value exists
    = analytical_estimate × uncertainty_factor + fixed_overhead_prior   otherwise
```

**Analytical estimate, architecture-aware:**

- *Weights on GPU* ≈ size of the GPU-assigned tensors (from GGUF tensor sizes and the layer split).
- *Attention KV cache*, summed **only over layers that have attention**:
  `KV = n_ctx × n_seq × Σ_attention_layers (n_kv_heads × (d_k + d_v)) × bytes(cache_type)`
  For Qwen3.5-4B at 4096 context, f16: `4096 × 1 × 8 × 4 × (256 + 256) × 2 B = 128 MiB`. **Matches the backend log exactly** [MEASURED]. A naive "all 32 layers" formula would give 512 MiB, 4× too high.
- *Recurrent state* (hybrid / SSM layers): fixed per sequence, does not grow with context. For Qwen3.5-4B: `24 layers × 32 heads × 128 × 128 × 4 B = 48 MiB`, plus a 2.25 MiB conv state. **Matches the log's 50.25 MiB** [MEASURED].
- *Compute buffer*: grows with batch size and context. Take it from the backend log; do not model it at first.
- *Per-process overhead* (CUDA context, cuBLAS workspace): a measured constant per backend; ~160 MiB observed for the second runner [MEASURED, n=1].
- KV cache type (`OLLAMA_KV_CACHE_TYPE` = q8_0 or q4_0) is **global in Ollama** [DOCUMENTED] and needs flash attention. It is a server setting, not something the manager can change per request.

The estimator is a table of per-architecture rules (standard GQA transformer, hybrid with an attention interval, embedding encoder), each tested against backend logs. No universal formula.

### Calibration without per-process VRAM (Windows)

Per-process VRAM reads `N/A` under WDDM, so the manager attributes memory by **serializing loads**: it allows at most one model load in flight per GPU, samples device used memory before and after, and stores the delta as that model's measured footprint for that (context bucket, backend version). This is a direct benefit of being the only dispatcher. Without that control, a delta could not be attributed to one model.

Limits: deltas include allocator caching and fragmentation; memory freed on unload may not return to the device immediately; compute buffers can grow during the first long prompt. So the ledger tracks **peak** after the first inference, not just after load.

### Admission rule

```
headroom = device_free_measured − safety_margin
safety_margin = max(384 MiB, 10% of total VRAM)      # configurable; starting value, to be tuned with E2
admit if predicted_peak ≤ headroom + reclaimable(idle, unpinned models)
```

Both the measured free memory and the ledger's own reservations are checked; the smaller headroom wins. This prevents two admissions in the same tick from both counting the same free memory.

**Eviction choice (cost-aware):** among idle, unpinned resident models, evict the one with the lowest
`expected_cost = P(needed again soon) × reload_seconds`,
where `P` comes from an exponentially decayed request rate per model. Compare against LRU and LFU in the simulator; LRU is the baseline because it is what llama-server uses.

**Offloading (CPU/GPU split):** the only supported lever is Ollama's `num_gpu` option (number of layers on GPU), which **forces a reload** [HYPOTHESIS: verify in E4 that changing it per request reloads the model]. The manager cannot move layers of a running model. In the MVP the split is a *configured* per-model value. A dynamic choice ("load Qwen with fewer GPU layers so bge-m3 can stay resident") is a Phase 3 experiment.

**Guarantees:** none. This lowers the chance of OOM; it does not prevent it. Backends can allocate memory we did not predict, and other programs (browsers, games) use the same GPU.

---

## G. Scheduling: comparison and recommendation

The machine runs essentially one GPU job at a time, and switching models can cost 4–12 s. That makes this closer to **single-machine scheduling with sequence-dependent setup times** than to classic CPU scheduling. Model switches are the setup times.

| Policy | Idea | Strength | Weakness here |
|---|---|---|---|
| FCFS | Arrival order | Fair, simple; is Ollama's current behavior → **baseline** | Interactive waits behind batches |
| Strict priority | Always the highest class first | Best interactive latency | Background can starve indefinitely |
| Priority + aging | Priority rises with waiting time | Bounded starvation, still simple | Aging rate needs tuning; ignores switch cost |
| Weighted fair queuing / deficit round robin | Share service time by class weight | Predictable throughput shares | Shares *time*, but interactive needs *latency*; awkward with switch costs |
| Shortest job first | Shortest predicted job first | Minimizes mean wait | Needs service-time prediction; starvation |
| **Proposed: priority + aging + switch-cost affinity** | Priority with aging; among jobs of similar priority, prefer one whose model is already resident; hard cap on maximum wait | Protects interactive, limits model thrashing, bounded starvation | More parameters; must prove itself against simpler policies |

**Recommendation:** implement FCFS, strict priority, and priority + aging first; they are each about 20 lines behind one policy interface. Add the switch-cost-aware policy only when the simulator shows how much thrashing the simpler ones cause. Deficit round robin is worth one simulator experiment, not production code.

Non-preemptive at request boundaries, always. Combined with the GPU interference gate and chunking, the worst case for an interactive request is roughly *remaining time of the current background chunk + model load time if not resident*. That bound can be stated and tested.

---

## H. Proposed repository structure (when implementation is approved)

```
local-ai-resource-manager/
  pyproject.toml
  config/default.yaml
  src/lrm/
    api/            gateway.py (FastAPI routes), status.py, ws.py
    core/           job.py, scheduler.py, policies.py, planner.py, estimator.py, events.py, clock.py
    dispatch/       dispatcher.py, gates.py
    adapters/       base.py, ollama.py            (whisper.py, llama_server.py later)
    telemetry/      sampler.py (NVML + psutil + backend status)
    sim/            engine.py, fake_backend.py, service_model.py, replay.py
    cli.py
  bench/            workloads/*.yaml, loadgen.py, run_experiment.py, analyze.py
  tests/            unit/, property/, sim/, integration/ (marked; need a live Ollama)
  docs/             architecture.md, experiments/E*.md (each with raw data and environment)
```

`core/` imports nothing from `api/`, `adapters/`, or `telemetry/`. That keeps the scheduling logic pure and lets the simulator reuse it.

---

## I. Technology stack

| Choice | Why | Alternative considered |
|---|---|---|
| Python 3.12, asyncio | Your strength; gateway overhead (~ms) is tiny next to 90 ms/token decoding | Go (what Ollama and llama-swap use): faster, but slows learning and gains nothing measurable here |
| FastAPI + uvicorn | Async HTTP, streaming responses, WebSockets | Starlette alone: fewer features for little gain |
| httpx | Async streaming client for backends | aiohttp: equivalent; httpx has the cleaner API |
| Pydantic v2 | Request schemas and config validation | dataclasses: no validation |
| nvidia-ml-py (official NVML bindings) | Device memory and utilization; per-process values are N/A on WDDM, so device-level only | Parsing nvidia-smi output: slower and fragile |
| psutil | System RAM and process RSS | — |
| PyYAML | Config | TOML: also fine |
| JSONL event files | Append-only, easy to diff and replay, no schema migrations | SQLite: add later if querying history becomes slow |
| pytest, pytest-asyncio, hypothesis | Unit, async, and property tests (e.g. "no job waits longer than the aging bound") | — |
| Dashboard (Phase 5): React + Vite + TypeScript + Recharts | Vite is enough for a local single-page app; no server rendering needed | Next.js: adds server-side features we do not use |

Not needed: Redis, Celery, Kafka, Docker, Kubernetes, Postgres, Prometheus (maybe an optional exporter later).

---

## J. MVP scope versus future work

**MVP: the smallest version worth building.** It answers one question: *can a single dispatcher in front of Ollama measurably protect interactive latency under mixed load, while bounding background starvation, and explain every wait?*

In the MVP:
- OpenAI-compatible `/v1/chat/completions` (streaming) and `/v1/embeddings`, Ollama backend only.
- Three classes, set by header or API key.
- Scheduler with FCFS, strict priority, and priority + aging policies, switchable in config.
- Dispatcher with per-backend in-flight limits and the GPU interference gate.
- Background chunking for embedding jobs.
- Resource Planner: analytical estimator for the two architectures we have (hybrid Qwen3.5, BERT-style bge-m3/nomic), calibration via serialized load deltas, admission rule, LRU eviction via `keep_alive: 0`.
- Event log (JSONL) with a reason on every decision; `GET /status`; a CLI `lrm status` and `lrm explain <job>`.
- Load generator and the E1 benchmark against direct Ollama.
- Simulator with the same policies, plus a fidelity check against one measured run.

Later:
- Phase 3: cost-aware eviction, dynamic `num_gpu` choice, pinning UI, RAM tier experiment (likely negative, see L).
- whisper.cpp server adapter with process supervision (the first test of cross-backend admission).
- llama-server adapter.
- Dashboard (read-only first), WebSocket updates, configuration controls, profiles.
- Image generation as an *exclusive* GPU class.

**Do not build:** an inference engine, layer migration between devices, GPU kernel preemption, model downloading, an ML model that predicts latency, multi-GPU or distributed support, profile presets before the policies are evaluated, or a dashboard before the core works.

---

## K. Benchmarking methodology and success criteria

**Baselines (same workload, same machine, same models):**
1. Direct Ollama, default settings (FCFS, its own eviction).
2. Direct Ollama, tuned (e.g. `OLLAMA_NUM_PARALLEL`, `OLLAMA_KEEP_ALIVE=-1`) — a fair baseline, not a straw man.
3. Our manager with FCFS (measures the proxy's own overhead).
4. Our manager with each policy.

**Workloads** (defined in YAML, seeded, replayable):
- W1 *Chat + indexing:* interactive chat requests (Poisson arrivals, e.g. one every 20 s, ~150 output tokens) while a background job embeds 2,000 chunks with bge-m3.
- W2 *Model churn:* alternating requests for Qwen and two embedding models, with too little VRAM for all three.
- W3 *Burst:* 10 interactive requests at once during background work.
- W4 (later) *Cross-backend:* chat + whisper transcription.

**Metrics:** TTFT and end-to-end latency per class (p50 / p95 / p99), queue wait, decode tokens/s, background throughput (chunks/min), maximum background wait (starvation), model loads and unloads, load failures and OOMs, peak device VRAM, proxy overhead (manager-FCFS vs direct).

**Rigor:**
- Laptop plugged in, fixed Windows power plan, GPU temperature logged; warm-up run discarded; at least 5 repetitions; report median with bootstrap 95% confidence intervals.
- Record Ollama version, driver version, and model digests with every result.
- Raw event logs are committed with each report, so every number can be recomputed.

**Success criteria (targets, not results):**
- S1: Interactive p95 TTFT under W1 is clearly lower than baseline 1, with the confidence intervals not overlapping.
- S2: Background throughput loss under W1 stays within a stated budget (target ≤ 25%, to be set after E1).
- S3: No background job waits longer than the configured aging bound in any run (also checked by property tests).
- S4: Proxy overhead under FCFS ≤ 5 ms median per request.
- S5: Simulator predicts p95 interactive wait within ±20% of the measured run for the same trace (otherwise simulator conclusions are not reported as predictions).
- S6: Fewer load failures or OOMs in W2 than baseline 1. This may turn out to be zero in both, which is an acceptable result to report.

---

## L. Risks, unknowns, and validation experiments

| # | Risk / unknown | Experiment | Decides |
|---|---|---|---|
| E1 | **Does background embedding actually hurt interactive latency?** (the project's main premise) | Measure Qwen TTFT and tokens/s alone, then with a continuous bge-m3 embedding loop, direct to Ollama. 20+ requests each. | If the slowdown is small (< ~15%), refocus on cross-backend admission and explainability. |
| E2 | Estimator error | For each model × context (2k/4k/8k/16k), compare predicted footprint vs device delta vs backend log. | Safety margin size; whether analytical estimates are usable without calibration. |
| E3 | Does a RAM "warm tier" help? Windows already caches the GGUF file in its standby memory, and the 11.9 s load happened with the file probably cached. | Cold load (after reboot) vs warm load vs load with the file pre-read; break down time from the log (CUDA init, upload, CPU repacking). | Most likely result: the OS page cache already provides this, so no warm tier gets built. |
| E4 | Can more of Qwen go on the GPU? Does `num_gpu` force a reload? | Load with `num_gpu` = 16, 20, 24, 28; record VRAM, tokens/s, and stability. | Whether offload choice is a useful lever and what it costs. |
| E5 | Cancellation: does Ollama stop computing when the client disconnects? | Start a 1,000-token generation, disconnect after 1 s, watch GPU utilization and logs. | Whether cancellation frees the GPU or only stops the response. |
| E6 | Cross-backend race | Run Ollama and whisper.cpp server; start loads at the same time with tight VRAM. | Strength of the cross-backend argument. |
| E7 | Windows VRAM spill to system memory instead of OOM | Load past VRAM capacity with the driver's sysmem fallback on vs off; measure tokens/s. | Whether "OOM prevention" really means "slowdown prevention" on Windows. |
| E8 | Load-failure rate (the crash seen today) | 30 cold loads in a loop; count failures and their causes. | Retry policy; whether a supervisor is needed for Ollama too. |
| R1 | Ollama behaves on its own under us (auto context sizing, keep-alive defaults, parallelism changes in 0.40.15) | Pin Ollama's version for experiments; always pass `num_ctx` and `keep_alive` explicitly. | — |
| R2 | Laptop thermal and power variation adds noise | Log temperature and clocks; repetitions and confidence intervals. | — |
| R3 | Simulator mismatches reality | Criterion S5; publish the fidelity error next to every simulated result. | — |

**E1, E2, E5, and E8 should run in Phase 1, before core implementation.**

---

## M. Milestones

| Milestone | Content | Exit criterion |
|---|---|---|
| M0 Feasibility (≈1 week) | Load generator script (the only code), E1, E2, E5, E8 | Written report with raw data; go / refocus decision on E1 |
| M1 Skeleton gateway | FastAPI proxy to Ollama, streaming, FCFS, event log | Proxy overhead measured (S4); streaming and disconnect tested |
| M2 Scheduling | Class queues, three policies, dispatch gates, embedding chunking | Property tests pass for the aging bound; W1 run against baselines |
| M3 Resource Planner | Estimator, load-delta calibration, admission rule, LRU eviction, reason records | E2 accuracy reported; W2 run |
| M4 Simulator | Discrete-event engine using the production policy code; service model fitted from M2/M3 logs | S5 fidelity reported; policy comparison report |
| M5 Second backend | whisper.cpp server adapter + supervision; E6 | Cross-backend admission demonstrated or ruled out |
| M6 Dashboard | Read-only views from the event log, then controls | — |
| M7 Release | Docs, benchmark report, demo | — |

Each milestone ends with tests that were actually run, and a short written note of what was learned.

---

## N. Concepts to learn, in the order they become relevant

1. **GPU memory basics:** VRAM vs shared system memory, CUDA contexts and their per-process cost, the caching allocator, fragmentation, and why device "used" ≠ the sum of tensor sizes. (M0)
2. **Transformer inference memory:** weights, attention KV cache, grouped-query attention (KV heads vs query heads), prefill vs decode, and why decode speed is limited by memory bandwidth. Then **hybrid / state-space models** and why their state does not grow with context. (M0)
3. **Quantization formats:** Q4_K_M, f16/q8_0 KV cache, and what they do to size and speed. (M0)
4. **CPU/GPU offloading:** how llama.cpp splits layers, and why PCIe transfer and slow CPU layers dominate when the model is split. (M0, E4)
5. **Windows GPU driver model (WDDM):** time-slicing between processes, VRAM residency, sysmem fallback, and why NVML cannot report per-process memory here. (M0)
6. **Asyncio internals:** the event loop, tasks, cancellation and `CancelledError`, backpressure, streaming responses, and why blocking calls (like NVML) belong in a thread. (M1)
7. **Queueing theory basics:** arrival rate, service rate, utilization, Little's law, and why tail latency rises sharply as utilization approaches 1. (M2)
8. **Scheduling theory:** priority inversion, starvation and aging, fairness (WFQ / DRR), setup-time scheduling. (M2)
9. **Caching and eviction:** LRU, LFU, cost-aware policies such as GreedyDual, and Belady's optimal policy as an upper bound in the simulator. (M3)
10. **Discrete-event simulation:** event queues, simulated clocks, determinism and seeding, and validating a model against reality. (M4)
11. **Performance measurement:** percentiles, confidence intervals, warm-up, coordinated omission in load generators. (M0 onward)
12. **Property-based testing:** stating invariants (no starvation beyond a bound, ledger never negative) and letting a tool search for counterexamples. (M2)

---

## Sources

- Ollama FAQ (concurrency, keep-alive, KV cache type): https://docs.ollama.com/faq
- Ollama scheduler source: https://github.com/ollama/ollama/blob/main/server/sched.go
- Ollama releases: https://github.com/ollama/ollama/releases
- llama-swap: https://github.com/mostlygeek/llama-swap
- llama.cpp server README (router mode, slots, cache): https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- vLLM priority scheduling PR: https://github.com/vllm-project/vllm/pull/5958
- Ollama scheduling issue discussion: https://github.com/ollama/ollama/issues/1956
- Environment data: nvidia-smi, `ollama show`, `/api/show`, and `%LOCALAPPDATA%\Ollama\server.log` on this machine, 2026-11-07.
