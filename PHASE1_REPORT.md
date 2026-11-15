# Phase 1 Report: Baseline Experiments on Plain Ollama

Date: 2026-11-07. Everything here is **measured** on this machine against Ollama directly. No resource manager exists yet. These are the baselines it must beat, and the facts its design must respect.

**Environment:** RTX 2050 laptop GPU (4 GB, WDDM, driver 577.00), i5-12450H, 16 GB RAM, Windows 11, on AC power, Balanced power plan. Ollama 0.40.14. Models: `qwen3.5:4b` (Q4_K_M), `bge-m3`, `nomic-embed-text`. Raw data: `bench/results/*.jsonl`. Scripts: `bench/` (standard library only).

**Limits that apply to every result:** one machine, one session, small samples (n given per result), one prompt shape. Medians come with bootstrap 95% confidence intervals where n ≥ 10. Treat these as strong indications, not final benchmarks.

---

## Summary

| # | Question | Answer | Effect on the design |
|---|---|---|---|
| E1 | Does background embedding (a different model, both resident) slow chat? | **Moderately.** First-token time unchanged; decode 8.05 → 6.94 tok/s (−14%); embedding throughput −17% | GPU interference gate drops from MVP default to an optional policy |
| E1b | Does background work on the **same** model block chat? | **Severely.** A chat request behind 4 background requests waited 31–35 s for its first token, vs 0.24 s alone (~127×) | **Confirms the main gap.** Priority scheduling for same-backend queues is the core of the MVP |
| E2 | Can memory be estimated? | KV cache and recurrent state formulas match the backend **exactly**. But Ollama keeps total GPU use constant (~2,620 MiB) by moving layers; ~987 MiB of that is a vision encoder that text requests never use | Estimator predicts the *split*; calibration measures the *footprint*. New experiment E9 |
| E5 | Does client disconnect stop generation? | **Yes.** Next request started in ~0.4 s; backend logs `cancel task` | Enables **cancel-and-requeue** preemption for background generation |
| E9 | What does the vision encoder cost text-only chat? | Encoder on GPU leaves only 12/34 layers on GPU; encoder on CPU → 26/34 layers. **Idle GPU: decode 7.1 → 15.6 tok/s (2.2×), prefill 2.1×**, no text-speed cost vs text-only (≈1.9× vs Ollama as installed, cross-experiment). Cause: llama.cpp keeps 1 GiB free plus the encoder's 987 MiB worst case | Placement decisions are the largest lever; requires the llama-server adapter, since Ollama doesn't expose them |
| E8 | How reliable are cold loads? | **2 of 40 failed (5%)**: one intermittent CUDA crash, one OOM despite Ollama's fit check. Load time 7–15.5 s | Load failure is a normal event: classify, retry or re-plan, explain |

---

## E1: Interference between two resident models

**Setup.** Qwen (16/34 layers on GPU) and bge-m3 (100% on GPU) both loaded. A chat request asks for exactly 64 tokens (temperature 0, all 24 requests produced 64). Four blocks alternate ALONE / WITH_BG, 6 requests each, so thermal drift affects both conditions. In WITH_BG, a thread sends 16-document bge-m3 batches back to back.

| Metric (n=12 each) | ALONE | WITH_BG | Change |
|---|---|---|---|
| TTFT median [95% CI] | 0.280 s [0.271, 0.295] | 0.271 s [0.258, 0.285] | none detectable |
| Decode tok/s median [95% CI] | 8.05 [7.86, 8.23] | 6.94 [6.47, 7.14] | **−14%** |
| End-to-end median / p95 | 8.24 s / 8.70 s | 9.51 s / 10.66 s | +15% / +23% |
| Embedding throughput | 16.6 inputs/s (alone, 45 s) | 13.8 inputs/s (blocks 1 and 3) | **−17%** |
| Embedding batch latency median | 0.88 s | 1.20 s | +36% |

**Reading.** Ollama runs the two models in separate processes at the same time, so neither waits in a queue. They share the GPU and CPU, so both get slower, but only by 14–17%. TTFT here is dominated by a short prompt (~60 tokens, ~0.25 s prefill); a long-prompt interactive request might be affected more. That is untested.

**Design change.** My Phase 0 doc made the "GPU interference gate" (stop background work while chat runs) a default. A 14% decode gain does not justify stopping background work by default. It becomes an optional policy, compared in the simulator.

**Side observation.** The first request of each ALONE block decoded faster (9.4–9.5 tok/s) than the following ones (~8). This is consistent with GPU boost clocks dropping as the chip warms up. The per-request clock and temperature samples are in the data; not analysed yet.

## E1b: Head-of-line blocking on the same model

**Setup.** Four background requests to Qwen (64 tokens each) are submitted at once. 0.5 s later, a short interactive request (8 tokens) arrives. Ollama 0.40.14 serves Qwen3.5 one request at a time.

| Condition | TTFT |
|---|---|
| Interactive request alone (n=3) | 0.24 s median |
| Behind 4 background requests (n=3) | **35.1 s, 30.9 s, 31.0 s** |

In all three trials, all 4 background requests finished before the interactive one got its first token. That is strict FIFO.

**What a scheduler can achieve here** (to be measured in M2, not claimed now):
- *Priority at request boundaries:* wait ≤ the remaining time of the one request in progress (≤ ~8 s here).
- *Priority plus cancel-and-requeue* (made possible by E5): wait ≈ cancellation + prefill (~0.4 s measured in E5). The cost is wasted background work: its prefill plus the tokens already generated must be redone.

This trade-off (latency vs wasted work) should be a first-class simulator metric.

## E2: Memory, estimated vs reported vs measured

Each configuration was loaded alone on an empty GPU. "Device Δ" = nvidia-smi used memory after load minus before.

| Model | ctx | GPU layers | KV: formula / backend log | Recurrent: formula / log | Backend GPU buffers* | Device Δ (peak) | Load time |
|---|---|---|---|---|---|---|---|
| qwen3.5:4b | 2048 | 17/34 | 64 / 64 MiB | 50.25 / 50.25 | 1700 MiB | 2647 (2651) | 10.8 s |
| qwen3.5:4b | 4096 | 16/34 | 128 / 128 | 50.25 / 50.25 | 1671 | 2617 (2621) | 11.2 s |
| qwen3.5:4b | 8192 | 15/34 | 256 / 256 | 50.25 / 50.25 | 1669 | 2619 (2623) | 11.5 s |
| qwen3.5:4b | 16384 | 13/34 | 512 / 512 | 50.25 / 50.25 | 1614 | 2617 (2621) | 12.2 s |
| bge-m3 | 4096 | 25/25 | no KV cache | — | 657 | 737 (755) | 4.1 s |
| nomic-embed-text | 2048 | 13/13 | no KV cache | — | 320 | 401 (421) | 1.7 s |

\* Sum of `CUDA0 model buffer` + `CUDA0 KV buffer` + `CUDA0 compute buffer` from the server log.

Findings:
1. **The architecture-aware KV formula is exact** at every context size: `ctx × 8 attention layers × 4 KV heads × (256 + 256) × 2 bytes`. A formula counting all 32 layers would be 4× too high.
2. **Ollama fills a fixed VRAM budget.** Device use stays at ~2,620 MiB from 2K to 16K context. Ollama moves layers to the CPU (17 → 13) to make room for the larger KV cache. Consequence: for Ollama, raising `num_ctx` costs *speed*, not memory. An estimator that predicts "more context = more VRAM" would be wrong for this backend.
3. **The vision encoder takes ~1 GiB of the 4 GB GPU.** The log says `estimated worst-case memory usage of mmproj is 986.67 MiB` and `CLIP using CUDA0 backend`. That accounts for nearly all of Qwen's ~950 MiB gap between reported buffers and device use, even though none of our requests use images. Remaining gap after that: about 0–60 MiB (the embedding models show ~80 MiB per process).
4. **Load times are consistent** at 10.8–12.2 s, rising with context. Your earlier 16 s was probably a first load after boot or a different context setting [HYPOTHESIS].

**Design consequences.**
- The Ollama adapter needs a calibrated, per-(model, ctx, version) **device footprint** table, built from serialized load deltas. Analytical formulas are still useful for predicting the GPU/CPU **layer split**, and therefore speed.
- **New experiment E9:** a text-only Qwen variant (same weights, no vision projector). Prediction: ~1 GiB freed, roughly 8–10 more layers on the GPU, noticeably faster decoding. This is a configuration fix, not a resource-manager feature. But it is probably the largest single speedup available on this machine, and the manager's explanations should point out cases like it.

## E5: Cancellation

**Setup.** Start a 600-token generation, close the socket 2 s after the first token (20–23 tokens received), immediately send an 8-token probe. If generation kept running, the probe would wait ~70 s (one request at a time).

| Trial | Probe TTFT after cancel | GPU util, 8 s after |
|---|---|---|
| 1 | 0.50 s | 4% |
| 2 | 0.40 s | 3% |
| 3 | 0.41 s | 1% |
| Baseline probe (no cancel) | 0.18–0.36 s | — |

The backend log shows `srv stop: cancel task, id_task = …` once per trial. **Client disconnect stops generation in Ollama 0.40.14**, with ~0.1–0.2 s extra delay. This is task-level cancellation; it says nothing about stopping a single GPU kernel, and it was tested only for streaming `/api/generate`. Embedding requests are short (~1 s), so waiting for them is fine.

## E8: Cold-load reliability

**Setup.** 40 trials. Each one unloads everything, stays idle 3 s (20 trials) or 45 s (20 trials), then loads Qwen with a 1-token request at 4096 context. Run 1 = 29 trials (20 × 3 s, 9 × 45 s); run 2 = 11 × 45 s, about 10 minutes later. My script crashed on run 1's failed trial (it didn't handle an HTTP 500); that row was rebuilt from the Ollama server log and is marked as such in `e8_run1.jsonl`. The script was fixed before run 2.

| Idle before load | Trials | Failures | Load time median (min–max), successful loads |
|---|---|---|---|
| 3 s | 20 | 0 | 8.1 s (7.1–10.2) |
| 45 s | 20 | 2 | 11.1 s (7.5–15.5) |
| **All** | **40** | **2 (5%; Wilson 95% CI 1.4–16.5%)** | |

The two failures were different:

1. **Run 1, trial 29: CUDA initialization crash.** `CUDA error: shared object initialization failed` in `ggml_cuda_kernel_can_use_pdl`, runner exit `0xc0000409`, HTTP 500 after 24.8 s. It is the same crash as the first load this morning, so it is reproducible but intermittent: 2 of the 41 loads today, counting this morning's.
2. **Run 2, trial 11: out of memory, despite Ollama's own fit check.** The log shows `cudaMalloc failed: out of memory` for a 1.45 GB GPU buffer. Ollama then retried with the vision encoder on the CPU (`retrying with projector CPU offload`), and that failed too: `failed to allocate CUDA_Host buffer of size 1970133760` (1.97 GB of pinned system RAM). HTTP 500 after 11.5 s.

**Hypothesis tested: "a laptop GPU waking from low power causes the CUDA crash."** *Not supported.* 10 trials started with the GPU in its low-power state (~3.7 W, 210 MHz), and none failed. The CUDA-crash trial had no power reading, because the script crashed before recording it. So the cause is unknown.

**The OOM was not explained either.** Ollama's pre-load check passed, then the actual allocation failed. The most likely cause is another program using GPU memory and/or RAM at that moment [HYPOTHESIS]. Run 2's load times were also slower (10.9–15.5 s vs 7.1–10.6 s in run 1), which fits a busier system. The script did not record free VRAM or free RAM before each trial, so this cannot be confirmed. That is a gap to fix in the harness.

**Design consequences.**
- About 1 in 20 cold loads failing, in two different ways, means **load failure is a normal event, not an edge case.** The adapter needs: classify the error (CUDA init crash vs OOM), retry once with backoff for the CUDA crash, re-plan memory for OOM (re-measure free memory first), and tell the client exactly which happened.
- This is direct evidence for a Phase 0 claim: **a backend's own fit estimate is not a guarantee**, because other programs share the GPU and RAM. The manager must sample *device* free memory right before a load, and treat its estimate as a safety check, not a promise.
- Load time varies 2× with system state (7–15.5 s). The simulator should take load time from a measured distribution, not a constant.
- Harness fix: record free VRAM, free RAM, and the top memory-using processes before every load.

## E9: Cost of the vision encoder for text-only work

**Setup.** The vision encoder is bundled inside the same 3.2 GB GGUF file as the language model (the Ollama manifest has a single model layer), so a Modelfile cannot remove it. Ollama 0.40 starts its bundled `llama-server.exe` with `--mmproj <same file>`. It adds `--no-mmproj-offload` (encoder on CPU) only as an automatic retry after an OOM (seen in E8); no setting exposes it. So E9 ran that **same bundled binary directly, with Ollama's exact arguments** copied from `server.log`, in three variants. llama.cpp's automatic fit chose the GPU layer count each time. 6 requests per variant, 64 tokens each, temperature 0; the control variant A ran first and last.

**Caveat.** During this run, an unrelated program on the machine (the `eval.run_noc4pc` Python job) held 0.8–1.4 GB of GPU memory and some GPU compute. All variants saw the same ~3.3 GB of free VRAM at fit time, so the comparison between them is fair. Absolute speeds may be lower than on an idle GPU. An earlier test run, before the CUDA path and logging were fixed, is excluded; the first two attempts ran CPU-only by mistake.

| Variant | GPU layers | Vision encoder | Decode tok/s median (range) | Prefill tok/s | Startup |
|---|---|---|---|---|---|
| A: as Ollama runs it (run 1 / run 2) | 12/34 | GPU | 8.06 (7.48–8.16) / 8.58 (8.13–8.67) | 107 / 114 | 13.6 / 10.9 s |
| B: encoder on CPU | **26/34** | CPU | **12.27** (11.72–13.74) | 204 | 9.5 s |
| C: text only | **26/34** | none | **13.07** (12.38–14.10) | 204 | 5.7 s |

**Why: llama.cpp's fit rule, read from the log.**
- It always aims to leave **1,024 MiB of VRAM free** (`cannot meet free memory target of 1024 MiB`). Ollama exposes this as `LLAMA_ARG_FIT_TARGET`.
- With the encoder on the GPU, it reserves the encoder's **worst case** on top of that (987 MiB, sized for a 1472×1472 image): `free memory target of 1985 MiB`.
- So on this 4 GB GPU, roughly 2 GB is held back for text-only chat. That explains the 1.3 GB that "stayed unused" in Phase 0.

**Results.**
1. Moving the encoder to the CPU more than doubles the layers on the GPU (12 → 26) and gives **+52% decode and +85% prefill** speed, while keeping vision usable. The image-processing speed with the encoder on the CPU was **not** measured.
2. Text-only adds a little more (+62% decode vs A) and starts about 2× faster.
3. Neither is available through Ollama's settings today. Lowering `LLAMA_ARG_FIT_TARGET` *is* available, but is untested here, and it trades directly against the OOM risk seen in E8.

**Design consequences.**
- **This is the strongest argument yet for the manager making placement decisions.** The biggest speedup on this machine (≈1.5× here, ≈2× on an idle GPU, see the repeat below) comes from a decision Ollama makes alone and doesn't expose. A manager that knows the workload is text-only could pick variant B or C. That requires the **llama-server adapter** (Phase 0 milestone M5), since Ollama hides these flags. Consider moving that adapter earlier.
- **The memory planner must model the fit rule** (free-memory target, plus the encoder's worst case when on GPU), not just tensor sizes. Otherwise it cannot predict how many layers a load will get.
- **Free VRAM changed under me during this run**, because an unrelated job used 0.8–1.4 GB. That is the cross-program problem the project targets, now observed directly: any footprint calibration must record the free memory at load time.

### E9 repeat on an idle GPU (2026-11-08)

**Setup.** Before each variant: GPU memory 0 MiB used, 3,964 MiB free, and no other process holding GPU memory (a ChatGPT desktop app appeared as a GPU process, with no memory). Each variant ran twice, in the balanced order A B C C B A. 6 requests per run, so n = 12 per variant; every request generated exactly 64 tokens. Raw data: `e9_projector_idle_2026-11-08.jsonl`. Yesterday's contended run is kept as `e9_projector_contended_2026-11-07.jsonl`.

| Variant | GPU layers | Device VRAM after load / peak | Decode tok/s median [95% CI] | Prefill tok/s median [95% CI] | Startup |
|---|---|---|---|---|---|
| A: encoder on GPU (Ollama's default) | 12/34 | 2343 / 2383 MiB | 7.14 [6.82, 8.07] | 103 [101, 112] | 12.7, 12.9 s |
| B: encoder on CPU | **26/34** | 2321 / 2345 MiB | **15.57 [15.05, 15.63]** | **213 [206, 219]** | 11.5, 8.5 s |
| C: text only | **26/34** | 2321 / 2345 MiB | **15.35 [14.82, 15.71]** | **228 [214, 245]** | 8.0, 7.2 s |

**Results.**
1. **Encoder on CPU vs on GPU: decode 2.2× faster (7.14 → 15.57 tok/s), prefill 2.1× faster.** The confidence intervals are far apart. The gain is larger than in the contended run (+52%), as expected once nothing else competes for the GPU.
2. **B and C are equally fast** (the intervals overlap). For text requests, an encoder on the CPU costs nothing measurable: the same 26 layers end up on the GPU. The only difference is startup, which is about 1–3 s faster without the encoder. So variant B keeps vision available at no text-speed cost. Image speed with B is still unmeasured.
3. **All three use about the same VRAM (~2.3 GB).** The fit rule fills the same budget either way; what changes is *what* fills it. In A, about 1 GB is held back for a possible image. In B and C, model layers use that space.
4. **The control does not exactly reproduce Ollama.** Run directly, variant A gets 12/34 layers. Ollama itself placed 16/34 for the same arguments (E2, E8). So Ollama's runner must get some extra setting from Ollama, such as a fit target via environment variable [HYPOTHESIS, not checked]. The fair comparison with Ollama itself is therefore across experiments: Ollama's 8.05 tok/s in E1 (n=12, same chat prompt and 64 tokens, though sent via `/api/generate` rather than a raw ChatML prompt) vs B's 15.57. That is **≈1.9× faster than Ollama as installed**. Treat this as an estimate, not a controlled result.
5. **Decode speed drifted within variant A** (round 1 median 8.07, round 2 6.82). The GPU temperature trace was not analysed. The balanced A…A order spreads this drift over the comparison, but doesn't remove it.

Still to do: measure image-request speed with variant B; find out what Ollama passes to its runner that gives 16 layers instead of 12; test `LLAMA_ARG_FIT_TARGET` at 512 MiB under Ollama (trades speed against OOM risk, see E8).

---

## Changes to the Phase 0 plan

1. **MVP focus narrows to the confirmed gap.** Priority scheduling across clients sharing the same backend model, with two preemption modes: wait for the request in progress, or cancel and requeue background generation. Measure wait time vs wasted work for each.
2. **GPU interference gate: optional, not default.** E1 showed only a 14% decode effect.
3. **Memory planner for Ollama predicts footprint from calibration, not formulas.** Formulas predict the layer split, which drives speed.
4. **E9 done, including the idle-GPU repeat (≈2× decode speedup).** The memory planner must model llama.cpp's fit rule, and a llama-server adapter that can control encoder placement is worth more than Phase 0 assumed. Proposed: build it right after the Ollama adapter, not in M5.
5. **Load-failure handling moves into the MVP** (E8): classify the error, retry or re-plan, and explain it to the client.
6. **Simulator models load time as a distribution** (7–15.5 s measured), plus a ~5% failure probability.

## Not yet tested

- Long-prompt interactive requests under background load (E1 used a ~60-token prompt).
- Interference when models do *not* fit together, forcing swaps (W2 in the Phase 0 doc).
- Cross-backend races (E6, needs whisper.cpp server), Windows VRAM spill to system memory (E7), cold load after reboot (E3), `num_gpu` override (E4).
- Ollama 0.40.15, which allows parallel requests for recurrent models. That would change E1b, so the Ollama version must be pinned for experiments.
