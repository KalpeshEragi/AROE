"""E2: estimated vs backend-reported vs device-measured memory per model and context size.

Method: unload everything, read device VRAM, load one model, read again (idle after load),
run one inference while sampling, record the peak. Parse Ollama's server.log for the
buffer sizes the backend itself reports.
"""
import json
import re
import statistics
import time

import olb

MIB = 1024 * 1024

CONFIGS = [
    ("qwen3.5:4b", 2048), ("qwen3.5:4b", 4096), ("qwen3.5:4b", 8192), ("qwen3.5:4b", 16384),
    ("bge-m3", None), ("nomic-embed-text", None),
]


def qwen_analytical_mib(ctx):
    # Qwen3.5-4B: 8 full-attention layers, 4 KV heads, d_k = d_v = 256, f16 cache
    kv = ctx * 1 * 8 * 4 * (256 + 256) * 2 / MIB
    recurrent = 24 * 32 * 128 * 128 * 4 / MIB + 2.25  # SSM state + conv state, fixed
    return {"kv_mib": kv, "recurrent_mib": recurrent}


def parse_log(text):
    out = {}
    pats = {
        "offloaded": r"offloaded (\d+)/(\d+) layers to GPU",
        "gpu_model_buf": r"CUDA0 model buffer size =\s+([\d.]+) MiB",
        "cpu_model_buf": r"(?<!CUDA_Host )CPU model buffer size =\s+([\d.]+) MiB",
        "host_model_buf": r"CUDA_Host model buffer size =\s+([\d.]+) MiB",
        "gpu_kv_buf": r"CUDA0 KV buffer size =\s+([\d.]+) MiB",
        "cpu_kv_buf": r"CPU KV buffer size =\s+([\d.]+) MiB",
        "kv_total": r"llama_kv_cache: size =\s+([\d.]+) MiB",
        "recurrent_total": r"llama_memory_recurrent: size =\s+([\d.]+) MiB",
        "gpu_compute_buf": r"CUDA0 compute buffer size =\s+([\d.]+) MiB",
        "host_compute_buf": r"CUDA_Host compute buffer size =\s+([\d.]+) MiB",
        "ollama_gpu_memory": r'msg="gpu memory".*?available="([^"]+)" free="([^"]+)" minimum="([^"]+)"',
        "n_ctx": r"llama_context: n_ctx\s+=\s+(\d+)",
        "flash_attn": r"llama_context: flash_attn\s+=\s+(\S+)",
    }
    for k, p in pats.items():
        ms = re.findall(p, text)
        if ms:
            out[k] = ms[-1]
    errs = re.findall(r"(CUDA error[^\n]*|Load failed[^\n]*|out of memory[^\n]*)", text)
    if errs:
        out["errors"] = errs[:5]
    return out


def mem_median(sampler, t0, t1):
    xs = [s["memory.used"] for s in sampler.between(t0, t1) if s["memory.used"] is not None]
    return statistics.median(xs) if xs else None


def main():
    rows = []
    env = olb.env_record()
    long_prompt = " ".join(olb.synthetic_docs(6, words_per_doc=150, seed=3))  # ~1000+ tokens
    for model, ctx in CONFIGS:
        olb.unload_all()
        time.sleep(3)
        base = olb.gpu_baseline()
        off = olb.log_offset()
        rec = {"model": model, "num_ctx": ctx, "baseline_used_mib": base}
        with olb.GpuSampler(200) as s:
            t_load0 = time.perf_counter()
            if model.startswith("qwen"):
                r = olb.stream_generate(model, "Say hi.", 1, num_ctx=ctx)
            else:
                r = olb.embed(model, ["warm up"], num_ctx=ctx)
            t_load1 = time.perf_counter()
            rec["load_request"] = {k: r.get(k) for k in ("load_duration", "e2e_s", "error", "http_status")}
            time.sleep(2.0)
            t_idle1 = time.perf_counter()
            rec["after_load_used_mib"] = mem_median(s, t_idle1 - 1.5, t_idle1)
            t_inf0 = time.perf_counter()
            if model.startswith("qwen"):
                r2 = olb.stream_generate(model, long_prompt, 16, num_ctx=ctx)
                rec["inference"] = {k: r2.get(k) for k in ("prompt_eval_count", "eval_count", "e2e_s", "error")}
            else:
                r2 = olb.embed(model, olb.synthetic_docs(16, seed=5), num_ctx=ctx)
                rec["inference"] = {k: r2.get(k) for k in ("prompt_eval_count", "e2e_s", "error")}
            t_inf1 = time.perf_counter()
            time.sleep(0.6)
            during = [x["memory.used"] for x in s.between(t_load0, t_inf1 + 0.5) if x["memory.used"]]
            rec["peak_used_mib"] = max(during) if during else None
        rec["ps"] = [{k: m.get(k) for k in ("name", "size", "size_vram", "context_length")} for m in olb.ps()]
        rec["log"] = parse_log(olb.log_since(off))
        if model.startswith("qwen"):
            rec["analytical"] = qwen_analytical_mib(ctx)
        rec["delta_after_load_mib"] = (rec["after_load_used_mib"] or 0) - base
        rec["delta_peak_mib"] = (rec["peak_used_mib"] or 0) - base
        rows.append(rec)
        print(json.dumps({k: rec[k] for k in ("model", "num_ctx", "delta_after_load_mib", "delta_peak_mib")}
                         | {"offloaded": rec["log"].get("offloaded"), "kv": rec["log"].get("kv_total"),
                            "gpu_w": rec["log"].get("gpu_model_buf"), "err": rec["log"].get("errors")}),
              flush=True)
    olb.unload_all()
    p = olb.write_jsonl("e2_memory.jsonl", [{"env": env}] + rows)
    print("wrote", p)


if __name__ == "__main__":
    main()
