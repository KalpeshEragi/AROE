"""E9: how much does Qwen3.5's vision encoder (mmproj) cost a text-only workload?

Runs Ollama's own bundled llama-server.exe directly, with the exact arguments Ollama
uses (copied from server.log), in three variants:
  A  as Ollama runs it      (--mmproj <blob>)              control: should match E2
  B  encoder on CPU         (--mmproj <blob> --no-mmproj-offload)
  C  text only              (no --mmproj)
llama.cpp's automatic fit (on by default) picks the GPU layer count in each case.
Ollama must have no models loaded so the GPU is otherwise empty.
"""
import http.client
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import olb

EXE = Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Ollama" / "lib" / "ollama" / "llama-server.exe"
BLOB = Path(os.environ["USERPROFILE"]) / ".ollama" / "models" / "blobs" / \
    "sha256-81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490"
PORT = 18089
BASE_ARGS = ["--model", str(BLOB), "--port", str(PORT), "--host", "127.0.0.1", "--no-webui",
             "--offline", "-c", "4096", "-np", "1", "--log-verbosity", "4", "--no-jinja", "--chat-template", "chatml",
             "--load-mode", "none", "--flash-attn", "auto", "-b", "512", "-ub", "512",
             "--context-shift", "--keep", "4"]
VARIANTS = {
    "A_ollama_default": ["--mmproj", str(BLOB)],
    "B_mmproj_on_cpu": ["--mmproj", str(BLOB), "--no-mmproj-offload"],
    "C_text_only": [],
}
N_REQ = 6
N_PREDICT = 64


def req(method, path, body=None, timeout=300):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    try:
        c.request(method, path, json.dumps(body) if body else None, {"Content-Type": "application/json"})
        r = c.getresponse()
        return r.status, json.loads(r.read() or b"{}")
    finally:
        c.close()


def wait_ready(proc, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            if req("GET", "/health", timeout=2)[0] == 200:
                return True
        except OSError:
            pass
        time.sleep(0.25)
    return False


def run_variant(name, extra):
    log_path = olb.RESULTS / f"e9_{name}.log"
    base = olb.gpu_baseline()
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        # Ollama's runner finds ggml-cuda.dll + cuBLAS in lib/ollama/cuda_v12; ggml searches the
        # working directory for backends and Windows searches PATH for the DLL dependencies.
        cuda_dir = EXE.parent / "cuda_v12"
        env = dict(os.environ, PATH=f"{cuda_dir};{EXE.parent};{os.environ['PATH']}")
        proc = subprocess.Popen([str(EXE)] + BASE_ARGS + extra, stdout=lf, stderr=subprocess.STDOUT,
                                cwd=str(cuda_dir), env=env)
    rec = {"variant": name, "args_extra": extra, "baseline_used_mib": base}
    try:
        ok = wait_ready(proc)
        rec["ready"] = ok
        rec["ready_s"] = time.perf_counter() - t0
        if not ok:
            rec["exit_code"] = proc.poll()
            return rec
        time.sleep(2)
        rec["after_load_used_mib"] = olb.gpu_baseline()
        prompt = f"<|im_start|>user\n{olb.CHAT_PROMPT}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        req("POST", "/completion", {"prompt": prompt, "n_predict": 8, "temperature": 0})  # warm-up
        runs = []
        with olb.GpuSampler(250) as gs:
            for _ in range(N_REQ):
                ts = time.perf_counter()
                st, out = req("POST", "/completion", {"prompt": prompt, "n_predict": N_PREDICT,
                                                     "temperature": 0, "seed": 42, "cache_prompt": False})
                tm = out.get("timings", {})
                runs.append({"status": st, "wall_s": time.perf_counter() - ts,
                             "predicted_n": tm.get("predicted_n"), "decode_tps": tm.get("predicted_per_second"),
                             "prompt_n": tm.get("prompt_n"), "prefill_tps": tm.get("prompt_per_second")})
                time.sleep(1)
            peak = max((s["memory.used"] for s in gs.samples if s["memory.used"]), default=None)
        rec["runs"] = runs
        rec["peak_used_mib"] = peak
        tps = [r["decode_tps"] for r in runs if r["decode_tps"]]
        pre = [r["prefill_tps"] for r in runs if r["prefill_tps"]]
        rec["decode_tps_median"] = statistics.median(tps)
        rec["decode_tps_range"] = (min(tps), max(tps))
        rec["prefill_tps_median"] = statistics.median(pre)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(3)
    log = log_path.read_text(encoding="utf-8", errors="replace")
    pats = {"offloaded": r"offloaded (\d+/\d+) layers", "gpu_model_buf": r"CUDA0 model buffer size =\s+([\d.]+)",
            "gpu_compute_buf": r"CUDA0 compute buffer size =\s+([\d.]+)",
            "mmproj_est": r"worst-case memory usage of mmproj is ([\d.]+) MiB",
            "clip_backend": r"CLIP using (\S+) backend", "fit": r"(llama_params_fit[^\n]*)"}
    for k, p in pats.items():
        m = re.findall(p, log)
        rec[k] = m[-1] if m else None
    rec["delta_after_load_mib"] = (rec.get("after_load_used_mib") or 0) - base
    rec["delta_peak_mib"] = (rec.get("peak_used_mib") or 0) - base
    return rec


def main():
    olb.unload_all()
    if olb.ps():
        sys.exit("Ollama still has models loaded")
    order = sys.argv[1:] or list(VARIANTS)
    rows = [{"env": olb.env_record(), "exe": str(EXE)}]
    for name in order:
        r = run_variant(name, VARIANTS[name])
        rows.append(r)
        print(json.dumps({k: r.get(k) for k in ("variant", "ready", "ready_s", "offloaded", "gpu_model_buf",
                                                 "clip_backend", "mmproj_est", "delta_after_load_mib",
                                                 "delta_peak_mib", "decode_tps_median", "decode_tps_range",
                                                 "prefill_tps_median")}), flush=True)
    olb.write_jsonl("e9_projector.jsonl", rows)


if __name__ == "__main__":
    main()
