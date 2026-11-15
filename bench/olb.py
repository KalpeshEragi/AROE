"""Small Ollama benchmark helpers. Standard library only.

Everything here talks to a local Ollama at OLLAMA_URL and to nvidia-smi.
All timestamps are time.perf_counter() seconds unless noted.
"""
import http.client
import json
import os
import random
import statistics
import subprocess
import threading
import time
from pathlib import Path

OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
SERVER_LOG = Path(os.environ.get("LOCALAPPDATA", "")) / "Ollama" / "server.log"
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

CHAT_MODEL = "qwen3.5:4b"
EMBED_MODEL = "bge-m3"

CHAT_PROMPT = (
    "You are a helpful assistant. Explain, in plain prose without lists, how an operating "
    "system scheduler decides which process runs next on a single CPU core, and why "
    "interactive programs usually feel responsive even when background jobs are running."
)


# ---------------------------------------------------------------- HTTP
def _conn(timeout=600):
    return http.client.HTTPConnection(OLLAMA_HOST, OLLAMA_PORT, timeout=timeout)


def post(path, body, timeout=600):
    c = _conn(timeout)
    try:
        c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read().decode("utf-8", "replace")
        try:
            obj = json.loads(data) if data else {}
        except json.JSONDecodeError:
            obj = {"raw": data}
        return r.status, obj
    finally:
        c.close()


def get(path, timeout=30):
    c = _conn(timeout)
    try:
        c.request("GET", path)
        r = c.getresponse()
        return r.status, json.loads(r.read() or b"{}")
    finally:
        c.close()


def ps():
    return get("/api/ps")[1].get("models", [])


def version():
    return get("/api/version")[1].get("version")


def unload(model, wait_s=60):
    post("/api/generate", {"model": model, "keep_alive": 0})
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if not any(m["name"].startswith(model) for m in ps()):
            return True
        time.sleep(0.25)
    return False


def unload_all():
    for m in ps():
        unload(m["name"])


def chat_options(num_predict, num_ctx=4096):
    # temperature 0 + fixed seed so token counts are as repeatable as the backend allows
    return {"num_predict": num_predict, "num_ctx": num_ctx, "temperature": 0, "seed": 42}


def stream_generate(model, prompt, num_predict, num_ctx=4096, keep_alive="30m",
                    cancel_after_first_token_s=None, timeout=600):
    """Streams /api/generate. Returns timing record.

    If cancel_after_first_token_s is set, closes the socket that many seconds after
    the first token (client-side disconnect) and returns immediately.
    """
    body = {"model": model, "prompt": prompt, "stream": True, "think": False,
            "keep_alive": keep_alive, "options": chat_options(num_predict, num_ctx)}
    rec = {"model": model, "num_predict": num_predict, "num_ctx": num_ctx,
           "t_send": time.perf_counter(), "wall_send": time.time()}
    c = _conn(timeout)
    try:
        c.request("POST", "/api/generate", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        rec["http_status"] = r.status
        if r.status != 200:
            rec["error"] = r.read().decode("utf-8", "replace")
            rec["t_end"] = time.perf_counter()
            rec["e2e_s"] = rec["t_end"] - rec["t_send"]
            return rec
        n_chunks = 0
        while True:
            line = r.readline()
            if not line:
                break
            msg = json.loads(line)
            if msg.get("error"):
                rec["error"] = msg["error"]
                break
            if msg.get("response") and "t_first" not in rec:
                rec["t_first"] = time.perf_counter()
            if msg.get("response"):
                n_chunks += 1
            if (cancel_after_first_token_s is not None and "t_first" in rec
                    and time.perf_counter() - rec["t_first"] >= cancel_after_first_token_s):
                rec["cancelled_by_client"] = True
                rec["chunks_before_cancel"] = n_chunks
                break
            if msg.get("done"):
                for k in ("load_duration", "prompt_eval_count", "prompt_eval_duration",
                          "eval_count", "eval_duration", "total_duration", "done_reason"):
                    rec[k] = msg.get(k)
                break
        rec["chunks"] = n_chunks
    except Exception as e:  # noqa: BLE001 - record any failure as data
        rec["error"] = repr(e)
    finally:
        c.close()  # closing the socket is how a client "disconnects"
    rec["t_end"] = time.perf_counter()
    if "t_first" in rec:
        rec["ttft_s"] = rec["t_first"] - rec["t_send"]
    rec["e2e_s"] = rec["t_end"] - rec["t_send"]
    if rec.get("eval_count") and rec.get("eval_duration"):
        rec["decode_tps"] = rec["eval_count"] / (rec["eval_duration"] / 1e9)
    return rec


def embed(model, inputs, num_ctx=None, keep_alive="30m"):
    body = {"model": model, "input": inputs, "keep_alive": keep_alive}
    if num_ctx:
        body["options"] = {"num_ctx": num_ctx}
    t0 = time.perf_counter()
    status, obj = post("/api/embed", body)
    rec = {"model": model, "n_inputs": len(inputs), "t_send": t0, "t_end": time.perf_counter(),
           "http_status": status, "load_duration": obj.get("load_duration"),
           "prompt_eval_count": obj.get("prompt_eval_count")}
    if status != 200 or "error" in obj:
        rec["error"] = obj.get("error", obj)
    rec["e2e_s"] = rec["t_end"] - t0
    return rec


# ---------------------------------------------------------------- synthetic documents
_WORDS = ("memory scheduler kernel process thread latency cache gpu model token queue "
          "priority request batch tensor layer context device driver allocation page "
          "buffer stream interrupt throughput fairness starvation eviction residency "
          "inference embedding index document vector search retrieval system resource").split()


def synthetic_docs(n, words_per_doc=180, seed=7):
    rng = random.Random(seed)
    return [" ".join(rng.choice(_WORDS) for _ in range(words_per_doc)) for _ in range(n)]


# ---------------------------------------------------------------- GPU sampling
GPU_FIELDS = ["memory.used", "memory.free", "utilization.gpu", "temperature.gpu",
              "clocks.sm", "power.draw"]


def gpu_once():
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10).stdout.strip()
    return _parse_gpu(out)


def _parse_gpu(line):
    vals = [v.strip() for v in line.split(",")]
    d = {}
    for k, v in zip(GPU_FIELDS, vals):
        try:
            d[k] = float(v)
        except ValueError:
            d[k] = None
    return d


def gpu_baseline(n=5, gap=0.3):
    xs = []
    for _ in range(n):
        xs.append(gpu_once()["memory.used"])
        time.sleep(gap)
    return statistics.median(xs)


class GpuSampler:
    """Runs `nvidia-smi -lms` in a subprocess and collects timestamped samples."""

    def __init__(self, period_ms=200):
        self.period_ms = period_ms
        self.samples = []
        self._p = None
        self._t = None

    def __enter__(self):
        self._p = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}",
             "--format=csv,noheader,nounits", f"-lms={self.period_ms}"],
            stdout=subprocess.PIPE, text=True)
        self._t = threading.Thread(target=self._read, daemon=True)
        self._t.start()
        return self

    def _read(self):
        for line in self._p.stdout:
            if line.strip():
                s = _parse_gpu(line)
                s["t"] = time.perf_counter()
                self.samples.append(s)

    def __exit__(self, *exc):
        self._p.terminate()
        self._p.wait(timeout=5)

    def between(self, t0, t1):
        return [s for s in self.samples if t0 <= s["t"] <= t1]


# ---------------------------------------------------------------- server log
def log_offset():
    return SERVER_LOG.stat().st_size if SERVER_LOG.exists() else 0


def log_since(offset):
    if not SERVER_LOG.exists():
        return ""
    with open(SERVER_LOG, "rb") as f:
        f.seek(offset)
        return f.read().decode("utf-8", "replace")


# ---------------------------------------------------------------- stats
def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def bootstrap_ci(xs, stat=statistics.median, n=5000, seed=1, alpha=0.05):
    rng = random.Random(seed)
    if len(xs) < 2:
        return (None, None)
    vals = sorted(stat([rng.choice(xs) for _ in xs]) for _ in range(n))
    return vals[int(n * alpha / 2)], vals[int(n * (1 - alpha / 2)) - 1]


def write_jsonl(name, rows):
    p = RESULTS / name
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def env_record():
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return {"ollama_version": version(), "gpu": smi, "wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
