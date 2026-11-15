"""M2 benchmark: Phase 1's E1b scenario (head-of-line blocking on one model), four ways.

Per trial: 4 background non-streaming chat requests to qwen3.5:4b (64 tokens each) are sent
at once; 0.5 s later an interactive streaming request (8 tokens) arrives. We measure the
interactive time-to-first-token, how long the background batch takes in total, and (via
the gateway's event log) how many tokens preemption threw away.

Configurations:
  direct        straight to Ollama (Phase 1 E1b baseline: 31-35 s)
  gw_fcfs       gateway --policy fcfs --no-preempt
  gw_aging      gateway --policy aging --no-preempt
  gw_preempt    gateway --policy aging (preemption on, the default)

python bench/m2_queue.py [trials]
"""
import http.client
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
import olb  # noqa: E402

GW_PORT = 18081
MODEL = olb.CHAT_MODEL
N_BG, BG_TOKENS, I_TOKENS, I_DELAY = 4, 64, 8, 0.5
CONFIGS = {
    "direct": None,
    "gw_fcfs": ["--policy", "fcfs", "--no-preempt"],
    "gw_aging": ["--policy", "aging", "--no-preempt"],
    "gw_preempt": ["--policy", "aging"],
}
BG_PROMPT = olb.CHAT_PROMPT
I_PROMPT = "Reply with one short sentence: what is a GPU?"


def post(port, path, body, headers=None, stream=False):
    """Returns (ttft_s or None, total_s, status)."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    t0 = time.perf_counter()
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json", **(headers or {})})
    r = c.getresponse()
    ttft = None
    if stream:
        while True:
            line = r.readline()
            if not line:
                break
            if ttft is None and b'"content"' in line and b'"content": ""' not in line and b'"content":""' not in line:
                ttft = time.perf_counter() - t0
    else:
        r.read()
    total = time.perf_counter() - t0
    c.close()
    return ttft, total, r.status


def requests_for(config):
    if config == "direct":
        def bg():
            return post(olb.OLLAMA_PORT, "/api/chat", {
                "model": MODEL, "messages": [{"role": "user", "content": BG_PROMPT}], "stream": False,
                "think": False, "keep_alive": "30m",
                "options": {"num_predict": BG_TOKENS, "temperature": 0, "seed": 42}})

        def inter():
            return post(olb.OLLAMA_PORT, "/api/chat", {
                "model": MODEL, "messages": [{"role": "user", "content": I_PROMPT}], "stream": True,
                "think": False, "keep_alive": "30m",
                "options": {"num_predict": I_TOKENS, "temperature": 0, "seed": 42}}, stream=True)
    else:
        def bg():
            return post(GW_PORT, "/v1/chat/completions", {
                "model": MODEL, "messages": [{"role": "user", "content": BG_PROMPT}], "stream": False,
                "think": False, "keep_alive": "30m", "max_tokens": BG_TOKENS, "temperature": 0, "seed": 42},
                headers={"X-LRM-Priority": "background"})

        def inter():
            return post(GW_PORT, "/v1/chat/completions", {
                "model": MODEL, "messages": [{"role": "user", "content": I_PROMPT}], "stream": True,
                "think": False, "keep_alive": "30m", "max_tokens": I_TOKENS, "temperature": 0, "seed": 42},
                headers={"X-LRM-Priority": "interactive"}, stream=True)
    return bg, inter


def trial(config):
    bg, inter = requests_for(config)
    bg_res = []
    threads = [threading.Thread(target=lambda: bg_res.append(bg())) for _ in range(N_BG)]
    t0 = time.perf_counter()
    for th in threads:
        th.start()
        time.sleep(0.05)
    time.sleep(I_DELAY)
    ttft, i_total, i_status = inter()
    for th in threads:
        th.join()
    return {"config": config, "interactive_ttft_s": ttft, "interactive_total_s": i_total,
            "interactive_status": i_status, "bg_statuses": [s for _, _, s in bg_res],
            "bg_batch_s": time.perf_counter() - t0, "bg_e2e_s": [t for _, t, _ in bg_res]}


def wait_port(port, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            c.request("GET", "/lrm/status")
            c.getresponse().read()
            return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("gateway did not start")


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    env = olb.env_record()
    olb.stream_generate(MODEL, "Say hi.", 1)  # make sure the model is resident
    rows = [{"env": env, "scenario": {"n_bg": N_BG, "bg_tokens": BG_TOKENS, "i_tokens": I_TOKENS,
                                      "i_delay_s": I_DELAY, "trials": trials}}]
    summary = []
    for config, flags in CONFIGS.items():
        proc, ev_path = None, None
        if flags is not None:
            ev_path = olb.RESULTS / f"m2_events_{config}.jsonl"
            ev_path.unlink(missing_ok=True)
            proc = subprocess.Popen([sys.executable, "-m", "lrm", "--port", str(GW_PORT),
                                     "--events", str(ev_path)] + flags, cwd=str(ROOT))
            wait_port(GW_PORT)
        try:
            res = []
            for t in range(trials):
                r = trial(config)
                r["trial"] = t
                res.append(r)
                print(config, t, "interactive TTFT %.2fs" % r["interactive_ttft_s"],
                      "batch %.1fs" % r["bg_batch_s"], "statuses", r["bg_statuses"], r["interactive_status"],
                      flush=True)
                time.sleep(2)
        finally:
            if proc:
                proc.terminate()
                proc.wait(10)
        wasted = preempts = 0
        if ev_path and ev_path.exists():
            evs = [json.loads(line) for line in open(ev_path, encoding="utf-8")]
            pre = [e for e in evs if e["type"] == "job.preempted"]
            preempts, wasted = len(pre), sum(e.get("wasted_tokens", 0) for e in pre)
        rows += res
        tt = [r["interactive_ttft_s"] for r in res]
        bb = [r["bg_batch_s"] for r in res]
        s = {"config": config, "trials": trials, "interactive_ttft_s": tt,
             "interactive_ttft_median_s": statistics.median(tt), "bg_batch_median_s": statistics.median(bb),
             "preemptions_total": preempts, "wasted_tokens_total": wasted}
        summary.append(s)
        print(json.dumps(s), flush=True)
    olb.write_jsonl("m2_queue.jsonl", rows + summary)


if __name__ == "__main__":
    main()
