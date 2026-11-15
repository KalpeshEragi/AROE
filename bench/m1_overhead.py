"""M1 exit criterion S4: gateway overhead per request.

Two setups, each alternating DIRECT (client -> backend /api/chat) and GATEWAY
(client -> lrm /v1/chat/completions -> backend) request by request, same client code:
  fake    backend = instant fake (bench/fake_ollama.py): isolates the gateway's own cost
  ollama  backend = real Ollama with qwen3.5:4b loaded, 1 token: overhead vs real latency
Gateway and fake backend run as separate processes (no shared GIL with the client).

python bench/m1_overhead.py fake 300
python bench/m1_overhead.py ollama 40
"""
import http.client
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
sys.path.insert(0, str(Path(__file__).parent))
import olb  # noqa: E402  (stats helpers)

GW_PORT, FAKE_PORT, OLLAMA_PORT = 18080, 18090, 11434
MODEL = "qwen3.5:4b"
MSG = [{"role": "user", "content": "Say hi."}]


def wait_port(port, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            c.request("GET", "/")
            c.getresponse().read()
            c.close()
            return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"port {port} not up")


KEEPALIVE = False
_conns = {}


def _conn(port):
    if not KEEPALIVE:
        return http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    if port not in _conns:
        _conns[port] = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    return _conns[port]


def one(port, path, body, stream):
    """Returns (ttft_s, total_s). ttft = first content bytes, total = response fully read."""
    c = _conn(port)
    t0 = time.perf_counter()
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    assert r.status == 200, (r.status, r.read()[:300])
    ttft = None
    if stream:
        while True:
            line = r.readline()
            if not line:
                break
            if ttft is None and b'"content"' in line and b'"content": ""' not in line \
                    and b'"content":""' not in line:
                ttft = time.perf_counter() - t0
    else:
        r.read()
        ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    if not KEEPALIVE:
        c.close()
    return ttft, total


def bodies(stream):
    opts = {"num_predict": 1, "temperature": 0, "seed": 1}
    direct = {"model": MODEL, "messages": MSG, "stream": stream, "think": False,
              "keep_alive": "30m", "options": opts}
    gw = {"model": MODEL, "messages": MSG, "stream": stream, "think": False, "keep_alive": "30m",
          "max_tokens": 1, "temperature": 0, "seed": 1}
    return direct, gw


def run(backend_port, n):
    out = {}
    for stream in (False, True):
        direct_b, gw_b = bodies(stream)
        for _ in range(5):  # warm-up both paths
            one(backend_port, "/api/chat", direct_b, stream)
            one(GW_PORT, "/v1/chat/completions", gw_b, stream)
        d, g = [], []
        for i in range(n):
            pair = [("d", backend_port, "/api/chat", direct_b), ("g", GW_PORT, "/v1/chat/completions", gw_b)]
            if i % 2:
                pair.reverse()  # alternate who goes first
            for who, port, path, b in pair:
                (d if who == "d" else g).append(one(port, path, b, stream)[0])
        key = "stream_ttft" if stream else "nonstream_total"
        diffs = [gi - di for gi, di in zip(g, d)]
        out[key] = {
            "n": n,
            "direct_median_ms": statistics.median(d) * 1e3, "gateway_median_ms": statistics.median(g) * 1e3,
            "direct_p95_ms": olb.pct(d, 95) * 1e3, "gateway_p95_ms": olb.pct(g, 95) * 1e3,
            "direct_p10_ms": olb.pct(d, 10) * 1e3, "gateway_p10_ms": olb.pct(g, 10) * 1e3,
            "overhead_median_ms": statistics.median(diffs) * 1e3,
            "overhead_of_medians_ms": (statistics.median(g) - statistics.median(d)) * 1e3,
            "overhead_of_p10_ms": (olb.pct(g, 10) - olb.pct(d, 10)) * 1e3,
            "overhead_ci95_ms": [x * 1e3 for x in olb.bootstrap_ci(diffs)],
        }
        print(key, json.dumps({k: (round(v, 2) if isinstance(v, float) else
                                   [round(x, 2) for x in v] if isinstance(v, list) else v)
                               for k, v in out[key].items()}), flush=True)
    return out


def main():
    global KEEPALIVE
    mode, n = sys.argv[1], int(sys.argv[2])
    KEEPALIVE = "--keepalive" in sys.argv
    gw_opts = []
    for flag in ("--client", "--http"):
        if flag in sys.argv:
            gw_opts += [flag, sys.argv[sys.argv.index(flag) + 1]]
    label = "_".join(gw_opts[1::2]) or "aiohttp"  # gateway's default client since 2026-11-08
    procs = []
    try:
        if mode == "fake":
            procs.append(subprocess.Popen([PY, str(ROOT / "bench" / "fake_ollama.py"), str(FAKE_PORT)]))
            backend = FAKE_PORT
        else:
            backend = OLLAMA_PORT
        procs.append(subprocess.Popen([PY, "-m", "lrm", "--port", str(GW_PORT), "--ollama",
                                       f"http://127.0.0.1:{backend}", "--events", ""] + gw_opts,
                                      cwd=str(ROOT)))
        wait_port(backend)
        wait_port(GW_PORT)
        res = run(backend, n)
        res["mode"] = mode
        res["keepalive"] = KEEPALIVE
        res["gateway_opts"] = gw_opts
        res["env"] = olb.env_record() if mode == "ollama" else {"python": sys.version}
        olb.write_jsonl(f"m1_overhead_{mode}{'_keepalive' if KEEPALIVE else ''}_{label}.jsonl", [res])
    finally:
        for p in procs:
            p.terminate()
            p.wait(10)


if __name__ == "__main__":
    main()
