"""Per-request cost of async HTTP clients against an instant local backend.

python bench/client_compare.py
Starts bench/fake_ollama.py in a separate process, then measures (keep-alive pools):
  post         non-streaming POST, full JSON read
  stream-open  streaming POST, time until response headers
for httpx and aiohttp, interleaved in rounds to spread drift. Stdlib http.client is
included as a floor (blocking, so it is not an option for the gateway).
"""
import asyncio
import http.client
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import httpx

sys.path.insert(0, str(Path(__file__).parent))
import olb  # noqa: E402

PORT = 18095
BASE = f"http://127.0.0.1:{PORT}"
BODY = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
ROUNDS, PER_ROUND = 6, 50


async def httpx_post(c):
    r = await c.post("/api/chat", json={**BODY, "stream": False})
    r.json()


async def httpx_stream(c):
    r = await c.send(c.build_request("POST", "/api/chat", json={**BODY, "stream": True}), stream=True)
    t = time.perf_counter()
    await r.aread(); await r.aclose()
    return t


async def aio_post(s):
    async with s.post(BASE + "/api/chat", json={**BODY, "stream": False}) as r:
        await r.json()


async def aio_stream(s):
    r = await s.post(BASE + "/api/chat", json={**BODY, "stream": True})
    t = time.perf_counter()
    await r.read(); r.release()
    return t


async def timed(fn, c, stream):
    t0 = time.perf_counter()
    t_head = await fn(c)
    return ((t_head if stream else time.perf_counter()) - t0) * 1e3


def stdlib_post(conn):
    t0 = time.perf_counter()
    conn.request("POST", "/api/chat", json.dumps({**BODY, "stream": False}), {"Content-Type": "application/json"})
    conn.getresponse().read()
    return (time.perf_counter() - t0) * 1e3


async def main():
    res = {k: [] for k in ("httpx post", "aiohttp post", "httpx stream-open", "aiohttp stream-open", "stdlib post")}
    conn = http.client.HTTPConnection("127.0.0.1", PORT)
    async with httpx.AsyncClient(base_url=BASE) as hc, aiohttp.ClientSession() as s:
        for fn, c, st in ((httpx_post, hc, 0), (aio_post, s, 0), (httpx_stream, hc, 1), (aio_stream, s, 1)):
            for _ in range(20):
                await timed(fn, c, st)  # warm-up
        for _ in range(ROUNDS):
            for name, fn, c, st in (("httpx post", httpx_post, hc, 0), ("aiohttp post", aio_post, s, 0),
                                    ("httpx stream-open", httpx_stream, hc, 1),
                                    ("aiohttp stream-open", aio_stream, s, 1)):
                for _ in range(PER_ROUND):
                    res[name].append(await timed(fn, c, st))
            res["stdlib post"] += [stdlib_post(conn) for _ in range(PER_ROUND)]
    out = {}
    for k, v in res.items():
        out[k] = {"n": len(v), "median_ms": statistics.median(v), "ci95": olb.bootstrap_ci(v),
                  "p10_ms": olb.pct(v, 10), "p90_ms": olb.pct(v, 90)}
        print(f"{k:20s} median {out[k]['median_ms']:5.2f} ms  CI [{out[k]['ci95'][0]:.2f}, {out[k]['ci95'][1]:.2f}]"
              f"  p10 {out[k]['p10_ms']:5.2f}  p90 {out[k]['p90_ms']:5.2f}  (n={len(v)})")
    out["versions"] = {"httpx": httpx.__version__, "aiohttp": aiohttp.__version__, "python": sys.version.split()[0]}
    olb.write_jsonl("client_compare.jsonl", [out])


if __name__ == "__main__":
    fake = subprocess.Popen([sys.executable, str(Path(__file__).parent / "fake_ollama.py"), str(PORT)])
    try:
        time.sleep(3)
        asyncio.run(main())
    finally:
        fake.terminate()
