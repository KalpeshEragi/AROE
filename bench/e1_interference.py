"""E1: does background embedding slow down interactive chat when both models are resident?

Both models are preloaded so this measures compute interference, not loading.
Blocks alternate ALONE / WITH_BG to spread thermal drift across both conditions.
A background thread sends bge-m3 embedding batches back to back during WITH_BG blocks.
"""
import json
import statistics
import sys
import threading
import time

import olb

N_BLOCKS = 4          # ALONE, BG, ALONE, BG, ... (each block = PER_BLOCK chat requests)
PER_BLOCK = 6
NUM_PREDICT = 64
GAP_S = 1.5
BATCH = 16            # embedding inputs per background request (one "chunk")


class BackgroundEmbedder:
    def __init__(self):
        self.stop = threading.Event()
        self.rows = []
        self.docs = olb.synthetic_docs(BATCH * 50, seed=11)

    def run(self):
        i = 0
        while not self.stop.is_set():
            batch = [self.docs[(i * BATCH + j) % len(self.docs)] for j in range(BATCH)]
            r = olb.embed(olb.EMBED_MODEL, batch)
            self.rows.append(r)
            i += 1


def bg_rate(rows, t0, t1):
    done = [r for r in rows if t0 <= r["t_end"] <= t1 and not r.get("error")]
    return len(done) * BATCH / (t1 - t0) if t1 > t0 else None


def main():
    env = olb.env_record()
    olb.unload_all()
    time.sleep(2)
    # preload both, chat first so it gets the GPU layers it gets by default
    print("preload", olb.stream_generate(olb.CHAT_MODEL, "Say hi.", 1).get("load_duration"), flush=True)
    print("preload", olb.embed(olb.EMBED_MODEL, ["warm"]).get("load_duration"), flush=True)
    print("ps", [(m["name"], round(m["size_vram"] / 2**20)) for m in olb.ps()], flush=True)
    # warm-up chat requests (discarded)
    for _ in range(2):
        olb.stream_generate(olb.CHAT_MODEL, olb.CHAT_PROMPT, NUM_PREDICT)

    rows, bg_meta = [], []
    with olb.GpuSampler(250) as gs:
        # background-alone throughput, 45 s
        bg = BackgroundEmbedder()
        th = threading.Thread(target=bg.run, daemon=True)
        t0 = time.perf_counter(); th.start(); time.sleep(45); bg.stop.set(); th.join()
        t1 = time.perf_counter()
        bg_alone = {"cond": "BG_ALONE", "chunks": len(bg.rows), "inputs_per_s": bg_rate(bg.rows, t0, t1),
                    "chunk_e2e_median_s": statistics.median(r["e2e_s"] for r in bg.rows)}
        print(json.dumps(bg_alone), flush=True)
        bg_meta.append(bg_alone)

        for b in range(N_BLOCKS):
            cond = "ALONE" if b % 2 == 0 else "WITH_BG"
            bg = None
            if cond == "WITH_BG":
                bg = BackgroundEmbedder()
                th = threading.Thread(target=bg.run, daemon=True)
                th.start(); time.sleep(3)  # let it reach steady state
            tb0 = time.perf_counter()
            for i in range(PER_BLOCK):
                r = olb.stream_generate(olb.CHAT_MODEL, olb.CHAT_PROMPT, NUM_PREDICT)
                r.update(cond=cond, block=b, i=i)
                samp = gs.between(r["t_send"], r["t_end"])
                r["gpu_util_mean"] = statistics.mean(s["utilization.gpu"] for s in samp) if samp else None
                r["gpu_temp_max"] = max((s["temperature.gpu"] for s in samp), default=None)
                r["sm_clock_mean"] = statistics.mean(s["clocks.sm"] for s in samp) if samp else None
                rows.append(r)
                print(cond, b, i, "ttft=%.3f" % r.get("ttft_s", -1), "tps=%.2f" % r.get("decode_tps", -1),
                      "e2e=%.2f" % r["e2e_s"], r.get("error", ""), flush=True)
                time.sleep(GAP_S)
            tb1 = time.perf_counter()
            if bg:
                bg.stop.set(); th.join()
                m = {"cond": "BG_DURING_CHAT", "block": b, "chunks": len(bg.rows),
                     "inputs_per_s": bg_rate(bg.rows, tb0, tb1),
                     "chunk_e2e_median_s": statistics.median(r["e2e_s"] for r in bg.rows),
                     "chunk_e2e_max_s": max(r["e2e_s"] for r in bg.rows),
                     "errors": sum(1 for r in bg.rows if r.get("error"))}
                print(json.dumps(m), flush=True)
                bg_meta.append(m)
        gpu_trace = gs.samples
    p = olb.write_jsonl("e1_interference.jsonl", [{"env": env}] + rows + bg_meta)
    olb.write_jsonl("e1_gpu_trace.jsonl", gpu_trace)
    print("wrote", p)

    def summarize(cond):
        rs = [r for r in rows if r["cond"] == cond and "ttft_s" in r and not r.get("error")]
        tt = [r["ttft_s"] for r in rs]
        tps = [r["decode_tps"] for r in rs if r.get("decode_tps")]
        e2e = [r["e2e_s"] for r in rs]
        pe = [r["prompt_eval_duration"] / 1e9 for r in rs if r.get("prompt_eval_duration")]
        return {"cond": cond, "n": len(rs),
                "ttft_median": statistics.median(tt), "ttft_ci95": olb.bootstrap_ci(tt),
                "ttft_p95": olb.pct(tt, 95), "ttft_max": max(tt),
                "prompt_eval_median_s": statistics.median(pe) if pe else None,
                "decode_tps_median": statistics.median(tps), "decode_tps_ci95": olb.bootstrap_ci(tps),
                "e2e_median": statistics.median(e2e), "e2e_p95": olb.pct(e2e, 95),
                "eval_count_set": sorted({r.get("eval_count") for r in rs})}

    s = [summarize("ALONE"), summarize("WITH_BG")]
    for x in s:
        print(json.dumps(x))
    olb.write_jsonl("e1_summary.jsonl", s + bg_meta)


if __name__ == "__main__":
    sys.exit(main())
