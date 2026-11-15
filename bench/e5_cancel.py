"""E5: does Ollama stop generating when the client disconnects?

Start a 600-token generation, close the socket 2 s after the first token, then send a
short probe request right away. Qwen3.5 on 0.40.14 serves one request at a time, so if
generation kept running, the probe would wait roughly the remaining ~70 s.
GPU utilization is sampled for 8 s after the disconnect as a second signal.
"""
import json
import statistics
import time

import olb

TRIALS = 3


def probe():
    return olb.stream_generate(olb.CHAT_MODEL, "Reply with one short sentence: what is a GPU?", 8)


def main():
    env = olb.env_record()
    olb.stream_generate(olb.CHAT_MODEL, "Say hi.", 1)
    off = olb.log_offset()
    rows = []
    base = [probe()["ttft_s"] for _ in range(3)]
    print("baseline probe ttft", [round(x, 3) for x in base], flush=True)
    with olb.GpuSampler(200) as gs:
        for t in range(TRIALS):
            long = olb.stream_generate(olb.CHAT_MODEL, "Write a very long story about a lighthouse keeper.",
                                       600, cancel_after_first_token_s=2.0)
            t_cancel = long["t_end"]
            p = probe()
            time.sleep(8)
            util_after = [s["utilization.gpu"] for s in gs.between(p["t_end"], p["t_end"] + 8)]
            row = {"trial": t, "long_chunks_before_cancel": long.get("chunks_before_cancel"),
                   "long_error": long.get("error"), "probe_ttft_s": p.get("ttft_s"),
                   "probe_sent_after_cancel_s": p["t_send"] - t_cancel,
                   "gpu_util_mean_8s_after_probe": statistics.mean(util_after) if util_after else None}
            rows.append(row)
            print(json.dumps(row), flush=True)
    log = olb.log_since(off)
    hints = [l for l in log.splitlines() if any(k in l.lower() for k in ("abort", "cancel", "client", "closed"))]
    olb.write_jsonl("e5_cancel.jsonl", [{"env": env, "baseline_probe_ttft": base}] + rows +
                    [{"log_hints": hints[-20:]}])
    print("log hints:", *hints[-8:], sep="\n  ")


if __name__ == "__main__":
    main()
