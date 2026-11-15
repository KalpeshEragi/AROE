"""E1b: head-of-line blocking on the same model.

N_BG background requests to the chat model are submitted at once (e.g. a batch
summarization job), then an interactive request arrives 0.5 s later. Ollama 0.40.14
serves Qwen3.5 one request at a time, so the question is how long the interactive one waits.
"""
import json
import statistics
import threading
import time

import olb

N_BG = 4
BG_PREDICT = 64
TRIALS = 3


def probe():
    return olb.stream_generate(olb.CHAT_MODEL, "Reply with one short sentence: what is a GPU?", 8)


def main():
    env = olb.env_record()
    olb.stream_generate(olb.CHAT_MODEL, "Say hi.", 1)  # ensure resident
    rows = []
    for _ in range(3):
        r = probe(); r["cond"] = "PROBE_ALONE"; rows.append(r)
        print("alone ttft=%.3f" % r["ttft_s"], flush=True)
    for t in range(TRIALS):
        bg_rows = []
        ths = [threading.Thread(target=lambda: bg_rows.append(
            olb.stream_generate(olb.CHAT_MODEL, olb.CHAT_PROMPT, BG_PREDICT)), daemon=True)
            for _ in range(N_BG)]
        t0 = time.perf_counter()
        for th in ths:
            th.start(); time.sleep(0.05)
        time.sleep(0.5)
        r = probe(); r["cond"] = "PROBE_BEHIND_BG"; r["trial"] = t
        r["arrived_after_bg_s"] = r["t_send"] - t0
        for th in ths:
            th.join()
        r["bg_finished_before_probe_first_token"] = sum(1 for b in bg_rows if b["t_end"] <= r.get("t_first", 0))
        rows.append(r)
        for b in bg_rows:
            b["cond"] = "BG"; b["trial"] = t
        rows += bg_rows
        print("trial", t, "probe ttft=%.2f" % r["ttft_s"], "bg done before it:",
              r["bg_finished_before_probe_first_token"], flush=True)
        time.sleep(2)
    olb.write_jsonl("e1b_queue.jsonl", [{"env": env}] + rows)
    alone = [r["ttft_s"] for r in rows if r["cond"] == "PROBE_ALONE"]
    behind = [r["ttft_s"] for r in rows if r["cond"] == "PROBE_BEHIND_BG"]
    print(json.dumps({"probe_alone_ttft_median": statistics.median(alone),
                      "probe_behind_bg_ttft": behind,
                      "slowdown_x": statistics.median(behind) / statistics.median(alone)}))


if __name__ == "__main__":
    main()
