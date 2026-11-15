"""E8: how often does a cold load of the chat model fail, and how variable is load time?

Usage: python e8_cold_loads.py <out.jsonl> <idle_s>x<count> [<idle_s>x<count> ...]
  e.g. python e8_cold_loads.py e8_run2.jsonl 45x11

Each trial: unload everything, stay idle for idle_s (45 s lets the laptop GPU drop
into a low-power state, one hypothesis for the CUDA init crash), then load the chat
model with a 1-token request. Rows are appended as they finish.
"""
import json
import re
import sys
import time

import olb


def main(out_name, plan):
    out = olb.RESULTS / out_name
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"env": olb.env_record()}) + "\n")
    for i, idle in enumerate(plan):
        olb.unload_all()
        time.sleep(idle)
        pre = olb.gpu_once()
        off = olb.log_offset()
        r = olb.stream_generate(olb.CHAT_MODEL, "Say hi.", 1, num_ctx=4096)
        log = olb.log_since(off)
        row = {"i": i, "idle_s": idle, "ok": "error" not in r, "error": r.get("error"),
               "http_status": r.get("http_status"),
               "load_s": (r.get("load_duration") or 0) / 1e9, "wall_s": r.get("e2e_s"),
               "pre_clock_sm": pre.get("clocks.sm"), "pre_power_w": pre.get("power.draw"),
               "cuda_errors": re.findall(r"CUDA error[^\n]*", log)[:3],
               "offloaded": (re.findall(r"offloaded (\d+/\d+) layers", log) or [None])[-1]}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    plan = []
    for spec in sys.argv[2:]:
        idle, n = spec.split("x")
        plan += [int(idle)] * int(n)
    main(sys.argv[1], plan)
