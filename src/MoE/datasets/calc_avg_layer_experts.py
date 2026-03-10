#!/usr/bin/env python3
"""Calculate average unique experts activated per layer per step from GPUReplayTraces."""

import json
import os
import sys

TRACE_BASE = os.path.join(os.path.dirname(__file__),
                          "ShareGPT_Vicuna/expert_traces/mixtral-8x7b")


def calc_avg_experts(trace_path):
    with open(trace_path) as f:
        data = json.load(f)

    total_unique = 0
    total_entries = 0

    for step in data['steps']:
        for lt in step['layers']:
            experts = set()
            for token_experts in lt['topk_ids']:
                experts.update(token_experts)
            total_unique += len(experts)
            total_entries += 1

    return total_unique / total_entries if total_entries else 0


if __name__ == "__main__":
    pcts = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [60, 70, 80]
    for pct in pcts:
        path = os.path.join(TRACE_BASE, f"cache{pct}pct/Belady-None.json")
        if not os.path.exists(path):
            print(f"Cache {pct}%: trace not found ({path})")
            continue
        avg = calc_avg_experts(path)
        print(f"Cache {pct}%: {avg:.2f} avg experts/layer/step")
