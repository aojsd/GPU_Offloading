"""Profile piecewise decode with nsys to understand per-layer cost breakdown.

Runs a few decode steps and emits NVTX ranges so nsys can show:
- Per-layer total time
- Stage1, Stage2, Stage4a, Stage4b breakdown
- Kernel-level timing within each stage
"""
import ctypes
ctypes.CDLL("/gpfs/radev/apps/avx512/software/GCCcore/13.3.0/lib64/libstdc++.so.6",
            mode=ctypes.RTLD_GLOBAL)

import os, sys
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from moe_engine import MoEEngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--experts-per-layer", type=int, default=None,
                        help="Experts per layer (None = no offloading)")
    parser.add_argument("--n-steps", type=int, default=10)
    parser.add_argument("--n-warmup", type=int, default=5)
    args = parser.parse_args()

    engine = MoEEngine(
        args.model, device="cuda:0",
        experts_per_layer=args.experts_per_layer,
    )

    with torch.inference_mode():
        engine.capture_prefill_cuda_graph(total_token_sizes=[128],
                                           use_torch_compile=True)
        engine.reset()
        engine.capture_mixed_cuda_graphs(total_token_sizes=[1, 128],
                                          use_torch_compile=True)

        # Prefill
        prompt = torch.randint(1, 1000, (128,), device="cuda")
        engine.reset()
        logits = engine.prefill_to_slot(0, prompt)
        next_token = logits[-1].argmax().unsqueeze(0)

        # Warmup decode steps
        for i in range(args.n_warmup):
            logits = engine.mixed_step(
                decode_seq_ids=[0], decode_token_ids=next_token,
                prefill_seq_ids=[], prefill_input_ids=[])
            next_token = logits[0].argmax().unsqueeze(0)

        # Profiled decode steps
        torch.cuda.cudart().cudaProfilerStart()
        for step in range(args.n_steps):
            torch.cuda.nvtx.range_push(f"decode_step_{step}")
            logits = engine.mixed_step(
                decode_seq_ids=[0], decode_token_ids=next_token,
                prefill_seq_ids=[], prefill_input_ids=[])
            next_token = logits[0].argmax().unsqueeze(0)
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

        print(f"Done: {args.n_steps} decode steps profiled")


if __name__ == "__main__":
    main()
