"""Validate CUDA graph correctness + measure performance vs eager and vLLM."""
import torch
from pathlib import Path
import sys

MODEL_DIR = str(Path(__file__).resolve().parent / "models" / "OLMoE-1B-7B")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from moe_engine import MoEEngine

N_STEPS = 200
WARMUP_SEQ = 128

# ── Part 1: Correctness — greedy generation must match eager ────────

print("=" * 70)
print("PART 1: Correctness validation (CUDA graph vs eager)")
print("=" * 70)

engine = MoEEngine(
    MODEL_DIR, max_batch_size=4, max_seq_len=1024, use_torch_compile=False)

# Eager greedy generation (no graph captured yet, so decode_step uses eager)
engine.reset()
prompt = torch.tensor([[50256, 510, 5765, 273, 6181, 310]], device="cuda")
tokens_eager = engine.generate(prompt, max_new_tokens=30)
print(f"Eager:  {tokens_eager[0].tolist()}")

# Capture CUDA graph WITHOUT torch.compile (exact match expected)
engine.capture_decode_cuda_graph(batch_size=1, warmup_seq_len=6,
                                  max_decode_tokens=50,
                                  use_torch_compile=False)

# Re-prefill with actual prompt, then generate using graph
engine.reset()
logits = engine.prefill(prompt)
next_token = logits[:, -1, :].argmax(dim=-1)
generated_graph = [next_token]

for i in range(29):
    positions = engine.seq_lens[:1].clone()
    logits = engine._decode_step_graphed(next_token, positions)
    next_token = logits.argmax(dim=-1)
    generated_graph.append(next_token)
    if next_token.item() == engine.eos_token_id:
        break

tokens_graph = torch.cat([prompt, torch.stack(generated_graph, dim=1)], dim=1)
print(f"Graph:  {tokens_graph[0].tolist()}")

match = torch.equal(tokens_eager[:, :tokens_graph.shape[1]], tokens_graph)
print(f"\nExact match: {match}")
if not match:
    min_len = min(tokens_eager.shape[1], tokens_graph.shape[1])
    for i in range(min_len):
        if tokens_eager[0, i] != tokens_graph[0, i]:
            print(f"  First divergence at position {i}: eager={tokens_eager[0, i].item()}, graph={tokens_graph[0, i].item()}")
            break

del engine
torch.cuda.empty_cache()

# ── Part 2: Timing ──────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"PART 2: Timing comparison ({N_STEPS} decode steps from pos {WARMUP_SEQ})")
print(f"{'='*70}")

token = torch.tensor([100], device="cuda")
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

def time_steps(eng, n, label, use_graph=False):
    """Time n decode steps."""
    for _ in range(20):
        pos = eng.seq_lens[:1].clone()
        if use_graph:
            eng._decode_step_graphed(token, pos)
        else:
            eng.decode_step(token, pos)
    torch.cuda.synchronize()

    start.record()
    for _ in range(n):
        pos = eng.seq_lens[:1].clone()
        if use_graph:
            eng._decode_step_graphed(token, pos)
        else:
            eng.decode_step(token, pos)
    end.record()
    torch.cuda.synchronize()
    t = start.elapsed_time(end) / n
    print(f"  {label:55s}  {t:.2f} ms/step  ({1000/t:.0f} tok/s)")
    return t

# --- Eager (FlashInfer, no graph) — fresh engine, no graph ---
eng_eager = MoEEngine(
    MODEL_DIR, max_batch_size=4, max_seq_len=1024, use_torch_compile=False)
eng_eager.reset()
eng_eager.prefill(torch.randint(1, 1000, (1, WARMUP_SEQ), device="cuda"))
t_eager = time_steps(eng_eager, N_STEPS, "Eager (FlashInfer, no graph)")
del eng_eager
torch.cuda.empty_cache()

# --- CUDA Graph (no compile) ---
eng_no_compile = MoEEngine(
    MODEL_DIR, max_batch_size=4, max_seq_len=WARMUP_SEQ + 512 + 256,
    use_torch_compile=False)
eng_no_compile.capture_decode_cuda_graph(
    batch_size=1, warmup_seq_len=WARMUP_SEQ, max_decode_tokens=512,
    use_torch_compile=False)
eng_no_compile.reset()
eng_no_compile.prefill(torch.randint(1, 1000, (1, WARMUP_SEQ), device="cuda"))
t_no_compile = time_steps(eng_no_compile, N_STEPS,
                           "CUDA Graph (no compile)", use_graph=True)
del eng_no_compile
torch.cuda.empty_cache()

# --- CUDA Graph + torch.compile ---
eng_compile = MoEEngine(
    MODEL_DIR, max_batch_size=4, max_seq_len=WARMUP_SEQ + 512 + 256,
    use_torch_compile=True)
eng_compile.capture_decode_cuda_graph(
    batch_size=1, warmup_seq_len=WARMUP_SEQ, max_decode_tokens=512,
    use_torch_compile=True)
eng_compile.reset()
eng_compile.prefill(torch.randint(1, 1000, (1, WARMUP_SEQ), device="cuda"))
t_compile = time_steps(eng_compile, N_STEPS,
                        "CUDA Graph + torch.compile", use_graph=True)
del eng_compile
torch.cuda.empty_cache()

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"  Eager (FlashInfer):            {t_eager:5.2f} ms/step  ({1000/t_eager:.0f} tok/s)")
print(f"  CUDA Graph (no compile):       {t_no_compile:5.2f} ms/step  ({1000/t_no_compile:.0f} tok/s)")
print(f"  CUDA Graph + torch.compile:    {t_compile:5.2f} ms/step  ({1000/t_compile:.0f} tok/s)")
print(f"  Compile speedup vs no-compile: {t_no_compile/t_compile:.2f}x")
