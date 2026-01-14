import torch
import sys
import os

# Add parent directory to sys.path to import paged_transformer
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from include.paged_transformer import PagedTransformer, PagedTransformerData, TransformerArgs

# ==========================================
# CONFIGURATION
# ==========================================
BATCH_SIZE = 1
SEQ_LEN = 500000  # Long sequence to stress BW
DIM = 8192
HEADS = 64
LAYERS = 4      # Keep small for quick testing, BW scales linearly
BLOCK_SIZE = 16
DTYPE = torch.float16
DEVICE = "cuda"

NUM_WARMUP = 10
NUM_TRIALS = 50

# ==========================================
# SETUP
# ==========================================
print(f"--- Setting up PagedTransformer ---")
print(f"Batch: {BATCH_SIZE}, Seq Len: {SEQ_LEN}, Dim: {DIM}, Layers: {LAYERS}")

# 1. Initialize Model
args = TransformerArgs(dim=DIM, n_heads=HEADS, n_layers=LAYERS)
model = PagedTransformer(args).to(DEVICE).to(DTYPE)
model.eval()

# 2. Initialize Data (KV Heap)
# Estimate blocks needed: Layers * Batch * (Seq + 1 / Block)
blocks_per_seq = (SEQ_LEN + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE
total_blocks = LAYERS * BATCH_SIZE * blocks_per_seq + 1024 # Buffer

print(f"Allocating KV Heap ({total_blocks} blocks)...")
data = PagedTransformerData(
    batch_size=BATCH_SIZE,
    seq_len=SEQ_LEN,
    max_num_blocks=total_blocks,
    num_layers=LAYERS,
    num_heads=HEADS,
    head_dim=DIM // HEADS,
    block_size=BLOCK_SIZE,
    dtype=DTYPE,
    device=DEVICE
)

# 3. Create Input
# [Batch, 1, Dim] - Single decode step
x_input = torch.randn((BATCH_SIZE, 1, DIM), dtype=DTYPE, device=DEVICE)

# ==========================================
# COMPILATION
# ==========================================
print("Compiling model (torch.compile)...")
compiled_model = torch.compile(model)

# Trigger compilation with a single run
with torch.no_grad():
    compiled_model(x_input, data)
torch.cuda.synchronize()
print("Compilation finished.")

# ==========================================
# PROFILING
# ==========================================
print("\n--- Starting Benchmark ---")

def run_step():
    with torch.no_grad():
        compiled_model(x_input, data)

# Warmup
for _ in range(NUM_WARMUP):
    run_step()
torch.cuda.synchronize()

# Timing
start_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_TRIALS)]
end_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_TRIALS)]

for i in range(NUM_TRIALS):
    start_events[i].record()
    run_step()
    end_events[i].record()

torch.cuda.synchronize()

# Stats
times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
avg_ms = sum(times) / len(times)

# ==========================================
# BANDWIDTH CALCULATION
# ==========================================
element_size = torch.tensor([], dtype=DTYPE).element_size() # 2 bytes for FP16

# 1. KV Cache Read (History)
# Size: Layers * Batch * Seq_Len * Dim * 2(K+V) * Bytes
kv_read_bytes = LAYERS * BATCH_SIZE * SEQ_LEN * DIM * 2 * element_size

# 2. KV Cache Write (New Token)
# Size: Layers * Batch * 1 * Dim * 2(K+V) * Bytes
kv_write_bytes = LAYERS * BATCH_SIZE * 1 * DIM * 2 * element_size

# 3. Model Weights Load
# Attention Weights: 4 matrices (Q, K, V, O) of shape (Dim, Dim) -> 4 * D^2
# MLP Weights: 3 matrices (Gate, Up, Down) of shape (Dim, 4*Dim) -> 3 * 4 * D^2 = 12 * D^2
# Total Params per Layer: 16 * D^2
weights_bytes = LAYERS * (16 * DIM * DIM) * element_size

# Total Bytes Transferred
total_bytes = kv_read_bytes + kv_write_bytes + weights_bytes
gb_s = (total_bytes / 1e9) / (avg_ms / 1000.0)

print(f"\nResults:")
print(f"Average Step Time:   {avg_ms:.4f} ms")
print(f"--------------------------------------------------")
print(f"Weight Load:         {weights_bytes / 1024**3:.4f} GB")
print(f"KV Cache Read:       {kv_read_bytes / 1024**3:.4f} GB")
print(f"KV Cache Write:      {kv_write_bytes / 1024**3:.6f} GB")
print(f"Total Memory IO:     {total_bytes / 1024**3:.4f} GB")
print(f"--------------------------------------------------")
print(f"Effective Bandwidth: {gb_s:.2f} GB/s")