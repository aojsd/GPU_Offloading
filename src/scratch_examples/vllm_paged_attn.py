import torch
from vllm import _custom_ops as ops
import argparse
import sys
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# ==========================================
# USER CONFIGURATION
# ==========================================
# Define the context length for each sequence in the batch.
# Edit this array to test different heuristics.
# Example High Throughput (v1 preferred): [100] * 128
# Example Low Latency (v2 preferred):     [10000, 10000]
SEQUENCE_LENGTHS = [20000, 12000, 4000, 2048, 1024, 500, 100, 50]

# Benchmarking
NUM_TRIALS = 100
NUM_WARMUP = 10

# Architecture
HIDDEN_DIM = 8192
NUM_HEADS = 32
NUM_KV_HEADS = 32
HEAD_SIZE = HIDDEN_DIM // NUM_HEADS
BLOCK_SIZE = 16
DTYPE = torch.float16
DEVICE = 'cuda'
PARTITION_SIZE = 512 # v2 specific

# ==========================================
# ARGUMENT PARSING
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--rand', action='store_true', help='Randomize physical block layout to test fragmentation.')
# Check if running in a notebook (sys.argv might contain kernel info)
if 'ipykernel' in sys.modules:
    args = parser.parse_args([])
else:
    args = parser.parse_args()

RANDOMIZE_BLOCKS = args.rand

# ==========================================
# SYSTEM SETUP
# ==========================================
NUM_SEQS = len(SEQUENCE_LENGTHS)
MAX_SEQ_LEN = max(SEQUENCE_LENGTHS)
scale = float(1.0 / (HEAD_SIZE ** 0.5))
element_size = torch.tensor([], dtype=DTYPE).element_size()
x = 16 // element_size

# Calculate Total Active KV Bytes for Bandwidth Calculation
# Formula: Total_Tokens * Num_KV_Heads * Head_Size * Bytes_Per_Element * 2 (Key + Value)
total_active_tokens = sum(SEQUENCE_LENGTHS)
total_kv_bytes = total_active_tokens * NUM_KV_HEADS * HEAD_SIZE * element_size * 2

print(f"--- Configuration ---")
print(f"Batch Size: {NUM_SEQS}")
print(f"Total Active Tokens: {total_active_tokens}")
print(f"Total KV Cache Size: {total_kv_bytes / 1024**3:.2f} GB")
print(f"Memory Layout: {'RANDOMIZED (Fragmented)' if RANDOMIZE_BLOCKS else 'SEQUENTIAL (Contiguous)'}")

# ==========================================
# MEMORY ALLOCATION
# ==========================================
# 1. Physical KV Cache Pool
total_blocks_needed = sum((l + BLOCK_SIZE - 1) // BLOCK_SIZE for l in SEQUENCE_LENGTHS)
num_physical_blocks = total_blocks_needed + 500 # Extra buffer

# vLLM Shape: (blocks, heads, head_size/x, block_size, x)
key_cache = torch.randn(
    (num_physical_blocks, NUM_KV_HEADS, HEAD_SIZE // x, BLOCK_SIZE, x), 
    dtype=DTYPE, device=DEVICE
)
value_cache = torch.randn(
    (num_physical_blocks, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE), 
    dtype=DTYPE, device=DEVICE
)

# 2. Block Tables (The Mapping)
max_blocks_per_seq = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
block_tables = torch.zeros(
    (NUM_SEQS, max_blocks_per_seq), 
    dtype=torch.int32, device=DEVICE
)

# --- MEMORY LAYOUT STRATEGY ---
if RANDOMIZE_BLOCKS:
    # Generate a random permutation of all available physical blocks
    # We take slices from this shuffled list to assign to sequences
    free_blocks = torch.randperm(num_physical_blocks, dtype=torch.int32, device=DEVICE)
else:
    # Sequential assignment
    free_blocks = torch.arange(num_physical_blocks, dtype=torch.int32, device=DEVICE)

current_offset = 0
for seq_id, seq_len in enumerate(SEQUENCE_LENGTHS):
    needed_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Grab the next 'N' blocks from our (shuffled or linear) pool
    assigned_blocks = free_blocks[current_offset : current_offset + needed_blocks]
    
    # Write to block table
    block_tables[seq_id, :needed_blocks] = assigned_blocks
    current_offset += needed_blocks

# 3. Inputs
query = torch.randn(NUM_SEQS, NUM_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)
context_lens = torch.tensor(SEQUENCE_LENGTHS, dtype=torch.int32, device=DEVICE)

# Create separate output buffers to compare results for correctness
output_v1 = torch.empty_like(query)
output_v2 = torch.empty_like(query)

k_scale = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
v_scale = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)

# 4. v2 Scratchpads
max_num_partitions = (MAX_SEQ_LEN + PARTITION_SIZE - 1) // PARTITION_SIZE
exp_sums = torch.empty((NUM_SEQS, NUM_HEADS, max_num_partitions), dtype=torch.float32, device=DEVICE)
max_logits = torch.empty_like(exp_sums)
tmp_output = torch.empty((NUM_SEQS, NUM_HEADS, max_num_partitions, HEAD_SIZE), dtype=output_v2.dtype, device=DEVICE)

# ==========================================
# FLEX ATTENTION SETUP
# ==========================================

# 1. Create Metadata Tables
owner_table = torch.full((num_physical_blocks,), -1, dtype=torch.int32, device=DEVICE)
logical_block_indices = torch.full((num_physical_blocks,), -1, dtype=torch.int32, device=DEVICE)

for seq_id in range(NUM_SEQS):
    num_blocks = (SEQUENCE_LENGTHS[seq_id] + BLOCK_SIZE - 1) // BLOCK_SIZE
    valid_blocks = block_tables[seq_id, :num_blocks].long()
    
    # Map Physical Block -> Owner Sequence
    owner_table[valid_blocks] = seq_id
    
    # Map Physical Block -> Logical Index (0, 1, 2...) in that sequence
    # This allows us to calculate the global token index later
    logical_block_indices[valid_blocks] = torch.arange(num_blocks, device=DEVICE, dtype=torch.int32)

# 2. Reshape Inputs for Flex
# Flex Query: (1, Heads, Num_Seqs, Head_Dim)
flex_query = query.unsqueeze(0).transpose(1, 2).contiguous()

# -------------------------------------------------------------------------
# Un-permute KV cache for FlexAttention Correctness
# vLLM Shape: (blocks, heads, head_size/x, block_size, x)
# Target:     (1, heads, total_physical_tokens, head_size)
# -------------------------------------------------------------------------
def unpermute_vllm_cache(cache_tensor):
    # Handle Key Cache (5D with 'x' layout optimization)
    # Shape: (blocks, heads, head_size//x, block_size, x)
    if cache_tensor.ndim == 5:
        # Permute to: (blocks, heads, block_size, head_size//x, x)
        t = cache_tensor.permute(0, 1, 3, 2, 4)
        # Flatten last two dims -> head_size
        t = t.flatten(3, 4)
        
    # Handle Value Cache (4D standard layout)
    # Shape: (blocks, heads, head_size, block_size)
    elif cache_tensor.ndim == 4:
        # Permute to: (blocks, heads, block_size, head_size)
        t = cache_tensor.permute(0, 1, 3, 2)
        
    else:
        raise ValueError(f"Unexpected cache dimension: {cache_tensor.ndim}")

    # Common Path: t is now (blocks, heads, block_size, head_size)
    # Target: (1, heads, total_physical_tokens, head_size)
    
    # 1. Permute to (heads, blocks, block_size, head_size)
    t = t.permute(1, 0, 2, 3)
    
    # 2. Flatten (blocks, block_size) -> total_tokens
    t = t.flatten(1, 2)
    
    # 3. Add Batch dim
    return t.unsqueeze(0).contiguous()

flex_key = unpermute_vllm_cache(key_cache)
flex_val = unpermute_vllm_cache(value_cache)

# 3. Define the Mask Mod (Fixed to respect context_len)
def paged_mask_mod(b, h, q_idx, kv_idx):
    # Map the dense Query Index (0..N) to Sequence ID
    seq_id = q_idx 
    
    # Map physical KV memory address to Block info
    physical_block_idx = kv_idx // BLOCK_SIZE
    block_offset = kv_idx % BLOCK_SIZE
    
    block_owner = owner_table[physical_block_idx]
    block_logical_idx = logical_block_indices[physical_block_idx]
    
    # 1. Ownership Check
    is_owned = block_owner == seq_id
    
    # 2. Boundary Check (Ignore garbage padding in the last block)
    # Calculate what "token number" this is in the sequence (e.g., token 105)
    token_logical_idx = (block_logical_idx * BLOCK_SIZE) + block_offset
    is_in_bounds = token_logical_idx < context_lens[seq_id]
    
    return is_owned & is_in_bounds

# 4. Create Block Mask
print("Generating BlockMask (this handles the layout mapping)...")
block_mask = create_block_mask(
    paged_mask_mod,
    B=1,
    H=NUM_HEADS,
    Q_LEN=NUM_SEQS,
    KV_LEN=num_physical_blocks * BLOCK_SIZE,
    device=DEVICE
)

# 5. Compile Flex Attention
# CHANGED: Use 'max-autotune-no-cudagraphs' to prevent crash on large inputs
@torch.compile(mode="max-autotune-no-cudagraphs")
def run_flex_compiled(q, k, v, mask):
    return flex_attention(q, k, v, block_mask=mask)

print("Compiling FlexAttention...")
run_flex_compiled(flex_query, flex_key, flex_val, block_mask)
print("Compilation finished.")


# ==========================================
# BENCHMARKING FUNCTIONS
# ==========================================

def run_v1():
    ops.paged_attention_v1(
        output_v1, query, key_cache, value_cache, NUM_KV_HEADS, scale,
        block_tables, context_lens, BLOCK_SIZE, MAX_SEQ_LEN,
        None, "auto", k_scale, v_scale
    )

def run_v2():
    ops.paged_attention_v2(
        output_v2, exp_sums, max_logits, tmp_output,
        query, key_cache, value_cache, NUM_KV_HEADS, scale,
        block_tables, context_lens, BLOCK_SIZE, MAX_SEQ_LEN,
        None, "auto", k_scale, v_scale,
        0, 0, 0, 0, 0 
    )

def run_flex():
    return run_flex_compiled(flex_query, flex_key, flex_val, block_mask)

def benchmark(name, func):
    # Warmup
    for _ in range(NUM_WARMUP):
        func()
    torch.cuda.synchronize()
    
    # Timing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_TRIALS)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_TRIALS)]
    
    for i in range(NUM_TRIALS):
        start_events[i].record()
        func()
        end_events[i].record()
    
    torch.cuda.synchronize()
    
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)] 
    avg_ms = sum(times) / len(times)
    
    # Bandwidth Calculation
    # GB/s = (Bytes / 1e9) / (Seconds)
    gb_s = (total_kv_bytes / 1e9) / (avg_ms / 1000.0)
    
    print(f"{name:<20} | Time: {avg_ms:.4f} ms | BW: {gb_s:.2f} GB/s")
    return gb_s

# ==========================================
# CORRECTNESS CHECKS
# ==========================================

print("\n--- Correctness Checks ---")

# Helper to run a kernel and catch immediate CUDA launch errors
def safe_run_kernel(name, func):
    try:
        func()
        torch.cuda.synchronize() # Force async errors to surface here
        return True
    except Exception as e:
        print(f">>> WARNING: {name} crashed or failed to launch.")
        print(f"    Error: {e}")
        return False

# 1. Run v1 Safely
# This is the risky one for long sequences
v1_status = safe_run_kernel("v1", run_v1)

# 2. Run v2 (Should always work)
run_v2()
torch.cuda.synchronize()

# 3. Run Flex
out_flex = run_flex()
torch.cuda.synchronize()

# --- Check v1 vs v2 ---
v1_valid = False
if v1_status:
    try:
        # Check for silent failures (NaNs or all zeros)
        if torch.isnan(output_v1).any() or output_v1.sum() == 0:
             print(">>> WARNING: v1 output contains NaNs or all Zeros. Marking as FAILED.")
             v1_valid = False
        else:
            # Use safe comparison
            is_close = torch.allclose(output_v1, output_v2, rtol=1e-2, atol=1e-2)
            diff_v1_v2 = torch.abs(output_v1 - output_v2).max().item()
            print(f"Max Diff (v1 vs v2): {diff_v1_v2:.6f}")
            
            if not is_close:
                print(">>> WARNING: v1 output differs significantly from v2.")
                v1_valid = False
            else:
                print(">>> v1 check passed.")
                v1_valid = True
    except Exception as e:
        print(f">>> WARNING: Error comparing v1 vs v2 (Context likely corrupted): {e}")
        v1_valid = False
else:
    print(">>> v1 check SKIPPED (Crashed on launch).")

# --- Check v2 vs Flex ---
# Flex output: (1, H, Q, D) -> Permute to (Q, H, D)
out_flex_reshaped = out_flex.squeeze(0).transpose(0, 1) 

flex_passed = torch.allclose(output_v2, out_flex_reshaped, rtol=1e-2, atol=1e-2)
diff_v2_flex = torch.abs(output_v2 - out_flex_reshaped).max().item()
print(f"Max Diff (v2 vs Flex): {diff_v2_flex:.6f}")

if not flex_passed:
    print(">>> WARNING: FlexAttention output differs significantly from v2.")
else:
    print(">>> FlexAttention sanity check passed.")


# ==========================================
# BENCHMARK EXECUTION
# ==========================================
print("\n--- Starting Benchmark ---")

# Only benchmark v1 if it actually works
if not v1_valid:
    print(f"{'v1 (Sequential)':<20} | SKIPPED (Failed Correctness/Launch Check)")
    bw1 = 0.0
else:
    # Wrap benchmark in try-except too, just in case
    try:
        bw1 = benchmark("v1 (Sequential)", run_v1)
    except Exception as e:
        print(f"{'v1 (Sequential)':<20} | CRASHED during benchmark: {e}")
        bw1 = 0.0

bw2 = benchmark("v2 (FlashDecoding)", run_v2)
bw3 = benchmark("FlexAttention", run_flex)

print("\n--- Summary ---")
best_bw = max(bw1, bw2, bw3)
if best_bw > 0:
    print(f"Peak Bandwidth: {best_bw:.2f} GB/s")
    if bw1 > 0: print(f"v1 vs Peak: {bw1/best_bw:.2f}x")
    print(f"v2 vs Peak: {bw2/best_bw:.2f}x")
    print(f"Flex vs Peak: {bw3/best_bw:.2f}x")
else:
    print("All kernels failed.")