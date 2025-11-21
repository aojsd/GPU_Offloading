import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
import copy
import argparse
import sys

# Check for GPU availability
if not torch.cuda.is_available():
    raise RuntimeError("This benchmark requires a CUDA GPU.")

def merge_attention_states(out1, lse1, out2, lse2):
    """
    Merges two partial attention outputs using their LogSumExp (LSE) statistics.
    """
    # lse shape is [Batch, Head, Q_Len], unsqueeze to match Output [B, H, Q, D]
    lse1 = lse1.unsqueeze(-1)
    lse2 = lse2.unsqueeze(-1)

    # 1. Calculate the new combined normalization factor (New LSE)
    new_lse = torch.logaddexp(lse1, lse2)

    # 2. Calculate contribution weights for each block
    # We subtract new_lse inside the exp for numerical stability
    w1 = torch.exp(lse1 - new_lse)
    w2 = torch.exp(lse2 - new_lse)

    # 3. Weighted sum of the partial outputs
    new_out = out1 * w1 + out2 * w2
    
    return new_out, new_lse.squeeze(-1)

# Wrap Full Attention to make it compilable/graphable easily
class FullAttentionOp(torch.nn.Module):
    def forward(self, q, k, v):
        return flex_attention(q, k, v, return_lse=True)

# Wrap Split Attention
class SplitAttentionOp(torch.nn.Module):
    def forward(self, q, k_A, v_A, k_B, v_B):
        # Phase A
        out_A, lse_A = flex_attention(q, k_A, v_A, return_lse=True)
        # Phase B
        out_B, lse_B = flex_attention(q, k_B, v_B, return_lse=True)
        # Merge
        return merge_attention_states(out_A, lse_A, out_B, lse_B)

def benchmark(args):
    # --- Configuration from Args ---
    B = args.batch_size
    H = args.heads
    D_HEAD = args.head_dim
    Q_len = args.q_len
    
    total_kv = args.total_kv_len
    offload_ratio = args.offload_ratio

    # Validate ratio
    if not (0.0 <= offload_ratio <= 1.0):
        raise ValueError("offload-ratio must be between 0.0 and 1.0")

    # Calculate Splits
    # Block B is the "offloaded" portion
    KV_len_B = int(total_kv * offload_ratio)
    # Block A is the "on-chip" portion
    KV_len_A = total_kv - KV_len_B

    # Handle edge cases where a split might be 0 (though flex_attention usually handles empty tensors, 
    # the split op logic implies we want two valid blocks for a meaningful benchmark)
    if KV_len_A == 0:
        KV_len_A = 1
        print("WARNING: KV Block A was 0. Forced to 1 to prevent empty tensor issues in benchmark.")
    if KV_len_B == 0:
        KV_len_B = 1
        print("WARNING: KV Block B was 0. Forced to 1 to prevent empty tensor issues in benchmark.")

    D_MODEL = H * D_HEAD
    dtype = torch.float16
    device = torch.device("cuda")

    # Safety check for kernel support
    if D_HEAD > 256:
        print(f"WARNING: Head dimension {D_HEAD} is very large (>256).")
        print("This may cause Triton compilation errors or poor performance.")
        print("Standard head dims are 64, 128, or 256.")

    print("=" * 60)
    print(f"{'FlexAttention Decoding Benchmark Configuration':^60}")
    print("=" * 60)
    print(f"{'Batch Size (B)':<25} : {B}")
    print(f"{'Attention Heads (H)':<25} : {H}")
    print(f"{'Head Dimension (D_head)':<25} : {D_HEAD}")
    print(f"{'Model Dimension (Derived)':<25} : {D_MODEL}")
    print(f"{'Query Length (Q_len)':<25} : {Q_len}")
    print("-" * 60)
    print(f"{'Total KV Length':<25} : {total_kv}")
    print(f"{'Offload Ratio':<25} : {offload_ratio:.2f}")
    print(f"{'KV Block A (On-Chip)':<25} : {KV_len_A}")
    print(f"{'KV Block B (Off-Chip)':<25} : {KV_len_B}")
    print("=" * 60)
    print("\nStarting Warmup and Compilation...\n")

    # --- Initialization ---
    q = torch.randn(B, H, Q_len, D_HEAD, device=device, dtype=dtype)
    k_A = torch.randn(B, H, KV_len_A, D_HEAD, device=device, dtype=dtype)
    v_A = torch.randn(B, H, KV_len_A, D_HEAD, device=device, dtype=dtype)
    k_B = torch.randn(B, H, KV_len_B, D_HEAD, device=device, dtype=dtype)
    v_B = torch.randn(B, H, KV_len_B, D_HEAD, device=device, dtype=dtype)

    # Concatenate for ground truth baseline
    k_full = torch.cat([k_A, k_B], dim=2)
    v_full = torch.cat([v_A, v_B], dim=2)

    # --- Warmup Eager Functions ---
    print("  -> Warming up Eager kernels (20 iters)...")
    full_op = FullAttentionOp().to(device)
    split_op = SplitAttentionOp().to(device)
    
    with torch.no_grad():
        for _ in range(20):
            full_op(q, k_full, v_full)
            split_op(q, k_A, v_A, k_B, v_B)
    torch.cuda.synchronize()
    
    # --- Timing Objects ---
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    num_iters = 200

    # ==========================================
    # 1. Benchmark Full Attention (Eager)
    # ==========================================
    with torch.no_grad():
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(num_iters):
            full_op(q, k_full, v_full)
        end_event.record()
        torch.cuda.synchronize()
    avg_full_eager_time = start_event.elapsed_time(end_event) / num_iters

    # ==========================================
    # 2. Benchmark Split & Merge (Eager)
    # ==========================================
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(num_iters):
        split_op(q, k_A, v_A, k_B, v_B)
    end_event.record()
    torch.cuda.synchronize()
    avg_split_eager_time = start_event.elapsed_time(end_event) / num_iters

    # ==========================================
    # 3. Benchmark Full Attention (Compiled/Graph)
    # ==========================================
    print("  -> Compiling FULL Attention (mode='reduce-overhead')...")
    full_op_compiled = torch.compile(full_op, mode="reduce-overhead")
    
    # Aggressive Warmup for Compile/Graph capture
    for _ in range(20):
        full_op_compiled(q, k_full, v_full)
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(num_iters):
        full_op_compiled(q, k_full, v_full)
    end_event.record()
    torch.cuda.synchronize()
    avg_full_graph_time = start_event.elapsed_time(end_event) / num_iters

    # ==========================================
    # 4. Benchmark Split & Merge (Compiled/Graph)
    # ==========================================
    print("  -> Compiling SPLIT Attention (mode='reduce-overhead')...")
    split_op_compiled = torch.compile(split_op, mode="reduce-overhead")
    
    # Aggressive Warmup for Compile/Graph capture
    for _ in range(20):
        split_op_compiled(q, k_A, v_A, k_B, v_B)
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(num_iters):
        split_op_compiled(q, k_A, v_A, k_B, v_B)
    end_event.record()
    torch.cuda.synchronize()
    avg_split_graph_time = start_event.elapsed_time(end_event) / num_iters

    # ==========================================
    # Results Output
    # ==========================================
    
    def calc_overhead(time_val):
        if time_val == avg_full_graph_time:
            return "Baseline"
        val = ((time_val - avg_full_graph_time) / avg_full_graph_time) * 100
        return f"{val:+.2f}%"

    print("\n" + "=" * 65)
    print(f"{'Benchmark Results (Avg of ' + str(num_iters) + ' runs)':^65}")
    print("=" * 65)
    print(f"{'Method':<30} | {'Time (ms)':<12} | {'Overhead':<15}")
    print("-" * 65)
    print(f"{'1. Full Attention (Eager)':<30} | {avg_full_eager_time:<12.4f} | {calc_overhead(avg_full_eager_time):<15}")
    print(f"{'2. Full Attention (Graph)':<30} | {avg_full_graph_time:<12.4f} | {calc_overhead(avg_full_graph_time):<15}")
    print(f"{'3. Split & Merge (Eager)':<30} | {avg_split_eager_time:<12.4f} | {calc_overhead(avg_split_eager_time):<15}")
    print(f"{'4. Split & Merge (Graph)':<30} | {avg_split_graph_time:<12.4f} | {calc_overhead(avg_split_graph_time):<15}")
    print("-" * 65)
    
    # Validation
    out_gt, _ = full_op(q, k_full, v_full)
    out_graph, _ = split_op_compiled(q, k_A, v_A, k_B, v_B)
    diff = (out_graph - out_gt).abs().max()
    
    print(f"\nValidation Delta (Max Diff): {diff.item():.6e}")
    if diff.item() < 1e-3:
        print("SUCCESS: Split attention matches Full attention within tolerance.")
    else:
        print("WARNING: Significant numerical divergence detected.")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FlexAttention for Split-KV Decoding")
    
    # Dimensions
    parser.add_argument("-b", "--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("-H", "--heads", type=int, default=48, help="Number of attention heads (default: 48)")
    parser.add_argument("-D", "--head-dim", type=int, default=256, help="Dimension per attention head (default: 256)")
    parser.add_argument("-q", "--q-len", type=int, default=1, help="Query sequence length (default: 1 for decoding)")
    
    # KV Cache Config
    parser.add_argument("--total-kv-len", type=int, default=10000, help="Total KV sequence length (default: 10000)")
    parser.add_argument("-r", "--offload-ratio", type=float, default=0.5, 
                        help="Fraction of KV cache offloaded/split (0.0 to 1.0). Example: 0.5 = 50%% split.")

    args = parser.parse_args()
    benchmark(args)