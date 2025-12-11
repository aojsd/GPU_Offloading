import torch
import torch.nn as nn
import logging
import argparse  # Added for command line arguments

# Set logging to error only
torch._logging.set_logs(all=logging.ERROR)
def compile_if_needed(module, compile_mode):
    if compile_mode is None:
        return module
    else:
        return torch.compile(module, mode=compile_mode)

class OffloadedMLP(nn.Module):
    def __init__(self, hidden_dim, num_layers, compile_mode=None,
                 offload_ratio_main=0.4, offload_ratio_secondary=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # --- 1. Calculate Splits ---
        self.M_O = int(hidden_dim * offload_ratio_main)      # Main Offload (Streamed)
        self.M_S = int(hidden_dim * offload_ratio_secondary) # Secondary (Prefetched)
        self.M_R = hidden_dim - self.M_O - self.M_S          # Resident (Static)
        
        if self.M_R < 0:
            raise ValueError("Offload ratios sum to > 1.0")

        # --- 2. Initialize Memory (Fast GPU Path) ---
        self.layers = []
        # Shared Scratchpad for Main Offload (L_O)
        self.scratchpad = torch.zeros(self.M_O, hidden_dim, device="cuda", dtype=torch.float16)
        
        print(f"Initializing MLP ({num_layers} layers) on GPU...")
        
        # We perform initialization on GPU to avoid slow CPU RNG, then move specific chunks to Host.
        for i in range(num_layers):
            # Generate FULL matrix on GPU [H, H]
            # In a real scenario, this might be loaded from disk layer-by-layer.
            scale_factor = 1.0 / (hidden_dim ** 0.5)
            W_full_gpu = torch.randn(hidden_dim, hidden_dim, device="cuda", dtype=torch.float16) * scale_factor
            
            # A. Resident Part (L_R + L_S Buffer)
            # We keep the top (M_R + M_S) rows on GPU.
            # We clone to ensure memory independence from the full tensor which we will discard.
            # W_resident_gpu = W_full_gpu[:self.M_R + self.M_S, :].clone()
            W_resident_gpu = W_full_gpu.clone()
            
            # B. Main Offload Part (L_O) -> Move to CPU
            # The bottom M_O rows.
            W_main_host = W_full_gpu[self.M_R + self.M_S:, :].to("cpu", non_blocking=True).pin_memory()
            
            # C. Secondary Source (L_S Source) -> Move to CPU
            # The slice corresponding to L_S (which sits just after M_R).
            W_sec_host = W_full_gpu[self.M_R : self.M_R + self.M_S, :].to("cpu", non_blocking=True).pin_memory()
            
            self.layers.append({
                "W_resident_gpu": W_resident_gpu, # [M_R + M_S, H]
                "W_main_host":    W_main_host,    # [M_O, H]
                "W_sec_host":     W_sec_host      # [M_S, H]
            })

        # Synchronize to ensure all CPU moves are finished before we start
        torch.cuda.synchronize()

        # --- 3. Streams & Events ---
        self.transfer_stream = torch.cuda.Stream()
        self.events_LO_ready = [torch.cuda.Event(enable_timing=False) for _ in range(num_layers)]
        self.events_LS_ready = [torch.cuda.Event(enable_timing=False) for _ in range(num_layers)]
        self.layer_started = [torch.cuda.Event(enable_timing=False) for _ in range(num_layers)]

        # Compiled subroutines
        self.forward_compute = compile_if_needed(self.forward_compute_, compile_mode)
        self.baseline = compile_if_needed(self.run_baseline, compile_mode)
        self.validation = compile_if_needed(self.run_validation, compile_mode)
        self.mm = compile_if_needed(torch.mm, compile_mode)
        self.relu = compile_if_needed(torch.relu, compile_mode)

    def start_transfers(self):
        """
        Queues all data transfers ahead of time.
        """
        self.events_LO_ready[0].record()  # First layer L_O is ready immediately
        for i in range(self.num_layers):
            layer_data = self.layers[i]
            with torch.cuda.stream(self.transfer_stream):
                # 1. Transfer Main Offload (L_O)
                self.transfer_stream.wait_event(self.layer_started[i])
                self.scratchpad.copy_(layer_data["W_main_host"], non_blocking=True)
                self.events_LO_ready[i].record()
                
                # 2. Transfer Next Layer's Secondary (L_S)
                if i < self.num_layers - 1:
                    next_layer = self.layers[i + 1]
                    ls_target = next_layer["W_resident_gpu"][self.M_R:, :]
                    ls_source = next_layer["W_sec_host"]
                    ls_target.copy_(ls_source, non_blocking=True)
                    self.events_LS_ready[i+1].record()
    
    def forward_compute_(self, x):
        """
        Forward pass through the offloaded MLP.
        """
        for i in range(self.num_layers):
            layer_data = self.layers[i]
            
            # 1. Resident Compute (L_R + L_S)
            self.layer_started[i].record(torch.cuda.current_stream())
            torch.cuda.current_stream().wait_event(self.events_LS_ready[i])
            y_resident = self.mm(layer_data["W_resident_gpu"][:self.M_R + self.M_S, :], x)
            
            # 2. Main Offload Compute (L_O)
            torch.cuda.current_stream().wait_event(self.events_LO_ready[i])
            y_offload = self.mm(self.scratchpad, x)

            # 3. Combine & Act
            y_combined = torch.cat([y_resident, y_offload], dim=0)
            x = self.relu(y_combined)
        return x

    def forward(self, x):
        # Start transfers
        self.start_transfers()

        # Loop through layers
        return self.forward_compute(x)

    def create_resident_baseline(self):
        """
        Reconstructs the full model on the GPU for validation.
        Returns a list of full GPU tensors [H, H].
        """
        full_layers = []
        for layer in self.layers:
            # 1. Get Resident part
            # part_res = layer["W_resident_gpu"] # [M_R + M_S, H]
            # # 2. Get Main Offload part (move back to GPU)
            # part_main = layer["W_main_host"].to("cuda") # [M_O, H]
            
            # # Concatenate to form original [H, H] matrix
            # # Note: Resident part already contains L_S at the bottom, so we just append Main.
            # W_full = torch.cat([part_res, part_main], dim=0)
            # full_layers.append(W_full)
            full_layers.append(layer["W_resident_gpu"].to("cuda"))
        return full_layers
    
    def run_baseline(self, x, full_weights):
        """
        Forward pass using fully resident weights.
        """
        for W in full_weights:
            y = self.mm(W, x)
            x = self.relu(y)
        return x
    
    def run_validation(self, x, full_weights):
        """
        Forward pass with fully resident weights with same computation as offloaded.
        Handles cases where GPU kernels are non-deterministic.
        """
        for W in full_weights:
            y_A = self.mm(W[:self.M_R + self.M_S, :], x)
            y_B = self.mm(W[self.M_R + self.M_S:, :], x)
            y = torch.cat([y_A, y_B], dim=0)
            # y = torch.cat([self.mm(W[:self.M_R, :], x), self.mm(W[self.M_R:, :], x)], dim=0)
            x = self.relu(y)
        return x

# ==========================================
# Validation & Benchmarking Logic
# ==========================================
def run_benchmark(args):
    # --- Configuration ---
    H = args.hidden_dim
    L = args.num_layers
    b = args.batch_size
    ratio = args.offload_ratio
    
    if args.compile_mode == 0:
        compile_mode = None
    elif args.compile_mode == 1:
        compile_mode = "reduce-overhead"
    else:
        compile_mode = "max-autotune"

    # Instantiate Model
    # We maintain the ratio logic: Main=ratio, Secondary=ratio^2
    model = OffloadedMLP(hidden_dim=H, num_layers=L, compile_mode=compile_mode,
                         offload_ratio_main=ratio, 
                         offload_ratio_secondary=ratio**2).cuda()

    x_input = torch.randn(H, b, device="cuda", dtype=torch.float16) * 0.01
    
    # Metrics Setup
    total_weights_elements = L * H * H
    total_data_bytes = total_weights_elements * 2 # FP16 = 2 bytes
    
    print("\n" + "="*85)
    print(f"{'Offloaded FFN Pipeline Benchmark':^85}")
    print("="*85)
    print(f"{'Batch Size (b)':<25} : {b}")
    print(f"{'Hidden Dim (H)':<25} : {H}")
    print(f"{'Layers (L)':<25} : {L}")
    print(f"{'Compile Mode':<25} : {compile_mode if compile_mode else 'Disabled'}")
    print("-" * 85)
    print(f"{'Offload Ratio (Main)':<25} : {ratio:.2f}")
    print(f"{'Total Weight Size':<25} : {total_data_bytes / 1024**3:.2f} GB")
    print("=" * 85)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # -------------------------------------------------------
    # 1. Fully Resident Baseline
    # -------------------------------------------------------
    print("  -> Benchmarking Fully Resident Baseline...")
    resident_weights = model.create_resident_baseline()

    # Warmup
    for _ in range(5):
        torch.compiler.cudagraph_mark_step_begin()
        _ = model.baseline(x_input, resident_weights)
    torch.cuda.synchronize()
    
    # Measure
    start_event.record()
    for _ in range(20):
        torch.compiler.cudagraph_mark_step_begin()
        out_baseline = model.baseline(x_input, resident_weights).clone()
    end_event.record()
    torch.cuda.synchronize()
    
    time_res = start_event.elapsed_time(end_event) / 20.0
    bw_res = (total_data_bytes / 1e9) / (time_res / 1000.0)
    
    # -------------------------------------------------------
    # 2. Offloaded FFN
    # -------------------------------------------------------
    print("  -> Benchmarking Offloaded Pipeline...")
    
    # Warmup
    for _ in range(5):
        torch.compiler.cudagraph_mark_step_begin()
        _ = model(x_input)
    torch.cuda.synchronize()
    
    # Measure
    start_event.record()
    for _ in range(20):
        torch.compiler.cudagraph_mark_step_begin()
        out_offload = model(x_input).clone()
    end_event.record()
    torch.cuda.synchronize()
    
    time_off = start_event.elapsed_time(end_event) / 20.0
    bw_off = (total_data_bytes / 1e9) / (time_off / 1000.0)
    
    # -------------------------------------------------------
    # 3. Results Output
    # -------------------------------------------------------
    def calc_diff(time_val):
        val = ((time_val - time_res) / time_res) * 100
        return f"{val:+.2f}%"

    print("\n" + "=" * 85)
    print(f"{'Benchmark Results':^85}")
    print("=" * 85)
    print(f"{'Metric':<30} | {'Time (ms)':<12} | {'Eff. BW':<18} | {'Note':<15}")
    print("-" * 85)
    print(f"{'1. Fully Resident GPU':<30} | {time_res:<12.4f} | {bw_res:<7.2f} GB/s {'':<8} | {'Baseline':<15}")
    print(f"{'2. Offloaded Pipeline':<30} | {time_off:<12.4f} | {bw_off:<7.2f} GB/s {'':<8} | {calc_diff(time_off):<15}")
    print("-" * 85)

    # -------------------------------------------------------
    # 4. Validation
    # -------------------------------------------------------
    max_diff = 1e-3
    if not args.default_baseline:
        out_baseline = model.validation(x_input, resident_weights)
        max_diff = 0.0 # Strict check if we use validation mode
    
    # If using default baseline (pure torch.mm loop), it might differ slightly from split logic
    # but here we usually compare against the 'validation' path unless arg says otherwise.
    # The original code compared `out_offload` vs `out_baseline` (from validation()).
    # out_baseline = model.baseline(x_input, resident_weights)
    diff = (out_offload - out_baseline).abs().max()
    
    print(f"\nValidation Delta (Max Diff): {diff.item():.6e}")
    if diff.item() <= max_diff:
        print("SUCCESS: Offloaded FFN matches Fully Resident FFN within tolerance.")
    else:
        print(f"WARNING: Numerical divergence detected (Max Diff: {diff.item():.6e}).")
    print("=" * 85)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Offloaded FFN Pipeline")
    
    parser.add_argument("-H", "--hidden-dim", type=int, default=16384, help="Hidden Dimension (H)")
    parser.add_argument("-L", "--num-layers", type=int, default=1, help="Number of Layers")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="Batch Size (b)")
    parser.add_argument("-r", "--offload-ratio", type=float, default=0.01, 
                        help="Fraction of matrix rows offloaded to CPU (Main). Secondary is ratio^2.")
    parser.add_argument("--default-baseline", action="store_true", help="Use default baseline.", default=False)
    parser.add_argument("-C", "--compile-mode", type=int, default=0,
                        choices=[0, 1, 2], help="Torch Compile Mode (0=none, 1=reduce-overhead, 2=max-autotune)")
    
    args = parser.parse_args()
    
    run_benchmark(args)