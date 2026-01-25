import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import sys

# ==========================================
#          PATH DISCOVERY LOGIC
# ==========================================
def find_project_root(target_name="GPU_Offloading"):
    current_path = os.path.abspath(os.getcwd())
    while True:
        head, tail = os.path.split(current_path)
        if tail == target_name:
            return current_path
        if head == current_path: 
            raise FileNotFoundError(f"Could not find directory '{target_name}' in any parent path.")
        current_path = head

try:
    PROJECT_ROOT = find_project_root()
    print(f"Project Root Detected: {PROJECT_ROOT}")
except FileNotFoundError as e:
    print(f"[Error] {e}")
    sys.exit(1)

# ==========================================
#                CONFIGURATION
# ==========================================

# --- NEW: Hardcoded Interconnect Bandwidth (GB/s) ---
INTERCONNECT_BW = 300.0  # e.g., PCIe Gen4 x16 ~24 GB/s effective. Adjust as needed.

# List of tuples: (path_to_csv, legend_label)
PREFIX = f"{PROJECT_ROOT}/data/full_transformers/gh200_32L_4096H_D150K"
FILES_TO_PLOT = [
    (f"{PREFIX}.csv", "Decode-Only"),
    (f"{PREFIX}_P256.csv", "256-Prefill"),
    (f"{PREFIX}_P512.csv", "512-Prefill"),
    (f"{PREFIX}_P768.csv", "768-Prefill"),
    # Add more files here...
]

OUTPUT_IMAGE = "transformer_throughput.pdf"

# Graph Styling
FIG_SIZE = (7.5, 4)
TITLE = "Throughput vs. Memory Offload Ratio"
X_LABEL = "Memory Offload Ratio"
Y_LABEL = "Throughput (steps/sec)"
GRID_STYLE = "--"

# ==========================================
#                PROCESSING
# ==========================================

def process_and_plot():
    plt.figure(figsize=FIG_SIZE)
    
    plotted_any = False
    theoretical_plotted = False # Flag to ensure we only plot theoretical line once

    for index, (filepath, label) in enumerate(FILES_TO_PLOT):
        if not os.path.exists(filepath):
            print(f"[Warning] File not found: {filepath}. Skipping.")
            continue

        try:
            df = pd.read_csv(filepath)
            
            # 1. Validation
            required_cols = ["Offload Ratio (%)", "Peak GPU Mem (Alloc GB)", "Avg Step Time (ms)", "Effective Memory BW (GB/s)"]
            if not all(col in df.columns for col in required_cols):
                print(f"[Skip] {filepath} missing required columns.")
                continue

            # 2. Find Baseline (0% Offload)
            baseline_row = df[df["Offload Ratio (%)"] == 0]
            if baseline_row.empty:
                print(f"[Skip] {filepath} has no row with 'Offload Ratio (%)' == 0.")
                continue
            
            # Extract Baseline Constants
            base_peak_mem = baseline_row.iloc[0]["Peak GPU Mem (Alloc GB)"] # M
            base_bw = baseline_row.iloc[0]["Effective Memory BW (GB/s)"]   # T
            
            if base_peak_mem == 0:
                print(f"[Skip] Baseline memory is 0 in {filepath}.")
                continue

            # 3. Calculate X-Axis (Calculated Offload Ratio)
            current_peaks = df["Peak GPU Mem (Alloc GB)"].astype(float)
            x_values = 1.0 - (current_peaks / base_peak_mem)
            
            # 4. Calculate Y-Axis (Measured Throughput)
            step_times = df["Avg Step Time (ms)"].astype(float)
            y_values = 1000.0 / step_times

            # 5. --- Theoretical Calculation (First File Only) ---
            if index == 0 and not theoretical_plotted:
                print(f"   -> Calculating Theoretical Curve using T={base_bw:.2f} GB/s, M={base_peak_mem:.2f} GB, Bus={INTERCONNECT_BW} GB/s")
                
                # Formula inputs:
                # T  = base_bw (GB/s)
                # M  = base_peak_mem (GB)
                # M_i = current_peaks (GB)
                # BW = INTERCONNECT_BW (GB/s)
                # t_i = max( (2*M - M_i)/T , (M - M_i)/BW )

                # Correction ratios
                compute_coef = 0.95
                IO_coef = 30/32 * 31/32
                
                # Vectorized calculation
                # Term 1: Compute Bound / Bandwidth Bound
                term1 = (2 * base_peak_mem - current_peaks) / base_bw / compute_coef
                
                # Term 2: Interconnect Bound
                term2 = (base_peak_mem - current_peaks) / INTERCONNECT_BW / IO_coef / compute_coef
                
                # Take element-wise max (Result is in seconds)
                t_i = pd.concat([term1, term2], axis=1).max(axis=1)
                
                # Convert to Throughput (steps/sec)
                y_theo = 1.0 / t_i
                
                # Sort for plotting
                theo_data = pd.DataFrame({"x": x_values, "y": y_theo})
                theo_data = theo_data.sort_values(by="x")
                
                plt.plot(
                    theo_data["x"],
                    theo_data["y"],
                    linestyle='--',
                    color='black',
                    linewidth=2,
                    label='Theoretical'
                )
                theoretical_plotted = True
            
            # 6. Plot Actual Data
            plot_data = pd.DataFrame({"x": x_values, "y": y_values})
            plot_data = plot_data.sort_values(by="x")

            plt.plot(
                plot_data["x"], 
                plot_data["y"], 
                marker='o', 
                linestyle='-', 
                linewidth=2, 
                markersize=6, 
                label=label
            )
            
            plotted_any = True
            print(f"Plotted: {label} (Base Mem: {base_peak_mem:.2f} GB)")

        except Exception as e:
            print(f"[Error] Failed to process {filepath}: {e}")

    # ==========================================
    #                FORMATTING
    # ==========================================
    
    if not plotted_any:
        print("No data was plotted. Please check CSV paths and content.")
        return

    plt.title(TITLE, fontsize=14, pad=15)
    plt.xlabel(X_LABEL, fontsize=12)
    plt.ylabel(Y_LABEL, fontsize=12)
    plt.grid(True, linestyle=GRID_STYLE, alpha=0.7)
    plt.ylim(bottom=0)
    plt.legend(fontsize=10)
    
    # Format X-axis as percentage
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"\nGraph saved to: {OUTPUT_IMAGE}")
    # plt.show() 

if __name__ == "__main__":
    process_and_plot()