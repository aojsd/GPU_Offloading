import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import sys
import math

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
#               CONFIGURATION
# ==========================================

PREFIX = f"{PROJECT_ROOT}/data/full_transformers/gh200_32L_4096H_D150K"
PREFILL = lambda x: (f"{PREFIX}_P{x}.csv", f"{x}-Prefill")
FILES_TO_PLOT = [
    (f"{PREFIX}.csv", "Decode-Only"),
    PREFILL(256),
    PREFILL(512),
    PREFILL(768),
    PREFILL(1024),
    PREFILL(1280),
    PREFILL(1536),
    PREFILL(1792),
    PREFILL(2048),
    # Add more files here...
]

OUTPUT_IMAGE = "bandwidth_util.pdf"
FIG_SIZE_PER_ROW = 2.5  # Height of each subplot row
FIG_WIDTH = 4

# Styling for the 3 metrics
LINE_STYLES = {
    "Total DRAM (%)": {"color": "black", "linestyle": "-", "linewidth": 2, "label": "Total"},
    "DRAM Read (%)":  {"color": "tab:blue", "linestyle": "--", "linewidth": 1.5, "label": "Read"},
    "DRAM Write (%)": {"color": "tab:red",  "linestyle": ":",  "linewidth": 1.5, "label": "Write"}
}

# ==========================================
#                PROCESSING
# ==========================================

def process_and_plot():
    num_files = len(FILES_TO_PLOT)
    if num_files == 0:
        print("No files to plot.")
        return

    # Create Subplots (Vertical Stack)
    fig, axes = plt.subplots(nrows=num_files, ncols=1, figsize=(FIG_WIDTH, FIG_SIZE_PER_ROW * num_files), sharex=True)
    
    # Handle single file case (axes is not a list)
    if num_files == 1:
        axes = [axes]

    plotted_any = False

    for ax, (filepath, label) in zip(axes, FILES_TO_PLOT):
        if not os.path.exists(filepath):
            ax.text(0.5, 0.5, f"File not found:\n{filepath}", ha='center', va='center')
            continue

        try:
            df = pd.read_csv(filepath)
            
            # 1. Validation
            required_cols = ["Offload Ratio (%)", "Peak GPU Mem (Alloc GB)", "Total DRAM (%)", "DRAM Read (%)", "DRAM Write (%)"]
            if not all(col in df.columns for col in required_cols):
                ax.text(0.5, 0.5, "Missing columns", ha='center', va='center')
                continue

            # 2. Find Baseline for X-Axis calc
            baseline_row = df[df["Offload Ratio (%)"] == 0]
            if baseline_row.empty:
                print(f"[Skip] {filepath} has no baseline (0% offload).")
                continue
            
            base_peak_mem = baseline_row.iloc[0]["Peak GPU Mem (Alloc GB)"]
            
            # 3. Calculate X-Axis
            current_peaks = df["Peak GPU Mem (Alloc GB)"].astype(float)
            x_values = 1.0 - (current_peaks / base_peak_mem)
            
            # 4. Prepare Plot Data
            plot_df = pd.DataFrame({
                "x": x_values,
                "Total DRAM (%)": df["Total DRAM (%)"],
                "DRAM Read (%)": df["DRAM Read (%)"],
                "DRAM Write (%)": df["DRAM Write (%)"]
            }).sort_values(by="x")

            # 5. Plot the 3 Lines
            for metric, style in LINE_STYLES.items():
                ax.plot(
                    plot_df["x"], 
                    plot_df[metric], 
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    label=style["label"]
                )

            # 6. Subplot Formatting
            ax.set_title(label, fontsize=12, fontweight='bold', pad=10)
            ax.set_ylabel("Utilization (%)", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_ylim(0, 100) # Percent is always 0-100
            
            # Add Legend to the first plot only (or all if preferred)
            ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
            
            plotted_any = True

        except Exception as e:
            print(f"[Error] Processing {filepath}: {e}")

    # ==========================================
    #             GLOBAL FORMATTING
    # ==========================================
    
    if not plotted_any:
        print("No valid data found to plot.")
        return

    # Set common X-label on the bottom-most plot
    axes[-1].set_xlabel("Memory Offload Ratio", fontsize=12)
    
    # Format X-axis as percentage
    axes[-1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0%}'))

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"\nGraph saved to: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    process_and_plot()