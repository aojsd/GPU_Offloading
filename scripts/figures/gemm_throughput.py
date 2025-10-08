import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

# --- Global Bandwidth Constants (GB/s) ---
PCIE_BW = 55      # Approx. peak for PCIe 5.0 x16
NVLINK_BW = 132   # Approx. peak for H100 NVLink (unidirectional)
C2C_BW = 360      # Approx. peak for GH200 C2C (unidirectional)

def main():
    """
    This script processes a single CSV file to plot GPU effective throughput vs. memory offload ratio.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Plot GPU effective throughput vs. memory offload ratio from a CSV file.')
    parser.add_argument('input_csv', type=str, help='Path to the input CSV file.')
    parser.add_argument('-o', '--output', type=str, default='gemm_throughput.png', help='Path to save the output plot file. Defaults to gemm_throughput.png.')
    parser.add_argument('-t', '--title', type=str, default='', help='Title for the plot.')
    args = parser.parse_args()

    # --- Plot Styling Parameters (Tunable) ---
    FONTSIZE_AXIS = 12
    FONTSIZE_TITLE = 12
    FIG_WIDTH = 8
    FIG_HEIGHT = 4

    # --- Data Loading and Processing ---
    try:
        data = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"Error: The file '{args.input_csv}' was not found. Aborting.")
        return
    except Exception as e:
        print(f"Error reading or processing file: {e}")
        return

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Plot the GPU Throughput vs. Offload Ratio
    ax.plot(
        data['OffloadRatio'] * 100,
        data['GPU_Throughput_GBs'],
        marker='o',
        markersize=5,
        linestyle='-',
        label='Aggregate GEMM Throughput'
    )

    # Add horizontal lines for reference bandwidths
    ax.axhline(y=PCIE_BW, color='g', linestyle='--', label=f'PCIe 5.0 x16 BW ({PCIE_BW} GB/s)')
    ax.axhline(y=NVLINK_BW, color='r', linestyle='--', label=f'NVLink BW ({NVLINK_BW} GB/s)')
    ax.axhline(y=C2C_BW, color='purple', linestyle='--', label=f'NVLink C2C BW ({C2C_BW} GB/s)')

    # --- Final Figure Adjustments ---
    ax.set_xlabel('Memory Offload Ratio (%)', fontsize=FONTSIZE_AXIS)
    ax.set_ylabel('Throughput (GB/s)', fontsize=FONTSIZE_AXIS)
    ax.set_title(args.title, fontweight='bold', fontsize=FONTSIZE_TITLE)
    ax.grid(True)
    ax.legend(fontsize=12)

    # Adjust layout to prevent labels/titles from overlapping
    plt.tight_layout()

    # Save the plot
    plt.savefig(args.output, bbox_inches='tight')
    print(f"\nPlot saved to '{args.output}'")

if __name__ == '__main__':
    main()