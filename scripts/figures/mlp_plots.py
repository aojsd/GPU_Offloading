import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def main():
    """
    This script processes one or more CSV files from MLP offloading experiments
    to plot normalized throughput vs. memory offload ratio. Each CSV file is
    plotted on a separate subplot, arranged horizontally.
    """
    # --- User Configuration ---
    # Get path of the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # List of CSV files to plot (update these paths to your MLP data files)
    csv_files = [
        os.path.join(script_dir, '../../data/H100/MLP/h100_MLP12288_S8.csv'),
        os.path.join(script_dir, '../../data/H100/MLP/h100_nvlink_MLP12288_S8.csv'),
        os.path.join(script_dir, '../../data/GH200/MLP/gh200_MLP12288_S8.csv')
    ]
    # Corresponding labels for each file, which will become the subplot titles
    labels = [
        'H100 (PCIe)', 'H100 (NVLink)', 'GH200 (NVLink C2C)'
    ]

    # --- Plot Styling Parameters (Tunable) ---
    FONTSIZE_AXIS = 14
    FONTSIZE_TITLE = 16
    FONTSIZE_LEGEND = 16
    H_SPACE_BETWEEN_PLOTS = 0.1  # Horizontal space between subplots
    V_SPACE_TITLE_TO_GRAPH = 6.0 # Vertical padding for subplot titles
    V_SPACE_LEGEND_TO_GRAPH = -0.25 # Vertical position of the legend relative to subplots

    parser = argparse.ArgumentParser(description='Process and plot MLP throughput data from one or more CSV files.')
    parser.add_argument('-o', '--output', type=str, default='mlp_plots.png', help='Path to save the output plot file. Defaults to mlp_plots.png in the current directory.')
    args = parser.parse_args()

    if len(csv_files) != len(labels):
        print("Error: The number of CSV files must match the number of labels.")
        return

    # Create subplots: 1 row, N columns, with a shared Y-axis
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(csv_files),
        figsize=(5 * len(csv_files), 3), # Adjust figure size based on number of plots
        sharey=True
    )

    # If there's only one plot, 'axes' is not an array, so we wrap it
    if len(csv_files) == 1:
        axes = [axes]

    for i, (csv_file, label) in enumerate(zip(csv_files, labels)):
        ax = axes[i]
        try:
            data = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"Error: The file '{csv_file}' was not found. Aborting.")
            plt.close(fig)  # Close the figure window to prevent a partial plot
            return

        # Find the row with the maximum 'OffloadRatio_r' to get the measured bandwidth
        max_offload_row = data.loc[data['OffloadRatio_r'].idxmax()]
        measured_bandwidth = max_offload_row['Observed_Transfer_BW_GBs']
        print(f"[{label}] Measured Communication Bandwidth: {measured_bandwidth:.2f} GB/s")

        # Calculate throughput for all data points and normalize
        data['Throughput'] = 1 / data['Avg_Forward_Pass_Time_ms']
        throughput_baseline = data['Throughput'].iloc[0]
        data['Normalized_Throughput'] = data['Throughput'] / throughput_baseline

        # Calculate predicted throughput
        non_off_gpu = data['Observed_GPU_Throughput_GBs'].iloc[0]
        x = non_off_gpu / measured_bandwidth
        lambda_pred_throughput = lambda R: min(1, 1 / ((x + 1) * R)) if R != 0 else 1
        data['Predicted_Throughput'] = data['OffloadRatio_r'].apply(lambda_pred_throughput)

        # --- Plotting on the current subplot (ax) ---

        # Add the baseline horizontal line
        ax.axhline(y=1, color='r', linestyle='--', label='No Offloading', zorder=3)

        # Plot the normalized throughput vs. offload ratio
        ax.plot(data['OffloadRatio_r'] * 100, data['Normalized_Throughput'], marker='o', markersize=5, linestyle='-', label='Measured', color='tab:blue', zorder=1)

        # Plot the predicted throughput vs. offload ratio
        ax.plot(data['OffloadRatio_r'] * 100, data['Predicted_Throughput'], marker='.', markersize=2, linestyle='-', label='Performance Model', color='tab:orange', zorder=2)

        # Add labels and title for the subplot
        ax.set_xlabel('Memory Offload Ratio (%)', fontsize=FONTSIZE_AXIS)
        ax.set_title(label, fontweight='bold', fontsize=FONTSIZE_TITLE, pad=V_SPACE_TITLE_TO_GRAPH)
        ax.grid(True)

    # --- Final Figure-Level Adjustments ---

    # Set the shared Y-axis label on the first plot
    axes[0].set_ylabel('Normalized Throughput', fontsize=FONTSIZE_AXIS)

    # Create a shared legend below the subplots
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc='lower center',
        bbox_to_anchor=(0.5, V_SPACE_LEGEND_TO_GRAPH),
        ncol=len(legend_labels), frameon=True, fontsize=FONTSIZE_LEGEND
    )

    plt.subplots_adjust(wspace=H_SPACE_BETWEEN_PLOTS)
    plt.savefig(args.output, bbox_inches='tight')
    print(f"\nPlot saved to '{args.output}'")

if __name__ == '__main__':
    main()