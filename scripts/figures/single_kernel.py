import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def main():
    """
    This script processes a CSV file to plot normalized throughput vs. memory offload ratio.
    The plot is saved in the same directory as the script.
    """
    parser = argparse.ArgumentParser(description='Process and plot throughput data from a CSV file.')
    parser.add_argument('csv_file', type=str, help='Path to the input CSV file')
    args = parser.parse_args()

    # --- Start of Changes ---

    # Get the absolute path of the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- End of Changes ---

    # Load the data from the CSV file
    try:
        data = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"Error: The file '{args.csv_file}' was not found.")
        return

    # Calculate throughput for all data points
    data['Throughput'] = 1 / data['Total_Kernel_Time_ms']
    
    # Set the baseline throughput from the first data point
    throughput_baseline = data['Throughput'].iloc[0]

    # Normalize the throughput
    data['Normalized_Throughput'] = data['Throughput'] / throughput_baseline

    # Create the plot
    plt.figure(figsize=(7, 4))
    plt.plot(data['OffloadRatio'] * 100, data['Normalized_Throughput'], marker='o', linestyle='-')

    # Add the baseline horizontal line
    plt.axhline(y=1, color='r', linestyle='--', label='Baseline Throughput')

    # Add labels and title
    plt.xlabel('Memory Offload Ratio (%)')
    plt.ylabel('Normalized Throughput')
    plt.title('Normalized Throughput vs. Memory Offload Ratio')
    plt.grid(True)
    plt.legend()
    
    # --- Start of Changes ---

    # Define the output filename and join it with the script's directory
    output_filename = 'single_kernel.png'
    output_path = os.path.join(script_dir, output_filename)

    # Save the plot with minimal whitespace
    plt.savefig(output_path, bbox_inches='tight')
    
    # --- End of Changes ---
    
    print(f"Plot saved to '{output_path}'")


if __name__ == '__main__':
    main()