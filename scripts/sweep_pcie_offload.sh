#!/bin/bash

# A script to sweep offload ratios for the 'pcie_overlap' binary
# and collect performance data into a CSV file.

# --- Configuration ---
EXP_ROOT="/home/michael/GPU_Offloading"
EXECUTABLE="$EXP_ROOT/src/cuda/pcie_overlap"
START_RATIO=0.005
STEP=0.005
END_RATIO=0.25

# --- Script Logic ---

# Function to display usage information
usage() {
  echo "Usage: $0 [-H <height>] [-N <width>] [-S <inner_dim>] [-d] <output_filename.csv>"
  echo "  (Arguments can be in any order)"
  echo
  echo "  -H, -N, -S: Optional arguments to pass to the '$EXECUTABLE' binary."
  echo "  -d:           Enable debug mode to print the full command and its output."
  echo "  <output_filename.csv>: Mandatory argument for the output data file."
  exit 1
}

# --- Smart Argument Parsing ---

# 1. Pre-parse arguments to separate the output file from the options
declare -a options_array=()
OUTPUT_FILE=""
prev_arg_was_value_flag=false

for arg in "$@"; do
  if [ "$prev_arg_was_value_flag" = true ]; then
    # This argument is the value for the preceding flag (e.g., the '1000' in '-H 1000')
    options_array+=("$arg")
    prev_arg_was_value_flag=false
  elif [[ "$arg" == "-H" || "$arg" == "-N" || "$arg" == "-S" ]]; then
    # This is a flag that requires a value. Note it for the next iteration.
    options_array+=("$arg")
    prev_arg_was_value_flag=true
  elif [[ "$arg" == -* ]]; then
    # This is a simple flag (like -d)
    options_array+=("$arg")
    prev_arg_was_value_flag=false
  else
    # This is not a flag or a value for a flag, so it must be the output file.
    if [ -n "$OUTPUT_FILE" ]; then
      echo "Error: Multiple output files specified ('$OUTPUT_FILE' and '$arg')." >&2
      usage
    fi
    OUTPUT_FILE="$arg"
  fi
done

# 2. Check if the mandatory output file was found
if [ -z "$OUTPUT_FILE" ]; then
  echo "Error: Output filename is required."
  usage
fi

# 3. Reset the script's positional parameters to only contain the options
set -- "${options_array[@]}"

# 4. Now, parse the options using getopts as before
declare -a extra_args=()
DEBUG=0
while getopts ":H:N:S:d" opt; do
  case ${opt} in
    H ) extra_args+=("-H" "$OPTARG") ;;
    N ) extra_args+=("-N" "$OPTARG") ;;
    S ) extra_args+=("-S" "$OPTARG") ;;
    d ) DEBUG=1 ;;
    \? ) echo "Invalid Option: -$OPTARG" >&2; usage ;;
    : ) echo "Invalid Option: -$OPTARG requires an argument" >&2; usage ;;
  esac
done

# --- Main Execution ---

# Check if the executable exists and is executable
if [ ! -x "$EXECUTABLE" ]; then
    echo "Error: Executable '$EXECUTABLE' not found or is not executable."
    echo "Please ensure it's in the same directory and has execute permissions (chmod +x $EXECUTABLE)."
    exit 1
fi

# Write the header row to the CSV file
HEADER="OffloadRatio,PCIe_Transfer_ms,Compute_Resident_ms,Compute_Offloaded_ms,PCIe_Bandwidth_GBs,GPU_Throughput_GBs,Total_Kernel_Time_ms,Total_Compute_Time_ms"
echo "$HEADER" > "$OUTPUT_FILE"

echo "Starting data collection..."
if [ ${#extra_args[@]} -gt 0 ]; then
    echo "Optional arguments passed to executable: ${extra_args[*]}"
fi
if [ $DEBUG -eq 1 ]; then
    echo "Debug mode is ON."
fi
echo "Output will be saved to '$OUTPUT_FILE'"

# --- Main Loop ---
for r in $(seq $START_RATIO $STEP $END_RATIO); do
  echo "Running with offload ratio: $r"

  full_command=("$EXECUTABLE" "${extra_args[@]}" -r "$r")
  program_output=$( "${full_command[@]}" )

  if [ $DEBUG -eq 1 ]; then
    printf "\n--- DEBUG START ---\n"
    printf "COMMAND: "
    printf "%q " "${full_command[@]}"
    printf "\n\n"
    echo "--- PROGRAM OUTPUT ---"
    echo "$program_output"
    echo "--- DEBUG END ---\n"
  fi

  results=$( echo "$program_output" | awk '
      /PCIe Transfer \(HtoD\):/    { pcie_transfer=$4 }
      /Compute \(Resident Data\):/ { compute_resident=$4 }
      /Compute \(Offloaded Data\):/ { compute_offloaded=$4 }
      /PCIe Bandwidth \(GB\/s\):/   { pcie_bw=$4 }
      /GPU Throughput \(GB\/s\):/  { gpu_tp=$4 }
      /Total Kernel Time:/          { kernel_time=$4 }
      /Total Compute Time =/       { compute_time=$5 }
      END {
          print pcie_transfer","compute_resident","compute_offloaded","pcie_bw","gpu_tp","kernel_time","compute_time
      }
  ')

  if [ -n "$results" ]; then
    echo "$r,$results" >> "$OUTPUT_FILE"
  else
    echo "Warning: No data parsed for ratio $r. Check debug output if enabled."
  fi
done

echo "--------------------------------------"
echo "✅ Sweep complete. Data saved to '$OUTPUT_FILE'."