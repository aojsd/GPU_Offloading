#!/bin/bash

# A script to sweep offload ratios for the 'memory_offload' binary
# and collect performance data into a CSV file.

# --- Configuration ---
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
EXP_ROOT="$SCRIPT_DIR/.."
EXECUTABLE=$(realpath "$EXP_ROOT/bin/memory_offload")
START_RATIO=0.00
STEP=0.005
END_RATIO=0.40

# --- Script Logic ---

# Function to display usage information
usage() {
  echo "Usage: $0 [-H <hidden dimension>] [-N <rows>] [-S <seq length/batch size>] [-d] <output_filename.csv>"
  echo "  (Arguments can be in any order)"
  echo
  echo "  -H, -N, -S:   Optional arguments to pass to the '$EXECUTABLE' binary."
  echo "  -d:           Enable debug mode to print the full command and its output."
  echo "  --nvlink      Enable NVLink transfers."
  echo "  <output_filename.csv>: Mandatory argument for the output data file."
  exit 1
}

# --- Argument Parsing ---
declare -a extra_args=()
OUTPUT_FILE=""
DEBUG=0

# Manually parse all arguments to support any order and long options
while (( "$#" )); do
  case "$1" in
    -H|-N|-S) # These flags take a value
      if [ -n "$2" ] && [[ "$2" != -* ]]; then
        extra_args+=("$1" "$2")
        shift 2
      else
        echo "Error: Argument for $1 is missing or invalid." >&2
        usage
      fi
      ;;
    --nvlink)
      extra_args+=("$1")
      shift 1
      ;;
    -d)
      DEBUG=1
      shift 1
      ;;
    -*) # Handle unknown options
      echo "Error: Unknown option $1" >&2
      usage
      ;;
    *) # Handle the positional argument (output file)
      if [ -n "$OUTPUT_FILE" ]; then
        echo "Error: Multiple output files specified ('$OUTPUT_FILE' and '$1')." >&2
        usage
      fi
      OUTPUT_FILE="$1"
      shift 1
      ;;
  esac
done

# Check if the mandatory output file was provided
if [ -z "$OUTPUT_FILE" ]; then
  echo "Error: Output filename is required."
  usage
fi

# --- Main Execution ---

# Check if the executable exists and is executable
if [ ! -x "$EXECUTABLE" ]; then
    echo "Error: Executable '$EXECUTABLE' not found or is not executable."
    echo "Please ensure it's in the same directory and has execute permissions (chmod +x $EXECUTABLE)."
    exit 1
fi

# Write the header row to the CSV file
HEADER="OffloadRatio,Data_Transfer_ms,Compute_Resident_ms,Compute_Offloaded_ms,Comm_Bandwidth_GBs,GPU_Throughput_GBs,Total_Kernel_Time_ms,Total_Compute_Time_ms"
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
      /PCIe Transfer \(H2D\):/      { data_transfer=$4 }
      /NVLink Transfer \(D2D\):/      { data_transfer=$4 }
      /Compute \(Resident Data\):/ { compute_resident=$4 }
      /Compute \(Offloaded Data\):/ { compute_offloaded=$4 }
      /Comm. Bandwidth \(GB\/s\):/   { comm_bw=$4 }
      /GPU Throughput \(GB\/s\):/  { gpu_tp=$4 }
      /Total Kernel Time:/          { kernel_time=$4 }
      /Total Compute Time:/       { compute_time=$4 }
      END {
          print data_transfer","compute_resident","compute_offloaded","comm_bw","gpu_tp","kernel_time","compute_time
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