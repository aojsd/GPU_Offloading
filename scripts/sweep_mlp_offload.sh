#!/bin/bash

# sweep_mlp_offload.sh

# A script to sweep communication ratios for the 'mlp_offload' binary
# and collect performance data into a CSV file.

# --- Configuration ---
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
EXP_ROOT="$SCRIPT_DIR/.."
EXECUTABLE=$(realpath "$EXP_ROOT/bin/mlp_offload")
START_RATIO=0.005
STEP=0.005
END_RATIO=0.25

# --- Script Logic ---

# Function to display usage information
usage() {
  echo "Usage: $0 [-H <hidden dimension>] [-N <rows>] [-B <batch size>] [-d] <output_filename.csv>"
  echo "  (Arguments can be in any order)"
  echo
  echo "  -H, -N, -B:   Optional arguments to pass to the '$EXECUTABLE' binary."
  echo "  -d:           Enable debug mode to print the full command and its output."
  echo "  --nvlink:     Enable NVLink transfers."
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
    -H|-N|-B) # These flags take a value
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
    echo "Please ensure the path is correct and it has execute permissions (chmod +x $EXECUTABLE)."
    exit 1
fi

# Write the header row to the CSV file
HEADER="OffloadRatio_r,TargetCommRatio_x,Compute1_Resident_ms,Compute2_Offloaded_ms,Transfer1_Main_ms,Transfer2_Prefetch_ms,Observed_Transfer_BW_GBs,Observed_GPU_Throughput_GBs,Avg_Per_Layer_Time_ms,Avg_Forward_Pass_Time_ms"
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
  # Calculate x = 1/r for the mlp_offload binary
  x=$(echo "scale=10; 1 / $r" | bc -l)

  echo "Running with offload ratio r=$r (x=$x)"

  full_command=("$EXECUTABLE" "${extra_args[@]}" -x "$x")
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

  # Use awk to parse the specific output format of mlp_offload
  results=$( echo "$program_output" | awk '
      /Compute1 \(Resident\):/      { compute1=$3 }
      /Compute2 \(Offloaded\):/      { compute2=$3 }
      /Transfer1 \(Main\):/          { transfer1=$3 }
      /Transfer2 \(Pre-fetch\):/     { transfer2=$3 }
      /Observed Transfer BW:/       { bw=$4 }
      /Observed GPU Throughput:/   { throughput=$4 }
      /Avg. Per-Layer Time:/       { layer_time=$4 }
      /Avg. Forward Pass Time:/    { pass_time=$5 }
      END {
          print compute1","compute2","transfer1","transfer2","bw","throughput","layer_time","pass_time
      }
  ')

  if [ -n "$results" ]; then
    # Prepend the input ratio (r) and calculated comm ratio (x) to the results
    echo "$r,$x,$results" >> "$OUTPUT_FILE"
  else
    echo "Warning: No data parsed for ratio r=$r. Check debug output if enabled."
  fi
done

echo "--------------------------------------"
echo "✅ Sweep complete. Data saved to '$OUTPUT_FILE'."