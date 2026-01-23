import subprocess
import sys
import re
import csv
import os
import sqlite3
import pandas as pd
import argparse
import time

# ==========================================
#               CONFIGURATION
# ==========================================

# 1. Sweep Parameters
VALUES_TO_SWEEP = list(range(0, 26))

# 2. Command Template
#    {ratio}      -> Replaced by "0.01", "0.05", etc.
#    {extra_args} -> Replaced by user input (default: "--decode 158000")
APP_CMD_TEMPLATE = "python ../src/benchmark_paged_transformer.py {extra_args} -r {ratio}"

# 3. Nsight Systems Settings
NSYS_CMD_TEMPLATE = (
    "nsys profile "
    "--trace=cuda,nvtx "
    "--gpu-metrics-devices=all "
    "--sample=process-tree "
    "--output={prefix_path} "
    "--force-overwrite=true "
    "{app_cmd}"
)

# 4. Extraction Settings
NVTX_TAG_FILTER = "Trial"
METRICS_CONFIG = {
    "DRAM Read Bandwidth": "DRAM Read (%)",
    "DRAM Write Bandwidth": "DRAM Write (%)",
    "SMs Active": "SM Active (%)",
    "Tensor Active": "Tensor Active (%)"
}

# ==========================================
#           PART 1: EXTRACTOR LOGIC
# ==========================================

def get_hardware_metrics(db_file):
    """
    Connects to the SQLite profile and calculates average hardware utilization.
    """
    results = {
        "Total DRAM (%)": 0.0,
        "DRAM Read (%)": 0.0,
        "DRAM Write (%)": 0.0,
        "SM Active (%)": 0.0,
        "Tensor Active (%)": 0.0
    }

    if not os.path.exists(db_file):
        print(f"    [Error] DB file {db_file} not found.")
        return results

    try:
        conn = sqlite3.connect(db_file)
        
        # 1. Find NVTX Ranges
        query_range = f"SELECT start, end FROM NVTX_EVENTS WHERE text = '{NVTX_TAG_FILTER}'"
        ranges = pd.read_sql_query(query_range, conn)

        if ranges.empty:
            print(f"    [Warn] No NVTX ranges named '{NVTX_TAG_FILTER}' found.")
            conn.close()
            return results

        # 2. Get Metric IDs
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(TARGET_INFO_GPU_METRICS)")
        cols = [info[1] for info in cursor.fetchall()]
        id_col = 'metricId' if 'metricId' in cols else 'id'
        name_col = 'metricName' if 'metricName' in cols else 'name'

        metric_map = {}
        for pattern, display_key in METRICS_CONFIG.items():
            query_id = f"""
            SELECT {id_col} FROM TARGET_INFO_GPU_METRICS 
            WHERE {name_col} LIKE '%{pattern}%'
            """
            try:
                m_id = pd.read_sql_query(query_id, conn).iloc[0][id_col]
                metric_map[display_key] = m_id
            except (IndexError, pd.errors.DatabaseError):
                metric_map[display_key] = None

        # 3. Calculate Averages
        trial_stats = []
        for _, row in ranges.iterrows():
            start_ns, end_ns = row['start'], row['end']
            current_trial = {}

            for display_key, m_id in metric_map.items():
                if m_id is None:
                    current_trial[display_key] = 0.0
                    continue

                query_data = f"""
                SELECT value FROM GPU_METRICS
                WHERE metricId = {m_id} AND timestamp >= {start_ns} AND timestamp <= {end_ns}
                """
                data = pd.read_sql_query(query_data, conn)
                current_trial[display_key] = data['value'].mean() if not data.empty else 0.0
            
            # Derived metric
            current_trial["Total DRAM (%)"] = current_trial.get("DRAM Read (%)", 0) + current_trial.get("DRAM Write (%)", 0)
            trial_stats.append(current_trial)

        # 4. Average across trials
        if trial_stats:
            df = pd.DataFrame(trial_stats)
            mean_stats = df.mean().to_dict()
            results.update(mean_stats)

        conn.close()
    except Exception as e:
        print(f"    [Error] SQLite extraction failed: {e}")
    
    return results


# ==========================================
#           PART 2: PARSER LOGIC
# ==========================================

def parse_application_log(log_text):
    """Parses the text output from the PyTorch script."""
    data = {}
    patterns = {
        "Avg Step Time (ms)": r"Average Step Time:\s+([\d\.]+)\s*ms",
        "Weights Loaded (GB)": r"Weights Loaded:\s+([\d\.]+)\s*GB",
        "KV Cache Read (GB)": r"KV Cache Read:\s+([\d\.]+)\s*GB",
        "KV Cache Written (GB)": r"KV Cache Written:\s+([\d\.]+)\s*GB",
        "Total Memory IO (GB)": r"Total Memory IO:\s+([\d\.]+)\s*GB",
        "Effective Memory BW (GB/s)": r"Effective Memory BW:\s+([\d\.]+)\s*GB/s",
        "Effective Tokens/s": r"Effective Tokens/s:\s+([\d\.]+)\s*tok/s",
    }
    
    for key, regex in patterns.items():
        match = re.search(regex, log_text)
        data[key] = float(match.group(1)) if match else 0.0
    
    return data

def confirm_overwrite(filename):
    while True:
        response = input(f"File '{filename}' already exists. Overwrite? (y/n): ").strip().lower()
        if response in ('y', 'yes'):
            return True
        elif response in ('n', 'no'):
            return False

# ==========================================
#           PART 3: MAIN DRIVER
# ==========================================

def main():
    # --- Arg Parsing ---
    parser = argparse.ArgumentParser(description="Benchmark Suite with Nsight Profiling")
    parser.add_argument("filename", help="Output CSV filename (Required)")
    
    # New Argument for extra flags
    parser.add_argument(
        "--extra_args", 
        default="--decode 158000", 
        help="Additional args to pass to the benchmark command (Default: '--decode 158000')"
    )
    
    args = parser.parse_args()
    csv_filename = args.filename
    extra_args = args.extra_args

    headers = [
        "Offload Ratio (%)",
        "Total DRAM (%)", "DRAM Read (%)", "DRAM Write (%)",
        "Effective Memory BW (GB/s)", "Effective Tokens/s",
        "SM Active (%)", "Tensor Active (%)",
        "Avg Step Time (ms)", "Total Memory IO (GB)",
        "Weights Loaded (GB)", "KV Cache Read (GB)", "KV Cache Written (GB)"
    ]

    # --- File Safety Check (Interactive) ---
    if os.path.exists(csv_filename):
        if not confirm_overwrite(csv_filename):
            print("Aborting.")
            sys.exit(0)
        else:
            print(f"Overwriting '{csv_filename}'...")

    # Initialize CSV
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

    print(f"--- Starting Benchmark Suite ({len(VALUES_TO_SWEEP)} runs) ---")
    print(f"Output: {csv_filename}")
    print(f"Extra Args: {extra_args}")
    print(f"Temp Directory: /tmp\n")

    for val in VALUES_TO_SWEEP:
        ratio_str = f"0.{val:02d}"
        
        # Define paths in /tmp
        prefix_name = f"offload_bench_{val}_{int(time.time())}"
        temp_dir = "/tmp"
        prefix_path = os.path.join(temp_dir, prefix_name)
        
        log_file = f"{prefix_path}_log.txt"
        rep_file = f"{prefix_path}.nsys-rep"
        sqlite_file = f"{prefix_path}.sqlite"
        
        print(f">>> Running: Ratio={ratio_str}")

        # A. Construct Commands
        # Inject extra_args here
        app_cmd = APP_CMD_TEMPLATE.format(ratio=ratio_str, extra_args=extra_args)
        
        full_nsys_cmd = NSYS_CMD_TEMPLATE.format(prefix_path=prefix_path, app_cmd=app_cmd)

        # B. Run NSYS
        try:
            with open(log_file, "w") as outfile:
                proc = subprocess.run(full_nsys_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                outfile.write(proc.stdout)
                
            if proc.returncode != 0:
                print(f"    [Error] Profiling failed. See {log_file}")
                continue
                
        except Exception as e:
            print(f"    [Error] Subprocess failed: {e}")
            continue

        # C. Export to SQLite
        export_cmd = f"nsys export --type sqlite --output {sqlite_file} {rep_file} --force-overwrite=true"
        subprocess.run(export_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # D. Analyze Data
        app_stats = {}
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                app_stats = parse_application_log(f.read())
        else:
             print("    [Error] Log file missing, skipping log parse.")
            
        hw_stats = get_hardware_metrics(sqlite_file)
        
        # E. Save to CSV
        row_data = {
            "Offload Ratio (%)": val,
            **app_stats,
            **hw_stats
        }
        
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writerow(row_data)

        # F. Cleanup
        try:
            if os.path.exists(sqlite_file): os.remove(sqlite_file)
            if os.path.exists(rep_file): os.remove(rep_file)
            if os.path.exists(log_file): os.remove(log_file)
        except OSError as e:
            print(f"    [Warning] Cleanup failed: {e}")
        
        print(f"    Done (Ratio {val}% saved).")

    print("\n--- Sweep Complete ---")

if __name__ == "__main__":
    main()