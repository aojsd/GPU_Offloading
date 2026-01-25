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

VALUES_TO_SWEEP = list(range(0, 51))

# Command Template (We will append --profile dynamically for Run 2)
APP_CMD_TEMPLATE = f"python {PROJECT_ROOT}/src/benchmark_paged_transformer.py {{extra_args}} -r {{ratio}}"

# Nsight Systems Template
NSYS_CMD_TEMPLATE = (
    "nsys profile "
    "--trace=cuda,nvtx "
    "--gpu-metrics-device=all "
    "--sample=process-tree "
    "--output={prefix_path} "
    "--force-overwrite=true "
    "{app_cmd}"
)

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
    results = {
        "Total DRAM (%)": 0.0, "DRAM Read (%)": 0.0, "DRAM Write (%)": 0.0,
        "SM Active (%)": 0.0, "Tensor Active (%)": 0.0
    }
    if not os.path.exists(db_file):
        print(f"    [Error] DB file {db_file} not found.")
        return results

    try:
        conn = sqlite3.connect(db_file)
        query_range = f"SELECT start, end FROM NVTX_EVENTS WHERE text = '{NVTX_TAG_FILTER}'"
        ranges = pd.read_sql_query(query_range, conn)

        if ranges.empty:
            print(f"    [Warn] No NVTX ranges named '{NVTX_TAG_FILTER}' found.")
            conn.close()
            return results

        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(TARGET_INFO_GPU_METRICS)")
        cols = [info[1] for info in cursor.fetchall()]
        id_col = 'metricId' if 'metricId' in cols else 'id'
        name_col = 'metricName' if 'metricName' in cols else 'name'

        metric_map = {}
        for pattern, display_key in METRICS_CONFIG.items():
            query_id = f"SELECT {id_col} FROM TARGET_INFO_GPU_METRICS WHERE {name_col} LIKE '%{pattern}%'"
            try:
                m_id = pd.read_sql_query(query_id, conn).iloc[0][id_col]
                metric_map[display_key] = m_id
            except (IndexError, pd.errors.DatabaseError):
                metric_map[display_key] = None

        trial_stats = []
        for _, row in ranges.iterrows():
            start_ns, end_ns = row['start'], row['end']
            current_trial = {}
            for display_key, m_id in metric_map.items():
                if m_id is None:
                    current_trial[display_key] = 0.0
                    continue
                query_data = f"SELECT value FROM GPU_METRICS WHERE metricId = {m_id} AND timestamp >= {start_ns} AND timestamp <= {end_ns}"
                data = pd.read_sql_query(query_data, conn)
                current_trial[display_key] = data['value'].mean() if not data.empty else 0.0
            
            current_trial["Total DRAM (%)"] = current_trial.get("DRAM Read (%)", 0) + current_trial.get("DRAM Write (%)", 0)
            trial_stats.append(current_trial)

        if trial_stats:
            df = pd.DataFrame(trial_stats)
            results.update(df.mean().to_dict())

        conn.close()
    except Exception as e:
        print(f"    [Error] SQLite extraction failed: {e}")
    
    return results

# ==========================================
#           PART 2: PARSER LOGIC
# ==========================================

def parse_time_to_ms(time_str):
    time_str = time_str.lower().strip()
    if time_str.endswith('us'): return float(time_str.replace('us', '')) / 1000.0
    elif time_str.endswith('ms'): return float(time_str.replace('ms', ''))
    elif time_str.endswith('s'): return float(time_str.replace('s', '')) * 1000.0
    return 0.0

def parse_baseline_log(log_text):
    """Parses standard metrics from Run 1."""
    data = {}
    patterns = {
        "Avg Step Time (ms)": r"Average Step Time:\s+([\d\.]+)\s*ms",
        "Weights Loaded (GB)": r"Weights Loaded:\s+([\d\.]+)\s*GB",
        "KV Cache Read (GB)": r"KV Cache Read:\s+([\d\.]+)\s*GB",
        "KV Cache Written (GB)": r"KV Cache Written:\s+([\d\.]+)\s*GB",
        "Total Memory IO (GB)": r"Total Memory IO:\s+([\d\.]+)\s*GB",
        "Effective Memory BW (GB/s)": r"Effective Memory BW:\s+([\d\.]+)\s*GB/s",
        "Effective Tokens/s": r"Effective Tokens/s:\s+([\d\.]+)\s*tok/s",
        "Peak GPU Mem (Alloc GB)": r"Peak GPU Mem \(Alloc\):\s+([\d\.]+)\s*GB",
    }
    for key, regex in patterns.items():
        match = re.search(regex, log_text)
        data[key] = float(match.group(1)) if match else 0.0
    return data

def parse_profile_log(log_text):
    """Parses ONLY the kernel table from Run 2."""
    paged_attn_time = 0.0
    lines = log_text.split('\n')
    for line in lines:
        if "vllm::paged_attention_v2_kernel" in line and "reduce" not in line:
            tokens = line.split()
            if len(tokens) > 12:
                try:
                    paged_attn_time = parse_time_to_ms(tokens[-10])
                except ValueError: pass
            break
    return {"Paged Attention Runtime (ms)": paged_attn_time}

def confirm_overwrite(filename):
    while True:
        r = input(f"File '{filename}' already exists. Overwrite? (y/n): ").strip().lower()
        if r in ('y', 'yes'): return True
        elif r in ('n', 'no'): return False

# ==========================================
#           PART 3: MAIN DRIVER
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark Suite: 3-Pass (Baseline, App Profile, Nsys)")
    parser.add_argument("filename", help="Output CSV filename (Required)")
    parser.add_argument("--extra_args", default="--decode 158000", help="Additional benchmark args")
    args = parser.parse_args()
    
    csv_filename = args.filename
    extra_args = args.extra_args

    headers = [
        "Offload Ratio (%)", "Total DRAM (%)", "DRAM Read (%)", "DRAM Write (%)",
        "Effective Memory BW (GB/s)", "Effective Tokens/s", "SM Active (%)", "Tensor Active (%)",
        "Avg Step Time (ms)", "Paged Attention Runtime (ms)", "Peak GPU Mem (Alloc GB)",
        "Total Memory IO (GB)", "Weights Loaded (GB)", "KV Cache Read (GB)", "KV Cache Written (GB)"
    ]

    if os.path.exists(csv_filename):
        if not confirm_overwrite(csv_filename):
            print("Aborting.")
            sys.exit(0)
        else:
            print(f"Overwriting '{csv_filename}'...")

    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

    print(f"--- Starting Benchmark Suite ({len(VALUES_TO_SWEEP)} runs) ---")
    print(f"Temp Directory: /tmp\n")

    for val in VALUES_TO_SWEEP:
        ratio_str = f"0.{val:02d}"
        prefix_name = f"offload_bench_{val}_{int(time.time())}"
        temp_dir = "/tmp"
        
        # Define 3 Log Files
        log_run1 = os.path.join(temp_dir, f"{prefix_name}_run1_base.txt")
        log_run2 = os.path.join(temp_dir, f"{prefix_name}_run2_prof.txt")
        log_run3 = os.path.join(temp_dir, f"{prefix_name}_run3_nsys.txt")
        
        prefix_nsys_out = os.path.join(temp_dir, prefix_name)
        sqlite_file = os.path.join(temp_dir, f"{prefix_name}.sqlite")
        rep_file = os.path.join(temp_dir, f"{prefix_name}.nsys-rep")

        print(f">>> Processing Ratio: {ratio_str}")
        
        # Base command for Runs 1 & 3
        base_cmd = APP_CMD_TEMPLATE.format(ratio=ratio_str, extra_args=extra_args)
        # Profile command for Run 2 (Add --profile)
        prof_cmd = f"{base_cmd} --profile"

        # --- RUN 1: BASELINE (Performance Metrics) ---
        print("    [1/3] Running Baseline (Silent)...")
        cmd_run1 = f"{base_cmd} > {log_run1} 2>&1"
        try:
            proc = subprocess.run(cmd_run1, shell=True)
            if proc.returncode != 0:
                print(f"    [Error] Run 1 failed.")
                continue
        except Exception as e:
            print(f"    [Error] Run 1 execution failed: {e}")
            continue

        # --- RUN 2: APP PROFILER (Kernel Table) ---
        print("    [2/3] Running App Profiler (Silent)...")
        cmd_run2 = f"{prof_cmd} > {log_run2} 2>&1"
        try:
            proc = subprocess.run(cmd_run2, shell=True)
            if proc.returncode != 0:
                print(f"    [Error] Run 2 failed.")
                continue
        except Exception as e:
            print(f"    [Error] Run 2 execution failed: {e}")
            continue

        # --- RUN 3: NSYS PROFILER (Hardware Metrics) ---
        # Note: We use base_cmd here to avoid overhead of --profile during nsys
        print("    [3/3] Running Nsight Systems (Silent)...")
        full_nsys_cmd = NSYS_CMD_TEMPLATE.format(prefix_path=prefix_nsys_out, app_cmd=base_cmd)
        cmd_run3 = f"{full_nsys_cmd} > {log_run3} 2>&1"
        try:
            proc = subprocess.run(cmd_run3, shell=True)
            if proc.returncode != 0:
                print(f"    [Error] Run 3 failed.")
                continue

            export_cmd = f"nsys export --type sqlite --output {sqlite_file} {rep_file} --force-overwrite=true"
            subprocess.run(export_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"    [Error] Run 3 execution failed: {e}")
            continue

        # --- DATA ANALYSIS ---
        combined_stats = {"Offload Ratio (%)": val}

        # 1. Parse Baseline
        if os.path.exists(log_run1):
            with open(log_run1, 'r') as f:
                combined_stats.update(parse_baseline_log(f.read()))
        else:
             print("    [Error] Run 1 log missing.")

        # 2. Parse Kernel Table
        if os.path.exists(log_run2):
            with open(log_run2, 'r') as f:
                combined_stats.update(parse_profile_log(f.read()))
        else:
             print("    [Error] Run 2 log missing.")

        # 3. Parse Hardware
        combined_stats.update(get_hardware_metrics(sqlite_file))
        
        # Save Row
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writerow(combined_stats)

        # --- CLEANUP ---
        try:
            for fpath in [log_run1, log_run2, log_run3, sqlite_file, rep_file]:
                if os.path.exists(fpath): os.remove(fpath)
        except OSError:
            pass
        
        print(f"    Done (Ratio {val}% saved).")

    print("\n--- Sweep Complete ---")

if __name__ == "__main__":
    main()