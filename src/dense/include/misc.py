import os
import sys
import torch
import pynvml
import threading
import time
import csv
import statistics
from tabulate import tabulate
from datetime import datetime
from contextlib import contextmanager

class GPUProfiler:
    def __init__(self, output_file=None, gpu_index=0, interval=0.1, show_live=False):
        self.output_file = output_file
        self.gpu_index = gpu_index
        self.interval = interval
        self.show_live = show_live
        
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._monitor)
        
        # History for summary table
        self.history = {}
        self.throttle_events = set() 

    def _monitor(self):
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            name = pynvml.nvmlDeviceGetName(handle)
        except pynvml.NVMLError as e:
            print(f"NVML Error: {e}")
            return

        print(f"\n--- Profiling GPU: {name} ---")
        if self.output_file:
            print(f"Logging all metrics to {self.output_file}...")
        else:
            print("CSV logging disabled (in-memory only).")

        start_time = time.time()
        
        # Helper to safely get data
        def safe_get(func, *args):
            try:
                return func(*args)
            except pynvml.NVMLError:
                return 0

        # Define headers
        field_names = [
            "Timestamp", "Elapsed_s", 
            "Power_Draw_W", "Power_Limit_W", "Energy_Total_J",
            "Temp_GPU_C", "Fan_Speed_Pct",
            "Util_GPU_Pct", "Util_Mem_Pct", "Util_Encoder_Pct", "Util_Decoder_Pct",
            "Clock_Graphics_MHz", "Clock_SM_MHz", "Clock_Mem_MHz", "Clock_Video_MHz",
            "Mem_Used_MB", "Mem_Free_MB",
            "PCIe_TX_MBs", "PCIe_RX_MBs", "PCIe_Gen", "PCIe_Width",
            "Throttle_Reason_Bitmask"
        ]

        # Initialize history lists (skipping timestamp cols)
        for f in field_names[2:]:
            self.history[f] = []

        # Setup File I/O conditionally
        f = None
        writer = None
        if self.output_file:
            try:
                f = open(self.output_file, 'w', newline='')
                writer = csv.writer(f)
                writer.writerow(field_names)
            except IOError as e:
                print(f"Error opening file {self.output_file}: {e}")
                return

        try:
            while not self.stop_event.is_set():
                now = time.time()
                elapsed = now - start_time
                
                # --- QUERY METRICS ---
                power = safe_get(pynvml.nvmlDeviceGetPowerUsage, handle) / 1000.0
                power_limit = safe_get(pynvml.nvmlDeviceGetEnforcedPowerLimit, handle) / 1000.0
                energy = safe_get(pynvml.nvmlDeviceGetTotalEnergyConsumption, handle) / 1000.0 
                
                temp = safe_get(pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU)
                fan = safe_get(pynvml.nvmlDeviceGetFanSpeed, handle)
                
                try:
                    util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    util_gpu = util_rates.gpu
                    util_mem = util_rates.memory
                except: util_gpu, util_mem = 0, 0
                
                util_enc = safe_get(pynvml.nvmlDeviceGetEncoderUtilization, handle)[0] if safe_get(pynvml.nvmlDeviceGetEncoderUtilization, handle) else 0
                util_dec = safe_get(pynvml.nvmlDeviceGetDecoderUtilization, handle)[0] if safe_get(pynvml.nvmlDeviceGetDecoderUtilization, handle) else 0

                clock_gr = safe_get(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_GRAPHICS)
                clock_sm = safe_get(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_SM)
                clock_mem = safe_get(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_MEM)
                clock_vid = safe_get(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_VIDEO)

                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used = mem_info.used / 1024**2
                    mem_free = mem_info.free / 1024**2
                except: mem_used, mem_free = 0, 0

                pcie_tx = safe_get(pynvml.nvmlDeviceGetPcieThroughput, handle, pynvml.NVML_PCIE_UTIL_TX_BYTES) / 1024
                pcie_rx = safe_get(pynvml.nvmlDeviceGetPcieThroughput, handle, pynvml.NVML_PCIE_UTIL_RX_BYTES) / 1024
                pcie_gen = safe_get(pynvml.nvmlDeviceGetCurrPcieLinkGeneration, handle)
                pcie_width = safe_get(pynvml.nvmlDeviceGetCurrPcieLinkWidth, handle)

                throttle = safe_get(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons, handle)
                if throttle > 0 and throttle != 0x0000000000000001: 
                    self.throttle_events.add(throttle)

                # --- AGGREGATE ---
                row_data = [
                    now, elapsed,
                    power, power_limit, energy,
                    temp, fan,
                    util_gpu, util_mem, util_enc, util_dec,
                    clock_gr, clock_sm, clock_mem, clock_vid,
                    mem_used, mem_free,
                    pcie_tx, pcie_rx, pcie_gen, pcie_width,
                    throttle
                ]

                # 1. Write to CSV (if enabled)
                if writer:
                    writer.writerow(row_data)

                # 2. Store for summary
                for i, key in enumerate(field_names[2:]):
                    self.history[key].append(row_data[i+2])

                # 3. Live Print
                if self.show_live:
                    print(f"T: {elapsed:.1f}s | Pwr: {power:.1f}W | Util: {util_gpu}% | Mem: {mem_used:.0f}MB")
                
                time.sleep(self.interval)

        finally:
            if f:
                f.close()
            pynvml.nvmlShutdown()

    def __enter__(self):
        self.stop_event.clear()
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        self.thread.join()
        self._print_full_summary()

    def _print_full_summary(self):
        if not self.history.get('Power_Draw_W'):
            print("No data collected.")
            return

        stats = []
        
        def get_stat_row(label, key, unit=""):
            data = self.history[key]
            if not data: return [label, "N/A", "N/A", "N/A"]
            return [
                label,
                f"{min(data):.1f}{unit}",
                f"{max(data):.1f}{unit}",
                f"{statistics.mean(data):.1f}{unit}"
            ]

        # Structure the summary groups
        groups = [
            ("POWER & THERMALS", [
                ("Power Draw", "Power_Draw_W", " W"),
                ("Temperature", "Temp_GPU_C", " C"),
                ("Fan Speed", "Fan_Speed_Pct", " %")
            ]),
            ("UTILIZATION", [
                ("GPU Util", "Util_GPU_Pct", " %"),
                ("Memory Util", "Util_Mem_Pct", " %"),
                ("Encoder Util", "Util_Encoder_Pct", " %")
            ]),
            ("MEMORY", [
                ("Memory Used", "Mem_Used_MB", " MB")
            ]),
            ("CLOCKS", [
                ("Graphics Clock", "Clock_Graphics_MHz", " MHz"),
                ("SM Clock", "Clock_SM_MHz", " MHz"),
                ("Memory Clock", "Clock_Mem_MHz", " MHz")
            ]),
            ("PCIE", [
                ("PCIe TX", "PCIe_TX_MBs", " MB/s"),
                ("PCIe RX", "PCIe_RX_MBs", " MB/s"),
                ("PCIe Gen", "PCIe_Gen", "")
            ])
        ]

        for group_name, rows in groups:
            stats.append([f"--- {group_name} ---", "", "", ""])
            for label, key, unit in rows:
                stats.append(get_stat_row(label, key, unit))

        print("\n" + "="*60)
        print("                FULL GPU PROFILING REPORT")
        print("="*60)
        print(tabulate(stats, headers=["Metric", "Min", "Max", "Avg"], tablefmt="simple"))
        
        # --- Decode Throttling ---
        print("\n" + "="*60)
        print("THROTTLING ANALYSIS")
        if not self.throttle_events:
            print("No throttling detected.")
        else:
            print("Warning: The following throttle reasons were detected:")
            reasons_map = {
                0x0000000000000001: "GPU_IDLE",
                0x0000000000000002: "APPLICATIONS_CLOCKS_SETTING",
                0x0000000000000004: "SW_POWER_CAP (Hit Power Limit)",
                0x0000000000000008: "HW_SLOWDOWN (General Hardware Brake)",
                0x0000000000000010: "SYNC_BOOST",
                0x0000000000000020: "SW_THERMAL_SLOWDOWN",
                0x0000000000000040: "HW_THERMAL_SLOWDOWN (Overheating!)",
                0x0000000000000080: "HW_POWER_BRAKE_SLOWDOWN",
                0x0000000000000100: "DISPLAY_CLOCK_SETTING"
            }
            
            for event in self.throttle_events:
                active_reasons = []
                for bit, reason in reasons_map.items():
                    if event & bit:
                        active_reasons.append(reason)
                print(f" - {', '.join(active_reasons)}")
        
        print("="*60)
        if self.output_file:
            print(f"Detailed timeline saved to: {self.output_file}\n")
        else:
            print("Detailed timeline not saved (output_file=None)\n")

# --- USAGE EXAMPLE ---
# with GPUProfiler(output_file='h100_run1.csv', interval=0.1):
#     # ... Your PyTorch Code ...
#     pass

@contextmanager
def suppress_all_output():
    """
    Redirects file descriptors 1 (stdout) and 2 (stderr) to /dev/null.
    This suppresses output from C++ extensions, Triton, and the Python interpreter.
    """
    # Open a pair of null files
    null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
    
    # Save the actual stdout (1) and stderr (2) file descriptors.
    save_fds = [os.dup(1), os.dup(2)]

    try:
        # Assign the null pointers to stdout and stderr.
        os.dup2(null_fds[0], 1)
        os.dup2(null_fds[1], 2)
        yield
    finally:
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(save_fds[0], 1)
        os.dup2(save_fds[1], 2)
        
        # Close the null files and the saved copies
        for fd in null_fds + save_fds:
            os.close(fd)

def compile_if_needed(module, compile_mode):
    if compile_mode is None:
        return module
    else:
        # Suppress compile output and logs
        with suppress_all_output():
            return torch.compile(module, mode=compile_mode)