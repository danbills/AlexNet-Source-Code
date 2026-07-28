"""
System Resource Telemetry Logger for AlexNet Training.
Logs CPU %, RAM %, Swap %, Disk I/O (Read/Write MB/s), Disk Space %, and GPU metrics to TensorBoard.
"""

import time
import os
import psutil
from tensorboardX import SummaryWriter


def run_system_monitoring(log_dir: str = "logs/system_telemetry", interval_sec: float = 2.0):
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"=== Starting System Telemetry Monitor (Logging to {log_dir}) ===")
    
    prev_io = psutil.disk_io_counters()
    prev_time = time.time()
    step = 0

    while True:
        time.sleep(interval_sec)
        curr_time = time.time()
        dt = curr_time - prev_time
        
        # CPU Metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_per_core = psutil.cpu_percent(percpu=True)
        
        # Memory Metrics
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk I/O Metrics
        curr_io = psutil.disk_io_counters()
        read_mb_s = ((curr_io.read_bytes - prev_io.read_bytes) / (1024 * 1024)) / dt if dt > 0 else 0.0
        write_mb_s = ((curr_io.write_bytes - prev_io.write_bytes) / (1024 * 1024)) / dt if dt > 0 else 0.0
        prev_io = curr_io
        prev_time = curr_time
        
        # Disk Space
        disk_usage = psutil.disk_usage('/')
        
        # Log to TensorBoard
        step += 1
        writer.add_scalar("System/CPU_Util_Total_%", cpu_percent, step)
        writer.add_scalar("System/RAM_Util_%", mem.percent, step)
        writer.add_scalar("System/RAM_Used_GB", mem.used / (1024**3), step)
        writer.add_scalar("System/Swap_Util_%", swap.percent, step)
        writer.add_scalar("System/Disk_Read_MB_s", read_mb_s, step)
        writer.add_scalar("System/Disk_Write_MB_s", write_mb_s, step)
        writer.add_scalar("System/Disk_Used_%", disk_usage.percent, step)
        
        for idx, core_pct in enumerate(cpu_per_core):
            writer.add_scalar(f"CPU_Cores/Core_{idx:02d}_%", core_pct, step)
            
        writer.flush()


if __name__ == "__main__":
    run_system_monitoring()
