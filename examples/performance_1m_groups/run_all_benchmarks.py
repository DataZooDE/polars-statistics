#!/usr/bin/env python3
"""Run all fit_predict benchmarks with memory monitoring.

Usage: python run_all_benchmarks.py
"""

import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil

SCRIPT_DIR = Path(__file__).parent

BENCHMARKS = [
    ("OLS", "benchmark_ols.py", "fit_intercept: true"),
    ("Ridge", "benchmark_ridge.py", "lambda: 1.0"),
    ("WLS", "benchmark_wls.py", "uniform weights"),
    ("RLS", "benchmark_rls.py", "forgetting_factor: 0.99"),
    ("Elastic Net", "benchmark_elasticnet.py", "lambda: 1.0, alpha: 0.5"),
    ("OLS Predict", "benchmark_ols_predict.py", "over window"),
]


def get_system_info():
    """Get system information."""
    cpu_info = "Unknown CPU"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_info = line.split(":")[1].strip()
                    break
    except Exception:
        pass

    mem_gb = psutil.virtual_memory().total / (1024**3)
    cores = psutil.cpu_count(logical=False)
    threads = psutil.cpu_count(logical=True)

    try:
        with open("/proc/version") as f:
            kernel = f.read().split()[2]
    except Exception:
        kernel = "Unknown"

    return {
        "cpu": cpu_info,
        "cores": cores,
        "threads": threads,
        "memory_gb": mem_gb,
        "kernel": kernel,
    }


def run_benchmark_with_memory(script_path: Path) -> dict:
    """Run a benchmark script and monitor peak memory usage."""
    peak_rss = 0
    process = None
    monitor_running = True

    def monitor_memory():
        nonlocal peak_rss
        while monitor_running:
            try:
                if process and process.poll() is None:
                    proc = psutil.Process(process.pid)
                    # Include child processes
                    rss = proc.memory_info().rss
                    for child in proc.children(recursive=True):
                        try:
                            rss += child.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    peak_rss = max(peak_rss, rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(0.5)

    # Start memory monitor
    monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
    monitor_thread.start()

    # Run benchmark
    start = time.perf_counter()
    process = subprocess.Popen(
        [sys.executable, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=SCRIPT_DIR,
    )
    stdout, _ = process.communicate()
    elapsed = time.perf_counter() - start

    monitor_running = False
    monitor_thread.join(timeout=1)

    # Parse time from output
    time_match = re.search(r"Time:\s+([\d.]+)s", stdout)
    benchmark_time = float(time_match.group(1)) if time_match else elapsed

    # Parse throughput
    throughput_match = re.search(r"Throughput:\s+([\d,]+)", stdout)
    throughput = throughput_match.group(1) if throughput_match else "N/A"

    return {
        "time": benchmark_time,
        "peak_memory_mb": peak_rss / (1024 * 1024),
        "throughput": throughput,
        "output": stdout,
        "success": process.returncode == 0,
    }


def main():
    print("=" * 60)
    print("BENCHMARK: 1M Groups (100M Rows), 3 Features")
    print("=" * 60)
    print()

    # System info
    sys_info = get_system_info()
    print(f"System: {sys_info['cpu']}")
    print(f"Cores: {sys_info['cores']} ({sys_info['threads']} threads)")
    print(f"Memory: {sys_info['memory_gb']:.0f} GB")
    print(f"Kernel: {sys_info['kernel']}")
    print(f"Date: {datetime.now().isoformat()}")
    print()

    results = []

    for name, script, config in BENCHMARKS:
        print(f"=== {name} ===")
        script_path = SCRIPT_DIR / script

        if not script_path.exists():
            print(f"  Script not found: {script}")
            results.append({
                "name": name,
                "config": config,
                "time": None,
                "memory": None,
                "throughput": None,
                "success": False,
            })
            continue

        result = run_benchmark_with_memory(script_path)

        if result["success"]:
            print(f"  Time: {result['time']:.1f}s")
            print(f"  Peak Memory: {result['peak_memory_mb']:.0f} MB")
            print(f"  Throughput: {result['throughput']} rows/sec")
        else:
            print(f"  FAILED")
            print(result["output"])

        results.append({
            "name": name,
            "config": config,
            "time": result["time"] if result["success"] else None,
            "memory": result["peak_memory_mb"] if result["success"] else None,
            "throughput": result["throughput"] if result["success"] else None,
            "success": result["success"],
        })
        print()

    # Summary table
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print(f"| {'Method':<15} | {'Time':>8} | {'Peak Memory':>12} | {'Configuration':<30} |")
    print(f"|{'-'*17}|{'-'*10}|{'-'*14}|{'-'*32}|")

    for r in results:
        time_str = f"{r['time']:.1f}s" if r['time'] else "FAILED"
        mem_str = f"{r['memory']:.0f} MB" if r['memory'] else "N/A"
        print(f"| {r['name']:<15} | {time_str:>8} | {mem_str:>12} | {r['config']:<30} |")

    print()
    print("Benchmark complete")


if __name__ == "__main__":
    main()
