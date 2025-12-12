#!/usr/bin/env python3
"""Elastic Net Fit Benchmark: 1M Groups, 100M Rows, 3 Features

Usage: python benchmark_elasticnet.py
"""

import time

import numpy as np
import polars as pl
import polars_statistics as ps

N_ROWS = 100_000_000
N_GROUPS = 1_000_000

print("=" * 50)
print("Elastic Net Fit Benchmark")
print("=" * 50)
print(f"Configuration: {N_GROUPS:,} groups, {N_ROWS:,} rows, 3 features")
print(f"Parameters: lambda=1.0, alpha=0.5")
print()

print("Generating data...")
gen_start = time.perf_counter()
np.random.seed(42)
df = pl.DataFrame({
    "group_id": np.arange(N_ROWS, dtype=np.int32) % N_GROUPS,
    "x1": np.random.random(N_ROWS).astype(np.float64) * 100,
    "x2": np.random.random(N_ROWS).astype(np.float64) * 50,
    "x3": np.random.random(N_ROWS).astype(np.float64) * 25,
    "y": np.random.random(N_ROWS).astype(np.float64) * 100,
})
gen_elapsed = time.perf_counter() - gen_start
print(f"Data generation: {gen_elapsed:.1f}s")
print()

print("Running Elastic Net regression per group...")
start = time.perf_counter()

result = df.group_by("group_id").agg(
    ps.elastic_net("y", "x1", "x2", "x3", lambda_=1.0, alpha=0.5).alias("model")
)

elapsed = time.perf_counter() - start

print(f"Time: {elapsed:.1f}s")
print(f"Groups processed: {len(result):,}")
print(f"Throughput: {N_ROWS / elapsed:,.0f} rows/sec")
print()
