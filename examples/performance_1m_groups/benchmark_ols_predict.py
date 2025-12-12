#!/usr/bin/env python3
"""OLS Predict Benchmark: 1M Groups, 100M Rows, 3 Features

Usage: python benchmark_ols_predict.py
"""

import time

import numpy as np
import polars as pl
import polars_statistics as ps

N_ROWS = 100_000_000
N_GROUPS = 1_000_000

print("=" * 50)
print("OLS Predict Benchmark (over window)")
print("=" * 50)
print(f"Configuration: {N_GROUPS:,} groups, {N_ROWS:,} rows, 3 features")
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

print("Running OLS predictions per group...")
start = time.perf_counter()

result = df.with_columns(
    ps.ols_predict("y", "x1", "x2", "x3").over("group_id").alias("pred")
)

elapsed = time.perf_counter() - start

print(f"Time: {elapsed:.1f}s")
print(f"Rows processed: {len(result):,}")
print(f"Throughput: {N_ROWS / elapsed:,.0f} rows/sec")
print()
