# Performance Benchmark: 1M Groups

Benchmarks for polars-statistics regression functions on large-scale grouped data.

The statistical functions are powered by [anofox-statistics](https://github.com/sipemu/anofox-statistics-rs), providing full API parity with R implementations.

## Test Configuration

- **Groups**: 1,000,000
- **Rows per group**: 100
- **Total rows**: 100,000,000
- **Features**: 3 (x1, x2, x3)

## Running the Benchmarks

### Prerequisites

```bash
cd examples/performance_1m_groups
uv sync
```

### Run Individual Benchmark

```bash
uv run python benchmark_ols.py
```

### Run All Benchmarks

```bash
uv run python run_all_benchmarks.py
```

## Benchmark Results (2025-12-12)

**System**: Intel Core i7-6800K @ 3.40GHz, 6 cores (12 threads), 63 GB RAM, Linux 5.15.196-2-MANJARO

### Model Fitting (group_by aggregation)

| Method      |    Time | Peak Memory | Throughput       | Configuration              |
|-------------|---------|-------------|------------------|----------------------------|
| OLS         |  402.8s |    19490 MB |   248,282 rows/s | fit_intercept: true        |
| Ridge       |   81.6s |    19476 MB | 1,225,793 rows/s | lambda: 1.0                |
| WLS         |  312.7s |          ~* |   319,761 rows/s | uniform weights            |
| RLS         |   87.7s |    19474 MB | 1,140,184 rows/s | forgetting_factor: 0.99    |
| Elastic Net |   77.6s |    19476 MB | 1,289,347 rows/s | lambda: 1.0, alpha: 0.5    |

*WLS memory not captured in batch run due to script fix

### Predictions (over window)

| Method      |    Time | Peak Memory | Throughput       |
|-------------|---------|-------------|------------------|
| OLS Predict |  479.3s |    16208 MB |   208,658 rows/s |

## Key Findings

- **Throughput**: 200K-1.3M rows/second depending on method
- **Regularized methods are faster**: Ridge, RLS, and Elastic Net are 4-5x faster than OLS
- **Memory**: ~19-20 GB peak, dominated by Polars' partitioning overhead for 1M groups
- **OLS vs OLS Predict**: Prediction with window functions is ~20% slower than fitting alone

## Why OLS is Slower than Ridge

OLS is 4-5x slower than regularized methods due to algorithmic differences in the [anofox-regression](https://github.com/sipemu/anofox-regression) solver:

| Aspect | OLS | Ridge/RLS/Elastic Net |
|--------|-----|----------------------|
| **Algorithm** | QR decomposition with column pivoting | Normal equations: (X'X + λI)⁻¹ X'y |
| **Rank handling** | Detects and marks aliased/collinear columns | λI guarantees full rank |
| **Column pivoting** | Yes (requires comparisons and swaps) | No |
| **Matrix condition** | May be ill-conditioned | Always well-conditioned |

The column-pivoted QR in OLS is designed to handle rank-deficient and collinear data gracefully, but this robustness comes at a computational cost. The regularization term in Ridge/RLS/Elastic Net makes the matrix always invertible, allowing a simpler and faster solve.

**Performance tip**: If your data is well-conditioned and you don't need collinearity detection, use `ridge(lambda_=1e-10)` for near-OLS results with Ridge performance.

## Comparison with DuckDB

The [anofox-statistics DuckDB benchmark](https://github.com/DataZooDE/anofox-statistics/tree/feature/update-examples-and-benchmarks/examples/performance_1m_groups) uses `ols_fit_predict()`, which is an **incremental streaming algorithm** designed for window functions. It updates sufficient statistics (X'X, X'y) row-by-row rather than performing batch QR decomposition per group.

This is a different algorithm optimized for different use cases:
- **DuckDB `ols_fit_predict`**: Incremental/streaming, O(p²) per row, ideal for window functions
- **Polars `ols`**: Batch fitting with full diagnostics, O(np² + p³) per group, robust to collinearity

Both use the same underlying [anofox-regression](https://github.com/sipemu/anofox-regression) crate, but different solver functions.

## Customization

Modify the `N_ROWS` and `N_GROUPS` constants in benchmark scripts to test different configurations.
