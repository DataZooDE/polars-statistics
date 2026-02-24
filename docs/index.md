# polars-statistics

High-performance statistical testing and regression for [Polars](https://pola.rs/) DataFrames, powered by Rust.

---

## Features

- **Native Polars Expressions** — Full support for `group_by`, `over`, and lazy evaluation
- **Statistical Tests** — Parametric, non-parametric, distributional, correlation, categorical, and TOST equivalence tests
- **Regression Models** — OLS, Ridge, Elastic Net, WLS, Quantile, Isotonic, GLMs, ALM (24+ distributions)
- **Formula Syntax** — R-style formulas with polynomial and interaction effects
- **Diagnostics** — Condition number, quasi-separation detection, count sparsity checks
- **High Performance** — Rust-powered with zero-copy data transfer and automatic parallelization

## Installation

```bash
pip install polars-statistics
```

## Quick Example

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "group": ["A"] * 50 + ["B"] * 50,
    "y": [...],
    "x1": [...],
    "x2": [...],
})

# OLS regression per group
result = df.group_by("group").agg(
    ps.ols("y", "x1", "x2").alias("model")
)

# Extract results
result.with_columns(
    pl.col("model").struct.field("r_squared"),
    pl.col("model").struct.field("coefficients"),
)
```

## Examples

| Example | Description |
|---------|-------------|
| [Hypothesis Testing](examples/hypothesis-testing.md) | Check assumptions, choose tests, interpret results |
| [Regression Workflow](examples/regression-workflow.md) | Fit, summarize, predict, diagnose |
| [Group-wise Analysis](examples/group-analysis.md) | `group_by` and `over` patterns |
| [A/B Testing](examples/ab-testing.md) | Proportions, equivalence, per-segment analysis |

## What's in the Docs

| Section | Description |
|---------|-------------|
| [Getting Started](getting-started.md) | Installation and first examples |
| [API Conventions](api/conventions.md) | Common patterns across all functions |
| [Statistical Tests](api/tests/parametric.md) | 30+ hypothesis tests |
| [Regression](api/regression/linear.md) | Linear, GLM, ALM, dynamic models |
| [Model Classes](api/classes/linear.md) | Direct Python class access |
| [R Validation](validation/index.md) | R-vs-Rust numerical agreement with reference values |
| [Output Structures](api/outputs.md) | Return type definitions |

## Links

- [GitHub Repository](https://github.com/DataZooDE/polars-statistics)
- [PyPI Package](https://pypi.org/project/polars-statistics/)
- [Polars Documentation](https://docs.pola.rs/)
