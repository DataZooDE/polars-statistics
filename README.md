# polars-statistics

[![CI](https://github.com/DataZooDE/polars-statistics/actions/workflows/ci.yml/badge.svg)](https://github.com/DataZooDE/polars-statistics/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/DataZooDE/polars-statistics/graph/badge.svg)](https://codecov.io/gh/DataZooDE/polars-statistics)
[![PyPI version](https://badge.fury.io/py/polars-statistics.svg)](https://badge.fury.io/py/polars-statistics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> **Note:** This extension is in early stage development. APIs may change and some features are experimental.

High-performance statistical testing and regression for [Polars](https://pola.rs/) DataFrames, powered by Rust.

## Features

- **Native Polars Expressions**: Full support for `group_by`, `over`, and lazy evaluation
- **Statistical Tests**: Parametric, non-parametric, distributional, and forecast comparison tests
- **Regression Models**: OLS, Ridge, Elastic Net, WLS, GLMs, ALM (24+ distributions)
- **Formula Syntax**: R-style formulas with polynomial and interaction effects
- **High Performance**: Rust-powered with zero-copy data transfer

## Installation

```bash
pip install polars-statistics
```

## Quick Start

All functions work as Polars expressions, integrating with `group_by` and `over`:

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "group": ["A"] * 50 + ["B"] * 50,
    "y": [...],
    "x1": [...],
    "x2": [...],
})

# Run OLS regression per group
result = df.group_by("group").agg(
    ps.ols("y", "x1", "x2").alias("model")
)

# Extract results from struct
result.with_columns(
    pl.col("model").struct.field("r_squared"),
    pl.col("model").struct.field("coefficients"),
)
```

## Statistical Tests

Statistical tests are powered by [anofox-statistics](https://github.com/sipemu/anofox-statistics-rs), providing full API parity with R's statistical functions and validated against R implementations.

```python
# Parametric tests
ps.ttest_ind("treatment", "control", alternative="two-sided")
ps.ttest_paired("before", "after")

# Non-parametric tests
ps.mann_whitney_u("x", "y")
ps.kruskal_wallis("group1", "group2", "group3")

# Normality tests
ps.shapiro_wilk("x")

# Forecast comparison
ps.diebold_mariano("errors1", "errors2", horizon=1)
```

All tests return a struct with `statistic` and `p_value` fields.

## Regression Models

Regression models are powered by [anofox-regression](https://github.com/sipemu/anofox-regression), providing validated implementations against R.

### Expression API

```python
# Linear models
ps.ols("y", "x1", "x2")
ps.ridge("y", "x1", "x2", lambda_=1.0)
ps.elastic_net("y", "x1", "x2", lambda_=1.0, alpha=0.5)

# GLM models
ps.logistic("y", "x1", "x2")      # Binary classification
ps.poisson("y", "x1", "x2")       # Count data

# ALM - 24+ distributions
ps.alm("y", "x1", "x2", distribution="laplace")  # Robust to outliers
```

### Formula Syntax

R-style formulas with polynomial and interaction effects:

```python
# Main effects + interaction
ps.ols_formula("y ~ x1 * x2")  # Expands to: x1 + x2 + x1:x2

# Polynomial regression (centered per group)
ps.ols_formula("y ~ poly(x, 2)")

# Explicit transform
ps.ols_formula("y ~ x1 + I(x^2)")
```

### Predictions with Intervals

```python
df.with_columns(
    ps.ols_predict("y", "x1", "x2", interval="prediction", level=0.95)
        .over("group").alias("pred")
).unnest("pred")  # Columns: prediction, lower, upper
```

### Tidy Coefficient Summary

```python
df.group_by("group").agg(
    ps.ols_summary("y", "x1", "x2").alias("coef")
).explode("coef").unnest("coef")
# Columns: term, estimate, std_error, statistic, p_value
```

## Model Classes

For direct model access outside Polars expressions:

```python
from polars_statistics import OLS, Ridge, Logistic, ALM

# Fit model
model = OLS(compute_inference=True).fit(X, y)
print(model.coefficients, model.r_squared, model.p_values)

# ALM with various distributions
alm = ALM.laplace().fit(X, y)  # Robust to outliers
```

## Test Model Classes

Statistical tests are also available as model classes with `.fit()`, `.statistic`, `.p_value`, and `.summary()`:

```python
from polars_statistics import TTestInd, ShapiroWilk, KruskalWallis
import numpy as np

# Two-sample t-test
test = TTestInd(alternative="two-sided").fit(x, y)
print(test.statistic, test.p_value)
print(test.summary())

# Normality test
test = ShapiroWilk().fit(x)
print(test.p_value)

# Multi-group comparison
test = KruskalWallis().fit(g1, g2, g3)
print(test.summary())
```

Available test classes: `TTestInd`, `TTestPaired`, `BrownForsythe`, `YuenTest`, `MannWhitneyU`, `WilcoxonSignedRank`, `KruskalWallis`, `BrunnerMunzel`, `ShapiroWilk`, `DAgostino`.

## API Reference

See [docs/API_REFERENCE.md](docs/API_REFERENCE.md) for complete documentation of all functions, parameters, and output structures.

## Performance

Built on high-performance Rust libraries:
- **[faer](https://github.com/sarah-ek/faer-rs)**: Fast linear algebra with SIMD
- **Zero-copy**: Direct memory sharing between Python and Rust
- **Automatic parallelization**: For `group_by` operations

## Development

```bash
git clone https://github.com/DataZooDE/polars-statistics.git
cd polars-statistics
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy polars pytest
maturin develop --release
pytest
```

## License

MIT License - see [LICENSE](LICENSE) for details.
