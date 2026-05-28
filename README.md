# polars-statistics

[![CI](https://github.com/DataZooDE/polars-statistics/actions/workflows/ci.yml/badge.svg)](https://github.com/DataZooDE/polars-statistics/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/DataZooDE/polars-statistics/graph/badge.svg)](https://codecov.io/gh/DataZooDE/polars-statistics)
[![PyPI version](https://badge.fury.io/py/polars-statistics.svg)](https://badge.fury.io/py/polars-statistics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> **Note:** This extension is in early stage development. APIs may change and some features are experimental.

High-performance statistical testing and regression for [Polars](https://pola.rs/) DataFrames, powered by Rust.

Usable from **Python** (as a Polars plugin) and from **Rust** (as an rlib that other Rust crates depend on directly — see [Use from Rust](#use-from-rust)).

## Features

- **Native Polars Expressions**: Full support for `group_by`, `over`, and lazy evaluation
- **Statistical Tests**: Parametric, non-parametric, distributional, and forecast comparison tests
- **Regression Models**: OLS, Ridge, Elastic Net, WLS, Quantile, Isotonic, GLMs, ALM (24+ distributions)
- **Diagnostics**: Condition number, quasi-separation detection for GLMs
- **Formula Syntax**: R-style formulas with polynomial and interaction effects
- **Hybrid crate**: `cdylib` (Python wheel) and `rlib` (Rust dependency) from the same source
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

# Correlation tests
ps.pearson("x", "y")                    # Pearson correlation with CI
ps.spearman("x", "y")                   # Spearman rank correlation
ps.kendall("x", "y", variant="b")       # Kendall's tau
ps.distance_cor("x", "y")               # Distance correlation (detects nonlinear)
ps.partial_cor("x", "y", ["z1", "z2"])  # Partial correlation

# Categorical tests
ps.binom_test(successes=7, n=10, p0=0.5)  # Exact binomial test
ps.chisq_test("counts", n_rows=2, n_cols=2)  # Chi-square independence
ps.fisher_exact(a=10, b=2, c=3, d=15)   # Fisher's exact test
ps.mcnemar_test(a=45, b=15, c=5, d=35)  # McNemar's test
ps.cohen_kappa("counts", n_categories=3) # Inter-rater agreement
ps.cramers_v("counts", n_rows=3, n_cols=3) # Association strength
```

All tests return a struct with `statistic` and `p_value` fields.

### TOST Equivalence Tests

Test for practical equivalence using Two One-Sided Tests (TOST) procedure:

```python
# t-test based equivalence
ps.tost_t_test_two_sample("x", "y", delta=0.5, alpha=0.05)
ps.tost_t_test_paired("before", "after", bounds_type="cohen_d", delta=0.3)

# Correlation equivalence (test if correlation is near zero)
ps.tost_correlation("x", "y", delta=0.3, method="pearson")

# Proportion equivalence
ps.tost_prop_two(successes1=45, n1=100, successes2=48, n2=100, delta=0.1)

# Non-parametric and robust equivalence
ps.tost_wilcoxon_paired("x", "y", delta=0.5)
ps.tost_yuen("x", "y", trim=0.2, delta=0.5)  # Trimmed means
ps.tost_bootstrap("x", "y", n_bootstrap=1000)  # Bootstrap-based
```

Returns struct with `estimate`, `ci_lower`, `ci_upper`, `tost_p_value`, `equivalent`.

## Regression Models

Regression models are powered by [anofox-regression](https://github.com/sipemu/anofox-regression), providing validated implementations against R.

### Expression API

```python
# Linear models
ps.ols("y", "x1", "x2")
ps.ridge("y", "x1", "x2", lambda_=1.0)
ps.elastic_net("y", "x1", "x2", lambda_=1.0, alpha=0.5)

# Robust regression
ps.quantile("y", "x1", "x2", tau=0.5)  # Median regression
ps.isotonic("y", "x")                   # Monotonic regression
ps.huber("y", "x1", epsilon=1.35)       # Huber M-estimator (outlier-robust)

# GLM models (with optional Ridge regularization)
ps.logistic("y", "x1", "x2", lambda_=0.1)             # Binary classification (BinomialRegressor)
ps.logistic_regression("y", "x1", "x2", penalty="l2", C=1.0)  # sklearn-style logistic
ps.poisson("y", "x1", "x2")                            # Count data

# ALM - 24+ distributions
ps.alm("y", "x1", "x2", distribution="laplace")  # Robust to outliers

# Diagnostics
ps.condition_number("x1", "x2")            # Multicollinearity check
ps.check_binary_separation("y", "x1")      # Quasi-separation detection
ps.check_count_sparsity("y", "x1")         # Sparse count data check
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

## Use from Rust

`polars-statistics` builds as both a Python extension (`cdylib`) and a Rust library (`rlib`). Other Rust crates can depend on it directly and call the same statistical and regression code that the Python plugin uses — no Python boundary, no FFI overhead.

### Cargo dependency

```toml
[dependencies]
polars = { version = "0.52", features = ["lazy", "partition_by"] }
polars-statistics = { version = "0.4", default-features = false }
```

`default-features = false` disables the `python` feature, so pyo3 / numpy are not linked.

### Calling the fit functions

Every Polars expression has a public Rust-callable counterpart named `<name>_fit` that accepts a `&[Series]` input slice matching the expression's input contract and returns a `PolarsResult<Series>` (a one-row struct with the model output).

```rust
use polars::prelude::*;
use polars_statistics::expressions::wls_fit;

fn main() -> PolarsResult<()> {
    let df = df!(
        "site"   => &["A", "A", "A", "B", "B", "B"],
        "y"      => &[1.0_f64, 3.0, 5.0, 2.0, 5.0, 8.0],
        "weight" => &[1.0_f64, 1.0, 1.0, 1.0, 1.0, 1.0],
        "x1"     => &[0.0_f64, 1.0, 2.0, 0.0, 1.0, 2.0],
    )?;

    for group in df.partition_by(["site"], true)? {
        let y  = group.column("y")?.as_materialized_series().clone();
        let w  = group.column("weight")?.as_materialized_series().clone();
        let x1 = group.column("x1")?.as_materialized_series().clone();
        let with_intercept = Series::new("with_intercept".into(), &[true]);
        let solver         = Series::new("solve_method".into(), &[None::<&str>]);

        let result = wls_fit(&[y, w, with_intercept, solver, x1])?;
        println!("{result:?}");
    }
    Ok(())
}
```

The full runnable version is in [`examples/rust_wls.rs`](examples/rust_wls.rs). Run with:

```bash
cargo run --example rust_wls --no-default-features
```

### Available fit functions

All ~95 Polars expressions have a `*_fit` Rust entry point in `polars_statistics::expressions`:

- **Regression**: `ols_fit`, `ridge_fit`, `elastic_net_fit`, `wls_fit`, `rls_fit`, `bls_fit`, `quantile_fit`, `isotonic_fit`, `huber_fit`
- **GLMs**: `logistic_fit`, `logistic_regression_fit`, `poisson_fit`, `negative_binomial_fit`, `tweedie_fit`, `probit_fit`, `cloglog_fit`, `alm_fit`
- **Diagnostics**: `condition_number_fit`, `check_binary_separation_fit`, `check_count_sparsity_fit`
- **Summaries / predictions**: `ols_summary_fit`, `ols_predict_fit`, … (one per model)
- **Hypothesis tests**: `ttest_ind_fit`, `ttest_paired_fit`, `mann_whitney_fit`, `wilcoxon_fit`, `kruskal_wallis_fit`, `brunner_munzel_fit`, `brown_forsythe_fit`, `yuen_fit`, `shapiro_wilk_fit`, `dagostino_fit`
- **Correlation**: `pearson_fit`, `spearman_fit`, `kendall_fit`, `distance_cor_fit`, `partial_cor_fit`, `semi_partial_cor_fit`, `icc_fit`
- **Categorical**: `binom_test_fit`, `prop_test_one_fit`, `prop_test_two_fit`, `chisq_test_fit`, `fisher_exact_fit`, `mcnemar_test_fit`, `cohen_kappa_fit`, `cramers_v_fit`, …
- **Forecast comparison / TOST / modern**: see `expressions::forecast`, `expressions::tost`, `expressions::modern`.

The input slice layout (which input is `y`, which are scalars, which are `x` columns) is documented above each function — same contract that the Polars plugin uses.

## Documentation

- **[API Reference](docs/api/README.md)** - Complete API documentation
  - [Statistical Tests](docs/api/tests/) - Parametric, non-parametric, TOST equivalence
  - [Regression Models](docs/api/regression/) - Linear, GLM, ALM, dynamic
  - [Model Classes](docs/api/classes/) - Python classes for direct access
  - [Output Structures](docs/api/outputs.md) - Return type definitions

For the legacy monolithic reference, see [docs/API_REFERENCE.md](docs/API_REFERENCE.md).

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
