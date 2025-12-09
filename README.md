# polars-statistics

[![PyPI version](https://badge.fury.io/py/polars-statistics.svg)](https://badge.fury.io/py/polars-statistics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

High-performance statistical testing for [Polars](https://pola.rs/) DataFrames, powered by Rust.

## Features

- **Native Polars Expressions**: Full support for `group_by`, `over`, and lazy evaluation
- **Comprehensive Statistical Tests**: Parametric, non-parametric, distributional, and forecast comparison tests
- **Modern Distribution Tests**: Energy Distance and Maximum Mean Discrepancy (MMD)
- **High Performance**: Rust-powered computations with zero-copy data transfer
- **Regression Models**: OLS, Ridge, Elastic Net, WLS, GLMs, and more (supplementary)

## Installation

```bash
pip install polars-statistics
```

## Quick Start

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "group": ["A"] * 50 + ["B"] * 50,
    "treatment": [1.2, 2.3, 1.8, 2.1, ...],
    "control": [1.0, 2.1, 1.5, 1.9, ...],
})

# Run t-test per group
result = df.group_by("group").agg(
    ps.ttest_ind(pl.col("treatment"), pl.col("control")).alias("ttest")
)

# Extract results
result.with_columns(
    pl.col("ttest").struct.field("statistic"),
    pl.col("ttest").struct.field("p_value"),
)
```

All test functions return a struct with `statistic` and `p_value` fields.

## Expression API

### Parametric Tests

```python
import polars as pl
import polars_statistics as ps

# Independent samples t-test (Welch's by default)
ps.ttest_ind(pl.col("x"), pl.col("y"), alternative="two-sided", equal_var=False)

# Paired samples t-test
ps.ttest_paired(pl.col("before"), pl.col("after"), alternative="two-sided")

# Brown-Forsythe test for equality of variances
ps.brown_forsythe(pl.col("x"), pl.col("y"))

# Yuen's test for trimmed means (robust to outliers)
ps.yuen_test(pl.col("x"), pl.col("y"), trim=0.2)
```

### Non-Parametric Tests

```python
# Mann-Whitney U test (Wilcoxon rank-sum)
ps.mann_whitney_u(pl.col("x"), pl.col("y"))

# Wilcoxon signed-rank test (paired)
ps.wilcoxon_signed_rank(pl.col("x"), pl.col("y"))

# Kruskal-Wallis H test (3+ groups)
ps.kruskal_wallis(pl.col("group1"), pl.col("group2"), pl.col("group3"))

# Brunner-Munzel test for stochastic equality
ps.brunner_munzel(pl.col("x"), pl.col("y"), alternative="two-sided")
```

### Distributional Tests

```python
# Shapiro-Wilk normality test
ps.shapiro_wilk(pl.col("x"))

# D'Agostino-Pearson normality test
ps.dagostino(pl.col("x"))
```

### Forecast Comparison Tests

```python
# Diebold-Mariano test for equal predictive accuracy
ps.diebold_mariano(pl.col("errors1"), pl.col("errors2"), loss="squared", horizon=1)

# Permutation t-test (non-parametric)
ps.permutation_t_test(pl.col("x"), pl.col("y"), n_permutations=999, seed=42)

# Clark-West test for nested model comparison
ps.clark_west(pl.col("restricted_errors"), pl.col("unrestricted_errors"), horizon=1)

# Superior Predictive Ability (SPA) test
ps.spa_test(
    pl.col("benchmark_loss"),
    pl.col("model1_loss"), pl.col("model2_loss"),
    n_bootstrap=999,
    block_length=5.0,
)

# Model Confidence Set (MCS)
ps.model_confidence_set(
    pl.col("model1_loss"), pl.col("model2_loss"), pl.col("model3_loss"),
    alpha=0.1,
    statistic="range",
)

# MSPE-Adjusted SPA test for nested models
ps.mspe_adjusted(
    pl.col("benchmark_errors"),
    pl.col("model1_errors"), pl.col("model2_errors"),
)
```

### Modern Distribution Tests

```python
# Energy Distance test
ps.energy_distance(pl.col("x"), pl.col("y"), n_permutations=999, seed=42)

# Maximum Mean Discrepancy (MMD) test
ps.mmd_test(pl.col("x"), pl.col("y"), n_permutations=999, seed=42)
```

## Working with Group Operations

The expression API integrates seamlessly with Polars' `group_by` and `over`:

```python
df = pl.DataFrame({
    "experiment": ["exp1"] * 100 + ["exp2"] * 100,
    "treatment": [...],
    "control": [...],
})

# Test per experiment
df.group_by("experiment").agg(
    ps.ttest_ind("treatment", "control").alias("ttest"),
    ps.mann_whitney_u("treatment", "control").alias("mwu"),
)

# Window function
df.with_columns(
    ps.shapiro_wilk("treatment").over("experiment").alias("normality")
)

# Lazy evaluation
df.lazy().group_by("experiment").agg(
    ps.brunner_munzel("treatment", "control")
).collect()
```

## API Reference

### Expression Functions

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| **Parametric** | | |
| `ttest_ind` | Independent samples t-test | `alternative`, `equal_var` |
| `ttest_paired` | Paired samples t-test | `alternative` |
| `brown_forsythe` | Test for equality of variances | - |
| `yuen_test` | Trimmed means comparison | `trim` |
| **Non-Parametric** | | |
| `mann_whitney_u` | Mann-Whitney U test | - |
| `wilcoxon_signed_rank` | Wilcoxon signed-rank test | - |
| `kruskal_wallis` | Kruskal-Wallis H test | - |
| `brunner_munzel` | Brunner-Munzel test | `alternative` |
| **Distributional** | | |
| `shapiro_wilk` | Shapiro-Wilk normality test | - |
| `dagostino` | D'Agostino-Pearson test | - |
| **Forecast** | | |
| `diebold_mariano` | Diebold-Mariano test | `loss`, `horizon` |
| `permutation_t_test` | Permutation t-test | `alternative`, `n_permutations`, `seed` |
| `clark_west` | Clark-West nested model test | `horizon` |
| `spa_test` | Superior Predictive Ability | `n_bootstrap`, `block_length`, `seed` |
| `model_confidence_set` | Model Confidence Set | `alpha`, `statistic`, `n_bootstrap` |
| `mspe_adjusted` | MSPE-Adjusted SPA test | `n_bootstrap`, `block_length`, `seed` |
| **Modern** | | |
| `energy_distance` | Energy Distance test | `n_permutations`, `seed` |
| `mmd_test` | Maximum Mean Discrepancy | `n_permutations`, `seed` |

---

## Regression Models (Supplementary)

For users who need regression models, polars-statistics also provides high-performance implementations:

### Linear Models

```python
import numpy as np
from polars_statistics import OLS, Ridge, ElasticNet, WLS, RLS, BLS

X = np.random.randn(100, 3)
y = X @ [1, 2, 3] + np.random.randn(100) * 0.1

# Ordinary Least Squares
ols = OLS(compute_inference=True).fit(X, y)
print(ols.coefficients, ols.r_squared, ols.p_values)

# Ridge Regression (L2)
ridge = Ridge(lambda_=0.1).fit(X, y)

# Elastic Net (L1 + L2)
enet = ElasticNet(lambda_=0.1, alpha=0.5).fit(X, y)

# Weighted Least Squares
weights = np.ones(100)
wls = WLS().fit(X, y, weights)

# Recursive Least Squares (online learning)
rls = RLS(forgetting_factor=0.99).fit(X, y)

# Bounded Least Squares / Non-negative LS
bls = BLS.nnls().fit(np.abs(X), y)  # coefficients >= 0
```

### Generalized Linear Models

```python
from polars_statistics import Logistic, Poisson, NegativeBinomial, Tweedie, Probit, Cloglog

# Logistic Regression
y_binary = (y > 0).astype(float)
logit = Logistic().fit(X, y_binary)
probs = logit.predict_proba(X)

# Poisson Regression (count data)
y_counts = np.random.poisson(5, 100).astype(float)
poisson = Poisson().fit(X, y_counts)

# Negative Binomial (overdispersed counts)
negbin = NegativeBinomial(estimate_theta=True).fit(X, y_counts)
print(negbin.theta_estimated)

# Tweedie GLM (flexible distribution)
tweedie = Tweedie(var_power=1.5).fit(X, y_counts)
# Or use factory methods:
gamma_glm = Tweedie.gamma().fit(X, np.abs(y))

# Probit Regression
probit = Probit().fit(X, y_binary)

# Complementary Log-Log
cloglog = Cloglog().fit(X, y_binary)
```

### Bootstrap Methods

```python
from polars_statistics import StationaryBootstrap, CircularBlockBootstrap

data = np.random.randn(100)

# Stationary bootstrap (random block lengths)
bootstrap = StationaryBootstrap(expected_block_length=5.0, seed=42)
samples = bootstrap.samples(data, n_samples=1000)

# Circular block bootstrap (fixed block length)
cbb = CircularBlockBootstrap(block_length=10, seed=42)
samples = cbb.samples(data, n_samples=1000)
```

### Model Properties

```python
model.coefficients      # Regression coefficients
model.intercept         # Intercept term (if fitted)
model.r_squared         # R-squared (linear models)
model.std_errors        # Standard errors (if compute_inference=True)
model.p_values          # P-values (if compute_inference=True)
model.aic               # Akaike Information Criterion
model.bic               # Bayesian Information Criterion
```

---

## Performance

polars-statistics is built on high-performance Rust libraries:

- **[regress-rs](https://github.com/sipemu/regress-rs)**: Fast regression using [faer](https://github.com/sarah-ek/faer-rs) linear algebra
- **[anofox-statistics](https://github.com/sipemu/anofox-statistics-rs)**: Statistical tests with SIMD optimizations
- **Zero-copy integration**: Direct memory sharing between Python and Rust

## Development

```bash
git clone https://github.com/sipemu/polars-statistics.git
cd polars-statistics
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy polars pytest
maturin develop --release
pytest
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Polars](https://pola.rs/) - Fast DataFrame library
- [PyO3](https://pyo3.rs/) - Rust bindings for Python
- [pyo3-polars](https://github.com/pola-rs/pyo3-polars) - Polars plugin framework
- [faer](https://github.com/sarah-ek/faer-rs) - High-performance linear algebra
