# polars-statistics

[![PyPI version](https://badge.fury.io/py/polars-statistics.svg)](https://badge.fury.io/py/polars-statistics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

High-performance statistical testing and regression for [Polars](https://pola.rs/) DataFrames, powered by Rust.

## Features

- **Regression Models**: OLS, Ridge, Elastic Net, WLS, Logistic, Poisson
- **Statistical Tests**: t-tests, Mann-Whitney U, Wilcoxon, Kruskal-Wallis, Shapiro-Wilk, D'Agostino, and more
- **Forecast Comparison**: Diebold-Mariano test, permutation t-test
- **Bootstrap Methods**: Stationary Bootstrap, Circular Block Bootstrap
- **Polars Integration**: Expression API with full `group_by` and `over` support
- **High Performance**: Rust-powered computations with zero-copy data transfer

## Installation

```bash
pip install polars-statistics
```

For development:
```bash
pip install polars-statistics[dev]
```

## Quick Start

### Regression Models

```python
import numpy as np
from polars_statistics import OLS, Ridge, Logistic

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 3)
y = X @ np.array([1, 2, 3]) + np.random.randn(100) * 0.1

# Ordinary Least Squares
model = OLS(compute_inference=True).fit(X, y)
print(f"Coefficients: {model.coefficients}")
print(f"R-squared: {model.r_squared:.4f}")
print(f"P-values: {model.p_values}")

# Ridge Regression
ridge = Ridge(lambda_=0.1).fit(X, y)
print(f"Ridge coefficients: {ridge.coefficients}")

# Logistic Regression
y_binary = (y > y.mean()).astype(float)
logit = Logistic().fit(X, y_binary)
probs = logit.predict_proba(X)
```

### Statistical Tests with Polars Expressions

The expression API integrates seamlessly with Polars' lazy evaluation and supports `group_by` operations:

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "group": ["A"] * 50 + ["B"] * 50,
    "treatment": [1.2, 2.3, 1.8, ...],  # treatment values
    "control": [1.0, 2.1, 1.5, ...],    # control values
})

# T-test within groups
result = df.group_by("group").agg(
    ps.ttest_ind(pl.col("treatment"), pl.col("control")).alias("ttest")
)
print(result)
# shape: (2, 2)
# ┌───────┬─────────────────────┐
# │ group ┆ ttest               │
# │ ---   ┆ ---                 │
# │ str   ┆ struct[2]           │
# ╞═══════╪═════════════════════╡
# │ A     ┆ {2.34, 0.021}       │
# │ B     ┆ {1.89, 0.062}       │
# └───────┴─────────────────────┘

# Access struct fields
result.with_columns(
    pl.col("ttest").struct.field("statistic"),
    pl.col("ttest").struct.field("p_value"),
)
```

### Available Statistical Tests

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "x": [...],
    "y": [...],
})

# Parametric tests
ps.ttest_ind(pl.col("x"), pl.col("y"))           # Independent t-test
ps.ttest_paired(pl.col("x"), pl.col("y"))        # Paired t-test
ps.brown_forsythe(pl.col("x"), pl.col("y"))      # Equality of variances

# Non-parametric tests
ps.mann_whitney_u(pl.col("x"), pl.col("y"))      # Mann-Whitney U test
ps.wilcoxon_signed_rank(pl.col("x"), pl.col("y")) # Wilcoxon signed-rank
ps.kruskal_wallis(pl.col("x"), pl.col("y"))      # Kruskal-Wallis H test

# Distributional tests
ps.shapiro_wilk(pl.col("x"))                     # Shapiro-Wilk normality
ps.dagostino(pl.col("x"))                        # D'Agostino-Pearson

# Forecast comparison
ps.diebold_mariano(pl.col("e1"), pl.col("e2"))   # Diebold-Mariano test
ps.permutation_t_test(pl.col("x"), pl.col("y"))  # Permutation t-test
```

### Bootstrap Methods

```python
import numpy as np
from polars_statistics import StationaryBootstrap, CircularBlockBootstrap

data = np.random.randn(100)

# Stationary bootstrap (random block lengths)
bootstrap = StationaryBootstrap(expected_block_length=5.0, seed=42)
sample = bootstrap.sample(data)
samples = bootstrap.samples(data, n_samples=1000)

# Circular block bootstrap (fixed block length)
cbb = CircularBlockBootstrap(block_length=10, seed=42)
sample = cbb.sample(data)
```

## API Reference

### Regression Models

| Model | Description | Key Parameters |
|-------|-------------|----------------|
| `OLS` | Ordinary Least Squares | `with_intercept`, `compute_inference`, `confidence_level` |
| `Ridge` | Ridge regression (L2) | `lambda_`, `with_intercept` |
| `ElasticNet` | Elastic Net (L1+L2) | `lambda_`, `alpha`, `max_iter`, `tol` |
| `WLS` | Weighted Least Squares | `with_intercept`, `compute_inference` |
| `Logistic` | Logistic regression | `with_intercept`, `max_iter`, `tol` |
| `Poisson` | Poisson regression | `with_intercept`, `max_iter`, `tol` |

### Model Properties

After fitting, models expose these properties:

```python
model.coefficients      # Regression coefficients
model.intercept         # Intercept term (if fitted)
model.r_squared         # R-squared (linear models)
model.std_errors        # Standard errors (if compute_inference=True)
model.p_values          # P-values (if compute_inference=True)
model.aic               # Akaike Information Criterion
model.bic               # Bayesian Information Criterion
```

### Expression Functions

All expression functions return a struct with `statistic` and `p_value` fields:

| Function | Description | Parameters |
|----------|-------------|------------|
| `ttest_ind` | Independent samples t-test | `alternative`, `equal_var` |
| `ttest_paired` | Paired samples t-test | `alternative` |
| `brown_forsythe` | Brown-Forsythe test | - |
| `mann_whitney_u` | Mann-Whitney U test | - |
| `wilcoxon_signed_rank` | Wilcoxon signed-rank test | - |
| `kruskal_wallis` | Kruskal-Wallis H test | - |
| `shapiro_wilk` | Shapiro-Wilk normality test | - |
| `dagostino` | D'Agostino-Pearson test | - |
| `diebold_mariano` | Diebold-Mariano test | `loss`, `horizon` |
| `permutation_t_test` | Permutation t-test | `alternative`, `n_permutations`, `seed` |

## Performance

polars-statistics is built on high-performance Rust libraries:

- **[regress-rs](https://github.com/sipemu/regress-rs)**: Fast regression using [faer](https://github.com/sarah-ek/faer-rs) linear algebra
- **[anofox-statistics](https://github.com/sipemu/anofox-statistics-rs)**: Statistical tests with SIMD optimizations
- **Zero-copy integration**: Direct memory sharing between Python and Rust

Benchmarks show 10-100x speedups compared to pure Python implementations for large datasets.

## Development

### Building from Source

```bash
# Clone the repository
git clone https://github.com/sipemu/polars-statistics.git
cd polars-statistics

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install maturin and build
pip install maturin numpy polars
maturin develop --release

# Run tests
pip install pytest scipy statsmodels
pytest
```

### Running Tests

```bash
pytest tests/ -v
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Polars](https://pola.rs/) - Fast DataFrame library
- [PyO3](https://pyo3.rs/) - Rust bindings for Python
- [pyo3-polars](https://github.com/pola-rs/pyo3-polars) - Polars plugin framework
- [faer](https://github.com/sarah-ek/faer-rs) - High-performance linear algebra
