# Getting Started

## Installation

```bash
pip install polars-statistics
```

Requires Python 3.9+.

## Statistical Tests

All test functions work as Polars expressions, integrating with `group_by` and `over`:

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "treatment": [23.1, 25.3, 22.8, 24.5, 26.1],
    "control": [20.2, 21.5, 19.8, 22.1, 20.9],
})

# Two-sample t-test
result = df.select(
    ps.ttest_ind("treatment", "control", alternative="two-sided")
        .alias("test")
)

# Extract results from the returned struct
result.with_columns(
    pl.col("test").struct.field("statistic"),
    pl.col("test").struct.field("p_value"),
)
```

All tests return a struct with `statistic` and `p_value` fields. See the [Statistical Tests](api/tests/parametric.md) section for the full list of available tests.

## Regression Models

### Expression API

Regression functions also work as Polars expressions:

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

# Extract R-squared per group
result.with_columns(
    pl.col("model").struct.field("r_squared"),
)
```

### Formula Syntax

R-style formulas with polynomial and interaction effects:

```python
# Main effects + interaction
ps.ols_formula("y ~ x1 * x2")  # Expands to: x1 + x2 + x1:x2

# Polynomial regression
ps.ols_formula("y ~ poly(x, 2)")
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

# Fit a model
model = OLS(compute_inference=True).fit(X, y)
print(model.coefficients, model.r_squared, model.p_values)

# ALM with various distributions
alm = ALM.laplace().fit(X, y)  # Robust to outliers
```

See the [Model Classes](api/classes/linear.md) section for all available classes.

## Next Steps

- [API Conventions](api/conventions.md) — Common patterns across all functions
- [Output Structures](api/outputs.md) — Return type definitions for all functions
- [Regression Diagnostics](api/regression/diagnostics.md) — Multicollinearity and separation checks
