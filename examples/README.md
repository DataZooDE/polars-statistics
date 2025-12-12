# polars-statistics Examples

This folder contains runnable examples demonstrating common use cases for polars-statistics.

## Quick Start

```bash
cd examples
uv sync
uv run python 01_ols_regression.py
```

## Examples Overview

### Regression

| Example | Description |
|---------|-------------|
| [01_ols_regression.py](01_ols_regression.py) | OLS regression basics: fitting, coefficients, predictions, and R-style formulas |
| [02_grouped_regression.py](02_grouped_regression.py) | Running regression per group with `group_by` and `over` |
| [03_glm_models.py](03_glm_models.py) | Generalized Linear Models: logistic (binary), Poisson (counts) |

### Statistical Tests

| Example | Description |
|---------|-------------|
| [04_statistical_tests.py](04_statistical_tests.py) | T-tests, Mann-Whitney U, Shapiro-Wilk, and other hypothesis tests |

### Forecasting

| Example | Description |
|---------|-------------|
| [05_demand_classification.py](05_demand_classification.py) | AID (Automatic Identification of Demand) for demand pattern classification |

## Use Cases

### 1. OLS Regression

Fit ordinary least squares regression and extract model statistics:

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "price": [100, 150, 200, 250, 300],
    "sqft": [1000, 1200, 1500, 1800, 2000],
    "bedrooms": [2, 2, 3, 3, 4],
})

# Fit OLS model
result = df.select(ps.ols("price", "sqft", "bedrooms").alias("model"))

# Access R-squared, coefficients
print(result["model"].struct.field("r_squared"))
print(result["model"].struct.field("coefficients"))
```

### 2. Grouped Regression

Fit separate models per group (e.g., per region, per product):

```python
# OLS per region
result = df.group_by("region").agg(
    ps.ols("sales", "price", "advertising").alias("model")
)

# Get predictions with .over()
df_with_pred = df.with_columns(
    ps.ols_predict("sales", "price", "advertising")
        .over("region")
        .alias("predictions")
).unnest("predictions")
```

### 3. Statistical Tests

Compare groups using t-tests, Mann-Whitney U, and other tests:

```python
# Welch's t-test
result = df.select(
    ps.ttest_ind("treatment", "control").alias("ttest")
)
print(f"p-value: {result['ttest'].struct.field('p_value')[0]}")

# Non-parametric alternative
result = df.select(
    ps.mann_whitney_u("treatment", "control").alias("mwu")
)
```

### 4. GLM Models

Binary classification with logistic regression:

```python
# Logistic regression for churn prediction
result = df.group_by("segment").agg(
    ps.logistic("churned", "tenure", "monthly_charges").alias("model")
)

# Get predicted probabilities
df_with_prob = df.with_columns(
    ps.logistic_predict("churned", "tenure", "monthly_charges")
        .over("segment")
        .alias("pred")
).unnest("pred")
```

Count data with Poisson regression:

```python
# Poisson regression for count outcomes
result = df.select(
    ps.poisson("claims", "age", "exposure").alias("model")
)
```

### 5. Demand Classification

Classify demand patterns for inventory management:

```python
# Classify demand per SKU
result = df.group_by("sku").agg(
    ps.aid("demand").alias("classification")
)

# Check demand type and recommended distribution
result.with_columns(
    pl.col("classification").struct.field("demand_type"),
    pl.col("classification").struct.field("distribution"),
    pl.col("classification").struct.field("is_intermittent"),
)
```

## API Pattern Summary

polars-statistics provides three main function patterns:

| Pattern | Use Case | Example |
|---------|----------|---------|
| `ps.ols(...)` | Fit model, get summary stats | `group_by(...).agg(ps.ols(...))` |
| `ps.ols_predict(...)` | Per-row predictions | `.with_columns(ps.ols_predict(...).over(...))` |
| `ps.ols_summary(...)` | Tidy coefficient table | `group_by(...).agg(ps.ols_summary(...))` |

The same pattern applies to all regression methods: `ridge`, `elastic_net`, `logistic`, `poisson`, etc.

## Other Regression Methods

The API for other regression methods is similar to OLS:

```python
# Ridge regression (L2 regularization)
ps.ridge("y", "x1", "x2", lambda_=1.0)

# Elastic Net (L1 + L2)
ps.elastic_net("y", "x1", "x2", lambda_=1.0, alpha=0.5)

# Weighted Least Squares
ps.wls("y", "weights", "x1", "x2")

# Recursive Least Squares (time-varying)
ps.rls("y", "x1", "x2", forgetting_factor=0.99)
```

## Performance Benchmarks

For large-scale benchmarks (1M groups, 100M rows), see the [performance_1m_groups](performance_1m_groups/) folder.
