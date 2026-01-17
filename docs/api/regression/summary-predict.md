# Summary and Prediction Functions

Functions for extracting coefficient statistics and making predictions with confidence intervals.

## Summary Functions

Return coefficient statistics in tidy format (like R's `broom::tidy`).

### Available Functions

```python
ps.ols_summary(y, *x, with_intercept=True) -> pl.Expr
ps.ridge_summary(y, *x, lambda_=1.0, with_intercept=True) -> pl.Expr
ps.elastic_net_summary(y, *x, lambda_=1.0, alpha=0.5, with_intercept=True) -> pl.Expr
ps.wls_summary(y, weights, *x, with_intercept=True) -> pl.Expr
ps.rls_summary(y, *x, forgetting_factor=0.99, with_intercept=True) -> pl.Expr
ps.bls_summary(y, *x, lower_bound=None, upper_bound=None, with_intercept=True) -> pl.Expr
ps.logistic_summary(y, *x, with_intercept=True) -> pl.Expr
ps.poisson_summary(y, *x, with_intercept=True) -> pl.Expr
ps.negative_binomial_summary(y, *x, theta=None, with_intercept=True) -> pl.Expr
ps.tweedie_summary(y, *x, var_power=1.5, with_intercept=True) -> pl.Expr
ps.probit_summary(y, *x, with_intercept=True) -> pl.Expr
ps.cloglog_summary(y, *x, with_intercept=True) -> pl.Expr
ps.alm_summary(y, *x, distribution="normal", with_intercept=True) -> pl.Expr
```

Formula variants also available: `ps.ols_formula_summary(formula, ...)`, etc.

**Returns:** See [Summary Output](../outputs.md#summary-output)

### Example

```python
# Get coefficient table per group
df.group_by("group").agg(
    ps.ols_summary("y", "x1", "x2").alias("coef")
).explode("coef").unnest("coef")
```

**Output:**
| group | term | estimate | std_error | statistic | p_value |
|-------|------|----------|-----------|-----------|---------|
| A | intercept | 1.234 | 0.123 | 10.03 | 0.000 |
| A | x1 | 0.567 | 0.045 | 12.60 | 0.000 |
| A | x2 | -0.234 | 0.067 | -3.49 | 0.001 |

---

## Prediction Functions

Return per-row predictions with optional confidence/prediction intervals.

### Signature

```python
ps.ols_predict(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    add_intercept: bool = True,
    interval: str | None = None,  # None, "confidence", "prediction"
    level: float = 0.95,
    null_policy: str = "drop",    # "drop", "drop_y_zero_x"
) -> pl.Expr
```

### Available Functions

All models have prediction functions:
- `ols_predict`, `ridge_predict`, `elastic_net_predict`
- `wls_predict`, `rls_predict`, `bls_predict`, `nnls_predict`
- `logistic_predict`, `poisson_predict`, `negative_binomial_predict`
- `tweedie_predict`, `probit_predict`, `cloglog_predict`
- `alm_predict`

Formula variants also available: `ps.ols_formula_predict(formula, ...)`, etc.

**Returns:** See [Prediction Output](../outputs.md#prediction-output)

### Parameters

| Parameter | Description |
|-----------|-------------|
| `interval` | `None` (point only), `"confidence"` (mean interval), `"prediction"` (individual interval) |
| `level` | Confidence level (default 0.95 for 95% intervals) |
| `null_policy` | How to handle missing values: `"drop"` or `"drop_y_zero_x"` |

### Example

```python
# Per-group predictions with 95% prediction intervals
df.with_columns(
    ps.ols_predict("y", "x1", "x2", interval="prediction", level=0.95)
        .over("group").alias("pred")
).unnest("pred")
```

**Output:**
| group | y | x1 | x2 | prediction | lower | upper |
|-------|---|----|----|------------|-------|-------|
| A | 5.2 | 1.0 | 2.0 | 5.15 | 3.21 | 7.09 |
| A | 3.8 | 0.5 | 1.5 | 3.92 | 1.98 | 5.86 |

---

## Interval Types

| Type | Description | Use Case |
|------|-------------|----------|
| `None` | Point prediction only | Fast prediction |
| `"confidence"` | Interval for mean response | Uncertainty about regression line |
| `"prediction"` | Interval for individual observation | Forecasting new observations |

**Note:** Prediction intervals are always wider than confidence intervals because they account for both uncertainty in the regression line and individual observation variance.

---

## See Also

- [Linear Models](linear.md) - Model fitting
- [GLM Models](glm.md) - GLM fitting
- [Formula Syntax](formula.md) - Formula interface
