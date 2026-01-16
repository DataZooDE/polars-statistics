# API Conventions

Common patterns and conventions used throughout the polars-statistics API.

## Expression API

All functions work as Polars expressions and integrate with `group_by`, `over`, and lazy evaluation:

```python
import polars as pl
import polars_statistics as ps

# With group_by (aggregation)
df.group_by("group").agg(ps.ols("y", "x1", "x2").alias("model"))

# With over (window function)
df.with_columns(ps.ols("y", "x1", "x2").over("group").alias("model"))

# Lazy evaluation
df.lazy().group_by("group").agg(ps.ttest_ind("x", "y")).collect()
```

## Column References

All functions accept column names as strings or `pl.Expr`:

```python
ps.ols("y", "x1", "x2")              # String column names
ps.ols(pl.col("y"), pl.col("x1"))   # Polars expressions
ps.ols("y", pl.col("x1") * 2)       # Mixed / transformed
```

## Return Types

### Statistical Tests

Return a struct with `statistic` and `p_value` fields:

```python
result = df.select(ps.ttest_ind("x", "y").alias("test"))
result.with_columns(
    pl.col("test").struct.field("statistic"),
    pl.col("test").struct.field("p_value"),
)
```

### Regression Models

Return a struct with model-specific fields. See [Output Structures](outputs.md) for details.

```python
result = df.group_by("group").agg(ps.ols("y", "x1").alias("model"))
result.with_columns(
    pl.col("model").struct.field("r_squared"),
    pl.col("model").struct.field("coefficients"),
)
```

### Summary Functions

Return `List[Struct]` with coefficient statistics (like R's `broom::tidy`):

```python
df.group_by("group").agg(
    ps.ols_summary("y", "x1", "x2").alias("coef")
).explode("coef").unnest("coef")
# Columns: term, estimate, std_error, statistic, p_value
```

### Prediction Functions

Return `Struct{prediction, lower, upper}` per row:

```python
df.with_columns(
    ps.ols_predict("y", "x1", "x2", interval="prediction", level=0.95)
        .over("group").alias("pred")
).unnest("pred")
# Columns: prediction, lower, upper
```

## Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `with_intercept` | Include intercept term | `True` |
| `alternative` | Test alternative: "two-sided", "less", "greater" | "two-sided" |
| `alpha` | Significance level | `0.05` |
| `conf_level` | Confidence level for intervals | `0.95` |
| `lambda_` | L2 (Ridge) regularization strength | `0.0` |

## See Also

- [Output Structures](outputs.md) - Detailed return type definitions
- [Model Classes](classes/linear.md) - Direct model access outside expressions
