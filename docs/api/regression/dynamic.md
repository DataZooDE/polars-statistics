# Dynamic Linear Model (LmDynamic)

Time-varying coefficient regression using information criterion-weighted model averaging with optional LOWESS smoothing.

## `lm_dynamic`

```python
ps.lm_dynamic(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    ic: str = "aicc",              # "aic", "aicc", "bic"
    distribution: str = "normal",  # "normal", "laplace", "student_t", etc.
    lowess_span: float = 0.3,      # LOWESS smoothing span (0.05-1.0)
    max_models: int = 64,          # Max candidate models
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [LmDynamic Output](../outputs.md#lmdynamic-output)

---

## Parameters

| Parameter | Description |
|-----------|-------------|
| `ic` | Information criterion for model weighting: `"aic"`, `"aicc"` (default), or `"bic"` |
| `distribution` | Error distribution for ALM fitting |
| `lowess_span` | LOWESS smoothing bandwidth (0.05-1.0). Lower = more local, Higher = smoother |
| `max_models` | Maximum number of candidate models to consider |

---

## How It Works

1. **Model Generation**: Creates candidate models with different predictor subsets
2. **Per-Observation IC**: Computes information criterion values at each observation
3. **Model Weighting**: Weights models based on IC values (Akaike weights)
4. **Coefficient Averaging**: Combines coefficients using weights at each time point
5. **LOWESS Smoothing**: Optionally smooths the time-varying weights

---

## Example

```python
# Fit dynamic model per group
result = df.group_by("group").agg(
    ps.lm_dynamic("y", "x1", "x2", lowess_span=0.3).alias("model")
).unnest("model")

# Access time-varying coefficients (via model class)
from polars_statistics import LmDynamic
model = LmDynamic(lowess_span=0.3)
model.fit(X, y)
print(model.dynamic_coefficients)  # (n_obs x n_coef) array
```

---

## See Also

- [ALM](alm.md) - Augmented Linear Model distributions
- [Linear Models](linear.md) - Standard regression
