# Augmented Linear Model (ALM)

Flexible regression supporting 24+ distributions with automatic link function selection.

## `alm`

```python
ps.alm(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    distribution: str = "normal",
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [ALM Output](../outputs.md#alm-output)

**Example:**
```python
# Fit with Laplace distribution for robust estimation
df.group_by("group").agg(
    ps.alm("y", "x1", "x2", distribution="laplace").alias("model")
)
```

---

## Supported Distributions

| Category | Distributions |
|----------|---------------|
| Continuous | `normal`, `laplace`, `student_t`, `logistic` |
| Positive | `lognormal`, `loglaplace`, `gamma`, `inverse_gaussian`, `exponential` |
| Bounded (0,1) | `beta` |
| Count | `poisson`, `negative_binomial`, `binomial`, `geometric` |

---

## Distribution Selection Guide

| Use Case | Recommended Distribution |
|----------|-------------------------|
| Standard regression | `normal` |
| Robust to outliers | `laplace`, `student_t` |
| Heavy tails | `student_t` |
| Positive continuous | `lognormal`, `gamma` |
| Right-skewed positive | `gamma`, `inverse_gaussian` |
| Proportions/rates | `beta` |
| Count data | `poisson`, `negative_binomial` |
| Overdispersed counts | `negative_binomial` |

---

## See Also

- [GLM Models](glm.md) - Standard GLM interface
- [Dynamic Linear Model](dynamic.md) - Time-varying coefficients
