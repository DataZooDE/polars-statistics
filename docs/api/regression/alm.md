# Augmented Linear Model (ALM)

Flexible regression supporting 25 distributions with configurable link functions and loss criteria.

## `alm`

```python
ps.alm(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    distribution: str = "normal",
    link: str | None = None,         # None = canonical link for the distribution
    loss: str = "likelihood",        # "likelihood" | "mse" | "mae" | "ham" | "role"
    role_trim: float | None = None,  # Trim fraction for the "role" loss
    extra_parameter: float | None = None,
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [ALM Output](../outputs.md#alm-output)

**Example:**
```python
# Robust regression via Laplace likelihood
df.group_by("group").agg(
    ps.alm("y", "x1", "x2", distribution="laplace").alias("model")
)

# Gamma regression with log link
df.group_by("group").agg(
    ps.alm("y", "x1", distribution="gamma", link="log").alias("model")
)
```

---

## Supported Distributions (25)

| Category | Distributions |
|----------|---------------|
| Continuous | `normal`, `laplace`, `student_t`, `logistic`, `asymmetric_laplace`, `generalised_normal`, `s` |
| Positive | `lognormal`, `loglaplace`, `logs`, `loggeneralisednormal`, `gamma`, `inverse_gaussian`, `exponential`, `folded_normal`, `rectified_normal` |
| Bounded (0,1) | `beta`, `logit_normal` |
| Count | `poisson`, `negative_binomial`, `binomial`, `geometric` |
| Ordinal | `cumulative_logistic`, `cumulative_normal` |
| Transformed | `boxcox_normal` |

---

## Link Functions

When `link=None` ALM picks the canonical link for the distribution. Explicit options:

| `link` | Inverse | Typical use |
|--------|---------|-------------|
| `"identity"` | `η` | Gaussian |
| `"log"` | `exp(η)` | Positive / count |
| `"logit"` | `1 / (1 + exp(-η))` | Binary, beta |
| `"probit"` | `Φ(η)` | Binary |
| `"inverse"` | `1 / η` | Gamma |
| `"sqrt"` | `η²` | Poisson |
| `"cloglog"` | `1 - exp(-exp(η))` | Binary (asymmetric) |

---

## Loss Functions

| `loss` | Description |
|--------|-------------|
| `"likelihood"` | Maximum likelihood (default) |
| `"mse"` | Mean squared error |
| `"mae"` | Mean absolute error |
| `"ham"` | Half-absolute-moment |
| `"role"` | Robust loss with trimming via `role_trim` (default 0.05) |

`extra_parameter` is required by distributions that take an auxiliary parameter (e.g. degrees of freedom for `student_t`, shape for `generalised_normal`).

---

## Distribution Selection Guide

| Use Case | Recommended Distribution |
|----------|-------------------------|
| Standard regression | `normal` |
| Robust to outliers | `laplace`, `student_t` |
| Heavy tails | `student_t` |
| Positive continuous | `lognormal`, `gamma` |
| Right-skewed positive | `gamma`, `inverse_gaussian` |
| Proportions/rates | `beta`, `logit_normal` |
| Count data | `poisson`, `negative_binomial` |
| Overdispersed counts | `negative_binomial` |
| Ordinal outcomes | `cumulative_logistic`, `cumulative_normal` |

---

## See Also

- [GLM Models](glm.md) - Standard GLM interface
- [Dynamic Linear Model](dynamic.md) - Time-varying coefficients
