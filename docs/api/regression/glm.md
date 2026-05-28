# Generalized Linear Models (GLM)

GLM models for binary classification, count data, and other non-normal response distributions.

## Penalized IRLS

All GLM expressions (`logistic`, `poisson`, `negative_binomial`, `tweedie`, `probit`, `cloglog`) accept a `lambda_` kwarg (default `0.0`) that adds an L2 (ridge) penalty on the coefficients inside the IRLS update. Set `lambda_ > 0` to stabilize estimation under collinearity or quasi-separation.

The sklearn-style `logistic_regression` exposes the same penalty via `C = 1 / lambda_` and an explicit `penalty` choice.

---

## `logistic`

Logistic regression for binary classification.

```python
ps.logistic(
    y: Union[pl.Expr, str],  # Binary (0/1)
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,        # L2 (Ridge) regularization strength
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [GLM Output](../outputs.md#glm-output)

**Example:**
```python
df.group_by("group").agg(ps.logistic("success", "x1", "x2").alias("model"))
```

---

## `logistic_regression`

Sklearn-style logistic regression. Distinct from [`logistic`](#logistic): uses inverse-strength regularization `C = 1 / lambda_` and an explicit `penalty` choice. Returns the same [GLM Output](../outputs.md#glm-output) schema.

```python
ps.logistic_regression(
    y: Union[pl.Expr, str],      # Binary (0/1)
    *x: Union[pl.Expr, str],
    penalty: str = "l2",          # "l2" or "none"
    C: float = 1.0,               # Inverse of regularization strength
    threshold: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-8,
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [GLM Output](../outputs.md#glm-output)

**Example:**
```python
df.group_by("group").agg(
    ps.logistic_regression("success", "x1", "x2", penalty="l2", C=0.5).alias("model")
)
```

---

## `poisson`

Poisson regression for count data.

```python
ps.poisson(
    y: Union[pl.Expr, str],  # Non-negative counts
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,        # L2 (Ridge) regularization strength
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [GLM Output](../outputs.md#glm-output)

---

## `negative_binomial`

Negative Binomial regression for overdispersed count data.

```python
ps.negative_binomial(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    theta: float | None = None,  # Dispersion; None = estimate
    lambda_: float = 0.0,        # L2 (Ridge) regularization strength
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [GLM Output](../outputs.md#glm-output)

---

## `tweedie`

Tweedie GLM for flexible variance structures.

```python
ps.tweedie(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    var_power: float = 1.5,      # 0=Gaussian, 1=Poisson, 2=Gamma, 3=InvGaussian
    lambda_: float = 0.0,        # L2 (Ridge) regularization strength
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [GLM Output](../outputs.md#glm-output)

**Variance Power Interpretation:**
| var_power | Distribution |
|-----------|--------------|
| 0 | Gaussian (Normal) |
| 1 | Poisson |
| (1, 2) | Compound Poisson-Gamma |
| 2 | Gamma |
| 3 | Inverse Gaussian |

---

## `probit`

Probit regression for binary classification.

```python
ps.probit(
    y: Union[pl.Expr, str],  # Binary (0/1)
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,        # L2 (Ridge) regularization strength
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [GLM Output](../outputs.md#glm-output)

---

## `cloglog`

Complementary log-log regression for binary classification.

```python
ps.cloglog(
    y: Union[pl.Expr, str],  # Binary (0/1)
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,        # L2 (Ridge) regularization strength
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [GLM Output](../outputs.md#glm-output)

---

## Regularization

All GLM models support L2 (Ridge) regularization via the `lambda_` parameter:

```python
# Unregularized logistic regression
ps.logistic("y", "x1", "x2")

# Ridge-regularized logistic regression
ps.logistic("y", "x1", "x2", lambda_=1.0)
```

Regularization helps with:
- Preventing overfitting
- Stabilizing estimation when predictors are correlated
- Handling quasi-separation in binary response models

---

## See Also

- [Linear Models](linear.md) - Standard linear regression
- [ALM](alm.md) - Augmented Linear Model with 24+ distributions
- [Diagnostics](diagnostics.md) - Quasi-separation detection
