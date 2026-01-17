# Generalized Linear Models (GLM)

GLM models for binary classification, count data, and other non-normal response distributions. All GLM models support optional Ridge regularization via the `lambda_` parameter.

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
