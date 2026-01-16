# Linear Regression Models

Standard linear regression models including OLS, Ridge, Elastic Net, and specialized variants.

## `ols`

Ordinary Least Squares regression.

```python
ps.ols(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Linear Model Output](../outputs.md#linear-model-output)

**Example:**
```python
df.group_by("group").agg(ps.ols("y", "x1", "x2").alias("model"))
```

---

## `ridge`

Ridge regression (L2 regularization).

```python
ps.ridge(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 1.0,
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Linear Model Output](../outputs.md#linear-model-output)

---

## `elastic_net`

Elastic Net regression (L1 + L2 regularization).

```python
ps.elastic_net(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 1.0,
    alpha: float = 0.5,  # L1 ratio (0 = Ridge, 1 = Lasso)
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Linear Model Output](../outputs.md#linear-model-output)

---

## `wls`

Weighted Least Squares regression.

```python
ps.wls(
    y: Union[pl.Expr, str],
    weights: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Linear Model Output](../outputs.md#linear-model-output)

---

## `rls`

Recursive Least Squares regression (online learning).

```python
ps.rls(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    forgetting_factor: float = 0.99,
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Linear Model Output](../outputs.md#linear-model-output)

---

## `bls`

Bounded Least Squares regression.

```python
ps.bls(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Linear Model Output](../outputs.md#linear-model-output)

---

## `nnls`

Non-negative Least Squares (shorthand for `bls` with `lower_bound=0`).

```python
ps.nnls(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Linear Model Output](../outputs.md#linear-model-output)

---

## `quantile`

Quantile regression for estimating conditional quantiles (e.g., median).

```python
ps.quantile(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    tau: float = 0.5,            # Quantile to estimate (0.5 = median)
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Quantile Regression Output](../outputs.md#quantile-regression-output)

---

## `isotonic`

Isotonic (monotonic) regression for calibration curves and monotone relationships.

```python
ps.isotonic(
    y: Union[pl.Expr, str],
    x: Union[pl.Expr, str],
    increasing: bool = True,     # True = increasing, False = decreasing
) -> pl.Expr
```

**Returns:** See [Isotonic Regression Output](../outputs.md#isotonic-regression-output)

---

## See Also

- [GLM Models](glm.md) - Generalized linear models
- [Formula Syntax](formula.md) - R-style formula interface
- [Diagnostics](diagnostics.md) - Model diagnostics
