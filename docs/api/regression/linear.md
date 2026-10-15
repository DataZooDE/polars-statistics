# Linear Regression Models

Standard linear regression models including OLS, Ridge, Elastic Net, and specialized variants.

!!! info "Behaviour changes in 0.8.0"
    The upstream `anofox-regression` 0.5.14/0.5.15 releases bundled in
    polars-statistics 0.8.0 changed several linear-solver statistics to match R:

    - **`log_likelihood`/`aic`/`bic` now equal R's `logLik.lm`/`AIC`/`BIC`**
      (ML variance `RSS / n`, `k = p + 1` parameters) for `ols` and the other
      linear solvers — previously a different (non-R-matching) formula.
    - **No-intercept `r_squared`/`adj_r_squared`/`f_statistic` now use the
      uncentred total sum of squares**, matching R `summary.lm`'s own
      no-intercept convention, for both `ols` and `wls`.
    - **A constant column without `with_intercept`** is no longer aliased to
      `NaN` — it now behaves exactly like an explicit intercept term.

!!! info "`alias_tolerance` (0.8.0)"
    `ols`/`wls` (and their `_summary`/`_predict`/formula variants) accept an
    `alias_tolerance: float = 1e-7` kwarg — R `lm`'s configurable rank-deficiency
    tolerance (`anofox_regression::utils::LM_ALIAS_TOLERANCE`). A later column that is a
    linear combination of the intercept and earlier kept columns within this tolerance is
    aliased (`NaN` coefficient, excluded from `rank`/`n_parameters`); a **tighter** tolerance
    (smaller value) aliases more columns, a **looser** tolerance (larger value) keeps more.
    The default reproduces the pre-0.8.0 behaviour exactly. See
    [Rank-deficient designs](../classes/linear.md#rank-deficient-designs-ols-wls) for the full
    column-ordering/aliasing semantics this tolerance controls.

## `ols`

Ordinary Least Squares regression.

```python
ps.ols(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
    alias_tolerance: float = 1e-7,
) -> pl.Expr
```

**`alias_tolerance`** (default `1e-7`, R `lm`'s own tolerance): see the note above.

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
    alias_tolerance: float = 1e-7,
) -> pl.Expr
```

**`alias_tolerance`** (default `1e-7`, R `lm`'s own tolerance): see the [`ols` note](#ols) above.

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
    initial_p_diagonal: float = 1e6,
) -> pl.Expr
```

**Returns:** See [Linear Model Output](../outputs.md#linear-model-output)

**`initial_p_diagonal`** (default `1e6`, matching the upstream crate's own default
— omitting it behaves identically): sets the diagonal value of the initial
covariance prior `P0 = initial_p_diagonal * I`. Also available on `rls_summary`,
`rls_predict`, and `rls_formula`. See [RLS](../classes/linear.md#rls) for the full
semantics.

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

## `huber`

Huber M-estimator (robust regression). Down-weights observations with `|residual / scale| > epsilon`.

```python
ps.huber(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    epsilon: float = 1.35,       # Huber threshold (must be > 1.0)
    alpha: float = 0.0001,       # L2 ridge penalty in the WLS update
    max_iter: int = 100,
    tol: float = 1e-5,
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Linear Model Output](../outputs.md#linear-model-output)

**Example:**
```python
df.group_by("group").agg(ps.huber("y", "x1", "x2").alias("model"))
```

---

## `pls`

Partial Least Squares — projects `X` onto `n_components` latent variables maximizing covariance with `y`. Useful for highly collinear features.

```python
ps.pls(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    n_components: int = 2,
    tol: float = 1e-6,
    scale: bool = True,          # Scale X to unit variance before fitting
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Linear Model Output](../outputs.md#linear-model-output)

**Example:**
```python
df.group_by("group").agg(ps.pls("y", "x1", "x2", "x3", n_components=2).alias("model"))
```

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
