# Regression Diagnostics

Tools for detecting multicollinearity, quasi-separation, influence and outliers, plus residual batteries for OLS and GLM fits.

## Augment (per-observation diagnostics)

### `augment`

One combined, row-aligned expression returning R `broom::augment`-style per-observation
diagnostics for any of nine supported models — an alternative to composing the individual
OLS residual/leverage/Cook's-distance functions below, and the only way to get these
diagnostics for a GLM fit in one call.

```python title="Signature"
ps.augment(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    model: str = "ols",
    weights: Union[pl.Expr, str, None] = None,
    residual_type: str | None = None,
    add_intercept: bool = True,
    theta: float | None = None,
    var_power: float = 1.5,
    link_power: float | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> pl.Expr
```

**Parameters:**
- `model` (default `"ols"`) — one of `"ols"`, `"wls"`, `"logistic"`, `"probit"`, `"cloglog"`,
  `"poisson"`, `"negative_binomial"`, `"tweedie"`, `"gamma"`.
- `weights` — prior weights; required (and only valid) with `model="wls"`.
- `residual_type` — `"response"`, `"pearson"`, or `"deviance"`. Defaults to `"response"` for
  `ols`/`wls` and `"deviance"` for every GLM (matching R's `broom::augment`).
- `add_intercept` (default `True`).
- `theta` — fixed negative-binomial dispersion (`model="negative_binomial"` only); `None`
  estimates it from the data.
- `var_power` (default `1.5`) — Tweedie variance power (`model="tweedie"` only): `0` Gaussian,
  `1` Poisson, `2` Gamma, `3` inverse Gaussian.
- `link_power` (default `None`) — Tweedie link power (`model="tweedie"` only). `None` uses the
  canonical link (`1 - var_power`), matching `ps.tweedie()` and `Tweedie(link_power=None)` — the
  same model the user fits with the same arguments. Pass an explicit value (e.g. `0.0` for the
  log link) to override it.
- `max_iter`/`tol` — IRLS iteration limit/tolerance (GLMs only).

**Returns:** `Struct{fitted, residual, std_residual, stud_residual, leverage, cooks_d, dffits}`
— see [Augment Output](../outputs.md#augment-output) for the R-name mapping
(`std_residual` = `rstandard`, `stud_residual` = `rstudent`, `leverage` = `hatvalues`,
`cooks_d` = `cooks.distance`). One row per input row (`returns_scalar=False`, like the
`*_predict` expressions).

**NA handling:** rows with a null/non-finite `y`, `x`, or (for `wls`) `weights` entry come back
`NaN` in every field; the remaining rows equal a fit computed without that row — this is not an
error condition, it is standard broom behaviour (a dropped observation legitimately changes
every remaining row's leverage/Cook's distance/etc.).

**R equivalent:** `broom::augment(model, type.predict = residual_type, type.residuals = residual_type)`.

**Cross-check:** for `model="ols"`, `augment`'s `leverage`/`cooks_d`/`dffits`/`std_residual`/
`stud_residual` agree elementwise with `ps.leverage`/`ps.cooks_distance`/`ps.dffits`/
`ps.studentized_residuals`/`ps.externally_studentized_residuals` below.

**PyModel equivalent:** every model listed above also has a `.augment(x, y, residual_type=None)`
method on its PyModel class (`WLS.augment` additionally takes `weights=None`, defaulting to the
fit's own weights) — see [Model Classes](../classes/linear.md) and
[GLM Model Classes](../classes/glm.md). It returns the same seven fields as a `dict` of numpy
arrays, plus a trailing `resid_type` key (the resolved `residual_type` string), and always
describes exactly what that PyModel instance was fit with.

**Example:**
```python
import numpy as np
import polars as pl
import polars_statistics as ps

np.random.seed(0)
n = 50
x1 = np.random.normal(0, 1, n)
y = 1.0 + 2.0 * x1 + np.random.normal(0, 0.5, n)
df = pl.DataFrame({"y": y, "x1": x1})

result = df.select(ps.augment("y", "x1", model="ols").alias("aug")).unnest("aug")
print(result.head(3))

# Per-group diagnostics
grouped = pl.DataFrame({"group": ["A"] * n, "y": y, "x1": x1}).group_by("group").agg(
    ps.augment("y", "x1", model="ols").alias("aug")
)
```

---

## `condition_number`

Compute condition number diagnostics to detect multicollinearity in design matrices.

```python
ps.condition_number(
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Condition Number Output](../outputs.md#condition-number-output)

**Example:**
```python
df.select(ps.condition_number("x1", "x2", "x3").alias("diagnostics"))
```

**Interpretation:**

| Condition Number | Severity | Recommendation |
|------------------|----------|----------------|
| < 30 | Well-conditioned | Numerically stable |
| 30 - 100 | Moderate | Monitor for instability |
| 100 - 1000 | High | Consider regularization or removing predictors |
| > 1000 | Severe | Strong multicollinearity present |

---

## `check_binary_separation`

Detect quasi-separation in binary response data (logistic/probit regression).

```python
ps.check_binary_separation(
    y: Union[pl.Expr, str],      # Binary (0/1)
    *x: Union[pl.Expr, str],
) -> pl.Expr
```

**Returns:** See [Separation Check Output](../outputs.md#separation-check-output)

**Example:**
```python
df.select(ps.check_binary_separation("success", "predictor1", "predictor2"))
```

**Separation Types:**
- `Complete`: Predictor perfectly divides the classes (MLE does not exist)
- `Quasi`: Nearly perfect separation with 1-2 observations crossing
- `MonotonicResponse`: Each predictor level has all observations in same class

**When Separation is Detected:**
- Consider using penalized regression (`lambda_ > 0`)
- Remove or combine problematic predictors
- Use Firth's bias-reduced logistic regression (if available)

---

## `check_count_sparsity`

Detect sparsity-induced separation in count data (Poisson/NegBin regression).

```python
ps.check_count_sparsity(
    y: Union[pl.Expr, str],      # Non-negative counts
    *x: Union[pl.Expr, str],
) -> pl.Expr
```

**Returns:** See [Separation Check Output](../outputs.md#separation-check-output)

**Example:**
```python
df.select(ps.check_count_sparsity("count", "x1", "x2"))
```

**When Sparsity is Detected:**
- Use regularization (`lambda_ > 0`)
- Consider zero-inflated models
- Check for sparse predictor-response combinations

---

## Multicollinearity

### `vif`

Variance Inflation Factor per predictor: `VIF_j = 1 / (1 - R²_j)`.

```python
ps.vif(*x: Union[pl.Expr, str]) -> pl.Expr
```

**Returns:** See [VIF Output](../outputs.md#vif-output)

**Example:**
```python
df.select(ps.vif("x1", "x2", "x3").alias("vif"))
```

VIF > 5 (moderate) / > 10 (severe) typically flags problematic collinearity. The intercept is not included.

---

### `generalized_vif`

Generalized VIF (GVIF) for grouped predictors such as one-hot dummies.

```python
ps.generalized_vif(
    *x: Union[pl.Expr, str],
    group_sizes: list[int],  # e.g. [1, 1, 3] for two scalars + a 3-level dummy
) -> pl.Expr
```

**Returns:** See [GVIF Output](../outputs.md#gvif-output)

**Example:**
```python
# x1, x2 are scalars; cat_a/cat_b/cat_c are dummies for one categorical
df.select(
    ps.generalized_vif("x1", "x2", "cat_a", "cat_b", "cat_c",
                       group_sizes=[1, 1, 3]).alias("gvif")
)
```

For single-column groups GVIF coincides with `vif`.

---

### `high_vif_predictors`

Boolean mask of features whose VIF exceeds `threshold` (default 10).

```python
ps.high_vif_predictors(
    *x: Union[pl.Expr, str],
    threshold: float = 10.0,
) -> pl.Expr
```

**Returns:** See [VIF Mask Output](../outputs.md#vif-mask-output)

**Example:**
```python
df.select(ps.high_vif_predictors("x1", "x2", "x3", threshold=5.0))
```

---

## OLS Residual Battery

Each function fits OLS internally and returns one residual per row.

### `standardized_residuals`

`r_i / sqrt(MSE)`.

```python
ps.standardized_residuals(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Residual Diagnostics Output](../outputs.md#residual-diagnostics-output)

**Example:**
```python
df.select(ps.standardized_residuals("y", "x1", "x2").alias("r_std"))
```

---

### `studentized_residuals`

Internally studentized: `r_i / sqrt(MSE * (1 - h_ii))`.

```python
ps.studentized_residuals(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Residual Diagnostics Output](../outputs.md#residual-diagnostics-output)

---

### `externally_studentized_residuals`

Leave-one-out studentized residuals; t-distributed with `n - p - 1` df under the null.

```python
ps.externally_studentized_residuals(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Residual Diagnostics Output](../outputs.md#residual-diagnostics-output)

---

### `residual_outliers`

Boolean outlier mask based on `|studentized residual| > threshold` (default 2.0).

```python
ps.residual_outliers(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    threshold: float = 2.0,
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Outlier Mask Output](../outputs.md#outlier-mask-output)

**Example:**
```python
df.select(ps.residual_outliers("y", "x1", "x2", threshold=2.5))
```

---

## Influence / Leverage

### `leverage`

Hat-matrix diagonal `h_ii`. Sums to `p`; flags `h_ii > 2p/n` are conventional.

```python
ps.leverage(
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Leverage Output](../outputs.md#leverage-output)

**Example:**
```python
df.select(ps.leverage("x1", "x2").alias("h"))
```

---

### `cooks_distance`

`D_i = (e_i² / (p · MSE)) · (h_ii / (1 - h_ii)²)`. Common cutoffs: `D_i > 4/n` or `D_i > 1`.

```python
ps.cooks_distance(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Cook's Distance Output](../outputs.md#cooks-distance-output)

**Example:**
```python
df.select(ps.cooks_distance("y", "x1", "x2").alias("d"))
```

---

### `dffits`

Scaled change in fitted value when observation `i` is dropped. Cutoff: `|DFFITS_i| > 2 * sqrt(p / n)`.

```python
ps.dffits(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [DFFITS Output](../outputs.md#dffits-output)

---

### `influential_cooks`

Boolean mask of observations with Cook's distance above `threshold` (default `4 / n`).

```python
ps.influential_cooks(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    threshold: float | None = None,
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Influence Mask Output](../outputs.md#influence-mask-output)

---

### `influential_dffits`

Boolean mask of observations with `|DFFITS|` above `threshold` (default `2 * sqrt(p / n)`).

```python
ps.influential_dffits(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    threshold: float | None = None,
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Influence Mask Output](../outputs.md#influence-mask-output)

---

### `high_leverage_points`

Boolean mask of high-leverage observations. Takes only feature columns; default threshold `2 * p / n`.

```python
ps.high_leverage_points(
    *x: Union[pl.Expr, str],
    threshold: float | None = None,
    add_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Influence Mask Output](../outputs.md#influence-mask-output)

---

## GLM Residuals

Each function fits the appropriate GLM internally and returns one residual per row. All accept `lambda_=0.0` for ridge-penalized IRLS.

### `logistic_pearson_residuals`

```python
ps.logistic_pearson_residuals(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Residual Diagnostics Output](../outputs.md#residual-diagnostics-output)

---

### `logistic_deviance_residuals`

```python
ps.logistic_deviance_residuals(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Residual Diagnostics Output](../outputs.md#residual-diagnostics-output)

---

### `logistic_working_residuals`

IRLS adjusted-dependent-variable residuals.

```python
ps.logistic_working_residuals(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Residual Diagnostics Output](../outputs.md#residual-diagnostics-output)

---

### `poisson_pearson_residuals`

```python
ps.poisson_pearson_residuals(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Residual Diagnostics Output](../outputs.md#residual-diagnostics-output)

---

### `poisson_deviance_residuals`

```python
ps.poisson_deviance_residuals(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Residual Diagnostics Output](../outputs.md#residual-diagnostics-output)

---

### `poisson_working_residuals`

```python
ps.poisson_working_residuals(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Residual Diagnostics Output](../outputs.md#residual-diagnostics-output)

---

## Goodness of Fit

### `pearson_chi_squared_logistic`

`Σ pearson_residual²` from a logistic fit, plus residual degrees of freedom. For a well-specified model `X² / df_resid ≈ 1`.

```python
ps.pearson_chi_squared_logistic(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Chi-Squared Output](../outputs.md#chi-squared-output)

---

### `pearson_chi_squared_poisson`

`Σ pearson_residual²` from a Poisson fit, plus residual degrees of freedom.

```python
ps.pearson_chi_squared_poisson(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Chi-Squared Output](../outputs.md#chi-squared-output)

---

## Gamma GLM Diagnostics

Residual and dispersion diagnostics for Gamma GLM fits. Each function fits a Gamma GLM
internally (log link, `lambda_=0.0` by default) and returns one value per row or a
scalar summary.

### `gamma_dispersion_deviance`

Deviance-based dispersion estimate: `D / (n - p)`. For a well-specified Gamma model this
should be close to 1.

```python
ps.gamma_dispersion_deviance(
    y: Union[pl.Expr, str],      # Positive response
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** `Struct{dispersion: Float64, deviance: Float64, df_residual: Float64}`

**Example:**
```python
df.select(ps.gamma_dispersion_deviance("y", "x1", "x2").alias("disp"))
```

---

### `gamma_dispersion_pearson`

Pearson-statistic-based dispersion estimate: `X² / (n - p)`.

```python
ps.gamma_dispersion_pearson(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** `Struct{dispersion: Float64, pearson_chi_squared: Float64, df_residual: Float64}`

---

### `gamma_pearson_chi_squared`

`Σ pearson_residual²` from a Gamma GLM fit, plus residual degrees of freedom.

```python
ps.gamma_pearson_chi_squared(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Chi-Squared Output](../outputs.md#chi-squared-output)

---

### `gamma_standardized_pearson_residuals`

Standardized Pearson residuals for Gamma GLM: `(y - μ̂) / (μ̂ · sqrt(φ̂))`.

```python
ps.gamma_standardized_pearson_residuals(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Residual Diagnostics Output](../outputs.md#residual-diagnostics-output)

**Example:**
```python
df.select(ps.gamma_standardized_pearson_residuals("y", "x1", "x2").alias("r_p"))
```

---

### `gamma_standardized_deviance_residuals`

Standardized deviance residuals for Gamma GLM: `sign(y - μ̂) · sqrt(d_i / φ̂)`.

```python
ps.gamma_standardized_deviance_residuals(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Residual Diagnostics Output](../outputs.md#residual-diagnostics-output)

**Example:**
```python
df.select(ps.gamma_standardized_deviance_residuals("y", "x1", "x2").alias("r_d"))
```

---

## Diagnostic Workflow

```python
import polars as pl
import polars_statistics as ps

# 1. Check multicollinearity before fitting
cond = df.select(ps.condition_number("x1", "x2", "x3"))
vif = df.select(ps.vif("x1", "x2", "x3"))

# 2. For binary outcomes, check separation
sep = df.select(ps.check_binary_separation("y", "x1", "x2"))
if sep["separation"][0]["has_separation"]:
    model = df.select(ps.logistic("y", "x1", "x2", lambda_=1.0))
else:
    model = df.select(ps.logistic("y", "x1", "x2"))

# 3. After fitting OLS, scan for outliers and influential points
df.select(ps.residual_outliers("y", "x1", "x2").alias("outliers"))
df.select(ps.influential_cooks("y", "x1", "x2").alias("influence"))

# 4. Or get the full per-observation diagnostic battery in one call (OLS or any GLM)
df.select(ps.augment("y", "x1", "x2", model="ols").alias("aug")).unnest("aug")
```

> **Note:** The list-shaped OLS diagnostics above (`leverage`, `cooks_distance`, `dffits`,
> `studentized_residuals`, `externally_studentized_residuals`) and `augment`'s equivalent
> row-aligned fields agree elementwise for the same data — they are two different shapes
> (`List[Float64]` per call vs. one `Struct` row per observation) over the same underlying
> computation, not two different implementations.

---

## See Also

- [GLM Models](glm.md) - GLMs with regularization
- [Linear Models](linear.md) - Linear regression variants
