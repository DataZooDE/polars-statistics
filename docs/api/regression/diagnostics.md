# Regression Diagnostics

Tools for detecting multicollinearity, quasi-separation, influence and outliers, plus residual batteries for OLS and GLM fits.

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
```

---

## See Also

- [GLM Models](glm.md) - GLMs with regularization
- [Linear Models](linear.md) - Linear regression variants
