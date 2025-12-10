# polars-statistics API Reference

> **Version:** 0.1.0
> **Last Updated:** 2025-12-10

Complete API reference for polars-statistics. For quick start examples, see the [README](../README.md).

---

## Table of Contents

- [Conventions](#conventions)
- [Statistical Tests](#statistical-tests)
  - [Parametric Tests](#parametric-tests)
  - [Non-Parametric Tests](#non-parametric-tests)
  - [Distributional Tests](#distributional-tests)
  - [Forecast Comparison Tests](#forecast-comparison-tests)
  - [Modern Distribution Tests](#modern-distribution-tests)
- [Regression Models](#regression-models)
  - [Linear Models](#linear-models)
  - [GLM Models](#glm-models)
  - [Augmented Linear Model (ALM)](#augmented-linear-model-alm)
  - [Dynamic Linear Model (LmDynamic)](#dynamic-linear-model-lmdynamic)
  - [Demand Classification (AID)](#demand-classification-aid)
  - [Formula Syntax](#formula-syntax)
  - [Summary Functions](#summary-functions)
  - [Prediction Functions](#prediction-functions)
- [Model Classes](#model-classes)
  - [Linear Model Classes](#linear-model-classes)
  - [GLM Model Classes](#glm-model-classes)
  - [ALM Class](#alm-class)
  - [LmDynamic Class](#lmdynamic-class)
  - [Aid Class](#aid-class)
  - [Test Model Classes](#test-model-classes)
- [Bootstrap Methods](#bootstrap-methods)
- [Output Structures](#output-structures)

---

## Conventions

### Expression API

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

### Column References

All functions accept column names as strings or `pl.Expr`:

```python
ps.ols("y", "x1", "x2")              # String column names
ps.ols(pl.col("y"), pl.col("x1"))   # Polars expressions
ps.ols("y", pl.col("x1") * 2)       # Mixed / transformed
```

### Return Types

- **Statistical tests** return a struct with `statistic` and `p_value` fields
- **Regression models** return a struct with model-specific fields (see [Output Structures](#output-structures))
- **Summary functions** return `List[Struct]` with coefficient statistics
- **Prediction functions** return `Struct{prediction, lower, upper}` per row

[↑ Back to top](#table-of-contents)

---

## Statistical Tests

### Parametric Tests

#### `ttest_ind`

Independent samples t-test (Welch's t-test by default).

```python
ps.ttest_ind(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    alternative: str = "two-sided",  # "two-sided", "less", "greater"
    equal_var: bool = False,         # False = Welch's, True = Student's
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

---

#### `ttest_paired`

Paired samples t-test.

```python
ps.ttest_paired(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    alternative: str = "two-sided",  # "two-sided", "less", "greater"
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

---

#### `brown_forsythe`

Brown-Forsythe test for equality of variances.

```python
ps.brown_forsythe(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

---

#### `yuen_test`

Yuen's test for trimmed means (robust to outliers).

```python
ps.yuen_test(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    trim: float = 0.2,  # Proportion to trim from each tail
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

---

### Non-Parametric Tests

#### `mann_whitney_u`

Mann-Whitney U test (Wilcoxon rank-sum test).

```python
ps.mann_whitney_u(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

---

#### `wilcoxon_signed_rank`

Wilcoxon signed-rank test for paired samples.

```python
ps.wilcoxon_signed_rank(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

---

#### `kruskal_wallis`

Kruskal-Wallis H test for comparing 3+ independent groups.

```python
ps.kruskal_wallis(
    *groups: Union[pl.Expr, str],  # 3 or more groups
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

---

#### `brunner_munzel`

Brunner-Munzel test for stochastic equality.

```python
ps.brunner_munzel(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    alternative: str = "two-sided",  # "two-sided", "less", "greater"
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

---

### Distributional Tests

#### `shapiro_wilk`

Shapiro-Wilk test for normality.

```python
ps.shapiro_wilk(
    x: Union[pl.Expr, str],
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

---

#### `dagostino`

D'Agostino-Pearson test for normality.

```python
ps.dagostino(
    x: Union[pl.Expr, str],
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

---

### Forecast Comparison Tests

#### `diebold_mariano`

Diebold-Mariano test for equal predictive accuracy.

```python
ps.diebold_mariano(
    errors1: Union[pl.Expr, str],
    errors2: Union[pl.Expr, str],
    loss: str = "squared",  # "squared", "absolute"
    horizon: int = 1,       # Forecast horizon
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

---

#### `permutation_t_test`

Permutation-based t-test (non-parametric).

```python
ps.permutation_t_test(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    alternative: str = "two-sided",
    n_permutations: int = 999,
    seed: int | None = None,
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

---

#### `clark_west`

Clark-West test for nested model comparison.

```python
ps.clark_west(
    restricted_errors: Union[pl.Expr, str],
    unrestricted_errors: Union[pl.Expr, str],
    horizon: int = 1,
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

---

#### `spa_test`

Superior Predictive Ability (SPA) test.

```python
ps.spa_test(
    benchmark_loss: Union[pl.Expr, str],
    *model_losses: Union[pl.Expr, str],
    n_bootstrap: int = 999,
    block_length: float = 5.0,
    seed: int | None = None,
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

---

#### `model_confidence_set`

Model Confidence Set (MCS) for model selection.

```python
ps.model_confidence_set(
    *model_losses: Union[pl.Expr, str],
    alpha: float = 0.1,
    statistic: str = "range",  # "range" or "semi-quadratic"
    n_bootstrap: int = 999,
    block_length: float = 5.0,
    seed: int | None = None,
) -> pl.Expr
```

**Returns:** `Struct{included: List[Boolean], p_values: List[Float64]}`

---

#### `mspe_adjusted`

MSPE-Adjusted SPA test for nested models.

```python
ps.mspe_adjusted(
    benchmark_errors: Union[pl.Expr, str],
    *model_errors: Union[pl.Expr, str],
    n_bootstrap: int = 999,
    block_length: float = 5.0,
    seed: int | None = None,
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

---

### Modern Distribution Tests

#### `energy_distance`

Energy Distance test for comparing distributions.

```python
ps.energy_distance(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    n_permutations: int = 999,
    seed: int | None = None,
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

---

#### `mmd_test`

Maximum Mean Discrepancy (MMD) test with Gaussian kernel.

```python
ps.mmd_test(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    n_permutations: int = 999,
    seed: int | None = None,
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

[↑ Back to top](#table-of-contents)

---

## Regression Models

### Linear Models

#### `ols`

Ordinary Least Squares regression.

```python
ps.ols(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Linear Model Output](#linear-model-output)

---

#### `ridge`

Ridge regression (L2 regularization).

```python
ps.ridge(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 1.0,
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Linear Model Output](#linear-model-output)

---

#### `elastic_net`

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

**Returns:** See [Linear Model Output](#linear-model-output)

---

#### `wls`

Weighted Least Squares regression.

```python
ps.wls(
    y: Union[pl.Expr, str],
    weights: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Linear Model Output](#linear-model-output)

---

#### `rls`

Recursive Least Squares regression (online learning).

```python
ps.rls(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    forgetting_factor: float = 0.99,
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Linear Model Output](#linear-model-output)

---

#### `bls`

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

**Returns:** See [Linear Model Output](#linear-model-output)

---

#### `nnls`

Non-negative Least Squares (shorthand for `bls` with `lower_bound=0`).

```python
ps.nnls(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Linear Model Output](#linear-model-output)

---

### GLM Models

#### `logistic`

Logistic regression for binary classification.

```python
ps.logistic(
    y: Union[pl.Expr, str],  # Binary (0/1)
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [GLM Output](#glm-output)

---

#### `poisson`

Poisson regression for count data.

```python
ps.poisson(
    y: Union[pl.Expr, str],  # Non-negative counts
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [GLM Output](#glm-output)

---

#### `negative_binomial`

Negative Binomial regression for overdispersed count data.

```python
ps.negative_binomial(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    theta: float | None = None,  # Dispersion; None = estimate
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [GLM Output](#glm-output)

---

#### `tweedie`

Tweedie GLM for flexible variance structures.

```python
ps.tweedie(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    var_power: float = 1.5,  # 0=Gaussian, 1=Poisson, 2=Gamma, 3=InvGaussian
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [GLM Output](#glm-output)

---

#### `probit`

Probit regression for binary classification.

```python
ps.probit(
    y: Union[pl.Expr, str],  # Binary (0/1)
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [GLM Output](#glm-output)

---

#### `cloglog`

Complementary log-log regression for binary classification.

```python
ps.cloglog(
    y: Union[pl.Expr, str],  # Binary (0/1)
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [GLM Output](#glm-output)

---

### Augmented Linear Model (ALM)

Flexible regression supporting 24+ distributions.

```python
ps.alm(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    distribution: str = "normal",
    with_intercept: bool = True,
) -> pl.Expr
```

**Supported distributions:**

| Category | Distributions |
|----------|---------------|
| Continuous | `normal`, `laplace`, `student_t`, `logistic` |
| Positive | `lognormal`, `loglaplace`, `gamma`, `inverse_gaussian`, `exponential` |
| Bounded (0,1) | `beta` |
| Count | `poisson`, `negative_binomial`, `binomial`, `geometric` |

**Returns:** See [ALM Output](#alm-output)

---

### Dynamic Linear Model (LmDynamic)

Time-varying coefficient regression using information criterion-weighted model averaging with optional LOWESS smoothing.

#### `lm_dynamic`

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

**Returns:** See [LmDynamic Output](#lmdynamic-output)

**Example:**

```python
# Fit dynamic model per group
df.group_by("group").agg(
    ps.lm_dynamic("y", "x1", "x2", lowess_span=0.3).alias("model")
).unnest("model")
```

---

### Demand Classification (AID)

Automatic Identification of Demand patterns with anomaly detection.

> **Reference:** Svetunkov, I. & Sroginis, A. (2025). *Why do zeroes happen? A model-based approach for demand classification*. [arXiv:2504.05894](https://arxiv.org/abs/2504.05894)

#### `aid`

Classifies demand time series as regular or intermittent, fits the best distribution, and detects anomalies.

```python
ps.aid(
    y: Union[pl.Expr, str],
    intermittent_threshold: float = 0.3,  # Zero proportion threshold
) -> pl.Expr
```

**Returns:** See [AID Output](#aid-output)

**Example:**

```python
# Classify demand per SKU
df.group_by("sku").agg(
    ps.aid("demand").alias("classification")
).unnest("classification")
```

---

#### `aid_anomalies`

Returns per-observation anomaly flags. Use with `.over()` to add anomaly columns to the original DataFrame.

```python
ps.aid_anomalies(
    y: Union[pl.Expr, str],
    intermittent_threshold: float = 0.3,
) -> pl.Expr
```

**Returns:** `Struct{stockout: Boolean, new_product: Boolean, obsolete_product: Boolean, high_outlier: Boolean, low_outlier: Boolean}` per row

**Anomaly Types:**

| Type | Description |
|------|-------------|
| `stockout` | Unexpected zero in otherwise positive demand |
| `new_product` | Leading zeros pattern (product launch) |
| `obsolete_product` | Trailing zeros pattern (product phase-out) |
| `high_outlier` | Unusually high demand value |
| `low_outlier` | Unusually low demand value |

**Example:**

```python
# Add per-row anomaly flags
result = df.with_columns(
    ps.aid_anomalies("demand").over("sku").alias("anomalies")
).unnest("anomalies")

# Filter to flagged observations
result.filter(pl.col("high_outlier") | pl.col("stockout"))
```

---

### Formula Syntax

Formula-based regression using R-style syntax. Supports polynomial and interaction effects that are computed per-group with `group_by`/`over`.

> **Note:** Unlike R, the intercept is controlled via the `with_intercept` parameter, not in the formula. R's `y ~ x - 1` or `y ~ 0 + x` syntax for removing the intercept is not supported. Use `with_intercept=False` instead.

#### Supported Formula Syntax

| Pattern | Example | Expansion |
|---------|---------|-----------|
| Main effects | `y ~ x1 + x2` | Two variables |
| Interaction only | `y ~ x1:x2` | Product term |
| Full crossing | `y ~ x1 * x2` | `x1 + x2 + x1:x2` |
| Polynomial (centered) | `y ~ poly(x, 2)` | Centered x, x^2 |
| Polynomial (raw) | `y ~ poly(x, 2, raw=True)` | x, x^2 |
| Explicit transform | `y ~ I(x^2)` | x squared |

#### Formula Functions

All regression models have a `*_formula` variant:

```python
ps.ols_formula(formula: str, with_intercept: bool = True) -> pl.Expr
ps.ridge_formula(formula: str, lambda_: float = 1.0, with_intercept: bool = True) -> pl.Expr
ps.elastic_net_formula(formula: str, lambda_: float = 1.0, alpha: float = 0.5, with_intercept: bool = True) -> pl.Expr
ps.wls_formula(formula: str, weights: Union[pl.Expr, str], with_intercept: bool = True) -> pl.Expr
ps.rls_formula(formula: str, forgetting_factor: float = 0.99, with_intercept: bool = True) -> pl.Expr
ps.bls_formula(formula: str, lower_bound: float | None = None, upper_bound: float | None = None, with_intercept: bool = True) -> pl.Expr
ps.nnls_formula(formula: str, with_intercept: bool = True) -> pl.Expr
ps.logistic_formula(formula: str, with_intercept: bool = True) -> pl.Expr
ps.poisson_formula(formula: str, with_intercept: bool = True) -> pl.Expr
ps.negative_binomial_formula(formula: str, theta: float | None = None, with_intercept: bool = True) -> pl.Expr
ps.tweedie_formula(formula: str, var_power: float = 1.5, with_intercept: bool = True) -> pl.Expr
ps.probit_formula(formula: str, with_intercept: bool = True) -> pl.Expr
ps.cloglog_formula(formula: str, with_intercept: bool = True) -> pl.Expr
ps.alm_formula(formula: str, distribution: str = "normal", with_intercept: bool = True) -> pl.Expr
```

**Example:**

```python
# Polynomial regression per group
df.group_by("group").agg(
    ps.ols_formula("y ~ poly(x, 2)").alias("model")
)

# Full interaction model
df.group_by("group").agg(
    ps.ols_formula("y ~ x1 * x2").alias("model")
)
```

---

### Summary Functions

Return coefficient statistics in tidy format (like R's `broom::tidy`).

```python
ps.ols_summary(y, *x, with_intercept=True) -> pl.Expr
ps.ridge_summary(y, *x, lambda_=1.0, with_intercept=True) -> pl.Expr
ps.elastic_net_summary(y, *x, lambda_=1.0, alpha=0.5, with_intercept=True) -> pl.Expr
ps.wls_summary(y, weights, *x, with_intercept=True) -> pl.Expr
ps.rls_summary(y, *x, forgetting_factor=0.99, with_intercept=True) -> pl.Expr
ps.bls_summary(y, *x, lower_bound=None, upper_bound=None, with_intercept=True) -> pl.Expr
ps.logistic_summary(y, *x, with_intercept=True) -> pl.Expr
ps.poisson_summary(y, *x, with_intercept=True) -> pl.Expr
ps.negative_binomial_summary(y, *x, theta=None, with_intercept=True) -> pl.Expr
ps.tweedie_summary(y, *x, var_power=1.5, with_intercept=True) -> pl.Expr
ps.probit_summary(y, *x, with_intercept=True) -> pl.Expr
ps.cloglog_summary(y, *x, with_intercept=True) -> pl.Expr
ps.alm_summary(y, *x, distribution="normal", with_intercept=True) -> pl.Expr
```

Formula variants also available: `ps.ols_formula_summary(formula, ...)`, etc.

**Returns:** `List[Struct{term: String, estimate: Float64, std_error: Float64, statistic: Float64, p_value: Float64}]`

**Example:**

```python
df.group_by("group").agg(
    ps.ols_summary("y", "x1", "x2").alias("coef")
).explode("coef").unnest("coef")
```

---

### Prediction Functions

Return per-row predictions with optional confidence/prediction intervals.

```python
ps.ols_predict(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    add_intercept: bool = True,
    interval: str | None = None,  # None, "confidence", "prediction"
    level: float = 0.95,
    null_policy: str = "drop",    # "drop", "drop_y_zero_x"
) -> pl.Expr
```

Available for all models: `ols_predict`, `ridge_predict`, `elastic_net_predict`, `wls_predict`, `rls_predict`, `bls_predict`, `nnls_predict`, `logistic_predict`, `poisson_predict`, `negative_binomial_predict`, `tweedie_predict`, `probit_predict`, `cloglog_predict`, `alm_predict`.

Formula variants also available: `ps.ols_formula_predict(formula, ...)`, etc.

**Returns:** `Struct{prediction: Float64, lower: Float64, upper: Float64}`

**Example:**

```python
# Per-group predictions with intervals
df.with_columns(
    ps.ols_predict("y", "x1", "x2", interval="prediction", level=0.95)
        .over("group").alias("pred")
).unnest("pred")
```

[↑ Back to top](#table-of-contents)

---

## Model Classes

For users who need direct model access outside of Polars expressions.

### Linear Model Classes

```python
from polars_statistics import OLS, Ridge, ElasticNet, WLS, RLS, BLS

# Fit
model = OLS(with_intercept=True, compute_inference=True)
model.fit(X, y)  # X: 2D numpy array, y: 1D numpy array

# Properties
model.coefficients      # np.ndarray
model.intercept         # float or None
model.r_squared         # float
model.adj_r_squared     # float
model.std_errors        # np.ndarray (if compute_inference=True)
model.p_values          # np.ndarray (if compute_inference=True)
model.aic               # float
model.bic               # float

# Predict
predictions = model.predict(X_new)
```

| Class | Parameters |
|-------|------------|
| `OLS` | `with_intercept`, `compute_inference` |
| `Ridge` | `lambda_`, `with_intercept`, `compute_inference` |
| `ElasticNet` | `lambda_`, `alpha`, `with_intercept`, `compute_inference` |
| `WLS` | `with_intercept`, `compute_inference` |
| `RLS` | `forgetting_factor`, `with_intercept` |
| `BLS` | `lower_bound`, `upper_bound`, `with_intercept` |

---

### GLM Model Classes

```python
from polars_statistics import Logistic, Poisson, NegativeBinomial, Tweedie, Probit, Cloglog

model = Logistic(with_intercept=True)
model.fit(X, y)

# Predict
predictions = model.predict(X_new)
probs = model.predict_proba(X_new)  # For classification models
```

| Class | Parameters |
|-------|------------|
| `Logistic` | `with_intercept` |
| `Poisson` | `with_intercept` |
| `NegativeBinomial` | `theta`, `estimate_theta`, `with_intercept` |
| `Tweedie` | `var_power`, `with_intercept` |
| `Probit` | `with_intercept` |
| `Cloglog` | `with_intercept` |

---

### ALM Class

```python
from polars_statistics import ALM

# Factory methods
model = ALM.normal()
model = ALM.laplace()
model = ALM.student_t(df=5.0)
model = ALM.gamma()
model = ALM.poisson()

# Or direct construction
model = ALM(
    distribution="inverse_gaussian",
    link="inverse",
    with_intercept=True,
    compute_inference=True,
)

model.fit(X, y)
print(model.log_likelihood, model.aic, model.bic)
```

---

### LmDynamic Class

Dynamic linear model with time-varying coefficients.

```python
from polars_statistics import LmDynamic

model = LmDynamic(
    ic="aicc",                # "aic", "aicc", "bic"
    distribution="normal",   # Error distribution
    lowess_span=0.3,         # Smoothing span (0.05-1.0), None to disable
    max_models=64,           # Max candidate models
    with_intercept=True,
)

model.fit(X, y)

# Properties
model.coefficients           # np.ndarray - final coefficients
model.intercept              # float or None
model.r_squared              # float
model.dynamic_coefficients   # np.ndarray (n_obs x n_coef) - time-varying coefficients
model.model_weights          # np.ndarray (n_obs x n_models) - raw IC weights
model.smoothed_weights       # np.ndarray or None - LOWESS smoothed weights
model.pointwise_ic           # np.ndarray (n_obs x n_models) - per-observation IC values

# Predict
predictions = model.predict(X_new)
```

---

### Aid Class

Automatic Identification of Demand patterns.

```python
from polars_statistics import Aid

classifier = Aid(
    intermittent_threshold=0.3,  # Zero proportion threshold
    ic="aicc",                   # "aic", "aicc", "bic"
)

result = classifier.classify(y)  # y: 1D numpy array

# AidResult properties
result.demand_type        # str: "regular" or "intermittent"
result.distribution       # str: best-fit distribution name
result.is_fractional      # bool: whether data has fractional values
result.mean               # float: estimated mean
result.variance           # float: estimated variance
result.zero_proportion    # float: proportion of zeros
result.n_observations     # int: number of observations
result.anomalies          # list[str]: per-observation anomaly flags
result.ic_values          # dict: IC values for each candidate distribution

# Convenience methods
result.is_regular()           # bool
result.is_intermittent()      # bool
result.has_stockouts()        # bool
result.is_new_product()       # bool (leading zeros pattern)
result.is_obsolete_product()  # bool (trailing zeros pattern)
result.anomaly_counts()       # dict: counts by anomaly type
```

**Supported Distributions:**

| Category | Distributions |
|----------|---------------|
| Count | `poisson`, `negative_binomial`, `geometric` |
| Continuous | `normal`, `gamma`, `lognormal`, `rectified_normal` |

---

### Test Model Classes

Statistical tests available as model classes with `.fit()`, `.statistic`, `.p_value`, and `.summary()` methods.

#### Parametric Test Classes

```python
from polars_statistics import TTestInd, TTestPaired, BrownForsythe, YuenTest

# Independent samples t-test
test = TTestInd(alternative="two-sided", equal_var=False)
test.fit(x, y)
print(test.statistic, test.p_value)
print(test.summary())

# Paired samples t-test
test = TTestPaired(alternative="two-sided")
test.fit(before, after)

# Brown-Forsythe test for variance equality
test = BrownForsythe().fit(x, y)

# Yuen's trimmed mean test
test = YuenTest(trim=0.2).fit(x, y)
```

| Class | Parameters | Input |
|-------|------------|-------|
| `TTestInd` | `alternative`, `equal_var` | `fit(x, y)` |
| `TTestPaired` | `alternative` | `fit(x, y)` |
| `BrownForsythe` | - | `fit(x, y)` |
| `YuenTest` | `trim` | `fit(x, y)` |

---

#### Non-Parametric Test Classes

```python
from polars_statistics import MannWhitneyU, WilcoxonSignedRank, KruskalWallis, BrunnerMunzel

# Mann-Whitney U test
test = MannWhitneyU().fit(x, y)

# Wilcoxon signed-rank test (paired)
test = WilcoxonSignedRank().fit(before, after)

# Kruskal-Wallis H test (multiple groups)
test = KruskalWallis().fit(g1, g2, g3)

# Brunner-Munzel test
test = BrunnerMunzel(alternative="two-sided").fit(x, y)
```

| Class | Parameters | Input |
|-------|------------|-------|
| `MannWhitneyU` | - | `fit(x, y)` |
| `WilcoxonSignedRank` | - | `fit(x, y)` |
| `KruskalWallis` | - | `fit(*groups)` |
| `BrunnerMunzel` | `alternative` | `fit(x, y)` |

---

#### Distributional Test Classes

```python
from polars_statistics import ShapiroWilk, DAgostino

# Shapiro-Wilk normality test
test = ShapiroWilk().fit(x)
print(test.statistic, test.p_value)

# D'Agostino-Pearson normality test
test = DAgostino().fit(x)
```

| Class | Parameters | Input |
|-------|------------|-------|
| `ShapiroWilk` | - | `fit(x)` |
| `DAgostino` | - | `fit(x)` |

---

#### Test Class Properties and Methods

All test classes share these common properties and methods:

```python
test.is_fitted()    # bool: Check if test has been performed
test.statistic      # float: Test statistic value
test.p_value        # float: P-value
test.summary()      # str: Formatted summary of results
```

**Example summary output:**

```
Independent Samples T-Test
==========================

Test statistic:       -2.3412
P-value:           2.1400e-02
Alternative:        two-sided
Equal variance:         False (Welch's t)
Sample sizes:    n1=50, n2=50

Result: Reject H0 at alpha=0.05
```

[↑ Back to top](#table-of-contents)

---

## Bootstrap Methods

```python
from polars_statistics import StationaryBootstrap, CircularBlockBootstrap

# Stationary bootstrap (random block lengths)
bootstrap = StationaryBootstrap(expected_block_length=5.0, seed=42)
samples = bootstrap.samples(data, n_samples=1000)

# Circular block bootstrap (fixed block length)
cbb = CircularBlockBootstrap(block_length=10, seed=42)
samples = cbb.samples(data, n_samples=1000)
```

[↑ Back to top](#table-of-contents)

---

## Output Structures

### Linear Model Output

```
Struct {
    intercept: Float64,
    coefficients: List[Float64],
    r_squared: Float64,
    adj_r_squared: Float64,
    mse: Float64,
    rmse: Float64,
    f_statistic: Float64,
    f_pvalue: Float64,
    aic: Float64,
    bic: Float64,
    n_observations: UInt32,
}
```

### GLM Output

```
Struct {
    intercept: Float64,
    coefficients: List[Float64],
    deviance: Float64,
    null_deviance: Float64,
    aic: Float64,
    bic: Float64,
    n_observations: UInt32,
}
```

### ALM Output

```
Struct {
    intercept: Float64,
    coefficients: List[Float64],
    aic: Float64,
    bic: Float64,
    log_likelihood: Float64,
    n_observations: UInt32,
}
```

### LmDynamic Output

```
Struct {
    intercept: Float64,
    coefficients: List[Float64],
    r_squared: Float64,
    adj_r_squared: Float64,
    mse: Float64,
    rmse: Float64,
    n_observations: UInt32,
}
```

### AID Output

```
Struct {
    demand_type: String,           # "regular" or "intermittent"
    is_intermittent: Boolean,
    is_fractional: Boolean,
    distribution: String,          # Best-fit distribution name
    mean: Float64,
    variance: Float64,
    zero_proportion: Float64,
    n_observations: UInt32,
    has_stockouts: Boolean,
    is_new_product: Boolean,
    is_obsolete_product: Boolean,
    stockout_count: UInt32,
    new_product_count: UInt32,
    obsolete_product_count: UInt32,
    high_outlier_count: UInt32,
    low_outlier_count: UInt32,
}
```

### AID Anomalies Output

Per-row struct (use with `.over()` and `.unnest()`):

```
Struct {
    stockout: Boolean,
    new_product: Boolean,
    obsolete_product: Boolean,
    high_outlier: Boolean,
    low_outlier: Boolean,
}
```

### Summary Output

```
List[Struct {
    term: String,
    estimate: Float64,
    std_error: Float64,
    statistic: Float64,
    p_value: Float64,
}]
```

### Prediction Output

```
Struct {
    prediction: Float64,
    lower: Float64,
    upper: Float64,
}
```

[↑ Back to top](#table-of-contents)

---

## Performance Notes

- **Rust-powered**: All computations in Rust via [faer](https://github.com/sarah-ek/faer-rs) linear algebra
- **Zero-copy**: Direct memory sharing between Python and Rust
- **Parallelization**: Automatic for `group_by` operations
- **SIMD**: Optimized statistical computations

[↑ Back to top](#table-of-contents)

---

## See Also

- [README](../README.md) - Quick start guide
- [Polars Documentation](https://docs.pola.rs/)
- [faer](https://github.com/sarah-ek/faer-rs) - Linear algebra backend

[↑ Back to top](#table-of-contents)
