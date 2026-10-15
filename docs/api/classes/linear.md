# Linear Model Classes

Direct model access outside of Polars expressions.

## Common Interface

All linear model classes share this interface:

```python
from polars_statistics import OLS, Ridge, ElasticNet, WLS, RLS, BLS, Quantile, Isotonic
from polars_statistics import TheilSen, RANSAC, BayesianRidge, ARD, LARS, PassiveAggressive, MomentAccumulator

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
model.model_info        # dict — see "model_info" below

# Predict
predictions = model.predict(X_new)
```

### `model_info`

Every class on this page (plus every [GLM Model Class](glm.md)) exposes
`model_info -> dict` — `{"model_type": str, "family": str | None, "link": str}` — read from
the fitted model's own upstream `HasModelInfo` implementation, raising if the model is unfit.
`family`/`link` default to `"gaussian"`/`"identity"` for classes without an explicit family
(e.g. `OLS`); some classes have no family concept at all (`family: None`).

| Class | `model_type` | `family` | `link` |
|-------|--------------|----------|--------|
| `OLS` | `"ols"` | `"gaussian"` | `"identity"` |
| `Ridge` | `"ridge"` | `"gaussian"` | `"identity"` |
| `ElasticNet` | `"elastic_net"` | `"gaussian"` | `"identity"` |
| `WLS` | `"wls"` | `"gaussian"` | `"identity"` |
| `RLS` | `"rls"` | `"gaussian"` | `"identity"` |
| `BLS` (`lower_bound`/`upper_bound` not both `0`/unbounded-below) | `"bls"` | `"gaussian"` | `"identity"` |
| `BLS(lower_bound=0.0)` (non-negative least squares) | `"nnls"` | `"gaussian"` | `"identity"` |
| `PLS` | `"pls"` | `"gaussian"` | `"identity"` |
| `Quantile` | `"quantile"` | `None` | `"identity"` |
| `Isotonic` | `"isotonic"` | `None` | `"identity"` |
| `Huber` | `"huber"` | `None` | `"identity"` |
| `TheilSen` | `"theil_sen"` | `None` | `"identity"` |
| `RANSAC` | `"ransac"` | `None` | `"identity"` |
| `BayesianRidge` | `"bayesian_ridge"` | `"gaussian"` | `"identity"` |
| `ARD` | `"ard"` | `"gaussian"` | `"identity"` |
| `LARS` | `"lars"` | `"gaussian"` | `"identity"` |
| `PassiveAggressive` | `"passive_aggressive"` | `None` | `"identity"` |

`BLS` reports `"nnls"` vs. `"bls"` per upstream's own bound-based dispatch (a pure
non-negativity constraint on every coefficient is treated as the NNLS special case) — this is
not a separate constructor kwarg, it is inferred from the `lower_bound`/`upper_bound` you pass.
See [GLM Model Classes](glm.md#common-interface) for the GLM-family `model_type`/`family`/
`link` table (`Logistic`/`Probit`/`Cloglog` share `family="binomial"` but differ by `link`).

---

## OLS

Ordinary Least Squares.

```python
from polars_statistics import OLS

model = OLS(
    with_intercept: bool = True,
    compute_inference: bool = True,
    alias_tolerance: float = 1e-7,
)
model.fit(X, y)

# Inference getters (populated when compute_inference=True)
model.intercept_std_error, model.intercept_t_statistic   # float | None
model.intercept_p_value, model.intercept_conf_interval    # float | None, tuple[float, float] | None
model.conf_interval_lower, model.conf_interval_upper      # np.ndarray | None — feature coefficients

# Per-observation diagnostics
diag = model.augment(X, y, residual_type=None)   # dict: fitted/residual/std_residual/
                                                   # stud_residual/leverage/cooks_d/dffits/resid_type
```

**`alias_tolerance`** (default `1e-7`, R `lm`'s own tolerance): see
[Rank-deficient designs](#rank-deficient-designs-ols-wls) below.

**`.augment(x, y, residual_type=None)`:** R `broom::augment`-style per-observation
diagnostics — see [`ps.augment`](../regression/diagnostics.md#augment-per-observation-diagnostics)
for the full field semantics; `residual_type` defaults to `"response"` for OLS/WLS.

### `variance_factor` / `leverage_new` (OLS, WLS, Ridge)

```python
model.variance_factor()       # np.ndarray | None — (p+1)x(p+1) with intercept, p x p without
model.leverage_new(x_new)     # np.ndarray — leverage h_ii for new rows
```

`variance_factor()` returns the matrix that, multiplied by the residual variance
(`inv(X'X)` for OLS, `inv(X'WX)` for WLS, the ridge sandwich `A @ X'X @ A'` for
Ridge), gives the coefficient covariance. Returns `None` when the model was fit
from moments (`fit_from_accumulator`) — the raw design matrix isn't available to
compute it.

`leverage_new(x_new)` computes the hat-matrix diagonal for new observations
(`x_new @ variance_factor() @ x_new.T`, diagonal only). Raises `ValueError` (never
a faer panic) when `x_new`'s column count doesn't match the fitted model's feature
count. Returns `NaN` for moments-fits (same reason as `variance_factor()`).

### `fit_from_accumulator` (OLS)

Fit OLS from precomputed sufficient statistics held in a `MomentAccumulator`. Useful for
streaming or distributed workflows where raw data is unavailable at fit time.

```python
model = OLS(with_intercept=True, compute_inference=True)
model.fit_from_accumulator(accumulator)  # accumulator: MomentAccumulator
```

### Rank-deficient designs (OLS, WLS)

Since 0.8.0 (anofox-regression 0.5.21 #64), OLS and WLS alias rank-deficient designs
like R's `lm`: columns are kept in their given order, and a column that is a linear
combination of the intercept and the earlier kept columns gets a `NaN` coefficient
(`coefficients[j] = NaN`) and is excluded from `rank`/`n_parameters`. When two or more
columns are collinear, the **later** column in argument order is aliased — this holds
for every `solve_method` (`"qr"`, `"svd"`, `"cholesky"`) and is independent of column
scaling (a scaled copy, e.g. `x2 = 2 * x1`, aliases exactly like an exact duplicate).

Previously, `solve_method="svd"` silently returned a minimum-norm split of collinear
columns with no `NaN` flag, and `"qr"` aliased the *earlier* column of a collinear
pair — both now match `"cholesky"` and R's `dqrdc2` convention.

```python
model = OLS(solve_method="qr").fit(X, y)   # X columns 1 and 2 are collinear
model.coefficients  # e.g. [1.49, nan] — column 2 (the later one) is aliased
```

**`alias_tolerance`** (default `1e-7`, matching `anofox_regression::utils::LM_ALIAS_TOLERANCE`
and R `lm`'s own tolerance) controls how close to exactly collinear a column must be before it
is aliased: a **tighter** tolerance (smaller value) aliases more columns, a **looser**
tolerance (larger value) keeps more. The default reproduces the behaviour documented above
exactly.

---

## Ridge

Ridge regression (L2 regularization).

```python
from polars_statistics import Ridge

model = Ridge(
    lambda_: float = 1.0,
    with_intercept: bool = True,
    compute_inference: bool = True,
)
model.fit(X, y)

# Inference (readable for the first time at lambda_ > 0)
model.std_errors           # np.ndarray | None — sandwich SE: sigma^2 * A @ X'X @ A'
model.intercept_std_error  # float | None
model.t_statistics         # np.ndarray | None — None when lambda_ > 0
model.p_values             # np.ndarray | None — None when lambda_ > 0
model.variance_factor()    # np.ndarray | None — the sandwich A @ X'X @ A' matrix
model.leverage_new(x_new)  # np.ndarray — see the shared note above
```

**`lambda_ > 0` inference rule:** `std_errors`/`intercept_std_error` are always the
ridge sandwich standard errors (matching R). `t_statistics`/`p_values` are `None`
when `lambda_ > 0` — a classical t-test on shrunken/biased coefficients is not
statistically valid. At `lambda_ = 0`, Ridge delegates fully to OLS inference
(`t_statistics`/`p_values` populated, matching OLS to machine precision).

### `hc_inference` (Ridge)

Compute HC-robust (heteroscedasticity-consistent) standard errors and p-values for an
already-fitted Ridge model. Supports HC0–HC3 sandwich estimators.

```python
from polars_statistics import Ridge

model = Ridge(lambda_=1.0)
model.fit(X, y)
hc = model.hc_inference(X, y, hc_type="hc3")  # Returns dict with se, t_stat, p_value
```

**Supported `hc_type` values:** `"hc0"`, `"hc1"`, `"hc2"`, `"hc3"` (default `"hc3"`).

### `fit_from_accumulator` (Ridge)

Fit Ridge from precomputed sufficient statistics.

```python
model = Ridge(lambda_=1.0)
model.fit_from_accumulator(accumulator)  # accumulator: MomentAccumulator
```

---

## ElasticNet

Elastic Net regression (L1 + L2 regularization).

```python
from polars_statistics import ElasticNet

model = ElasticNet(
    lambda_: float = 1.0,
    alpha: float = 0.5,  # L1 ratio (0 = Ridge, 1 = Lasso)
    with_intercept: bool = True,
    compute_inference: bool = True,
)
model.fit(X, y)
```

---

## WLS

Weighted Least Squares.

```python
from polars_statistics import WLS

model = WLS(
    with_intercept: bool = True,
    compute_inference: bool = True,
    alias_tolerance: float = 1e-7,
)
model.fit(X, y, weights)  # weights: 1D numpy array

model.variance_factor()    # np.ndarray | None — inv(X'WX); see the OLS note above
model.leverage_new(x_new)  # np.ndarray — weighted hat-diagonal leverage; see the OLS note above

# Inference getters (populated when compute_inference=True)
model.std_errors, model.t_statistics, model.p_values      # np.ndarray | None — feature coefficients
model.conf_interval_lower, model.conf_interval_upper       # np.ndarray | None
model.intercept_std_error, model.intercept_t_statistic     # float | None
model.intercept_p_value, model.intercept_conf_interval      # float | None, tuple[float, float] | None

# Per-observation diagnostics
diag = model.augment(X, y, weights=None, residual_type=None)  # weights=None reuses the fit's own weights
```

**`alias_tolerance`** (default `1e-7`): same rank-deficiency tolerance as OLS — see
[Rank-deficient designs](#rank-deficient-designs-ols-wls) below.

**`.augment(x, y, weights=None, residual_type=None)`:** same R `broom::augment` semantics as
`OLS.augment` — see [`ps.augment`](../regression/diagnostics.md#augment-per-observation-diagnostics).
`weights=None` (default) reuses the fit's own weights rather than requiring them again.

> **Known limitation:** `WLS.hc_inference` raises `NotImplementedError` — HC-robust inference
> is not yet implemented for WLS. Use `OLS.hc_inference` or `Ridge.hc_inference` instead.

---

## RLS

Recursive Least Squares (online learning).

```python
from polars_statistics import RLS

model = RLS(
    forgetting_factor: float = 0.99,
    with_intercept: bool = True,
    initial_p_diagonal: float = 1e6,
)
model.fit(X, y)

model.initial_p_diagonal_value  # float — the P0 = delta * I value used
```

**`initial_p_diagonal`** (default `1e6`, matching the upstream crate's own default —
omitting it is identical to passing `1e6`): sets the diagonal value `delta` for the
initial covariance prior `P0 = delta * I`. A smaller value (e.g. `1e-3`) is a
"tighter" prior that measurably shifts early coefficient estimates toward zero
before enough observations have accumulated; a larger value is a "looser"
(less informative) prior. An invalid `delta` (negative or `NaN`) raises `ValueError`
at `.fit()` time (never a panic).

---

## BLS

Bounded Least Squares.

```python
from polars_statistics import BLS

model = BLS(
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    with_intercept: bool = True,
)
model.fit(X, y)
```

---

## Huber

Huber M-estimator (robust regression).

```python
from polars_statistics import Huber

model = Huber(
    epsilon: float = 1.35,
    alpha: float = 0.0001,
    with_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-5,
    compute_inference: bool = False,
    confidence_level: float = 0.95,
)
model.fit(X, y)

# Properties
model.coefficients      # np.ndarray
model.intercept         # float or None
model.scale             # float — robust scale estimate
model.epsilon           # float
model.outliers          # np.ndarray[bool] — down-weighted observations
model.n_outliers        # int
model.r_squared         # float
model.mse               # float
model.rmse              # float
model.n_observations    # int

# Inference (only populated when compute_inference=True; all None otherwise)
model.std_errors               # np.ndarray | None
model.t_statistics              # np.ndarray | None
model.p_values                  # np.ndarray | None
model.conf_interval_lower       # np.ndarray | None
model.conf_interval_upper       # np.ndarray | None
model.intercept_std_error       # float | None
model.intercept_t_statistic     # float | None
model.intercept_p_value         # float | None
model.intercept_conf_interval   # tuple[float, float] | None
```

**`compute_inference=True`** validates against R `MASS::rlm(method = "XtX")`'s
covariance (the nine getters above). Default `False` keeps every getter `None`
(no inference computed, faster fit). `confidence_level` controls the width of
`conf_interval_lower`/`conf_interval_upper`/`intercept_conf_interval` — a narrower
`confidence_level` (e.g. `0.90`) gives a strictly narrower interval around the
same point estimate.

---

## PLS

Partial Least Squares regression.

```python
from polars_statistics import PLS

model = PLS(
    n_components: int = 2,
    with_intercept: bool = True,
    tol: float = 1e-6,
    scale: bool = True,
)
model.fit(X, y)

# Methods
predictions = model.predict(X_new)
latent      = model.transform(X_new)   # Project into latent space

# Properties
model.coefficients              # np.ndarray
model.intercept                 # float or None
model.n_components              # int
model.explained_variance_ratio  # np.ndarray (one entry per component)
model.r_squared                 # float
model.n_observations            # int
```

---

## Quantile

Quantile regression.

```python
from polars_statistics import Quantile

model = Quantile(
    tau: float = 0.5,  # Quantile to estimate
    with_intercept: bool = True,
)
model.fit(X, y)

# Additional properties
model.tau              # float
model.pseudo_r_squared # float
model.check_loss       # float
```

---

## Isotonic

Isotonic (monotonic) regression.

```python
from polars_statistics import Isotonic

model = Isotonic(
    increasing: bool = True,
)
model.fit(x, y)  # x, y: 1D numpy arrays

# Properties
model.r_squared        # float
model.fitted_values    # np.ndarray
```

---

## MomentAccumulator

Online sufficient-statistic accumulator for OLS and Ridge. Accumulate batches of data
and pass to `fit_from_accumulator` to fit a model without materialising the full dataset.

```python
from polars_statistics import MomentAccumulator

acc = MomentAccumulator(n_features: int, with_intercept: bool = True)
acc.update(X_batch, y_batch)  # Call repeatedly with data batches
acc.update(X_batch2, y_batch2)

# Fit OLS or Ridge from accumulated statistics
from polars_statistics import OLS, Ridge
model = OLS().fit_from_accumulator(acc)
```

**Properties:**
```python
acc.n_samples    # int — total observations accumulated
acc.n_features   # int — number of features
```

**Example:**
```python
import numpy as np
from polars_statistics import MomentAccumulator, OLS

acc = MomentAccumulator(n_features=2)
for chunk in data_chunks:
    acc.update(chunk[:, :2], chunk[:, 2])

model = OLS().fit_from_accumulator(acc)
print(model.coefficients, model.r_squared)
```

---

## TheilSen

Theil-Sen median-of-slopes estimator — highly robust to outliers (up to ~29% contamination).

```python
from polars_statistics import TheilSen

model = TheilSen(
    with_intercept: bool = True,
    max_iter: int = 300,
    tol: float = 1e-3,
    univariate_pairwise: bool = True,
)
model.fit(X, y)

# Properties
model.coefficients      # np.ndarray
model.intercept         # float or None
model.r_squared         # float
model.n_observations    # int
```

**`univariate_pairwise`** (default `True`): for a single-feature `X`, use the
classical Theil-Sen estimator — the median of all pairwise slopes (Sen, 1968).
Set `univariate_pairwise=False` to use sklearn's spatial-median-over-subsamples
estimator instead (the original behaviour for single-feature data; this is always
the estimator used for multi-feature `X` regardless of this flag).

**Rank-deficient design (anofox-regression 0.5.15 #50):** a column that is exactly
linearly dependent on another is aliased to `NaN` in `coefficients` (not a huge
near-`1e14` value) — the same aliasing convention used elsewhere in this library
for genuinely rank-deficient designs.

**Example:**
```python
import numpy as np
from polars_statistics import TheilSen

X = np.column_stack([np.linspace(0, 10, 50)])
y = 2.5 * X[:, 0] + 1.0 + np.random.normal(0, 0.5, 50)
y[0] = 100.0  # outlier

model = TheilSen()
model.fit(X, y)
print(model.coefficients)  # ~[2.5]
```

---

## RANSAC

Random Sample Consensus — fits a model to an inlier subset by iteratively sampling
random minimal subsets and identifying consensus inliers.

```python
from polars_statistics import RANSAC

model = RANSAC(
    min_samples: int | float | None = None,  # Fraction or count of min samples per trial
    residual_threshold: float | None = None, # Max residual to count as inlier
    max_trials: int = 100,
    stop_probability: float = 0.99,
    stop_n_inliers: int | None = None,
    random_state: int = 0,
    with_intercept: bool = True,
    compute_inference: bool = False,
    confidence_level: float = 0.95,
)
model.fit(X, y)

# Properties — describe the inlier-only fit (anofox-regression 0.5.14 #35)
model.coefficients      # np.ndarray (fitted on inlier set)
model.intercept         # float or None
model.inlier_mask       # np.ndarray[bool] — per-observation inlier flag
model.n_inliers         # int
model.r_squared         # float (on inlier set)
model.n_observations    # int — now reports n_inliers, not the full sample size

# Inference (only populated when compute_inference=True; same 9 getters as Huber)
model.std_errors               # np.ndarray | None
model.t_statistics              # np.ndarray | None
model.p_values                  # np.ndarray | None
model.conf_interval_lower       # np.ndarray | None
model.conf_interval_upper       # np.ndarray | None
model.intercept_std_error       # float | None
model.intercept_t_statistic     # float | None
model.intercept_p_value         # float | None
model.intercept_conf_interval   # tuple[float, float] | None

# Inlier-OLS fit statistics (now meaningful — computed on the inlier subset)
model.adj_r_squared     # float
model.mse               # float
model.rmse              # float
model.f_statistic       # float
model.f_pvalue          # float
model.aic               # float
model.bic               # float
model.log_likelihood    # float

model.inlier_fit()      # OLS — a real fitted OLS instance on just the inliers
```

**0.8.0 semantics (anofox-regression 0.5.14 #35):** `r_squared`, `n_observations`,
and every fit-stat getter above describe the **inlier-only OLS fit**
(`|residual| <= residual_threshold`), not the full sample — `n_observations` equals
`n_inliers`, not the original row count. `residuals` still covers every row (not
just inliers). A constant-response design (sklearn's MAD=0 semantics) fits cleanly.

**`inlier_fit()`** returns a real, independently usable `OLS` instance (via
`PyOLS::from_fitted`) fit on exactly the inlier rows — its `coefficients`,
`intercept`, and `std_errors` match RANSAC's own to `1e-12`. Use it when you want
the full `OLS` API (e.g. `.summary()`, `.leverage_new()`) on the inlier subset.

**Example:**
```python
from polars_statistics import RANSAC

model = RANSAC(residual_threshold=2.0, random_state=42, compute_inference=True)
model.fit(X, y)
print(f"Inliers: {model.n_inliers}/{model.n_observations}")
print(model.std_errors, model.p_values)

inlier_ols = model.inlier_fit()
print(inlier_ols.summary())
```

---

## BayesianRidge

Bayesian Ridge regression with automatic regularization via evidence maximisation.
Provides posterior mean coefficients and per-prediction uncertainty.

```python
from polars_statistics import BayesianRidge

model = BayesianRidge(
    max_iter: int = 300,
    tol: float = 1e-3,
    alpha_1: float = 1e-6,   # Prior on noise precision (Gamma shape)
    alpha_2: float = 1e-6,   # Prior on noise precision (Gamma rate)
    lambda_1: float = 1e-6,  # Prior on weight precision (Gamma shape)
    lambda_2: float = 1e-6,  # Prior on weight precision (Gamma rate)
    with_intercept: bool = True,
    compute_score: bool = False,
)
model.fit(X, y)

# Properties
model.coefficients        # np.ndarray (posterior mean)
model.intercept           # float or None
model.alpha               # float — estimated noise precision
model.lambda_             # float — estimated weight precision
model.sigma               # np.ndarray — posterior covariance of weights
model.scores              # list[float] — evidence lower bound history (if compute_score=True)
model.r_squared           # float
model.n_observations      # int
```

---

## ARD

Automatic Relevance Determination (ARD) regression — sparse Bayesian model with per-feature
precision hyperparameters. Automatically prunes irrelevant features by driving their
precision to infinity.

```python
from polars_statistics import ARD

model = ARD(
    max_iter: int = 300,
    tol: float = 1e-3,
    alpha_1: float = 1e-6,
    alpha_2: float = 1e-6,
    lambda_1: float = 1e-6,
    lambda_2: float = 1e-6,
    threshold_lambda: float = 1e4,   # Features pruned when precision > threshold
    with_intercept: bool = True,
)
model.fit(X, y)

# Properties
model.coefficients         # np.ndarray (posterior mean; pruned features are 0)
model.intercept            # float or None
model.lambda_              # np.ndarray — per-feature precision
model.active_features      # np.ndarray[int] — indices of retained features
model.r_squared            # float
model.n_observations       # int
```

---

## LARS

Least Angle Regression — efficiently computes the full piecewise-linear regularization
path from null model to OLS.

```python
from polars_statistics import LARS

model = LARS(
    n_nonzero_coefs: int | None = None,  # Stop when this many features enter; None = full path
    with_intercept: bool = True,
)
model.fit(X, y)

# Properties
model.coefficients         # np.ndarray (at stopping point)
model.intercept            # float or None
model.coef_path            # list[np.ndarray] — coefficients at each knot
model.alphas               # list[float] — regularization values at each knot
model.r_squared            # float
model.adj_r_squared        # float (0.8.0) — R convention, df = active coefficients + intercept
model.mse                  # float (0.8.0)
model.rmse                 # float (0.8.0)
model.model_info           # dict (0.8.0) — {"model_type": "lars", "family": "gaussian", "link": "identity"}
model.n_observations       # int
```

**`adj_r_squared`/`mse`/`rmse` (0.8.0, anofox-regression 0.5.21 #65):** use R's `summary(lm())`
convention, `df = active coefficients + intercept`. For the full LAR path this matches
`summary(lm())` exactly; for `n_nonzero_coefs` stopping early (the active-set df formula) or a
fit without an intercept, `rmse`/`adj_r_squared` match `OLS(add_intercept=False)` fit on the
same active features.

**Example:**
```python
from polars_statistics import LARS

model = LARS(n_nonzero_coefs=5)
model.fit(X, y)
print(f"Active features: {(model.coefficients != 0).sum()}")
```

**Collinear columns** (anofox-regression 0.5.19 #62): when two feature columns are
exactly collinear (e.g. `x2 = 2 * x1`), LARS aliases the *later* column — its
coefficient is `NaN` and the fit equals the reduced model without it — instead of a
platform-dependent result that could previously raise on one architecture and fit on
another.

---

## PassiveAggressive

Online Passive-Aggressive regression — updates only when the current prediction error
exceeds `epsilon` (PA) or combines hinge loss with L2 regularization (PA-I / PA-II).

```python
from polars_statistics import PassiveAggressive

model = PassiveAggressive(
    C: float = 1.0,                  # Regularization parameter (smaller = more regularization)
    epsilon: float = 0.1,            # Insensitive zone width
    variant: str = "pa",             # "pa", "pa1", "pa2"
    max_iter: int = 1000,
    tol: float = 1e-3,
    shuffle: bool = True,
    seed: int | None = None,
    with_intercept: bool = True,
)
model.fit(X, y)

# Properties
model.coefficients         # np.ndarray
model.intercept            # float or None
model.n_iter               # int — actual iterations until convergence
model.r_squared            # float
model.n_observations       # int
```

**Example:**
```python
from polars_statistics import PassiveAggressive

# Online-style update for large datasets
model = PassiveAggressive(C=0.1, epsilon=0.05, variant="pa2")
model.fit(X_train, y_train)
```

---

## Class Summary

| Class | Parameters |
|-------|------------|
| `OLS` | `with_intercept`, `compute_inference`, `alias_tolerance` |
| `Ridge` | `lambda_`, `with_intercept`, `compute_inference` |
| `ElasticNet` | `lambda_`, `alpha`, `with_intercept`, `compute_inference` |
| `WLS` | `with_intercept`, `compute_inference`, `alias_tolerance` |
| `RLS` | `forgetting_factor`, `with_intercept`, `initial_p_diagonal` |
| `BLS` | `lower_bound`, `upper_bound`, `with_intercept` |
| `Huber` | `epsilon`, `alpha`, `with_intercept`, `max_iter`, `tol`, `compute_inference`, `confidence_level` |
| `PLS` | `n_components`, `with_intercept`, `tol`, `scale` |
| `Quantile` | `tau`, `with_intercept` |
| `Isotonic` | `increasing` |
| `MomentAccumulator` | `n_features`, `with_intercept` |
| `TheilSen` | `with_intercept`, `max_iter`, `tol`, `univariate_pairwise` |
| `RANSAC` | `min_samples`, `residual_threshold`, `max_trials`, `stop_probability`, `stop_n_inliers`, `random_state`, `with_intercept`, `compute_inference`, `confidence_level` |
| `BayesianRidge` | `max_iter`, `tol`, `alpha_1`, `alpha_2`, `lambda_1`, `lambda_2`, `with_intercept`, `compute_score` |
| `ARD` | `max_iter`, `tol`, `alpha_1`, `alpha_2`, `lambda_1`, `lambda_2`, `threshold_lambda`, `with_intercept` |
| `LARS` | `n_nonzero_coefs`, `with_intercept` |
| `PassiveAggressive` | `C`, `epsilon`, `variant`, `max_iter`, `tol`, `shuffle`, `seed`, `with_intercept` |

---

## See Also

- [GLM Model Classes](glm.md)
- [ALM Class](alm.md)
- [Test Model Classes](tests.md)
