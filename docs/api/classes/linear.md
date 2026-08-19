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

# Predict
predictions = model.predict(X_new)
```

---

## OLS

Ordinary Least Squares.

```python
from polars_statistics import OLS

model = OLS(
    with_intercept: bool = True,
    compute_inference: bool = True,
)
model.fit(X, y)
```

### `fit_from_accumulator` (OLS)

Fit OLS from precomputed sufficient statistics held in a `MomentAccumulator`. Useful for
streaming or distributed workflows where raw data is unavailable at fit time.

```python
model = OLS(with_intercept=True, compute_inference=True)
model.fit_from_accumulator(accumulator)  # accumulator: MomentAccumulator
```

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
```

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
)
model.fit(X, y, weights)  # weights: 1D numpy array
```

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
)
model.fit(X, y)
```

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
```

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
)
model.fit(X, y)

# Properties
model.coefficients      # np.ndarray
model.intercept         # float or None
model.r_squared         # float
model.n_observations    # int
```

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
    min_samples: int | float = 0.5,    # Fraction or count of min samples per trial
    residual_threshold: float = 1.0,   # Max residual to count as inlier
    max_iter: int = 100,
    stop_n_inliers: int | None = None,
    seed: int | None = None,
    with_intercept: bool = True,
)
model.fit(X, y)

# Properties
model.coefficients      # np.ndarray (fitted on inlier set)
model.intercept         # float or None
model.inlier_mask       # np.ndarray[bool] — per-observation inlier flag
model.n_inliers         # int
model.r_squared         # float (on inlier set)
model.n_observations    # int
```

**Example:**
```python
from polars_statistics import RANSAC

model = RANSAC(residual_threshold=2.0, seed=42)
model.fit(X, y)
print(f"Inliers: {model.n_inliers}/{model.n_observations}")
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
model.n_observations       # int
```

**Example:**
```python
from polars_statistics import LARS

model = LARS(n_nonzero_coefs=5)
model.fit(X, y)
print(f"Active features: {(model.coefficients != 0).sum()}")
```

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
| `OLS` | `with_intercept`, `compute_inference` |
| `Ridge` | `lambda_`, `with_intercept`, `compute_inference` |
| `ElasticNet` | `lambda_`, `alpha`, `with_intercept`, `compute_inference` |
| `WLS` | `with_intercept`, `compute_inference` |
| `RLS` | `forgetting_factor`, `with_intercept` |
| `BLS` | `lower_bound`, `upper_bound`, `with_intercept` |
| `Huber` | `epsilon`, `alpha`, `with_intercept`, `max_iter`, `tol` |
| `PLS` | `n_components`, `with_intercept`, `tol`, `scale` |
| `Quantile` | `tau`, `with_intercept` |
| `Isotonic` | `increasing` |
| `MomentAccumulator` | `n_features`, `with_intercept` |
| `TheilSen` | `with_intercept`, `max_iter`, `tol` |
| `RANSAC` | `min_samples`, `residual_threshold`, `max_iter`, `stop_n_inliers`, `seed`, `with_intercept` |
| `BayesianRidge` | `max_iter`, `tol`, `alpha_1`, `alpha_2`, `lambda_1`, `lambda_2`, `with_intercept`, `compute_score` |
| `ARD` | `max_iter`, `tol`, `alpha_1`, `alpha_2`, `lambda_1`, `lambda_2`, `threshold_lambda`, `with_intercept` |
| `LARS` | `n_nonzero_coefs`, `with_intercept` |
| `PassiveAggressive` | `C`, `epsilon`, `variant`, `max_iter`, `tol`, `shuffle`, `seed`, `with_intercept` |

---

## See Also

- [GLM Model Classes](glm.md)
- [ALM Class](alm.md)
- [Test Model Classes](tests.md)
