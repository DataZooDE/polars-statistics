# Linear Model Classes

Direct model access outside of Polars expressions.

## Common Interface

All linear model classes share this interface:

```python
from polars_statistics import OLS, Ridge, ElasticNet, WLS, RLS, BLS, Quantile, Isotonic

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

---

## See Also

- [GLM Model Classes](glm.md)
- [ALM Class](alm.md)
- [Test Model Classes](tests.md)
