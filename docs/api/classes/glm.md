# GLM Model Classes

Direct GLM model access outside of Polars expressions.

All GLM classes (`Logistic`, `Poisson`, `NegativeBinomial`, `Tweedie`, `Probit`, `Cloglog`) accept a `lambda_=0.0` kwarg for L2 (ridge) penalty applied inside the IRLS update. The sklearn-style [`LogisticRegression`](#logisticregression) class exposes the same penalty via `C = 1 / lambda_` and an explicit `penalty` choice.

## Common Interface

```python
from polars_statistics import Logistic, Poisson, NegativeBinomial, Tweedie, Probit, Cloglog

model = Logistic(with_intercept=True)
model.fit(X, y)

# Properties
model.coefficients      # np.ndarray
model.intercept         # float or None
model.deviance          # float
model.null_deviance     # float
model.aic               # float
model.bic               # float

# Predict
predictions = model.predict(X_new)
probs = model.predict_proba(X_new)  # For classification models
```

---

## Logistic

Logistic regression for binary classification.

```python
from polars_statistics import Logistic

model = Logistic(
    lambda_: float = 0.0,  # L2 regularization
    with_intercept: bool = True,
)
model.fit(X, y)  # y: binary (0/1)

predictions = model.predict(X_new)      # Class predictions
probabilities = model.predict_proba(X_new)  # Probability estimates
```

---

## LogisticRegression

Sklearn-style logistic regression. Distinct from [`Logistic`](#logistic): uses inverse-strength regularization `C = 1 / lambda_` and an explicit `penalty` choice.

```python
from polars_statistics import LogisticRegression

model = LogisticRegression(
    penalty: str = "l2",                # "l2" or "none"
    C: float = 1.0,                     # Inverse of regularization strength
    threshold: float = 0.5,
    with_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-8,
    compute_inference: bool = True,
    confidence_level: float = 0.95,
)
model.fit(X, y)                         # y: binary (0/1)

# Methods
classes      = model.predict(X_new)             # 0/1 predictions
probs        = model.predict_proba(X_new)       # Probability estimates
scores       = model.decision_function(X_new)   # Linear scores (log-odds)
accuracy     = model.score(X_new, y_new)        # Mean accuracy

# Properties
model.coefficients    # np.ndarray
model.intercept       # float or None
model.n_iter          # int — IRLS iterations until convergence
```

---

## Poisson

Poisson regression for count data.

```python
from polars_statistics import Poisson

model = Poisson(
    lambda_: float = 0.0,
    with_intercept: bool = True,
)
model.fit(X, y)  # y: non-negative counts
```

---

## NegativeBinomial

Negative Binomial regression for overdispersed count data.

```python
from polars_statistics import NegativeBinomial

model = NegativeBinomial(
    theta: float | None = None,  # Dispersion; None = estimate
    estimate_theta: bool = True,
    lambda_: float = 0.0,
    with_intercept: bool = True,
)
model.fit(X, y)

# Additional property
model.theta  # Estimated dispersion parameter
```

---

## Tweedie

Tweedie GLM for flexible variance structures.

```python
from polars_statistics import Tweedie

model = Tweedie(
    var_power: float = 1.5,
    lambda_: float = 0.0,
    with_intercept: bool = True,
)
model.fit(X, y)
```

---

## Probit

Probit regression for binary classification.

```python
from polars_statistics import Probit

model = Probit(
    lambda_: float = 0.0,
    with_intercept: bool = True,
)
model.fit(X, y)  # y: binary (0/1)
```

---

## Cloglog

Complementary log-log regression for binary classification.

```python
from polars_statistics import Cloglog

model = Cloglog(
    lambda_: float = 0.0,
    with_intercept: bool = True,
)
model.fit(X, y)  # y: binary (0/1)
```

---

## Gamma

Gamma GLM (log link) for positive, right-skewed response data. Suitable for cost, duration,
or concentration data where variance scales with the mean squared.

```python
from polars_statistics import Gamma

model = Gamma(
    lambda_: float = 0.0,        # L2 regularization
    with_intercept: bool = True,
)
model.fit(X, y)   # y: strictly positive

# Properties
model.coefficients      # np.ndarray
model.intercept         # float or None
model.deviance          # float
model.null_deviance     # float
model.dispersion        # float — estimated dispersion parameter
model.aic               # float
model.bic               # float
model.n_observations    # int

# Predict
predictions = model.predict(X_new)  # Predicted means (exp(X @ coef + intercept))
```

**Example:**
```python
import numpy as np
from polars_statistics import Gamma

X = np.column_stack([np.random.normal(0, 1, 200)])
y = np.exp(0.5 * X[:, 0] + np.random.normal(0, 0.3, 200))

model = Gamma()
model.fit(X, y)
print(f"Dispersion: {model.dispersion:.4f}")
```

---

## GLMM

Generalized Linear Mixed Model — extends GLM with random effects for grouped/clustered data.

```python
from polars_statistics import GLMM

model = GLMM(
    family: str = "gaussian",     # "gaussian", "poisson", "binomial"
    link: str | None = None,      # None = canonical link for family
    with_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-4,
)
model.fit(X, y, groups)   # groups: 1D array of group labels (int or str)

# Properties
model.coefficients          # np.ndarray — fixed effects
model.intercept             # float or None
model.random_effects        # dict — per-group random intercept estimates
model.n_observations        # int
model.n_groups              # int
```

**Example:**
```python
import numpy as np
from polars_statistics import GLMM

groups = np.repeat(np.arange(10), 20)   # 10 groups, 20 obs each
X = np.random.normal(0, 1, (200, 2))
y = (X[:, 0] + np.random.normal(0, 0.5, 200) > 0).astype(float)

model = GLMM(family="binomial")
model.fit(X, y, groups)
```

---

## PSpline

Penalized spline (P-spline) regression — fits a smooth function of one variable using
B-spline basis functions with a difference-based roughness penalty.

```python
from polars_statistics import PSpline

model = PSpline(
    n_knots: int = 20,            # Number of interior knots
    degree: int = 3,              # B-spline degree (3 = cubic)
    lambda_: float = 1.0,        # Smoothing penalty
    with_intercept: bool = True,
)
model.fit(x, y)   # x, y: 1D numpy arrays

# Properties
model.coefficients      # np.ndarray — B-spline coefficients
model.intercept         # float or None
model.knots             # np.ndarray — knot locations
model.r_squared         # float
model.n_observations    # int

# Predict
smooth = model.predict(x_new)
```

**Example:**
```python
import numpy as np
from polars_statistics import PSpline

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + np.random.normal(0, 0.1, 100)

model = PSpline(n_knots=15, lambda_=0.1)
model.fit(x, y)
smooth = model.predict(x)
```

---

## Class Summary

| Class | Parameters |
|-------|------------|
| `Logistic` | `lambda_`, `with_intercept` |
| `LogisticRegression` | `penalty`, `C`, `threshold`, `with_intercept`, `max_iter`, `tol`, `compute_inference`, `confidence_level` |
| `Poisson` | `lambda_`, `with_intercept` |
| `NegativeBinomial` | `theta`, `estimate_theta`, `lambda_`, `with_intercept` |
| `Tweedie` | `var_power`, `lambda_`, `with_intercept` |
| `Probit` | `lambda_`, `with_intercept` |
| `Cloglog` | `lambda_`, `with_intercept` |
| `Gamma` | `lambda_`, `with_intercept` |
| `GLMM` | `family`, `link`, `with_intercept`, `max_iter`, `tol` |
| `PSpline` | `n_knots`, `degree`, `lambda_`, `with_intercept` |

---

## See Also

- [Linear Model Classes](linear.md)
- [ALM Class](alm.md)
