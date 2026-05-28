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

---

## See Also

- [Linear Model Classes](linear.md)
- [ALM Class](alm.md)
