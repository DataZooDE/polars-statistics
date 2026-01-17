# ALM Class

Augmented Linear Model with 24+ distributions.

## Basic Usage

```python
from polars_statistics import ALM

# Factory methods (recommended)
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
```

---

## Factory Methods

```python
ALM.normal()              # Normal distribution
ALM.laplace()             # Laplace (robust to outliers)
ALM.student_t(df=5.0)     # Student's t (heavy tails)
ALM.logistic()            # Logistic distribution
ALM.gamma()               # Gamma (positive continuous)
ALM.poisson()             # Poisson (count data)
ALM.negative_binomial()   # Negative binomial (overdispersed counts)
ALM.lognormal()           # Log-normal (positive continuous)
ALM.beta()                # Beta (bounded 0-1)
ALM.exponential()         # Exponential (positive continuous)
ALM.inverse_gaussian()    # Inverse Gaussian
```

---

## Properties

```python
model.coefficients      # np.ndarray
model.intercept         # float or None
model.log_likelihood    # float
model.aic               # float
model.bic               # float
model.std_errors        # np.ndarray (if compute_inference=True)
model.p_values          # np.ndarray (if compute_inference=True)
```

---

## Example

```python
import numpy as np
from polars_statistics import ALM

# Generate data
np.random.seed(42)
X = np.random.randn(100, 2)
y = 1 + 0.5 * X[:, 0] - 0.3 * X[:, 1] + np.random.laplace(0, 0.5, 100)

# Fit with Laplace distribution (robust to outliers)
model = ALM.laplace()
model.fit(X, y)

print(f"Intercept: {model.intercept:.3f}")
print(f"Coefficients: {model.coefficients}")
print(f"AIC: {model.aic:.2f}")
print(f"Log-likelihood: {model.log_likelihood:.2f}")
```

---

## Supported Distributions

| Category | Distributions |
|----------|---------------|
| Continuous | `normal`, `laplace`, `student_t`, `logistic` |
| Positive | `lognormal`, `loglaplace`, `gamma`, `inverse_gaussian`, `exponential` |
| Bounded (0,1) | `beta` |
| Count | `poisson`, `negative_binomial`, `binomial`, `geometric` |

---

## See Also

- [ALM Expression](../regression/alm.md)
- [Linear Model Classes](linear.md)
- [GLM Model Classes](glm.md)
