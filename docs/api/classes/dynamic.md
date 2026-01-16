# LmDynamic Class

Dynamic linear model with time-varying coefficients.

## Usage

```python
from polars_statistics import LmDynamic

model = LmDynamic(
    ic: str = "aicc",              # "aic", "aicc", "bic"
    distribution: str = "normal",  # Error distribution
    lowess_span: float = 0.3,      # Smoothing span (0.05-1.0), None to disable
    max_models: int = 64,          # Max candidate models
    with_intercept: bool = True,
)

model.fit(X, y)
```

---

## Properties

```python
# Standard properties
model.coefficients           # np.ndarray - final coefficients
model.intercept              # float or None
model.r_squared              # float
model.adj_r_squared          # float
model.mse                    # float
model.rmse                   # float

# Dynamic-specific properties
model.dynamic_coefficients   # np.ndarray (n_obs x n_coef) - time-varying coefficients
model.model_weights          # np.ndarray (n_obs x n_models) - raw IC weights
model.smoothed_weights       # np.ndarray or None - LOWESS smoothed weights
model.pointwise_ic           # np.ndarray (n_obs x n_models) - per-observation IC values
```

---

## Parameters

| Parameter | Description |
|-----------|-------------|
| `ic` | Information criterion: `"aic"`, `"aicc"` (default), or `"bic"` |
| `distribution` | Error distribution for ALM fitting |
| `lowess_span` | LOWESS smoothing bandwidth (0.05-1.0). Lower = more local |
| `max_models` | Maximum number of candidate models |

---

## Example

```python
import numpy as np
from polars_statistics import LmDynamic

# Generate data with time-varying coefficients
np.random.seed(42)
n = 200
t = np.linspace(0, 1, n)
X = np.random.randn(n, 2)
# True coefficient changes over time
beta = np.column_stack([0.5 + 0.5 * t, -0.3 * (1 - t)])
y = 1 + np.sum(X * beta, axis=1) + np.random.randn(n) * 0.5

# Fit dynamic model
model = LmDynamic(lowess_span=0.3)
model.fit(X, y)

print(f"Final coefficients: {model.coefficients}")
print(f"R-squared: {model.r_squared:.3f}")
print(f"Dynamic coefficients shape: {model.dynamic_coefficients.shape}")
```

---

## See Also

- [lm_dynamic Expression](../regression/dynamic.md)
- [ALM Class](alm.md)
