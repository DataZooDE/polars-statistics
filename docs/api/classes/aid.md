# Aid Class

Automatic Identification of Demand patterns.

## Usage

```python
from polars_statistics import Aid

classifier = Aid(
    intermittent_threshold: float = 0.3,  # Zero proportion threshold
    ic: str = "aicc",                     # "aic", "aicc", "bic"
)

result = classifier.classify(y)  # y: 1D numpy array
```

---

## AidResult Properties

```python
result.demand_type        # str: "regular" or "intermittent"
result.distribution       # str: best-fit distribution name
result.is_fractional      # bool: whether data has fractional values
result.mean               # float: estimated mean
result.variance           # float: estimated variance
result.zero_proportion    # float: proportion of zeros
result.n_observations     # int: number of observations
result.anomalies          # list[str]: per-observation anomaly flags
result.ic_values          # dict: IC values for each candidate distribution
```

---

## Convenience Methods

```python
result.is_regular()           # bool
result.is_intermittent()      # bool
result.has_stockouts()        # bool
result.is_new_product()       # bool (leading zeros pattern)
result.is_obsolete_product()  # bool (trailing zeros pattern)
result.anomaly_counts()       # dict: counts by anomaly type
```

---

## Example

```python
import numpy as np
from polars_statistics import Aid

# Simulate intermittent demand
np.random.seed(42)
demand = np.random.poisson(2, 100)
demand[np.random.rand(100) < 0.3] = 0  # 30% zeros

# Classify
classifier = Aid(intermittent_threshold=0.3)
result = classifier.classify(demand)

print(f"Demand type: {result.demand_type}")
print(f"Distribution: {result.distribution}")
print(f"Zero proportion: {result.zero_proportion:.2%}")
print(f"Has stockouts: {result.has_stockouts()}")
print(f"Anomaly counts: {result.anomaly_counts()}")
```

---

## Supported Distributions

| Category | Distributions |
|----------|---------------|
| Count | `poisson`, `negative_binomial`, `geometric` |
| Continuous | `normal`, `gamma`, `lognormal`, `rectified_normal` |

---

## See Also

- [aid Expression](../regression/aid.md)
- [aid_anomalies Expression](../regression/aid.md#aid_anomalies)
