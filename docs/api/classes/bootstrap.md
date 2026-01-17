# Bootstrap Classes

Bootstrap resampling methods for time series and dependent data.

## StationaryBootstrap

Stationary bootstrap with random block lengths (Politis & Romano, 1994).

```python
from polars_statistics import StationaryBootstrap

bootstrap = StationaryBootstrap(
    expected_block_length: float = 5.0,
    seed: int | None = None,
)

samples = bootstrap.samples(data, n_samples=1000)
```

Block lengths are drawn from a geometric distribution with mean `expected_block_length`.

---

## CircularBlockBootstrap

Circular block bootstrap with fixed block length.

```python
from polars_statistics import CircularBlockBootstrap

bootstrap = CircularBlockBootstrap(
    block_length: int = 10,
    seed: int | None = None,
)

samples = bootstrap.samples(data, n_samples=1000)
```

---

## Example

```python
import numpy as np
from polars_statistics import StationaryBootstrap

# Time series data
np.random.seed(42)
data = np.cumsum(np.random.randn(100))  # Random walk

# Generate bootstrap samples
bootstrap = StationaryBootstrap(expected_block_length=5.0, seed=42)
samples = bootstrap.samples(data, n_samples=1000)

# Compute bootstrap confidence interval for mean
means = [s.mean() for s in samples]
ci_lower = np.percentile(means, 2.5)
ci_upper = np.percentile(means, 97.5)
print(f"95% CI for mean: [{ci_lower:.2f}, {ci_upper:.2f}]")
```

---

## Choosing Block Length

| Data Characteristics | Recommendation |
|---------------------|----------------|
| Low autocorrelation | Shorter blocks (3-5) |
| High autocorrelation | Longer blocks (10-20) |
| Seasonal patterns | Block length >= season length |

---

## See Also

- [TOST Bootstrap](../tests/tost.md#tost_bootstrap) - Bootstrap equivalence test
- [Forecast Tests](../tests/forecast.md) - Tests using block bootstrap
