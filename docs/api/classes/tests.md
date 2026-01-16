# Test Model Classes

Statistical tests available as model classes with `.fit()`, `.statistic`, `.p_value`, and `.summary()` methods.

## Common Interface

All test classes share these properties and methods:

```python
test.is_fitted()    # bool: Check if test has been performed
test.statistic      # float: Test statistic value
test.p_value        # float: P-value
test.summary()      # str: Formatted summary of results
```

---

## Parametric Test Classes

### TTestInd

Independent samples t-test.

```python
from polars_statistics import TTestInd

test = TTestInd(
    alternative: str = "two-sided",  # "two-sided", "less", "greater"
    equal_var: bool = False,         # False = Welch's, True = Student's
)
test.fit(x, y)
```

### TTestPaired

Paired samples t-test.

```python
from polars_statistics import TTestPaired

test = TTestPaired(
    alternative: str = "two-sided",
)
test.fit(before, after)
```

### BrownForsythe

Brown-Forsythe test for variance equality.

```python
from polars_statistics import BrownForsythe

test = BrownForsythe()
test.fit(x, y)
```

### YuenTest

Yuen's trimmed mean test.

```python
from polars_statistics import YuenTest

test = YuenTest(
    trim: float = 0.2,  # Proportion to trim
)
test.fit(x, y)
```

---

## Non-Parametric Test Classes

### MannWhitneyU

Mann-Whitney U test (Wilcoxon rank-sum).

```python
from polars_statistics import MannWhitneyU

test = MannWhitneyU()
test.fit(x, y)
```

### WilcoxonSignedRank

Wilcoxon signed-rank test for paired samples.

```python
from polars_statistics import WilcoxonSignedRank

test = WilcoxonSignedRank()
test.fit(before, after)
```

### KruskalWallis

Kruskal-Wallis H test for multiple groups.

```python
from polars_statistics import KruskalWallis

test = KruskalWallis()
test.fit(g1, g2, g3)  # 3+ groups
```

### BrunnerMunzel

Brunner-Munzel test for stochastic equality.

```python
from polars_statistics import BrunnerMunzel

test = BrunnerMunzel(
    alternative: str = "two-sided",
)
test.fit(x, y)
```

---

## Distributional Test Classes

### ShapiroWilk

Shapiro-Wilk normality test.

```python
from polars_statistics import ShapiroWilk

test = ShapiroWilk()
test.fit(x)
```

### DAgostino

D'Agostino-Pearson normality test.

```python
from polars_statistics import DAgostino

test = DAgostino()
test.fit(x)
```

---

## Class Summary

| Class | Parameters | Input |
|-------|------------|-------|
| `TTestInd` | `alternative`, `equal_var` | `fit(x, y)` |
| `TTestPaired` | `alternative` | `fit(x, y)` |
| `BrownForsythe` | - | `fit(x, y)` |
| `YuenTest` | `trim` | `fit(x, y)` |
| `MannWhitneyU` | - | `fit(x, y)` |
| `WilcoxonSignedRank` | - | `fit(x, y)` |
| `KruskalWallis` | - | `fit(*groups)` |
| `BrunnerMunzel` | `alternative` | `fit(x, y)` |
| `ShapiroWilk` | - | `fit(x)` |
| `DAgostino` | - | `fit(x)` |

---

## Example Summary Output

```python
from polars_statistics import TTestInd
import numpy as np

x = np.random.randn(50)
y = np.random.randn(50) + 0.5

test = TTestInd(alternative="two-sided", equal_var=False)
test.fit(x, y)
print(test.summary())
```

Output:
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

---

## See Also

- [Parametric Tests](../tests/parametric.md) - Expression API
- [Non-Parametric Tests](../tests/nonparametric.md) - Expression API
- [Distributional Tests](../tests/distributional.md) - Expression API
