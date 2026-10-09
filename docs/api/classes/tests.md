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

### Non-finite input handling (anofox-statistics 0.4.4+)

All test classes raise `ValueError` ("non-finite value") on `NaN` input — they
never silently drop it like the matching expression builders do. `+/-Inf` input is
**not** rejected: rank-based classes (`MannWhitneyU`, `WilcoxonSignedRank`,
`KruskalWallis`, `BrunnerMunzel`) rank `+/-Inf` the same way R does. Constant input
(zero variance) yields a `NaN` `p_value` rather than raising or dividing by zero.
This differs from the Polars expression builders (`ps.mann_whitney_u`, etc.), which
filter out non-finite values with `is_finite()` before computing and never raise.

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

test = MannWhitneyU(
    alternative: str = "two-sided",   # "two-sided", "less", "greater"
    continuity_correction: bool = True,
    exact: bool = False,
    conf_level: float | None = None,  # Hodges-Lehmann CI level, e.g. 0.95
    mu: float | None = None,          # hypothesized location shift
)
test.fit(x, y)

test.estimate   # float | None — Hodges-Lehmann shift estimate
test.conf_int   # np.ndarray | None — [lower, upper], shape (2,)
```

**Hodges-Lehmann `estimate`/`conf_int` (anofox-statistics 0.4.5):** validated on
126 R `wilcox.test(conf.int = TRUE)` reference cases. The asymptotic interval is
root-found with the continuity correction applied; one-sided `alternative` values
give an interval with an infinite bound; `exact=True` is honoured only for
`n1 * n2 <= 10,000` (above that, the normal approximation is used silently); an
infinite observation makes `estimate`/`conf_int` `NaN`. Computing the CI on 20,000
observations per group completes in O(n) memory (previously O(n^2), ~3.2 GB).

**Since 0.8.0, `ps.mann_whitney_u` returns the same values** as trailing
`estimate`/`ci_lower`/`ci_upper` struct fields (additive; `statistic`/`p_value`
unchanged) — see [`mann_whitney_u`](../tests/nonparametric.md#mann_whitney_u) and
[Mann-Whitney / Wilcoxon Output](../outputs.md#mann-whitney-wilcoxon-output).

### WilcoxonSignedRank

Wilcoxon signed-rank test for paired samples.

```python
from polars_statistics import WilcoxonSignedRank

test = WilcoxonSignedRank(
    alternative: str = "two-sided",
    continuity_correction: bool = True,
    exact: bool = False,
    conf_level: float | None = None,
    mu: float | None = None,
)
test.fit(before, after)

test.estimate   # float | None — Hodges-Lehmann pseudomedian estimate
test.conf_int   # np.ndarray | None — [lower, upper], shape (2,)
```

**Hodges-Lehmann `estimate`/`conf_int` (anofox-statistics 0.4.5):** same R
`wilcox.test(conf.int = TRUE)` parity as `MannWhitneyU` above. The interval is
computed on `x - y` (not shifted by `mu`); `exact=True` is honoured only for
`<= 300` non-zero differences. O(n log n) time, O(n) memory — 100,000 observations
per group is fine.

**Since 0.8.0, `ps.wilcoxon_signed_rank` returns the same values** as trailing
`estimate`/`ci_lower`/`ci_upper` struct fields (additive; `statistic`/`p_value`
unchanged) — see
[`wilcoxon_signed_rank`](../tests/nonparametric.md#wilcoxon_signed_rank) and
[Mann-Whitney / Wilcoxon Output](../outputs.md#mann-whitney-wilcoxon-output).

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
