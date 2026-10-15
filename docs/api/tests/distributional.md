# Distributional Tests

Tests for checking distributional properties of data, particularly normality. Use these tests to verify assumptions before applying parametric methods.

> **Validation:** All distributional tests are validated against R implementations (shapiro.test, moments package).

## `shapiro_wilk`

Shapiro-Wilk test for normality.

The most powerful test for detecting departures from normality in small to medium samples (n < 5000). Based on the correlation between the data and the corresponding normal quantiles.

```python
ps.shapiro_wilk(
    x: Union[pl.Expr, str],
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**Null hypothesis:** The data comes from a normally distributed population.

**Statistic (W):** Ranges from 0 to 1. Values close to 1 indicate normality; values significantly less than 1 indicate non-normality.

**Sample size:** Best for n = 3 to 5000. For larger samples, even trivial departures from normality become significant.

**When to use:**
- Checking normality assumptions before t-tests, ANOVA, or linear regression
- Small to medium samples where power matters
- Exploratory data analysis

**Limitations:**
- Highly sensitive with large samples (may reject normality for practically normal data)
- Does not identify *how* the data deviates from normality (skewness vs kurtosis)

**Example:**
```python
# Check if residuals are normal
df.select(ps.shapiro_wilk("residuals").alias("normality_test"))

# Per-group normality check
df.group_by("treatment").agg(
    ps.shapiro_wilk("response").alias("normality")
)
```

**Interpretation:**
| p-value | Conclusion |
|---------|------------|
| > 0.05 | Cannot reject normality; data is consistent with normal distribution |
| ≤ 0.05 | Significant deviation from normality; consider non-parametric methods |

**Note:** Always combine with visual inspection (Q-Q plots, histograms) as p-values alone don't tell the full story.

!!! info "Behaviour change in 0.8.0"
    For `4 <= n <= 11`, p-values now follow R's `swilk.c` reference implementation
    exactly (anofox-statistics 0.4.3 #7) instead of an approximation. For example,
    on `x = [4.1, 5.3, 3.8, 6.9, 5.0, 4.4, 9.2, 5.7, 4.9, 6.1]` (n=10),
    `shapiro_wilk` now reports `p_value = 0.1606` instead of the previous `0.070`
    (statistic `W = 0.8879`, unchanged).

---

## `dagostino`

D'Agostino-Pearson omnibus test for normality.

Tests whether a sample has the skewness and kurtosis matching a normal distribution. Combines separate tests for skewness and kurtosis into an overall test of normality.

```python
ps.dagostino(
    x: Union[pl.Expr, str],
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**Null hypothesis:** The data has skewness and kurtosis consistent with a normal distribution.

**Statistic (K²):** Chi-square distributed with 2 degrees of freedom under the null hypothesis.

**Sample size:** Requires at least 20 observations. Works well for larger samples where Shapiro-Wilk becomes overly sensitive.

**What it detects:**
- Skewness: Asymmetry in the distribution (left or right tails)
- Kurtosis: Heavy or light tails compared to normal (leptokurtic vs platykurtic)

**When to use:**
- Larger samples (n > 50) where Shapiro-Wilk may be too sensitive
- When you want to detect departures from both symmetry and tail behavior
- Automated normality checking in pipelines

**Example:**
```python
# Test normality for larger dataset
df.select(ps.dagostino("measurements"))

# Compare normality across conditions
df.group_by("condition").agg(
    ps.dagostino("outcome").alias("normality")
)
```

---

## `jarque_bera`

Jarque-Bera omnibus test for normality.

An asymptotic chi-squared(2) test on sample skewness and excess kurtosis (population
moments), matching R `tseries::jarque.bera.test` (also `moments::jarque.test`).

```python
ps.jarque_bera(
    x: Union[pl.Expr, str],
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64, skewness: Float64, kurtosis: Float64, n: UInt32}`

**Null hypothesis:** The data has zero skewness and zero excess kurtosis (consistent with normality).

**Statistic (JB):** Chi-square distributed with 2 degrees of freedom under the null hypothesis. `JB = n/6 * (skewness^2 + kurtosis^2/4)`.

**Requirements:** Requires at least 3 finite, non-constant values. Fewer than 3 values, a non-finite input, or constant data yield NaN `statistic`/`p_value`/`skewness`/`kurtosis` and `n == 0` (never an exception).

**When to use:**
- Similar use case to `dagostino` — larger samples, and when skewness/kurtosis themselves are of interest (both are returned as separate fields)
- Automated normality checking in pipelines (R/`tseries` interoperability)

**Example:**
```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "values": [2.1, 3.4, 1.9, 5.6, 4.4, 3.3, 2.8, 9.7, 3.0, 4.1],
})

print(df.select(ps.jarque_bera("values")))
```

**Validated against R:** `tseries::jarque.bera.test` gives `statistic = 6.1833442919626558`,
`p_value = 0.045425932072353667`, `skewness = 1.6519836946632753`,
`kurtosis = 1.980864909825267` (excess kurtosis) for the sample above.

---

## Choosing a Normality Test

| Situation | Recommended Test |
|-----------|-----------------|
| Small samples (n < 50) | `shapiro_wilk` |
| Medium samples (50 ≤ n < 300) | Either test |
| Large samples (n ≥ 300) | `dagostino` (or rely on CLT) |
| Need to identify skewness/kurtosis | `dagostino` or `jarque_bera` (returns both as separate fields) |
| Most powerful test | `shapiro_wilk` |
| R `tseries`/`moments` interoperability | `jarque_bera` |

**Practical advice:**
- With large samples, minor non-normality is often detected but may not matter practically
- The Central Limit Theorem provides robustness for means with n > 30
- Visual inspection (Q-Q plots) often more informative than p-values
- Consider the consequences: t-tests and ANOVA are fairly robust to mild non-normality

---

## See Also

- [Parametric Tests](parametric.md) - Tests that assume normality
- [Non-Parametric Tests](nonparametric.md) - Distribution-free alternatives when normality fails
