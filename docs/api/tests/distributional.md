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

## Choosing a Normality Test

| Situation | Recommended Test |
|-----------|-----------------|
| Small samples (n < 50) | `shapiro_wilk` |
| Medium samples (50 ≤ n < 300) | Either test |
| Large samples (n ≥ 300) | `dagostino` (or rely on CLT) |
| Need to identify skewness/kurtosis | `dagostino` |
| Most powerful test | `shapiro_wilk` |

**Practical advice:**
- With large samples, minor non-normality is often detected but may not matter practically
- The Central Limit Theorem provides robustness for means with n > 30
- Visual inspection (Q-Q plots) often more informative than p-values
- Consider the consequences: t-tests and ANOVA are fairly robust to mild non-normality

---

## See Also

- [Parametric Tests](parametric.md) - Tests that assume normality
- [Non-Parametric Tests](nonparametric.md) - Distribution-free alternatives when normality fails
