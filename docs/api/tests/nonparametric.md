# Non-Parametric Tests

Distribution-free statistical tests that make no assumptions about the underlying distribution. Use these when data is ordinal, heavily skewed, or violates normality assumptions.

> **Validation:** All tests are validated against R implementations in the anofox-statistics crate.

## `mann_whitney_u`

Mann-Whitney U test (also known as Wilcoxon rank-sum test) for comparing two independent samples.

Tests whether one distribution is stochastically greater than another. Based on ranks, so robust to outliers and non-normality. The non-parametric alternative to the independent samples t-test.

```python
ps.mann_whitney_u(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**Null hypothesis:** The distributions of x and y are equal (P(X > Y) = P(Y > X)).

**Assumptions:**
- Independent observations
- Ordinal or continuous data
- Similar distribution shapes (tests location shift)

**When to use:**
- Non-normal data or small samples where normality is uncertain
- Ordinal data (e.g., Likert scales)
- Data with outliers
- When you want to compare medians rather than means

**Example:**
```python
# Compare two groups
df.select(ps.mann_whitney_u("treatment", "control"))

# Per-experiment comparison
df.group_by("experiment").agg(
    ps.mann_whitney_u("treatment", "control").alias("test")
)
```

---

## `wilcoxon_signed_rank`

Wilcoxon signed-rank test for paired samples.

Tests whether the median difference between paired observations is zero. The non-parametric alternative to the paired t-test. Uses ranks of absolute differences, signed by direction.

```python
ps.wilcoxon_signed_rank(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**Null hypothesis:** The median of paired differences is zero.

**Assumptions:**
- Paired observations
- Differences are symmetric around the median
- Continuous or ordinal differences

**When to use:**
- Before/after measurements on the same subjects
- Matched pairs design
- Non-normal difference distribution
- Ordinal paired data

**Example:**
```python
# Test if treatment changed scores
df.select(ps.wilcoxon_signed_rank("before", "after"))

# Per-subject analysis
df.group_by("subject_group").agg(
    ps.wilcoxon_signed_rank("pre_test", "post_test").alias("test")
)
```

---

## `kruskal_wallis`

Kruskal-Wallis H test for comparing 3 or more independent groups.

The non-parametric alternative to one-way ANOVA. Tests whether at least one group differs from the others based on rank sums. If significant, follow up with pairwise comparisons.

```python
ps.kruskal_wallis(
    *groups: Union[pl.Expr, str],  # 3 or more groups
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**Null hypothesis:** All groups have the same distribution.

**Assumptions:**
- Independent observations within and between groups
- Ordinal or continuous data
- Similar distribution shapes across groups

**When to use:**
- Comparing 3+ groups with non-normal data
- Ordinal response variable
- Unequal group sizes with skewed data

**Note:** A significant result indicates at least one group differs, but doesn't identify which groups differ. Use pairwise Mann-Whitney tests with multiple comparison correction for post-hoc analysis.

**Example:**
```python
# Compare three treatment groups
df.select(ps.kruskal_wallis("placebo", "low_dose", "high_dose"))

# Compare across categories
df.select(ps.kruskal_wallis("region_a", "region_b", "region_c", "region_d"))
```

---

## `brunner_munzel`

Brunner-Munzel test for stochastic equality.

A generalization of the Mann-Whitney test that is valid even when variances differ between groups. Tests whether P(X > Y) = 0.5 (neither distribution tends to produce larger values).

```python
ps.brunner_munzel(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    alternative: str = "two-sided",  # "two-sided", "less", "greater"
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**Null hypothesis:** P(X > Y) + 0.5 * P(X = Y) = 0.5 (stochastic equality).

**Advantages over Mann-Whitney:**
- Valid under heteroscedasticity (unequal variances)
- More accurate p-values for small samples
- Does not require similar distribution shapes

**When to use:**
- Two independent groups with possibly different variances
- When Mann-Whitney assumptions are questionable
- Small to moderate sample sizes
- Heteroscedastic non-normal data

**Example:**
```python
# Two-sided test
df.select(ps.brunner_munzel("treatment", "control"))

# Directional test: treatment stochastically greater?
df.select(ps.brunner_munzel("treatment", "control", alternative="greater"))
```

---

## Choosing a Test

| Situation | Recommended Test |
|-----------|-----------------|
| Two independent groups | `mann_whitney_u` or `brunner_munzel` |
| Two independent groups, unequal variances | `brunner_munzel` |
| Paired/repeated measurements | `wilcoxon_signed_rank` |
| 3+ independent groups | `kruskal_wallis` |
| Normal data, equal variances | Consider [Parametric Tests](parametric.md) |

---

## Comparison with Parametric Tests

| Non-Parametric | Parametric Equivalent | When to Prefer Non-Parametric |
|---------------|----------------------|-------------------------------|
| `mann_whitney_u` | `ttest_ind` | Non-normal data, ordinal data, outliers |
| `wilcoxon_signed_rank` | `ttest_paired` | Non-normal differences, ordinal data |
| `kruskal_wallis` | One-way ANOVA | Non-normal data, ordinal response |
| `brunner_munzel` | Welch's t-test | Heteroscedastic non-normal data |

---

## See Also

- [Parametric Tests](parametric.md) - Tests with distributional assumptions
- [TOST Equivalence Tests](tost.md) - Non-parametric equivalence tests (`tost_wilcoxon_*`)
- [Distributional Tests](distributional.md) - Test normality to help choose between parametric/non-parametric
