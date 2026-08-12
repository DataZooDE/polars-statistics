# Parametric Tests

Parametric statistical tests that assume specific distributional properties of the data. These tests are generally more powerful than non-parametric alternatives when their assumptions are met.

> **Validation:** All tests are validated against R implementations in the anofox-statistics crate.

## `one_way_anova`

One-way ANOVA for comparing means across two or more independent groups.

Supports two variants: **Fisher's ANOVA** (classic F-test, assumes equal variances) and
**Welch's ANOVA** (does not assume equal variances). Use `df.select(ps.one_way_anova(...))` on
the full DataFrame — pass each group as its own column, not as a long-format column.

```python
ps.one_way_anova(
    *groups: Union[pl.Expr, str],   # Two or more column expressions or names
    kind: str = "fisher",           # "fisher" (default) or "welch"
) -> pl.Expr
```

**Returns:**
`Struct{statistic: Float64, df_between: Float64, df_within: Float64, p_value: Float64,
ss_between: Float64, ss_within: Float64, ms_between: Float64, ms_within: Float64,
eta_squared: Float64, n_groups: UInt32}`

> **Note:** When `kind="welch"`, the fields `ss_between`, `ss_within`, `ms_between`,
> `ms_within`, and `eta_squared` are `NaN` — Welch's method does not decompose sums of squares.

**Assumptions:**
- Independent observations within and between groups
- For Fisher variant: equal variances (homoscedasticity)
- Data approximately normally distributed per group

**When to use:**
- Comparing means across three or more groups in one step
- Use `kind="welch"` when group variances differ (heteroscedasticity)
- Check variance equality first with [`brown_forsythe`](#brown_forsythe)

**Example:**
```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "control":     [1.0, 2.0, 3.0],
    "treatment_a": [4.0, 5.0, 6.0],
    "treatment_b": [7.0, 8.0, 9.0],
})

# Fisher ANOVA (assumes equal variances)
df.select(ps.one_way_anova("control", "treatment_a", "treatment_b"))

# Welch ANOVA (does not assume equal variances)
df.select(ps.one_way_anova("control", "treatment_a", "treatment_b", kind="welch"))
```

---

## `ttest_ind`

Independent samples t-test for comparing means of two groups.

**Welch's t-test** (default, `equal_var=False`) does not assume equal variances and is generally recommended. **Student's t-test** (`equal_var=True`) assumes equal variances and has slightly more power when the assumption holds.

```python
ps.ttest_ind(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    alternative: str = "two-sided",  # "two-sided", "less", "greater"
    equal_var: bool = False,         # False = Welch's, True = Student's
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**Assumptions:**
- Independent observations within and between groups
- Data approximately normally distributed (robust to violations with n > 30)
- For Student's t: equal variances (homoscedasticity)

**Interpretation:**
- `p_value < alpha`: Reject null hypothesis of equal means
- `alternative="less"`: Test if mean(x) < mean(y)
- `alternative="greater"`: Test if mean(x) > mean(y)

**Example:**
```python
# Compare treatment vs control group means
df.select(ps.ttest_ind("treatment", "control", alternative="two-sided"))

# Per-group comparison
df.group_by("experiment").agg(
    ps.ttest_ind("treatment", "control").alias("test")
)
```

---

## `ttest_paired`

Paired samples t-test for comparing means of two related measurements.

Tests whether the mean difference between paired observations differs from zero. More powerful than independent t-test when observations are naturally paired (e.g., before/after measurements on the same subjects).

```python
ps.ttest_paired(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    alternative: str = "two-sided",  # "two-sided", "less", "greater"
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**Assumptions:**
- Paired observations (same subjects measured twice)
- Differences approximately normally distributed
- Independence between pairs

**When to use:**
- Before/after measurements on the same subjects
- Matched pairs design
- Repeated measures with two time points

**Example:**
```python
# Test if treatment changed scores
df.select(ps.ttest_paired("before", "after"))

# Directional test: did scores increase?
df.select(ps.ttest_paired("before", "after", alternative="less"))
```

---

## `brown_forsythe`

Brown-Forsythe test for equality of variances (homoscedasticity).

A robust alternative to Levene's test that uses deviations from group medians instead of means. Recommended for checking variance homogeneity before using tests that assume equal variances.

```python
ps.brown_forsythe(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**Interpretation:**
- `p_value > 0.05`: Variances can be considered equal
- `p_value < 0.05`: Evidence of unequal variances (heteroscedasticity)

**When to use:**
- Before ANOVA or Student's t-test to verify equal variance assumption
- When comparing spread of two distributions
- More robust than Levene's test for non-normal data

**Example:**
```python
# Check if variances are equal before using equal_var=True
variance_test = df.select(ps.brown_forsythe("group1", "group2"))
```

---

## `yuen_test`

Yuen's test for trimmed means—a robust alternative to the independent t-test.

Compares trimmed means of two groups, reducing the influence of outliers. The test trims a proportion of observations from each tail before computing means and standard errors.

```python
ps.yuen_test(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    trim: float = 0.2,  # Proportion to trim from each tail
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**Parameters:**
- `trim=0.2` (default): Removes 20% from each tail, using the middle 60% of data
- `trim=0.1`: Removes 10% from each tail, using the middle 80%
- `trim=0.0`: Equivalent to Student's t-test

**When to use:**
- Data contains outliers or heavy tails
- Distribution is skewed
- You want robust inference without removing outliers manually

**Note:** The trimmed mean with 20% trimming is approximately 95% as efficient as the mean for normal data, but much more robust to contamination.

**Example:**
```python
# Robust comparison with default 20% trimming
df.select(ps.yuen_test("treatment", "control"))

# Less trimming for cleaner data
df.select(ps.yuen_test("treatment", "control", trim=0.1))
```

---

## Choosing a Test

| Situation | Recommended Test |
|-----------|-----------------|
| Two independent groups, normal data | `ttest_ind` (Welch's) |
| Two independent groups, outliers present | `yuen_test` |
| Paired/repeated measurements | `ttest_paired` |
| Check variance equality | `brown_forsythe` |
| Non-normal data | Consider [Non-Parametric Tests](nonparametric.md) |

---

## See Also

- [Non-Parametric Tests](nonparametric.md) - Distribution-free alternatives (Mann-Whitney, Wilcoxon)
- [TOST Equivalence Tests](tost.md) - Test for practical equivalence rather than difference
- [Distributional Tests](distributional.md) - Test normality assumptions
