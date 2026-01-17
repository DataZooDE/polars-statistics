# Categorical Tests

Tests for categorical data, proportions, and contingency tables. Use these for analyzing count data, survey responses, and classification outcomes.

> **Validation:** All categorical tests are validated against R implementations (binom.test, prop.test, chisq.test, fisher.test, mcnemar.test).

## Proportion Tests

### `binom_test`

Exact binomial test for a single proportion.

Tests whether the observed proportion differs from a hypothesized value using exact binomial probabilities. Preferred over normal approximation for small samples or proportions near 0 or 1.

```python
ps.binom_test(
    successes: int,
    n: int,
    p0: float = 0.5,
    alternative: str = "two-sided",  # "two-sided", "less", "greater"
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64, statistic: Float64, p_value: Float64, ci_lower: Float64, ci_upper: Float64, n: UInt32}`

**Null hypothesis:** The true proportion equals p0.

**When to use:**
- Small sample sizes (n < 30)
- Proportions near 0 or 1 (where normal approximation fails)
- When exact p-values are required

**Example:**
```python
# Test if success rate differs from 50%
ps.binom_test(successes=15, n=20, p0=0.5)

# Test if defect rate exceeds 5%
ps.binom_test(successes=8, n=100, p0=0.05, alternative="greater")
```

---

### `prop_test_one`

One-sample proportion test using normal approximation.

Tests whether an observed proportion differs from a hypothesized value using the z-test approximation. Faster than exact test but requires adequate sample size.

```python
ps.prop_test_one(
    successes: int,
    n: int,
    p0: float = 0.5,
    alternative: str = "two-sided",
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64, statistic: Float64, p_value: Float64, ci_lower: Float64, ci_upper: Float64, n: UInt32}`

**Null hypothesis:** The true proportion equals p0.

**Assumptions:**
- np ≥ 5 and n(1-p) ≥ 5 (rule of thumb for normal approximation)
- Independent observations

**When to use:**
- Larger samples where normal approximation is valid
- Quick approximate inference

---

### `prop_test_two`

Two-sample proportion test for comparing two groups.

Tests whether two independent proportions are equal using the z-test. Useful for A/B testing and comparing success rates between groups.

```python
ps.prop_test_two(
    successes1: int,
    n1: int,
    successes2: int,
    n2: int,
    alternative: str = "two-sided",
    correction: bool = False,  # Yates' continuity correction
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64, statistic: Float64, p_value: Float64, ci_lower: Float64, ci_upper: Float64, n: UInt32}`

**Null hypothesis:** The two proportions are equal (p1 = p2).

**Parameters:**
- `correction=True`: Applies Yates' continuity correction, making the test more conservative. Recommended for small samples.

**When to use:**
- A/B testing (comparing conversion rates)
- Comparing treatment vs control success rates
- Any comparison of two independent proportions

**Example:**
```python
# A/B test: 45/500 conversions vs 38/500
ps.prop_test_two(successes1=45, n1=500, successes2=38, n2=500)

---

## Contingency Table Tests

### `chisq_test`

Pearson's chi-square test for independence in contingency tables.

Tests whether two categorical variables are independent. The most common test for analyzing relationships between categorical variables.

```python
ps.chisq_test(
    data: Union[pl.Expr, str],  # Flattened contingency table (row-major)
    n_rows: int = 2,
    n_cols: int = 2,
    correction: bool = False,  # Yates' continuity correction
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64, df: Float64, n: UInt32}`

**Null hypothesis:** The row and column variables are independent (no association).

**Assumptions:**
- Expected cell counts ≥ 5 for most cells (rule of thumb)
- Independent observations
- Categories are mutually exclusive

**Parameters:**
- `correction=True`: Applies Yates' continuity correction for 2x2 tables

**Degrees of freedom:** (n_rows - 1) × (n_cols - 1)

**Example:**
```python
# 2x2 table: [[10, 20], [30, 40]] flattened to [10, 20, 30, 40]
df.select(ps.chisq_test("counts", n_rows=2, n_cols=2))

# 3x3 table analysis
df.select(ps.chisq_test("frequencies", n_rows=3, n_cols=3))
```

**When to avoid:**
- Small expected counts (< 5): use Fisher's exact test
- Ordered categories: consider ordinal tests

---

### `chisq_goodness_of_fit`

Chi-square goodness-of-fit test for categorical distributions.

Tests whether observed frequencies match expected frequencies from a theoretical distribution.

```python
ps.chisq_goodness_of_fit(
    observed: Union[pl.Expr, str],
    expected: Union[pl.Expr, str] | None = None,  # None = uniform distribution
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64, df: Float64, n: UInt32}`

**Null hypothesis:** The observed distribution matches the expected distribution.

**When to use:**
- Testing if dice are fair (uniform distribution)
- Checking if observed proportions match theoretical ratios
- Validating assumptions about category frequencies

**Example:**
```python
# Test if observations follow uniform distribution
df.select(ps.chisq_goodness_of_fit("category_counts"))

# Test against specific expected proportions
df.select(ps.chisq_goodness_of_fit("observed", "expected"))
```

---

### `g_test`

G-test (likelihood ratio test) for independence in contingency tables.

An alternative to chi-square test based on log-likelihood ratios. Asymptotically equivalent to chi-square but can be more accurate for sparse tables.

```python
ps.g_test(
    data: Union[pl.Expr, str],
    n_rows: int = 2,
    n_cols: int = 2,
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64, df: Float64, n: UInt32}`

**Null hypothesis:** The row and column variables are independent.

**Advantages over chi-square:**
- Better additivity (G-statistics can be decomposed)
- More accurate for sparse tables
- Preferred in some fields (ecology, linguistics)

**When to use:**
- Log-linear model analysis
- When additivity of test statistics matters
- Alternative to chi-square for robustness checking

---

### `fisher_exact`

Fisher's exact test for 2x2 contingency tables.

Computes exact p-values using the hypergeometric distribution. The gold standard for small samples where chi-square approximation fails.

```python
ps.fisher_exact(
    a: int, b: int, c: int, d: int,  # 2x2 table cells
    alternative: str = "two-sided",
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64 (odds ratio), p_value: Float64}`

**Null hypothesis:** The odds ratio equals 1 (no association).

**Table layout:**
```
         | Col1 | Col2 |
---------|------|------|
Row1     |  a   |  b   |
Row2     |  c   |  d   |
```

**When to use:**
- Any 2x2 table with small expected counts (< 5)
- Small total sample size
- When exact p-values are required

**Example:**
```python
# Treatment success vs failure
# Treated: 8 success, 2 failure
# Control: 3 success, 7 failure
ps.fisher_exact(a=8, b=2, c=3, d=7)
```

**Note:** Computationally intensive for large counts; use chi-square test for large samples.

---

## Paired Proportion Tests

### `mcnemar_test`

McNemar's test for paired proportions (marginal homogeneity).

Tests whether the marginal proportions in a 2x2 table are equal when observations are paired. Used for before/after studies with binary outcomes.

```python
ps.mcnemar_test(
    a: int, b: int, c: int, d: int,  # 2x2 table cells
    correction: bool = False,  # Edwards' continuity correction
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64, df: Float64, n: UInt32}`

**Null hypothesis:** The marginal proportions are equal (p1+ = p+1).

**Table layout for paired data:**
```
           | After: Yes | After: No |
-----------|------------|-----------|
Before: Yes|     a      |     b     |
Before: No |     c      |     d     |
```

**Key insight:** Only the discordant cells (b and c) matter for the test.

**When to use:**
- Before/after studies with binary outcomes on the same subjects
- Matched case-control studies
- Testing treatment effect in paired designs

**Example:**
```python
# Did training change pass/fail outcomes?
# Before training: 30 pass, 20 fail
# After training: 40 pass, 10 fail
# Concordant: 25 pass-pass, 5 fail-fail
# Discordant: 5 pass-fail, 15 fail-pass
ps.mcnemar_test(a=25, b=5, c=15, d=5)
```

---

### `mcnemar_exact`

McNemar's exact test for paired proportions using binomial distribution.

Provides exact p-values when discordant cell counts are small.

```python
ps.mcnemar_exact(
    a: int, b: int, c: int, d: int,
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**When to use:**
- Small number of discordant pairs (b + c < 25)
- When exact inference is required

---

## Agreement and Association Measures

### `cohen_kappa`

Cohen's Kappa coefficient for inter-rater agreement.

Measures agreement between two raters beyond what would be expected by chance. Essential for assessing reliability of categorical classifications.

```python
ps.cohen_kappa(
    data: Union[pl.Expr, str],  # Flattened confusion matrix
    n_categories: int = 2,
    weighted: bool = False,  # Linear weights
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64 (kappa), statistic: Float64 (se), p_value: Float64}`

**Null hypothesis:** Agreement is no better than chance (κ = 0).

**Parameters:**
- `weighted=True`: Uses linear weights for ordinal categories (penalizes disagreements by distance)

**Interpretation (Landis & Koch, 1977):**
| κ | Agreement Level |
|---|-----------------|
| < 0.00 | Poor (worse than chance) |
| 0.00-0.20 | Slight |
| 0.21-0.40 | Fair |
| 0.41-0.60 | Moderate |
| 0.61-0.80 | Substantial |
| 0.81-1.00 | Almost perfect |

**When to use:**
- Evaluating inter-rater reliability
- Assessing diagnostic test agreement
- Validating classification models (confusion matrix)

**Example:**
```python
# Two raters classifying items into 3 categories
# Confusion matrix: [[20, 5, 0], [3, 15, 2], [0, 1, 14]]
# Flattened: [20, 5, 0, 3, 15, 2, 0, 1, 14]
df.select(ps.cohen_kappa("rater_matrix", n_categories=3))
```

---

### `cramers_v`

Cramér's V coefficient for association strength in contingency tables.

A normalized measure of association between two categorical variables, ranging from 0 (no association) to 1 (perfect association). Works for tables of any size.

```python
ps.cramers_v(
    data: Union[pl.Expr, str],
    n_rows: int = 2,
    n_cols: int = 2,
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64, statistic: Float64, p_value: Float64}`

**Interpretation:**
| V | Association |
|---|-------------|
| 0.0-0.1 | Negligible |
| 0.1-0.2 | Weak |
| 0.2-0.4 | Moderate |
| 0.4-0.6 | Strong |
| 0.6-1.0 | Very strong |

**When to use:**
- Comparing association strength across tables of different sizes
- Effect size measure for chi-square tests
- Any nominal × nominal variable relationship

**Note:** Unlike phi, Cramér's V is normalized to [0, 1] regardless of table dimensions.

---

### `phi_coefficient`

Phi coefficient (φ) for 2x2 contingency tables.

The correlation coefficient for two binary variables. Equivalent to Pearson correlation computed on 0/1 coded variables.

```python
ps.phi_coefficient(
    a: int, b: int, c: int, d: int,
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64, statistic: Float64, p_value: Float64}`

**Range:** -1 to +1 (can be negative unlike Cramér's V)
- φ = +1: Perfect positive association
- φ = 0: No association
- φ = -1: Perfect negative association

**When to use:**
- 2x2 tables only
- When direction of association matters
- Effect size for 2x2 chi-square tests

**Relationship to chi-square:** φ² = χ²/n

---

### `contingency_coef`

Pearson's contingency coefficient (C) for association in contingency tables.

An older measure of association based on chi-square. Limited in that its maximum value depends on table dimensions.

```python
ps.contingency_coef(
    data: Union[pl.Expr, str],
    n_rows: int = 2,
    n_cols: int = 2,
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64, statistic: Float64, p_value: Float64}`

**Range:** 0 to √((k-1)/k) where k = min(n_rows, n_cols)

**Note:** Cramér's V is generally preferred as it can reach 1.0 regardless of table size.

---

## Choosing a Test

| Situation | Recommended Test |
|-----------|-----------------|
| Single proportion, small sample | `binom_test` |
| Single proportion, large sample | `prop_test_one` |
| Two independent proportions | `prop_test_two` |
| 2x2 table, adequate expected counts | `chisq_test` |
| 2x2 table, small counts | `fisher_exact` |
| Larger contingency table | `chisq_test` or `g_test` |
| Goodness-of-fit | `chisq_goodness_of_fit` |
| Paired binary outcomes | `mcnemar_test` |
| Inter-rater agreement | `cohen_kappa` |
| Association effect size | `cramers_v` |

---

## See Also

- [TOST Equivalence Tests](tost.md) - Equivalence tests for proportions
- [Correlation Tests](correlation.md) - Tests for continuous association
