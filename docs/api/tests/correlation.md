# Correlation Tests

Tests for measuring and testing linear, monotonic, and general associations between variables.

> **Validation:** All correlation tests are validated against R implementations (cor.test, ppcor, energy packages).

## `pearson`

Pearson product-moment correlation coefficient with hypothesis test.

Measures the strength and direction of the linear relationship between two continuous variables. Ranges from -1 (perfect negative) to +1 (perfect positive), with 0 indicating no linear relationship.

```python
ps.pearson(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    conf_level: float = 0.95,
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64, statistic: Float64, p_value: Float64, ci_lower: Float64, ci_upper: Float64, n: UInt32}`

**Null hypothesis:** True correlation is zero (no linear association).

**Assumptions:**
- Continuous variables
- Linear relationship
- Bivariate normal distribution (for inference)
- No extreme outliers

**Interpretation:**
| |r| | Strength |
|-----|----------|
| 0.0-0.1 | Negligible |
| 0.1-0.3 | Weak |
| 0.3-0.5 | Moderate |
| 0.5-0.7 | Strong |
| 0.7-1.0 | Very strong |

**Example:**
```python
# Basic correlation
df.select(ps.pearson("x", "y").alias("cor"))

# Per-group correlation
df.group_by("category").agg(
    ps.pearson("sales", "advertising").alias("correlation")
)
```

---

## `spearman`

Spearman rank correlation coefficient with hypothesis test.

Measures the strength and direction of the monotonic relationship between two variables using ranks. More robust than Pearson to outliers and non-linear monotonic relationships.

```python
ps.spearman(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    conf_level: float = 0.95,
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64, statistic: Float64, p_value: Float64, ci_lower: Float64, ci_upper: Float64, n: UInt32}`

**Null hypothesis:** No monotonic association between variables.

**When to use:**
- Ordinal data
- Non-linear but monotonic relationships
- Data with outliers
- When normality assumption is violated

**Example:**
```python
# Rank correlation
df.select(ps.spearman("satisfaction_rating", "purchase_frequency"))
```

---

## `kendall`

Kendall's tau correlation coefficient with hypothesis test.

A non-parametric measure of association based on concordant and discordant pairs. More robust than Spearman for small samples and handles ties well.

```python
ps.kendall(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    variant: str = "b",  # "a", "b", or "c"
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64, statistic: Float64, p_value: Float64, ci_lower: Float64, ci_upper: Float64, n: UInt32}`

**Variants:**
- `"a"` (tau-a): Does not adjust for ties; use only when no ties exist
- `"b"` (tau-b): Adjusts for ties; most commonly used (default)
- `"c"` (tau-c): Stuart's tau-c for rectangular contingency tables

**When to use:**
- Small sample sizes
- Many tied values
- Ordinal data
- When you need a more interpretable measure (proportion of concordant vs discordant pairs)

**Example:**
```python
# Kendall's tau-b (default)
df.select(ps.kendall("rank_a", "rank_b"))

# For data with no ties
df.select(ps.kendall("x", "y", variant="a"))
```

---

## `distance_cor`

Distance correlation with permutation test.

Measures both linear and nonlinear associations between variables. Unlike Pearson and Spearman, distance correlation equals zero if and only if the variables are statistically independent.

```python
ps.distance_cor(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    n_permutations: int = 999,
    seed: int | None = None,
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64, statistic: Float64, p_value: Float64, ci_lower: Float64, ci_upper: Float64, n: UInt32}`

**Key property:** Distance correlation = 0 ⟺ X and Y are independent.

This is a fundamental advantage over Pearson/Spearman, which can be zero even when variables are dependent (e.g., quadratic relationships).

**When to use:**
- Detecting any type of association (linear, nonlinear, complex)
- When you suspect nonlinear relationships
- For exploratory analysis when relationship form is unknown

**Note:** Computationally more expensive than Pearson/Spearman. The permutation test provides the p-value.

**Example:**
```python
# Detect any association
df.select(ps.distance_cor("x", "y", n_permutations=999))

# Reproducible result
df.select(ps.distance_cor("x", "y", seed=42))
```

---

## `partial_cor`

Partial correlation controlling for one or more covariates.

Measures the association between two variables after removing the linear effects of other variables. Useful for understanding direct relationships while controlling for confounders.

```python
ps.partial_cor(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    covariates: list[Union[pl.Expr, str]],
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64, statistic: Float64, p_value: Float64, ci_lower: Float64, ci_upper: Float64, n: UInt32}`

**Interpretation:** The correlation between X and Y that remains after accounting for the linear influence of the covariates on both X and Y.

**When to use:**
- Controlling for confounding variables
- Understanding direct vs indirect relationships
- Causal inference (with appropriate design)

**Example:**
```python
# Correlation between x and y, controlling for z1 and z2
df.select(ps.partial_cor("x", "y", ["z1", "z2"]))

# Control for age when examining income-happiness relationship
df.select(ps.partial_cor("income", "happiness", ["age", "education"]))
```

---

## `semi_partial_cor`

Semi-partial (part) correlation.

Controls for covariates in Y only, while leaving X uncontrolled. Measures the unique contribution of X to Y after other predictors have been accounted for.

```python
ps.semi_partial_cor(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    covariates: list[Union[pl.Expr, str]],
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64, statistic: Float64, p_value: Float64, ci_lower: Float64, ci_upper: Float64, n: UInt32}`

**Difference from partial correlation:**
- **Partial**: Controls for covariates in both X and Y
- **Semi-partial**: Controls for covariates in Y only

**When to use:**
- Assessing unique contribution of a predictor in regression
- The squared semi-partial correlation equals the increase in R² when adding X to a model with the covariates

---

## `icc`

Intraclass Correlation Coefficient (ICC) for reliability and agreement.

Measures the consistency or agreement of measurements made by different raters or at different times. Essential for assessing inter-rater reliability and measurement consistency.

```python
ps.icc(
    values: Union[pl.Expr, str],
    icc_type: str = "icc1",  # "icc1", "icc2", "icc3", "icc2k", "icc3k"
    conf_level: float = 0.95,
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64, statistic: Float64, p_value: Float64, ci_lower: Float64, ci_upper: Float64, n: UInt32}`

**ICC Types:**

| Type | Model | Definition | Use Case |
|------|-------|------------|----------|
| `icc1` | One-way random | Absolute agreement | Different raters for each subject |
| `icc2` | Two-way random | Absolute agreement | Same raters, raters are random sample |
| `icc3` | Two-way mixed | Consistency | Same raters, raters are fixed |
| `icc2k` | Two-way random | Mean of k raters | Reliability of average ratings |
| `icc3k` | Two-way mixed | Mean of k raters | Reliability of average ratings |

**Interpretation:**
| ICC | Reliability |
|-----|-------------|
| < 0.50 | Poor |
| 0.50-0.75 | Moderate |
| 0.75-0.90 | Good |
| > 0.90 | Excellent |

---

## Choosing a Correlation Measure

| Situation | Recommended |
|-----------|-------------|
| Linear relationship, continuous data | `pearson` |
| Monotonic relationship, ordinal data | `spearman` |
| Small samples, many ties | `kendall` |
| Unknown relationship type | `distance_cor` |
| Control for confounders | `partial_cor` |
| Unique contribution in regression | `semi_partial_cor` |
| Inter-rater reliability | `icc` |

---

## See Also

- [TOST Equivalence Tests](tost.md) - `tost_correlation` for testing if correlation is practically zero
- [Parametric Tests](parametric.md) - Tests for comparing means
