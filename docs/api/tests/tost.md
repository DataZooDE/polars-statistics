# TOST Equivalence Tests

Two One-Sided Tests (TOST) for testing practical equivalence. Unlike traditional hypothesis tests that test for *difference*, TOST tests whether effects are small enough to be considered *equivalent*.

> **Validation:** All TOST tests are validated against R's TOSTER package and equivalence package implementations.

## Understanding TOST

**The problem with traditional tests:** A non-significant result (p > 0.05) does NOT prove equivalence—it only means we failed to detect a difference. Absence of evidence is not evidence of absence.

**TOST solution:** Define an equivalence region [-δ, +δ] representing "practically equivalent." If we can reject both:
- H₀₁: effect ≤ -δ (effect is too negative)
- H₀₂: effect ≥ +δ (effect is too positive)

Then we conclude the effect lies within [-δ, +δ], establishing equivalence.

**Key insight:** TOST has the burden of proof reversed—the null hypothesis is *non-equivalence*, so we need evidence to claim equivalence.

## Common Parameters

| Parameter | Description |
|-----------|-------------|
| `bounds_type` | `"symmetric"` (±delta), `"raw"` (lower/upper), or `"cohen_d"` |
| `delta` | Equivalence margin for symmetric/cohen_d bounds |
| `lower`, `upper` | Explicit bounds for raw bounds_type |
| `alpha` | Significance level (default 0.05) |

**Choosing bounds:**
- `"symmetric"`: Use when the effect should be within ±delta of zero (or reference)
- `"raw"`: Use when bounds are asymmetric (e.g., -0.3 to +0.5)
- `"cohen_d"`: Use when delta represents a standardized effect size (Cohen's d)

## Return Structure

All TOST tests return:
```
Struct{
    estimate: Float64,      # Point estimate
    ci_lower: Float64,      # CI lower bound
    ci_upper: Float64,      # CI upper bound
    bound_lower: Float64,   # Equivalence lower bound
    bound_upper: Float64,   # Equivalence upper bound
    tost_p_value: Float64,  # TOST p-value (max of two one-sided)
    equivalent: Boolean,    # True if equivalence established
    alpha: Float64,
    n: UInt32,
}
```

**Decision rule:** If `tost_p_value < alpha`, equivalence is established (`equivalent = True`).

---

## T-Test Based TOST

### `tost_t_test_one_sample`

One-sample TOST equivalence test for comparing a mean to a reference value.

Tests whether the population mean is equivalent to a reference value μ (within ±delta).

```python
ps.tost_t_test_one_sample(
    x: Union[pl.Expr, str],
    mu: float = 0.0,
    bounds_type: str = "symmetric",
    delta: float = 0.5,
    lower: float = -0.5,
    upper: float = 0.5,
    alpha: float = 0.05,
) -> pl.Expr
```

**Null hypothesis:** The mean differs from μ by more than delta (|mean - μ| > delta).

**When to use:**
- Testing if a batch mean is equivalent to a target value
- Validating that a process is operating within acceptable limits
- Quality control applications

**Example:**
```python
# Test if mean is within ±0.5 of target (10.0)
df.select(ps.tost_t_test_one_sample("measurements", mu=10.0, delta=0.5))
```

---

### `tost_t_test_two_sample`

Two-sample TOST equivalence test for comparing two independent groups.

Tests whether the difference between two group means is practically zero (within ±delta).

```python
ps.tost_t_test_two_sample(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    bounds_type: str = "symmetric",
    delta: float = 0.5,
    lower: float = -0.5,
    upper: float = 0.5,
    alpha: float = 0.05,
    pooled: bool = False,
) -> pl.Expr
```

**Null hypothesis:** The difference between means exceeds the equivalence bounds.

**Parameters:**
- `pooled=False` (default): Welch's approach, doesn't assume equal variances
- `pooled=True`: Student's approach, assumes equal variances

**When to use:**
- Demonstrating bioequivalence (generic vs brand drug)
- Showing two methods produce equivalent results
- Validating that a change didn't meaningfully affect outcomes

**Example:**
```python
# Test if treatment and control differ by less than 0.5 units
df.select(ps.tost_t_test_two_sample("treatment", "control", delta=0.5))

# Using Cohen's d for standardized equivalence bounds
df.select(ps.tost_t_test_two_sample("treatment", "control",
                                     bounds_type="cohen_d", delta=0.3))
```

---

### `tost_t_test_paired`

Paired-samples TOST equivalence test for repeated measures.

Tests whether the mean difference between paired observations is practically zero.

```python
ps.tost_t_test_paired(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    bounds_type: str = "symmetric",
    delta: float = 0.5,
    lower: float = -0.5,
    upper: float = 0.5,
    alpha: float = 0.05,
) -> pl.Expr
```

**Null hypothesis:** The mean paired difference exceeds the equivalence bounds.

**When to use:**
- Before/after studies showing no meaningful change
- Method comparison on the same samples
- Test-retest reliability assessment

**Example:**
```python
# Test if method A and B give equivalent measurements
df.select(ps.tost_t_test_paired("method_a", "method_b", delta=0.25))

---

## Correlation TOST

### `tost_correlation`

Correlation TOST equivalence test using Fisher's z-transformation.

Tests whether the correlation between two variables is practically equivalent to a reference value (typically zero). Useful for demonstrating negligible association.

```python
ps.tost_correlation(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    method: str = "pearson",  # "pearson" or "spearman"
    rho_null: float = 0.0,
    bounds_type: str = "symmetric",
    delta: float = 0.3,
    lower: float = -0.3,
    upper: float = 0.3,
    alpha: float = 0.05,
) -> pl.Expr
```

**Null hypothesis:** The correlation differs from rho_null by more than delta.

**Parameters:**
- `method`: "pearson" for linear correlation, "spearman" for monotonic
- `rho_null`: Reference correlation to test against (default 0)
- `delta=0.3`: Common choice based on |r| < 0.3 being "weak" correlation

**When to use:**
- Demonstrating independence between variables
- Showing confounders have negligible association
- Validating discriminant validity (measures should not correlate)

**Example:**
```python
# Test if correlation is practically zero (|r| < 0.3)
df.select(ps.tost_correlation("x", "y", delta=0.3))

# Test if correlation is equivalent to a target (0.8)
df.select(ps.tost_correlation("scale_a", "scale_b",
                               rho_null=0.8, delta=0.1))

---

## Proportion TOST

### `tost_prop_one`

One-proportion TOST equivalence test.

Tests whether an observed proportion is practically equivalent to a target proportion.

```python
ps.tost_prop_one(
    successes: int,
    n: int,
    p0: float = 0.5,
    bounds_type: str = "symmetric",
    delta: float = 0.1,
    lower: float = -0.1,
    upper: float = 0.1,
    alpha: float = 0.05,
) -> pl.Expr
```

**Null hypothesis:** The true proportion differs from p0 by more than delta.

**When to use:**
- Validating that a process meets a target rate within tolerance
- Quality control: defect rate equivalent to acceptable level
- Survey validation: response rate equivalent to expected

**Example:**
```python
# Test if success rate is within ±10% of 80%
ps.tost_prop_one(successes=82, n=100, p0=0.8, delta=0.1)
```

---

### `tost_prop_two`

Two-proportion TOST equivalence test.

Tests whether the difference between two proportions is practically zero.

```python
ps.tost_prop_two(
    successes1: int,
    n1: int,
    successes2: int,
    n2: int,
    bounds_type: str = "symmetric",
    delta: float = 0.1,
    lower: float = -0.1,
    upper: float = 0.1,
    alpha: float = 0.05,
) -> pl.Expr
```

**Null hypothesis:** The difference between proportions exceeds the equivalence bounds.

**When to use:**
- A/B testing: showing no meaningful difference between variants
- Comparing success rates across groups
- Non-inferiority/equivalence trials with binary endpoints

**Example:**
```python
# Test if conversion rates are equivalent (within ±5%)
ps.tost_prop_two(successes1=120, n1=1000,
                 successes2=115, n2=1000, delta=0.05)

---

## Non-Parametric TOST

### `tost_wilcoxon_paired`

Wilcoxon signed-rank TOST equivalence test for paired samples (non-parametric).

Tests equivalence without assuming normality. Based on the median of paired differences.

```python
ps.tost_wilcoxon_paired(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    bounds_type: str = "symmetric",
    delta: float = 0.5,
    lower: float = -0.5,
    upper: float = 0.5,
    alpha: float = 0.05,
) -> pl.Expr
```

**Null hypothesis:** The pseudo-median of differences exceeds the equivalence bounds.

**When to use:**
- Paired data with non-normal differences
- Ordinal data or ranks
- Robust alternative to `tost_t_test_paired`

**Note:** Bounds refer to the Hodges-Lehmann pseudo-median, not the mean.

---

### `tost_wilcoxon_two_sample`

Wilcoxon rank-sum TOST equivalence test for two independent samples (non-parametric).

Tests equivalence using ranks, robust to non-normality and outliers.

```python
ps.tost_wilcoxon_two_sample(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    bounds_type: str = "symmetric",
    delta: float = 0.5,
    lower: float = -0.5,
    upper: float = 0.5,
    alpha: float = 0.05,
) -> pl.Expr
```

**Null hypothesis:** The location shift between groups exceeds the equivalence bounds.

**When to use:**
- Non-normal data with outliers
- Ordinal response variables
- Robust alternative to `tost_t_test_two_sample`

---

## Robust TOST

### `tost_bootstrap`

Bootstrap TOST equivalence test using resampling for inference.

Distribution-free method that makes no parametric assumptions. Uses bootstrap confidence intervals for equivalence testing.

```python
ps.tost_bootstrap(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    bounds_type: str = "symmetric",
    delta: float = 0.5,
    lower: float = -0.5,
    upper: float = 0.5,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    seed: int | None = None,
) -> pl.Expr
```

**Null hypothesis:** The mean difference exceeds the equivalence bounds.

**Parameters:**
- `n_bootstrap`: Number of bootstrap resamples (default 1000; use 10000 for publication)
- `seed`: For reproducible results

**When to use:**
- Unknown or complex distribution shapes
- When parametric assumptions are questionable
- Small samples where asymptotic approximations fail

**Note:** Computationally more expensive than parametric tests.

---

### `tost_yuen`

Yuen TOST equivalence test comparing trimmed means (robust to outliers).

Uses trimmed means and Winsorized variances for robust equivalence testing.

```python
ps.tost_yuen(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    trim: float = 0.2,
    bounds_type: str = "symmetric",
    delta: float = 0.5,
    lower: float = -0.5,
    upper: float = 0.5,
    alpha: float = 0.05,
) -> pl.Expr
```

**Null hypothesis:** The difference in trimmed means exceeds the equivalence bounds.

**Parameters:**
- `trim=0.2` (default): Remove 20% from each tail, using middle 60%

**When to use:**
- Data contains outliers
- Heavy-tailed distributions
- Robust inference on central tendency

---

## Choosing a TOST Test

| Situation | Recommended Test |
|-----------|-----------------|
| Normal data, one sample | `tost_t_test_one_sample` |
| Normal data, two independent groups | `tost_t_test_two_sample` |
| Normal data, paired measurements | `tost_t_test_paired` |
| Non-normal, two groups | `tost_wilcoxon_two_sample` |
| Non-normal, paired | `tost_wilcoxon_paired` |
| Outliers present | `tost_yuen` |
| Unknown distribution | `tost_bootstrap` |
| Correlation equivalence | `tost_correlation` |
| Proportion equivalence | `tost_prop_one`, `tost_prop_two` |

## Setting Equivalence Bounds

**Critical decision:** The choice of δ should be made *before* seeing the data, based on:
- Clinical/practical significance thresholds
- Regulatory requirements (e.g., bioequivalence uses 80-125% for AUC)
- Subject matter expertise

**Common conventions:**
| Domain | Typical δ |
|--------|----------|
| Cohen's d (standardized) | 0.3-0.5 (small effect) |
| Correlation | 0.3 (weak correlation) |
| Proportions | 0.05-0.10 (5-10%) |
| Bioequivalence | 20% (0.80-1.25 ratio) |

---

## Interpretation

- **equivalent = True**: The effect is within the equivalence bounds at the specified alpha level. Equivalence established.
- **equivalent = False**: Cannot conclude equivalence. Either:
  - The effect is truly outside the bounds, OR
  - Sample size insufficient to establish equivalence (low power)
- **tost_p_value**: The maximum of the two one-sided p-values; reject non-equivalence if < alpha

**Important:** A non-significant TOST result does NOT mean the groups differ—it means we cannot conclude equivalence with the current data.

---

## See Also

- [Parametric Tests](parametric.md) - Traditional t-tests (for testing difference)
- [Non-Parametric Tests](nonparametric.md) - Traditional non-parametric tests
- [Correlation Tests](correlation.md) - Correlation methods
