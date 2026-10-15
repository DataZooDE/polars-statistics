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
    alternative: str = "two-sided",  # "two-sided", "less", "greater"
    continuity_correction: bool = True,
    exact: bool = False,
    conf_level: float | None = None,
    mu: float | None = None,
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64, estimate: Float64, ci_lower: Float64, ci_upper: Float64}`

**Null hypothesis:** The distributions of x and y are equal (P(X > Y) = P(Y > X)).

**`estimate`/`ci_lower`/`ci_upper` (trailing fields added in 0.8.0):** the Hodges-Lehmann shift
estimate and its confidence interval, matching R `wilcox.test(conf.int = TRUE)` on 126 upstream
reference cases. They are `NaN` unless `conf_level` is given (R itself only reports them with
`conf.int = TRUE`); a one-sided `alternative` gives an interval with an infinite bound. The
first two fields (`statistic`, `p_value`) are unchanged in order and value. See
[Mann-Whitney / Wilcoxon Output](../outputs.md#mann-whitney-wilcoxon-output) for the full
schema — it is identical to the
[`MannWhitneyU` class](../classes/tests.md#mannwhitneyu)'s `estimate`/`conf_int`.

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

# Hodges-Lehmann estimate and 95% confidence interval
result = df.select(
    ps.mann_whitney_u("treatment", "control", conf_level=0.95).alias("test")
).unnest("test")
result.select("estimate", "ci_lower", "ci_upper")
```

**Effect size:** pair with [`rank_biserial`](effect-sizes.md#rank_biserial) for a
standardised effect size, or [`rank_biserial_from_u`](effect-sizes.md#rank_biserial_from_u)
if you already have the `statistic`.

---

## `wilcoxon_signed_rank`

Wilcoxon signed-rank test for paired samples.

Tests whether the median difference between paired observations is zero. The non-parametric alternative to the paired t-test. Uses ranks of absolute differences, signed by direction.

```python
ps.wilcoxon_signed_rank(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    alternative: str = "two-sided",  # "two-sided", "less", "greater"
    continuity_correction: bool = True,
    exact: bool = False,
    conf_level: float | None = None,
    mu: float | None = None,
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64, estimate: Float64, ci_lower: Float64, ci_upper: Float64}`

**Null hypothesis:** The median of paired differences is zero.

**`estimate`/`ci_lower`/`ci_upper` (trailing fields added in 0.8.0):** as with `mann_whitney_u`,
the Hodges-Lehmann pseudo-median estimate of `x - y` and its confidence interval, matching R
`wilcox.test(conf.int = TRUE)` on 126 upstream reference cases. They are `NaN` unless
`conf_level` is given, and a one-sided `alternative` gives an interval with an infinite bound.
The first two fields (`statistic`, `p_value`) are unchanged. Equal to the
[`WilcoxonSignedRank` class](../classes/tests.md#wilcoxonsignedrank)'s `estimate`/`conf_int`.
See [Mann-Whitney / Wilcoxon Output](../outputs.md#mann-whitney-wilcoxon-output).

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

# Hodges-Lehmann pseudo-median estimate and 95% confidence interval
result = df.select(
    ps.wilcoxon_signed_rank("before", "after", conf_level=0.95).alias("test")
).unnest("test")
result.select("estimate", "ci_lower", "ci_upper")
```

**Effect size:** pair with [`matched_pairs_rank_biserial`](effect-sizes.md#matched_pairs_rank_biserial)
for a standardised paired effect size.

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

**Note:** A significant result indicates at least one group differs, but doesn't identify which groups differ. Follow up with [`dunn_test`](posthoc.md#dunn_test) for pairwise rank-based comparisons with multiple comparison correction.

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

!!! info "Behaviour changes in 0.8.0"
    The upstream `anofox-statistics` 0.4.3–0.4.6 releases bundled in polars-statistics
    0.8.0 changed `mann_whitney_u`/`wilcoxon_signed_rank` (expressions and the
    `MannWhitneyU`/`WilcoxonSignedRank` classes) in several ways:

    - **Exact two-sided Mann-Whitney p fixed** (0.4.3 #12) — e.g. on
      `x = [1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30]`,
      `y = [0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.07, 3.14, 1.28, 4.1]` with
      `exact=True`, the statistic `W = 58` now gives `p = 0.3154` (previously `1`).
    - **No continuity correction at the null centre** — `MannWhitneyU` on data whose
      ranks tie exactly at the centre (or are entirely tied) now correctly reports
      `p_value == 1.0`, matching R's edge-case behaviour (0.4.3 #13).
    - **Hodges-Lehmann `estimate`/`conf_int` match R `wilcox.test(conf.int = TRUE)`**
      (anofox-statistics 0.4.5, validated on 126 R reference cases — see
      [`MannWhitneyU`/`WilcoxonSignedRank`](../classes/tests.md) for the full
      semantics: root-finding on the asymptotic path, one-sided intervals with an
      infinite bound, continuity correction applied to the interval). Originally
      class-only; since 0.8.0 `ps.mann_whitney_u`/`ps.wilcoxon_signed_rank` return the
      identical `estimate`/`ci_lower`/`ci_upper` as trailing struct fields — see
      above and [Mann-Whitney / Wilcoxon Output](../outputs.md#mann-whitney-wilcoxon-output).
    - **`exact=True` is honoured only below a sample-size threshold** — above
      `n1 * n2 > 10,000` (rank sum / Mann-Whitney) or `300` non-zero differences
      (signed rank / Wilcoxon), `exact=True` is silently ignored and the normal
      approximation is used instead. Below the threshold, exact and asymptotic
      p-values differ meaningfully; above it they are identical.
    - **Exact null distributions no longer overflow** for larger samples
      (previously: `n1 + n2 > ~62` for rank sum, `n >= 64` for signed rank).
    - **Non-finite input handling**: the expression builders (`ps.mann_whitney_u`,
      `ps.wilcoxon_signed_rank`) drop non-finite values before computing — they
      never raise. The `MannWhitneyU`/`WilcoxonSignedRank` **classes** instead rank
      `+/-Inf` like R and raise `ValueError` on `NaN` input (see
      [Test Model Classes](../classes/tests.md) for the full non-finite contract).

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
- [Post-hoc Tests & Multiple Comparisons](posthoc.md) - `dunn_test` (pairwise follow-up after `kruskal_wallis`)
- [TOST Equivalence Tests](tost.md) - Non-parametric equivalence tests (`tost_wilcoxon_*`)
- [Distributional Tests](distributional.md) - Test normality to help choose between parametric/non-parametric
- [Effect Sizes & Bootstrap CIs](effect-sizes.md) - `rank_biserial`, `matched_pairs_rank_biserial` effect sizes
- [Test Model Classes](../classes/tests.md) - `MannWhitneyU`/`WilcoxonSignedRank` classes (Hodges-Lehmann `estimate`/`conf_int`)
