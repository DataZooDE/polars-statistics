---
name: polars-statistics-hypothesis-tests
description: Use polars-statistics hypothesis tests — run t-tests, Mann-Whitney, Kruskal-Wallis, normality tests, correlation tests, chi-square, Fisher's exact, TOST equivalence tests, MMD, energy distance, effect sizes (Cohen's d, rank-biserial), Jarque-Bera normality, or bootstrap confidence intervals as Polars expressions.
version: "0.8.0"
user-invocable: false
---

## Overview

polars-statistics exposes 52 hypothesis-test expression functions across eight families plus
10 PyModel imperative classes (count: every public function in `exprs/parametric.py`,
`nonparametric.py`, `distributional.py`, `correlation.py`, `categorical.py`, `modern.py`,
`tost.py`, and `resampling.py`, excluding the three ANOVA builders documented under
polars-statistics-anova and the private `_to_expr` helper). Expression functions integrate
with Polars `group_by`, `over`, and lazy pipelines; PyModel classes follow an sklearn-style
`.fit(x, y)` interface on NumPy arrays.

- **Expression API**: `df.select(ps.ttest_ind("a", "b"))` or `df.group_by("cohort").agg(ps.pearson("x", "y"))`
- **PyModel API**: `from polars_statistics import TTestInd; m = TTestInd().fit(x, y)`

All expression results are Polars Struct columns. Use `ps.unnest(df, "col")` or
`.struct.field("p_value")` to extract individual fields. See
polars-statistics-results for full result-extraction patterns.

---

## When to use this skill

Route here when the user asks to:

- Run a t-test, Welch's test, or paired t-test (`ttest_ind`, `ttest_paired`)
- Check variance equality (`brown_forsythe`, `yuen_test`)
- Compare two non-parametric samples (`mann_whitney_u`, `wilcoxon_signed_rank`, `brunner_munzel`)
- Compare three or more groups non-parametrically (`kruskal_wallis`)
- Test for normality or distributional fit (`shapiro_wilk`, `dagostino`)
- Compute correlation with significance (Pearson, Spearman, Kendall, distance, partial, semi-partial, ICC)
- Test proportions or contingency tables (`binom_test`, `chisq_test`, `fisher_exact`, `mcnemar_test`, etc.)
- Measure agreement (`cohen_kappa`, `cramers_v`, `phi_coefficient`)
- Compare distributions with a modern kernel method (`mmd_test`, `energy_distance`, `energy_distance_nd`)
- Run equivalence or TOST tests (`tost_t_test_one_sample`, `tost_bootstrap`, etc.)
- Compute an effect size (`cohens_d`, `cohens_d_one_sample`, `rank_biserial`, `matched_pairs_rank_biserial`, `rank_biserial_from_u`)
- Test for normality via Jarque-Bera (`jarque_bera`)
- Build a bootstrap confidence interval for a mean, median, or standard deviation (`bootstrap_mean_ci`, `bootstrap_ci`)
- Get a conditional (R `fisher.test`-style) odds ratio (`fisher_exact_conditional`)
- Use any of the keywords: hypothesis test, p-value, statistical significance, nonparametric, normality, correlation coefficient, equivalence, two-one-sided test, chi-square, Fisher exact, effect size, Cohen's d, rank-biserial, Jarque-Bera, bootstrap confidence interval

**Do NOT route here for:**
- ANOVA (one-way, two-way, repeated-measures) -> polars-statistics-anova
- Regression models -> polars-statistics-regression
- Forecast comparison tests (Diebold-Mariano, Clark-West, SPA, MCS) -> out of scope for this milestone

---

## API Reference

### Group A — Parametric Tests (`exprs/parametric.py`)

All Group A functions return `struct{statistic: f64, p_value: f64}` unless noted.

```python
# verify: skip -- function signature reference (not runnable Python; type annotations in call context)
ps.ttest_ind(x, y, alternative="two-sided", equal_var=False, mu=0.0, conf_level=0.95)
ps.ttest_paired(x, y, alternative="two-sided", mu=0.0, conf_level=0.95)
ps.brown_forsythe(x, y)
ps.yuen_test(x, y, trim=0.2, alternative="two-sided", conf_level=0.95)
```

**Parameters:**

| Parameter | Type | Values | Default |
|-----------|------|--------|---------|
| `alternative` | str | `"two-sided"`, `"less"`, `"greater"` | `"two-sided"` |
| `equal_var` | bool | `True` (Student), `False` (Welch) | `False` |
| `mu` | float | location shift under H0 | `0.0` |
| `conf_level` | float | confidence level for CI | `0.95` |
| `trim` | float | proportion to trim each tail (Yuen) | `0.2` |

**Notes:**
- `ttest_ind` with `equal_var=False` runs Welch's t-test (default). Use `equal_var=True` for
  Student's t-test (assumes equal variances).
- `brown_forsythe` takes only two sample columns and has no `alternative` parameter.
- ANOVA functions (`one_way_anova`, `two_way_anova`, `repeated_measures_anova`) live in
  `exprs/parametric.py` but are documented under polars-statistics-anova.

**Output struct fields** (ttest_ind and ttest_paired): `statistic`, `p_value`

---

### Group B — Non-parametric Tests (`exprs/nonparametric.py`)

```python
# verify: skip -- function signature reference (not runnable Python; type annotations in call context)
ps.mann_whitney_u(x, y, alternative="two-sided", continuity_correction=True, exact=False, conf_level=None, mu=None)
ps.wilcoxon_signed_rank(x, y, alternative="two-sided", continuity_correction=True, exact=False, conf_level=None, mu=None)
ps.kruskal_wallis(*groups)
ps.brunner_munzel(x, y, alternative="two-sided", alpha=None)
```

**Parameters:**

| Parameter | Type | Values / Notes | Default |
|-----------|------|---------------|---------|
| `alternative` | str | `"two-sided"`, `"less"`, `"greater"` | `"two-sided"` |
| `continuity_correction` | bool | apply continuity correction | `True` |
| `exact` | bool | compute exact p-value (slow for large n) | `False` |
| `conf_level` | float or None | pass a float to get Hodges-Lehmann CI | `None` |
| `mu` | float or None | location shift under H0 | `None` |
| `alpha` | float or None | significance level for Brunner-Munzel CI | `None` |

**Notes:**
- `conf_level` and `mu` for `mann_whitney_u` and `wilcoxon_signed_rank` are `Optional[float]`
  defaulting to `None`. Pass them only when you want a confidence interval or non-zero location
  shift — omitting them skips the Hodges-Lehmann estimator (faster).
- `kruskal_wallis` accepts two or more group columns as positional arguments (wide format — one
  column per group). See the gotchas section about stacked / long data.

**Output struct fields:**

| Function | Fields |
|----------|--------|
| `mann_whitney_u`, `wilcoxon_signed_rank` | `statistic`, `p_value`, `estimate`, `ci_lower`, `ci_upper` |
| `kruskal_wallis`, `brunner_munzel` | `statistic`, `p_value` |

`estimate` is the Hodges-Lehmann location-shift estimate; `estimate`/`ci_lower`/`ci_upper` are
`NaN` unless `conf_level` is given, matching R's `wilcox.test`, which only reports
`estimate`/`conf.int` when `conf.int = TRUE`. A one-sided `alternative` ("less"/"greater") may
give an infinite bound. Values match R `wilcox.test(..., conf.int = TRUE)` exactly (confirmed
against the live `df.select(...).schema`: both expressions return a 5-field struct).

---

### Group C — Distributional Tests (`exprs/distributional.py`)

```python
# verify: skip -- function signature reference (not runnable Python; type annotations in call context)
ps.shapiro_wilk(x)
ps.dagostino(x)
ps.jarque_bera(x)
```

**Notes:**
- `shapiro_wilk`: tests normality using the Shapiro-Wilk W statistic. Appropriate for
  `n = 3..5000`. Returns `struct{statistic: f64, p_value: f64}`. Since anofox-statistics
  0.4.3, `n <= 11` p-values are computed from the R-exact small-sample distribution
  (previously an asymptotic approximation for all `n`).
- `dagostino`: D'Agostino-Pearson K² omnibus test combining skewness and kurtosis.
  Requires `n >= 20`. Returns `struct{statistic: f64, p_value: f64}`.
- `jarque_bera`: asymptotic chi-squared(2) test on sample skewness and excess kurtosis,
  matching R `tseries::jarque.bera.test`. Returns
  `struct{statistic: f64, p_value: f64, skewness: f64, kurtosis: f64, n: u32}`. Requires
  at least 3 finite, non-constant values; fewer values, non-finite input, or constant
  data degrade to an all-NaN struct with `n == 0` (never an exception).

---

### Group D — Correlation Tests (`exprs/correlation.py`)

```python
# verify: skip -- function signature reference (not runnable Python; covariates: list uses type annotation syntax)
ps.pearson(x, y, conf_level=0.95)
ps.spearman(x, y, conf_level=0.95)
ps.kendall(x, y, variant="b")
ps.distance_cor(x, y, n_permutations=999, seed=None)
ps.partial_cor(x, y, covariates: list)
ps.semi_partial_cor(x, y, covariates: list)
ps.icc(*rater_cols, icc_type="icc2")
```

**Parameters:**

| Parameter | Type | Values / Notes | Default |
|-----------|------|---------------|---------|
| `conf_level` | float | confidence level for CI bounds | `0.95` |
| `variant` | str | `"a"` (tau-a), `"b"` (adjusted for ties), `"c"` (Stuart tau-c) | `"b"` |
| `n_permutations` | int | permutations for distance correlation p-value | `999` |
| `seed` | int or None | RNG seed for reproducibility | `None` |
| `covariates` | list | column names or expressions to partial out | — |
| `icc_type` | str | see ICC type table below | `"icc2"` |

**ICC type values:**

| `icc_type` | Model | Unit |
|------------|-------|------|
| `"icc1"` | One-way random | Single rater |
| `"icc2"` | Two-way random (default) | Single rater |
| `"icc3"` | Two-way mixed | Single rater |
| `"icc1k"` | One-way random | Average of k raters |
| `"icc2k"` | Two-way random | Average of k raters |
| `"icc3k"` | Two-way mixed | Average of k raters |

**Output struct fields:**

| Function | Fields |
|----------|--------|
| `pearson`, `spearman` | `estimate`, `statistic`, `p_value`, `ci_lower`, `ci_upper`, `n` |
| `kendall` | `estimate`, `statistic`, `p_value`, `ci_lower`, `ci_upper`, `n` |
| `distance_cor` | `estimate`, `statistic`, `p_value`, `ci_lower`, `ci_upper`, `n` |
| `partial_cor`, `semi_partial_cor` | `estimate`, `statistic`, `p_value`, `ci_lower`, `ci_upper`, `n` |
| `icc` | `icc`, `f_value`, `df1`, `df2`, `p_value`, `ci_lower`, `ci_upper`, `n_subjects`, `n_raters` |

**ICC gotcha — separate rater columns required:**
Pass each rater as a separate positional column argument. Rows represent subjects.
Do NOT pass a single matrix column — that gives silently wrong results.

```python
import polars as pl
import polars_statistics as ps
import numpy as np

rng = np.random.default_rng(11)
n_subjects = 20
df_icc_ref = pl.DataFrame({
    "rater1": rng.normal(5, 1, n_subjects).tolist(),
    "rater2": (rng.normal(5, 1, n_subjects) + rng.normal(0, 0.3, n_subjects)).tolist(),
    "rater3": (rng.normal(5, 1, n_subjects) + rng.normal(0, 0.3, n_subjects)).tolist(),
})

# Correct: three separate rater columns (rows = subjects)
df_icc_ref.select(ps.icc("rater1", "rater2", "rater3", icc_type="icc2"))

# Wrong: single matrix column — silently incorrect, no error raised
# df.select(ps.icc("ratings_matrix_column"))  # DO NOT DO THIS
```

**Note on ICC in ANOVA context:** ICC is computed from an ANOVA variance decomposition.
`icc_type="icc1"` assumes a one-way random-effects model (equivalent to one-way ANOVA);
`icc_type="icc2"` and `icc_type="icc3"` assume two-way models that account for rater effects.
For ANOVA expressions themselves (F-statistic, eta-squared), see polars-statistics-anova.

---

### Group E — Categorical Tests (`exprs/categorical.py`)

#### Scalar-argument functions

These functions take scalar integer values, not column references. Call them in
`pl.select(...)` or standalone — no DataFrame row context:

```python
# verify: skip -- function signature reference (not runnable Python; successes: int, a: int etc. use type annotation syntax)
ps.binom_test(successes: int, n: int, p0=0.5, alternative="two-sided", conf_level=0.95)
ps.prop_test_one(successes: int, n: int, p0=0.5, alternative="two-sided", conf_level=0.95)
ps.prop_test_two(successes1: int, n1: int, successes2: int, n2: int, alternative="two-sided", correction=False, conf_level=0.95)
ps.fisher_exact(a: int, b: int, c: int, d: int, alternative="two-sided", conf_level=0.95)
ps.fisher_exact_conditional(a: int, b: int, c: int, d: int, alternative="two-sided", conf_level=0.95)
ps.mcnemar_test(a: int, b: int, c: int, d: int, correction=False)
ps.mcnemar_exact(a: int, b: int, c: int, d: int)
ps.phi_coefficient(a: int, b: int, c: int, d: int)
```

For `fisher_exact`, `fisher_exact_conditional`, `mcnemar_test`, `mcnemar_exact`, and
`phi_coefficient`, `a`, `b`, `c`, `d` are the four cells of a 2x2 contingency table in
row-major order: `[[a, b], [c, d]]`.

**`conf_level` (default `0.95`, unchanged behaviour at the default):** controls the
reported interval on `binom_test` (Clopper-Pearson), `prop_test_one`/`prop_test_two`
(Wilson score), and `fisher_exact`/`fisher_exact_conditional`. One-sided `alternative`
values (`"less"`/`"greater"`) give a one-sided interval on all of these, matching R's
`binom.test`/`prop.test`/`fisher.test`. An invalid `conf_level` (outside `(0, 1)`) yields
a NaN interval, never an exception.

**`fisher_exact` vs `fisher_exact_conditional`:** `fisher_exact` reports the *sample* odds
ratio (`ad/bc`) with a two-sided Woolf (logit) interval at `conf_level` — fast, but not
full R `fisher.test` semantics. `fisher_exact_conditional` reports R `fisher.test`'s
conditional maximum-likelihood odds ratio and the exact conditional interval (one-sided
for one-sided `alternative`: `"greater"` gives `ci_upper = +inf`, `"less"` gives
`ci_lower = 0.0`). The two share the same `p_value`.

#### Contingency-table column functions

These functions accept a flattened contingency table stored in a Polars column called `data`,
along with dimension parameters:

```python
# verify: skip -- function signature reference (not runnable Python; type annotations in call context)
ps.chisq_test(data, n_rows=2, n_cols=2, correction=False)
ps.g_test(data, n_rows=2, n_cols=2)
ps.cohen_kappa(data, n_categories=2, weighted=False)
ps.cramers_v(data, n_rows=2, n_cols=2)
ps.contingency_coef(data, n_rows=2, n_cols=2)
```

The `data` argument is a column name containing the contingency table cells flattened
row-by-row into a UInt32 column. For a 2x2 table `[[a, b], [c, d]]` pass four rows.

#### Goodness-of-fit

```python
# verify: skip -- function signature reference (not runnable Python; type annotations in call context)
ps.chisq_goodness_of_fit(observed, expected=None)
```

If `expected` is `None`, the test assumes a uniform distribution over all categories.
Pass an explicit `expected` column for non-uniform expected frequencies.

**`alternative` parameter values** (where applicable): `"two-sided"`, `"less"`, `"greater"`

**Output struct fields:**

| Function | Fields |
|----------|--------|
| `binom_test`, `prop_test_one`, `prop_test_two` | `estimate`, `statistic`, `p_value`, `ci_lower`, `ci_upper`, `n` |
| `chisq_test`, `g_test`, `mcnemar_test` | `statistic`, `p_value`, `df`, `n` |
| `fisher_exact` | `statistic` (sample odds ratio `ad/bc`), `p_value`, `ci_lower`, `ci_upper` (Woolf interval) |
| `fisher_exact_conditional` | `statistic` (conditional MLE odds ratio), `p_value`, `ci_lower`, `ci_upper` (exact conditional interval) |
| `mcnemar_exact` | `statistic`, `p_value` |
| `phi_coefficient`, `cramers_v`, `cohen_kappa`, `contingency_coef` | `estimate`, `statistic`, `p_value` |

---

### Group F — Modern Tests (`exprs/modern.py`)

```python
# verify: skip -- function signature reference (not runnable Python; x_cols: list, y_cols: list use type annotation syntax)
ps.energy_distance(x, y, n_permutations=999, seed=None)
ps.energy_distance_nd(x_cols: list, y_cols: list, n_permutations=999, seed=None)
ps.mmd_test(x, y, n_permutations=999, seed=None)
```

**Parameters:**

| Parameter | Type | Notes | Default |
|-----------|------|-------|---------|
| `n_permutations` | int | permutation iterations for p-value | `999` |
| `seed` | int or None | RNG seed for reproducibility | `None` |
| `x_cols` | list | column names for sample X (multivariate) | — |
| `y_cols` | list | column names for sample Y (multivariate) | — |

**Notes:**
- `energy_distance`: univariate two-sample energy distance test. Detects differences in
  location, scale, and shape simultaneously.
- `energy_distance_nd`: multivariate extension. `x_cols` and `y_cols` must have equal length
  (the dimension `d`). Raises `ValueError` if lengths differ or either list is empty.
- `mmd_test`: Maximum Mean Discrepancy test using a Gaussian kernel. More sensitive than
  energy distance for high-dimensional or scale-matched distributions.
- All three return `struct{statistic: f64, p_value: f64}`.

---

### Group G — TOST Equivalence Tests (`exprs/tost.py`)

```python
# verify: skip -- function signature reference (not runnable Python; successes: int, n: int use type annotation syntax)
ps.tost_t_test_one_sample(x, mu=0.0, bounds_type="symmetric", delta=0.5, lower=-0.5, upper=0.5, alpha=0.05)
ps.tost_t_test_two_sample(x, y, bounds_type="symmetric", delta=0.5, lower=-0.5, upper=0.5, alpha=0.05, pooled=False)
ps.tost_t_test_paired(x, y, bounds_type="symmetric", delta=0.5, lower=-0.5, upper=0.5, alpha=0.05)
ps.tost_correlation(x, y, method="pearson", rho_null=0.0, bounds_type="symmetric", delta=0.3, lower=-0.3, upper=0.3, alpha=0.05)
ps.tost_prop_one(successes: int, n: int, p0=0.5, bounds_type="symmetric", delta=0.1, lower=-0.1, upper=0.1, alpha=0.05)
ps.tost_prop_two(successes1: int, n1: int, successes2: int, n2: int, bounds_type="symmetric", delta=0.1, lower=-0.1, upper=0.1, alpha=0.05)
ps.tost_wilcoxon_paired(x, y, bounds_type="symmetric", delta=0.5, lower=-0.5, upper=0.5, alpha=0.05)
ps.tost_wilcoxon_two_sample(x, y, bounds_type="symmetric", delta=0.5, lower=-0.5, upper=0.5, alpha=0.05)
ps.tost_bootstrap(x, y, bounds_type="symmetric", delta=0.5, lower=-0.5, upper=0.5, alpha=0.05, n_bootstrap=1000, seed=None)
ps.tost_yuen(x, y, trim=0.2, bounds_type="symmetric", delta=0.5, lower=-0.5, upper=0.5, alpha=0.05)
```

**`bounds_type` values:**

| Value | Behaviour |
|-------|-----------|
| `"symmetric"` | Uses `delta` as both bounds: lower = `-delta`, upper = `+delta` |
| `"raw"` | Uses `lower` and `upper` directly |
| `"cohen_d"` | Interprets `delta` as Cohen's d effect size (t-test variants only) |

**Additional parameters:**

| Parameter | Applies to | Notes | Default |
|-----------|------------|-------|---------|
| `method` | `tost_correlation` | `"pearson"` or `"spearman"` | `"pearson"` |
| `rho_null` | `tost_correlation` | null correlation value | `0.0` |
| `pooled` | `tost_t_test_two_sample` | use pooled variance | `False` |
| `trim` | `tost_yuen` | proportion to trim each tail | `0.2` |
| `n_bootstrap` | `tost_bootstrap` | bootstrap iterations | `1000` |
| `seed` | `tost_bootstrap` | RNG seed | `None` |
| `alpha` | all | significance level | `0.05` |

**Output struct fields (all TOST functions):**
`estimate`, `ci_lower`, `ci_upper`, `bound_lower`, `bound_upper`, `tost_p_value`, `equivalent`, `alpha`, `n`

- `equivalent`: boolean — `True` if TOST concludes equivalence at level `alpha`.
- `tost_p_value`: the larger of the two one-sided p-values (equivalence requires both less than `alpha`).

---

### Group I — Effect Sizes & Bootstrap CIs (`exprs/parametric.py`, `nonparametric.py`, `resampling.py`)

```python
# verify: skip -- function signature reference (not runnable Python; type annotations in call context)
ps.cohens_d(x, y, kind="student", mu=0.0)
ps.cohens_d_one_sample(x, mu=0.0)
ps.rank_biserial(x, y, mu=None)
ps.matched_pairs_rank_biserial(x, y, mu=None)
ps.rank_biserial_from_u(u1, n1, n2)
ps.bootstrap_mean_ci(x, n_bootstrap=1000, conf_level=0.95, block_length=0, seed=None)
ps.bootstrap_ci(x, statistic="mean", n_bootstrap=1000, conf_level=0.95, seed=None)
```

**Parameters:**

| Parameter | Type | Values / Notes | Default |
|-----------|------|---------------|---------|
| `kind` (`cohens_d`) | str | `"student"` (pooled-SD), `"welch"` (unequal-variance), `"paired"` (`d_z` on `x - y`, jointly filtered) | `"student"` |
| `mu` | float or None | null value of the mean / location-shift difference | `0.0` (Cohen's d) / `None` (rank-biserial) |
| `block_length` (`bootstrap_mean_ci`) | int | `0` = IID bootstrap, `1` = stationary bootstrap, `>1` = circular block bootstrap (dependent/time-series data) | `0` |
| `statistic` (`bootstrap_ci`) | str | `"mean"`, `"median"`, `"std"` (sample SD, `n - 1` denominator) — fixed dispatch only, no Python callables | `"mean"` |
| `n_bootstrap` | int | number of bootstrap resamples | `1000` |
| `conf_level` | float | confidence level for the percentile interval | `0.95` |
| `seed` | int or None | RNG seed; omit for non-deterministic resamples | `None` |

**Output struct fields:**

| Function | Fields |
|----------|--------|
| `cohens_d`, `cohens_d_one_sample`, `rank_biserial`, `matched_pairs_rank_biserial`, `rank_biserial_from_u` | `estimate` |
| `bootstrap_mean_ci`, `bootstrap_ci` | `estimate`, `se`, `ci_lower`, `ci_upper`, `conf_level`, `n_bootstrap` |

**Notes:**
- `cohens_d`: `"student"` matches R `effectsize::cohens_d(x, y, pooled_sd = TRUE)`;
  `"welch"` matches `pooled_sd = FALSE`; `"paired"` matches
  `effectsize::cohens_d(x, y, paired = TRUE)`. Returns NaN when the standardiser is zero
  or a sample has fewer than 2 observations.
- `cohens_d_one_sample`: `d = (mean(x) - mu) / sd(x)`, matching
  `effectsize::cohens_d(x, mu = mu)`.
- `rank_biserial` / `matched_pairs_rank_biserial` / `rank_biserial_from_u`: see the sign-
  convention gotcha below. `rank_biserial_from_u` takes a precomputed Mann-Whitney U
  statistic (e.g. `MannWhitneyU().fit(x, y).statistic`) instead of raw samples —
  O(1) instead of O(n1 * n2).
- `bootstrap_mean_ci` / `bootstrap_ci`: both drop non-finite values before resampling;
  `statistic="mean"` on `bootstrap_ci` is bit-identical to
  `bootstrap_mean_ci(block_length=0)` for the same seed (same Rust mean closure).
  Invalid parameters (`n_bootstrap < 2`, `conf_level` outside `(0, 1)`, fewer than 2
  finite observations) degrade to an all-NaN struct, never an exception.

---

### Group H — PyModel Test Classes

These parallel the expression functions but operate on NumPy arrays directly.
They are stateful (fitted results stored on the object) and follow sklearn-style
`.fit(x, y)` interface. Import from the top-level package:

```python
from polars_statistics import (
    TTestInd, TTestPaired, BrownForsythe, YuenTest,
    MannWhitneyU, WilcoxonSignedRank, KruskalWallis, BrunnerMunzel,
    ShapiroWilk, DAgostino,
)
```

**TTestInd**
```python
# verify: skip -- PyModel reference (indented method chaining is illustrative syntax, not valid Python)
TTestInd(alternative="two-sided", equal_var=False, mu=0.0, conf_level=0.95)
  .fit(x, y) -> TTestInd
  .statistic, .p_value, .df, .conf_int, .mean_x, .mean_y, .null_value
  .is_fitted() -> bool
  .summary() -> str
  .to_dict() -> dict
```

**TTestPaired**
```python
# verify: skip -- PyModel reference (indented method chaining is illustrative syntax, not valid Python)
TTestPaired(alternative="two-sided", mu=0.0, conf_level=0.95)
  .fit(x, y) -> TTestPaired
  .statistic, .p_value, .df, .conf_int, .mean_x, .mean_y, .null_value
  .is_fitted() -> bool
  .summary() -> str
  .to_dict() -> dict
```

**BrownForsythe**
```python
# verify: skip -- PyModel reference (indented method chaining is illustrative syntax, not valid Python)
BrownForsythe()
  .fit(x, y) -> BrownForsythe
  .statistic, .p_value
  .is_fitted() -> bool
  .summary() -> str
  .to_dict() -> dict
```

**YuenTest**
```python
# verify: skip -- PyModel reference (indented method chaining is illustrative syntax, not valid Python)
YuenTest(trim=0.2, alternative="two-sided", conf_level=0.95)
  .fit(x, y) -> YuenTest
  .statistic, .p_value, .df, .diff, .trimmed_mean_x, .trimmed_mean_y, .conf_int
  .is_fitted() -> bool
  .summary() -> str
  .to_dict() -> dict
```

**MannWhitneyU**
```python
# verify: skip -- PyModel reference (indented method chaining is illustrative syntax, not valid Python)
MannWhitneyU(alternative="two-sided", continuity_correction=True, exact=False, conf_level=None, mu=None)
  .fit(x, y) -> MannWhitneyU
  .statistic, .p_value, .estimate, .conf_int, .null_value
  .is_fitted() -> bool
  .summary() -> str
  .to_dict() -> dict
```

**WilcoxonSignedRank**
```python
# verify: skip -- PyModel reference (indented method chaining is illustrative syntax, not valid Python)
WilcoxonSignedRank(alternative="two-sided", continuity_correction=True, exact=False, conf_level=None, mu=None)
  .fit(x, y) -> WilcoxonSignedRank
  .statistic, .p_value, .estimate, .conf_int, .null_value
  .is_fitted() -> bool
  .summary() -> str
  .to_dict() -> dict
```

**KruskalWallis**
```python
# verify: skip -- PyModel reference (indented method chaining is illustrative syntax, not valid Python)
KruskalWallis()
  .fit(*groups) -> KruskalWallis    # varargs: one numpy array per group
  .statistic, .p_value
  .is_fitted() -> bool
  .summary() -> str
  .to_dict() -> dict
```

**BrunnerMunzel**
```python
# verify: skip -- PyModel reference (indented method chaining is illustrative syntax, not valid Python)
BrunnerMunzel(alternative="two-sided", alpha=None)
  .fit(x, y) -> BrunnerMunzel
  .statistic, .p_value, .df, .estimate, .conf_int
  .is_fitted() -> bool
  .summary() -> str
  .to_dict() -> dict
```

**ShapiroWilk**
```python
# verify: skip -- PyModel reference (indented method chaining is illustrative syntax, not valid Python)
ShapiroWilk()
  .fit(x) -> ShapiroWilk    # single array argument, not two
  .statistic, .p_value
  .is_fitted() -> bool
  .summary() -> str
  .to_dict() -> dict
```

**DAgostino**
```python
# verify: skip -- PyModel reference (indented method chaining is illustrative syntax, not valid Python)
DAgostino()
  .fit(x) -> DAgostino      # single array argument
  .statistic, .p_value
  .is_fitted() -> bool
  .summary() -> str
  .to_dict() -> dict
```

All PyModel classes raise `PyRuntimeError("Model not fitted")` if you access a fitted
property before calling `.fit()`. Guard with `model.is_fitted()` in conditional paths.

---

## Worked Examples

### Example 1: Independent t-test (expression API)

```python
import polars as pl
import polars_statistics as ps
import numpy as np

rng = np.random.default_rng(42)
df = pl.DataFrame({
    "control": rng.normal(0.0, 1.0, 30).tolist(),
    "treatment": rng.normal(0.5, 1.0, 30).tolist(),
})

# Whole-frame Welch's t-test (default equal_var=False)
result = df.select(ps.ttest_ind("control", "treatment").alias("t"))
print(result.unnest("t"))  # statistic, p_value

# Per-group t-test with group_by
df2 = pl.DataFrame({
    "cohort": ["A"] * 30 + ["B"] * 30,
    "control": rng.normal(0.0, 1.0, 60).tolist(),
    "treatment": rng.normal(0.5, 1.0, 60).tolist(),
})
per_group = df2.group_by("cohort").agg(
    ps.ttest_ind("control", "treatment").alias("t")
)
ps.unnest(per_group, "t")   # statistic, p_value per cohort
```

### Example 2: Mann-Whitney U (nonparametric two-sample)

```python
import polars as pl
import polars_statistics as ps
import numpy as np

rng = np.random.default_rng(0)
df = pl.DataFrame({
    "before": rng.exponential(1.0, 25).tolist(),
    "after": rng.exponential(1.5, 25).tolist(),
})

# With Hodges-Lehmann confidence interval
result = df.select(
    ps.mann_whitney_u("before", "after", alternative="less", conf_level=0.95).alias("mwu")
).unnest("mwu")
print(result)  # statistic, p_value, estimate, ci_lower, ci_upper

row = result.row(0, named=True)
assert row["ci_lower"] <= row["estimate"] <= row["ci_upper"]
```

### Example 3: Normality test (Shapiro-Wilk)

```python
import polars as pl
import polars_statistics as ps
import numpy as np

rng = np.random.default_rng(7)
df = pl.DataFrame({
    "normal_data": rng.normal(0, 1, 50).tolist(),
    "skewed_data": rng.exponential(1, 50).tolist(),
})

print("Normal data:", df.select(ps.shapiro_wilk("normal_data").alias("w")))
print("Skewed data:", df.select(ps.shapiro_wilk("skewed_data").alias("w")))
# Low p_value on skewed_data -> reject normality
```

### Example 4: Pearson correlation and ICC

```python
import polars as pl
import polars_statistics as ps
import numpy as np

rng = np.random.default_rng(99)
n = 40
df = pl.DataFrame({
    "x": rng.normal(0, 1, n).tolist(),
    "y": (rng.normal(0, 1, n) + rng.normal(0, 0.5, n)).tolist(),
})

# Pearson with CI
corr = df.select(ps.pearson("x", "y").alias("r"))
print(corr.unnest("r"))  # estimate, statistic, p_value, ci_lower, ci_upper, n

# ICC: pass each rater as a separate positional column argument (rows = subjects)
ratings = pl.DataFrame({
    "rater1": rng.normal(5, 1, 20).tolist(),
    "rater2": (rng.normal(5, 1, 20) + rng.normal(0, 0.3, 20)).tolist(),
    "rater3": (rng.normal(5, 1, 20) + rng.normal(0, 0.3, 20)).tolist(),
})
icc_result = ratings.select(ps.icc("rater1", "rater2", "rater3", icc_type="icc2").alias("icc"))
print(icc_result.unnest("icc"))
# Fields: icc, f_value, df1, df2, p_value, ci_lower, ci_upper, n_subjects, n_raters
```

### Example 5: Chi-square test (categorical)

```python
import polars as pl
import polars_statistics as ps

# 2x2 contingency table flattened row-major: [[10, 20], [30, 40]]
df = pl.DataFrame({"counts": [10, 20, 30, 40]})
result = df.select(ps.chisq_test("counts", n_rows=2, n_cols=2).alias("chi"))
print(result.unnest("chi"))  # statistic, p_value, df, n

# Fisher's exact test for small cell counts (scalar API, no DataFrame needed)
result_fe = pl.select(ps.fisher_exact(10, 2, 3, 15, alternative="two-sided").alias("fe"))
print(result_fe.unnest("fe"))
```

### Example 6: MMD distribution comparison

```python
import polars as pl
import polars_statistics as ps
import numpy as np

rng = np.random.default_rng(42)
df = pl.DataFrame({
    "sample_a": rng.normal(0, 1, 50).tolist(),
    "sample_b": rng.normal(1, 1, 50).tolist(),   # shifted mean
})

result = df.select(
    ps.mmd_test("sample_a", "sample_b", n_permutations=999, seed=42).alias("mmd")
)
print(result.unnest("mmd"))  # statistic, p_value
# Low p_value -> distributions differ
```

### Example 7: TOST equivalence test

```python
import polars as pl
import polars_statistics as ps
import numpy as np

rng = np.random.default_rng(10)
df = pl.DataFrame({
    "old_process": rng.normal(100.0, 2.0, 40).tolist(),
    "new_process": rng.normal(100.2, 2.0, 40).tolist(),  # nearly identical
})

# Claim equivalence within plus/minus 1.0 raw units
result = df.select(
    ps.tost_t_test_two_sample(
        "old_process", "new_process",
        bounds_type="raw",
        lower=-1.0, upper=1.0,
        alpha=0.05,
    ).alias("tost")
)
print(result.unnest("tost"))
# equivalent=True when both one-sided tests pass at alpha=0.05
```

### Example 8: PyModel t-test (imperative API)

```python
import numpy as np
from polars_statistics import TTestInd

rng = np.random.default_rng(42)
x = rng.normal(0.0, 1.0, 30)
y = rng.normal(0.5, 1.0, 30)

m = TTestInd(alternative="two-sided", equal_var=False).fit(x, y)
print(m.summary())
print(f"statistic={m.statistic:.4f}, p={m.p_value:.4f}")
print(m.to_dict())
```

### Example 9: Effect sizes, Jarque-Bera and a bootstrap CI

```python
import polars as pl
import polars_statistics as ps
import numpy as np

rng = np.random.default_rng(42)
df = pl.DataFrame({
    "control": rng.normal(0.0, 1.0, 30).tolist(),
    "treatment": rng.normal(0.5, 1.0, 30).tolist(),
})

# Cohen's d: negative because treatment tends to be larger than control
d = df.select(ps.cohens_d("control", "treatment").alias("d"))
assert d.unnest("d")["estimate"][0] < 0.0

# Rank-biserial effect size for the same comparison (opposite sign convention vs
# R's effectsize::rank_biserial -- positive here means treatment tends to be larger)
rb = df.select(ps.rank_biserial("control", "treatment").alias("rb"))
print(rb.unnest("rb"))

# Jarque-Bera normality test on the control sample
jb = df.select(ps.jarque_bera("control").alias("jb"))
print(jb.unnest("jb"))  # statistic, p_value, skewness, kurtosis, n

# Bootstrap confidence interval for the control mean (reproducible via seed)
bc = df.select(ps.bootstrap_mean_ci("control", seed=42).alias("bc"))
print(bc.unnest("bc"))  # estimate, se, ci_lower, ci_upper, conf_level, n_bootstrap

# Conditional (R fisher.test) odds ratio for a 2x2 table
fe = pl.select(ps.fisher_exact_conditional(18, 7, 9, 16).alias("fe"))
print(fe.unnest("fe"))  # statistic, p_value, ci_lower, ci_upper
```

---

## Gotchas / Notes

### ICC requires separate rater columns — not a matrix

The `ps.icc()` expression takes one column per rater as positional arguments, where rows
correspond to subjects. A single column containing a matrix (List or Array dtype) produces
wrong results without raising an error.

```python
import polars as pl
import polars_statistics as ps
import numpy as np

rng = np.random.default_rng(22)
n = 15
df_raters = pl.DataFrame({
    "rater1": rng.normal(5, 1, n).tolist(),
    "rater2": (rng.normal(5, 1, n) + rng.normal(0, 0.3, n)).tolist(),
    "rater3": (rng.normal(5, 1, n) + rng.normal(0, 0.3, n)).tolist(),
})

# Correct
df_raters.select(ps.icc("rater1", "rater2", "rater3"))

# Wrong (no error, incorrect answer)
# df.select(ps.icc("ratings_matrix_column"))
```

### `kruskal_wallis` and wide-format input

`kruskal_wallis` (like `one_way_anova`) expects one column per group (wide format). It does
not accept a value column plus a group-label column (long format).

```python
import polars as pl
import polars_statistics as ps
import numpy as np

rng = np.random.default_rng(33)
df_wide = pl.DataFrame({
    "ctrl": rng.normal(0, 1, 25).tolist(),
    "treat_a": rng.normal(0.5, 1, 25).tolist(),
    "treat_b": rng.normal(1.0, 1, 25).tolist(),
})

# Correct: separate columns per group
df_wide.select(ps.kruskal_wallis("ctrl", "treat_a", "treat_b"))

# Wrong: stacked long format inside group_by
# df_long.group_by("group").agg(ps.kruskal_wallis("value"))  # DO NOT DO THIS
```

### `conf_level` and `mu` in Mann-Whitney / Wilcoxon default to `None`

Passing `conf_level=None` (the default) skips computing the Hodges-Lehmann location
estimator and confidence interval, which improves performance for large samples. Pass a
float (e.g., `0.95`) only when you need the CI.

### Rank-biserial sign convention is opposite R's `effectsize::rank_biserial`

`rank_biserial`, `matched_pairs_rank_biserial`, and `rank_biserial_from_u` report `r > 0`
when the *second* sample (`y`) tends to be larger than the first (`x`). R's
`effectsize::rank_biserial` reports the opposite sign for the same comparison. Negate the
estimate when porting an R-authored analysis.

### Expression builders drop non-finite values; PyModel test classes rank them like R

`ps.mann_whitney_u`, `ps.wilcoxon_signed_rank`, `ps.kruskal_wallis`, and the other Group A/B
expression builders silently filter non-finite (`NaN`/`Inf`) rows before computing. The
corresponding PyModel classes (`MannWhitneyU`, `WilcoxonSignedRank`, `KruskalWallis`, ...)
instead rank `+Inf`/`-Inf` as the largest/smallest values (matching R's own `rank()`
behaviour, anofox-statistics 0.4.4) and raise `ValueError` if a `NaN` is present — they do
not silently drop rows. Pick the PyModel class when you need R-matching treatment of
infinite values.

### `bootstrap_ci` supports only `"mean"` / `"median"` / `"std"` — no Python callables

The upstream `anofox_statistics::bootstrap_ci` is generic over a native Rust closure, which
cannot cross the Python FFI boundary (and calling back into Python from a Polars worker
thread risks a GIL-reentrancy deadlock). Only the fixed `{"mean", "median", "std"}`
dispatch is exposed. For any other statistic, resample manually in Python/NumPy.

### One-sided `alternative` gives a one-sided interval on `binom_test` / `prop_test_*` / `fisher_exact*`

Since this plan, passing `alternative="less"` or `alternative="greater"` to `binom_test`,
`prop_test_one`, `prop_test_two`, `fisher_exact`, or `fisher_exact_conditional` returns a
one-sided interval (e.g. `ci_upper = +inf` for `"greater"`), matching R's
`binom.test`/`prop.test`/`fisher.test`. Only `alternative="two-sided"` (the default) gives
a conventional two-sided interval.

### Mann-Whitney / Wilcoxon 0.4.5-0.4.6 retarget — R `wilcox.test` parity, exact gating, `mcnemar_exact`

Since anofox-statistics 0.4.5/0.4.6:
- The `MannWhitneyU` / `WilcoxonSignedRank` PyModel classes' `estimate` and `conf_int` now
  follow R `wilcox.test(conf.int = TRUE)` exactly: a root-finding interval, continuity
  correction applied to the interval (not only the test statistic), one-sided alternatives
  give an interval with an infinite bound, and the signed-rank interval is computed on
  `x - y`.
- `exact=True` is honoured only for `n1 * n2 <= 10,000` (rank-sum) or `<= 300` non-zero
  differences (signed-rank) — larger samples silently fall back to the normal
  approximation (the exact null distribution no longer overflows below these thresholds,
  but is not computed above them).
- `ps.mcnemar_exact` is exact and finite for any `b + c` (previously NaN for large
  discordant-pair counts) — it computes the R `binom.test(b, b+c, 0.5)` p-value via the
  binomial CDF rather than enumerating the binomial PMF.

### Mann-Whitney / Wilcoxon expressions now return the Hodges-Lehmann estimate + CI (0.8.0, UPST-07)

Since this release, `ps.mann_whitney_u` / `ps.wilcoxon_signed_rank` return a 5-field struct —
`{statistic, p_value, estimate, ci_lower, ci_upper}` — on **both** the expression layer and the
`MannWhitneyU` / `WilcoxonSignedRank` PyModel classes. `estimate`/`ci_lower`/`ci_upper` are `NaN`
unless `conf_level` is passed (matching R's `wilcox.test`, which only reports
`estimate`/`conf.int` when `conf.int = TRUE`); both layers are validated against R
`wilcox.test(conf.int = TRUE)` on the same 126-case reference fixture, including infinite
one-sided bounds. The PyModel classes additionally expose `null_value` (the hypothesized
location shift under H0, i.e. `mu`). No layer is limited to `{statistic, p_value}` anymore.

### Scalar-argument categorical functions (`binom_test`, `fisher_exact`, etc.)

These functions take scalar integers directly, not Polars column references. Use them in
`pl.select(ps.binom_test(7, 10))` or as standalone expressions without a row iterator.
Passing column names will raise a type error.

### TOST `bounds_type="cohen_d"` only for t-test variants

The Cohen's d bounds type is supported only for `tost_t_test_one_sample`,
`tost_t_test_two_sample`, and `tost_t_test_paired`. Using it with other TOST functions
(Wilcoxon, bootstrap, Yuen) will raise an error.

### `chisq_goodness_of_fit` with default expected

When `expected=None`, the test assumes all categories are equally likely. Provide an
explicit `expected` column when the theoretical distribution is non-uniform.

### Struct column naming after group_by

After `df.group_by(...).agg(ps.ttest_ind("a","b").alias("t"))`, pass the column name
explicitly to `ps.unnest()`:

```python
import polars as pl
import polars_statistics as ps
import numpy as np

rng = np.random.default_rng(44)
df_ab = pl.DataFrame({
    "a": rng.normal(0.0, 1.0, 30).tolist(),
    "b": rng.normal(0.5, 1.0, 30).tolist(),
    "cohort": ["X"] * 15 + ["Y"] * 15,
})
result = df_ab.group_by("cohort").agg(ps.ttest_ind("a", "b").alias("t"))

ps.unnest(result, "t")   # correct
# ps.unnest(result)      # raises ValueError when multiple Struct columns are present
```

### Accessing PyModel properties before fitting

All PyModel test classes raise `PyRuntimeError("Model not fitted")` when accessing fitted
properties on an unfitted instance. Use `model.is_fitted()` before accessing results in
conditional contexts.

---

## See also

- polars-statistics-anova
- polars-statistics-regression
- polars-statistics-results
- polars-statistics-getting-started
