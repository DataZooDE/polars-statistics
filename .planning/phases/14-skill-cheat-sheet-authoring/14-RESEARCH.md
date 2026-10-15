# Phase 14: Skill Cheat-Sheet Authoring - Research

**Researched:** 2026-09-21
**Domain:** Claude Code Skills authoring — polars-statistics v0.8.0 API surface documentation
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Coverage Depth**
- Exhaustive signatures: every public function/class in a family gets a signature line. Worked, runnable examples for representative cases per family (not one-example-per-symbol).
- Regression's 40+ PyModel classes are grouped by family with a signature line each and worked examples for the common ones.

**Accuracy While Authoring**
- Spot-check as you write: import the installed package and run key snippets per skill during authoring to catch wrong signatures/kwargs early. Full every-example gate is Phase 15.

**Signature Source of Truth**
- The shipped Python surface is authoritative: `python/polars_statistics/` (exprs/*.py, regression.py, results.py, etc.) plus `python/polars_statistics/_polars_statistics.pyi`. Never invent kwargs — if a parameter is not in the shipped signature, it does not exist.
- Cross-reference `docs/` cookbook and `examples/` scripts for idiomatic usage, but the signature itself comes from the Python source / `.pyi`.

**Frontmatter & Structure**
- Each SKILL.md frontmatter: `name` (matches directory), `description` (task/keyword-rich single sentence describing WHAT TASKS route here), `version` pinned to `0.8.0`, `user-invocable: false` (auto-surfaced).
- Consistent cheat-sheet structure across all five: Overview → When to use → API reference by group with signatures → Worked examples → Gotchas/notes → Cross-links to sibling skills.

### Claude's Discretion

None specified beyond the locked decisions above.

### Deferred Ideas (OUT OF SCOPE)

- Full "every example runs" execution gate and description-routing verification → Phase 15.
- Any new API surface — this phase documents only the shipped v0.8.0 (== v0.7.0) surface.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SKILL-01 | `polars-statistics-hypothesis-tests/SKILL.md` covers expression-API tests — parametric, nonparametric, distributional, correlation, categorical, modern (MMD/energy), and TOST/equivalence — with accurate signatures and runnable examples | Full symbol inventory with exact signatures in §Skill 1 below |
| SKILL-02 | `polars-statistics-anova/SKILL.md` covers one-way, two-way, and repeated-measures ANOVA plus ICC, with worked examples | Exact signatures, struct output fields, ANOVA gotchas in §Skill 2 |
| SKILL-03 | `polars-statistics-regression/SKILL.md` covers 40+ PyModel classes, regression expressions, R-style formula parser | Complete PyModel inventory with .pyi signatures; all expression groups; formula syntax in §Skill 3 |
| SKILL-04 | `polars-statistics-results/SKILL.md` covers result ergonomics — `to_dict()`/`summary()`/`__repr__`, unnest/struct helpers, contextual errors, typed `.pyi` API | `unnest`/`struct_to_dict` exact signatures; PyModel method patterns; error taxonomy in §Skill 4 |
| SKILL-05 | `polars-statistics-getting-started/SKILL.md` covers install, the model-selection decision matrix, and sklearn migration | Install path, version, two-API-style patterns, sklearn comparison in §Skill 5 |
| SKILL-06 | Every `SKILL.md` carries correct frontmatter (`name`, `description`, `version` pinned to package version, `user-invocable: false`) and a consistent cheat-sheet structure | Frontmatter spec, reference SKILL.md structure, version value in §Frontmatter |
</phase_requirements>

---

## Summary

This phase authors five `SKILL.md` cheat-sheets that replace the Phase 13 stub placeholders in `plugins/polars-statistics/skills/`. The files are pure documentation — no Rust or Python code changes. The entire research effort is directed at enumerating the exhaustive, exact API surface from the shipped v0.7.0/v0.8.0 source so no kwargs are invented.

The library exposes two orthogonal API styles: **expression-first** (`df.select(ps.ttest_ind("a", "b"))`, `df.group_by(...).agg(ps.ols(...))`) and **PyModel-class** (`from polars_statistics import OLS; m = OLS().fit(X, y)`). Both styles must be represented in the relevant cheat-sheets. Expression results are Polars Struct columns; the `results.py` helpers (`unnest`, `struct_to_dict`) are the ergonomic bridge from Struct to plain Python.

**Primary recommendation:** Write the skills in order: hypothesis-tests → anova → regression → results → getting-started, cross-linking as you go. Spot-check each family's worked example against the installed package before committing.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| SKILL.md authoring | Documentation (files in plugins/) | None | Purely static Markdown files |
| Version pinning | Configuration (frontmatter) | None | `version: "0.8.0"` in YAML frontmatter |
| API surface inventory | Python source truth | .pyi stubs | Source is canonical; .pyi is authoritative for class/method shapes |
| Example spot-checks | Local Python env | None | `python -c "import polars_statistics as ps; ..."` against installed wheel |

---

## Frontmatter Specification

Every SKILL.md must open with exactly this YAML frontmatter block (no extra fields):

```yaml
---
name: <directory-name>
description: <task/keyword-rich single sentence — see per-skill descriptions below>
version: "0.8.0"
user-invocable: false
---
```

**Version value:** `"0.8.0"` — confirmed from `plugins/polars-statistics/.claude-plugin/plugin.json` line 4: `"version": "0.8.0"`. The source `__version__ = "0.7.0"` in `python/polars_statistics/__init__.py` line 258 reflects the current wheel; the skill files are authored targeting the 0.8.0 release milestone. Planner must keep the version value consistent with the final plugin.json. [VERIFIED: /home/simonm/projects/rust/polars-statistics/plugins/polars-statistics/.claude-plugin/plugin.json:4] [VERIFIED: /home/simonm/projects/rust/polars-statistics/python/polars_statistics/__init__.py:258]

**Reference structure pattern** (from installed fdars plugin): `name`, optional `description`, no `user-invocable` field shown in fdars — but the milestone decision locks `user-invocable: false` for all five skills. [ASSUMED: fdars SKILL.md does not show `user-invocable`; the polars-statistics decision is locked independently per CONTEXT.md]

**Consistent cheat-sheet structure** (to apply to all five files):
```
## Overview
## When to use this skill
## API Reference
  ### [Group 1]
  ### [Group 2]
  ...
## Worked Examples
## Gotchas / Notes
## See also
```

---

## Standard Stack

No external packages are installed by this phase. The stack for authoring:

| Item | Value | Source |
|------|-------|--------|
| Package | `polars-statistics` | shipped wheel |
| Version (skill files) | `0.8.0` | plugin.json |
| Python import alias | `import polars_statistics as ps` | idiomatic convention throughout docs/ |
| Polars import | `import polars as pl` | all expression examples |
| Source of truth | `python/polars_statistics/` + `_polars_statistics.pyi` | CONTEXT.md decision |

---

## Package Legitimacy Audit

> This phase installs NO packages. Skip.

---

## Skill 1: polars-statistics-hypothesis-tests

**Description field:** `Use polars-statistics hypothesis tests — run t-tests, Mann-Whitney, Kruskal-Wallis, normality tests, correlation tests, chi-square, Fisher's exact, TOST equivalence tests, MMD, or energy distance as Polars expressions.`

**Skill file:** `plugins/polars-statistics/skills/polars-statistics-hypothesis-tests/SKILL.md`

**Symbol count: 44 expression functions** (all sourced from `exprs/parametric.py`, `exprs/nonparametric.py`, `exprs/distributional.py`, `exprs/correlation.py`, `exprs/categorical.py`, `exprs/modern.py`, `exprs/tost.py`)

### Group A — Parametric tests (exprs/parametric.py)

All return `struct{statistic: f64, p_value: f64}` unless noted.

```python
ps.ttest_ind(x, y, alternative="two-sided", equal_var=False, mu=0.0, conf_level=0.95)
ps.ttest_paired(x, y, alternative="two-sided", mu=0.0, conf_level=0.95)
ps.brown_forsythe(x, y)
ps.yuen_test(x, y, trim=0.2, alternative="two-sided", conf_level=0.95)
```
[VERIFIED: python/polars_statistics/exprs/parametric.py:241-480]

- `alternative` values: `"two-sided"`, `"less"`, `"greater"` [VERIFIED: python/polars_statistics/exprs/parametric.py:242]
- `ttest_ind` with `equal_var=True` uses Student's t (assumes equal variances); `False` uses Welch's (default).
- `brown_forsythe` takes only two samples (x, y); no `alternative` parameter.

**Note:** ANOVA functions are in `exprs/parametric.py` but are documented under Skill 2 (ANOVA).

### Group B — Non-parametric tests (exprs/nonparametric.py)

```python
ps.mann_whitney_u(x, y, alternative="two-sided", continuity_correction=True, exact=False, conf_level=None, mu=None)
ps.wilcoxon_signed_rank(x, y, alternative="two-sided", continuity_correction=True, exact=False, conf_level=None, mu=None)
ps.kruskal_wallis(*groups)
ps.brunner_munzel(x, y, alternative="two-sided", alpha=None)
```
[VERIFIED: python/polars_statistics/exprs/nonparametric.py:14-261]

- `kruskal_wallis` accepts 2+ group columns as positional args (same pattern as `one_way_anova`).
- `conf_level`/`mu` are `Optional[float]` defaulting to `None` — pass them only to get confidence intervals / custom location shift.

### Group C — Distributional tests (exprs/distributional.py)

```python
ps.shapiro_wilk(x)
ps.dagostino(x)
```
[VERIFIED: python/polars_statistics/exprs/distributional.py:14-105]

- `shapiro_wilk`: appropriate for n=3..5000.
- `dagostino`: requires n >= 20.
- Both return `struct{statistic: f64, p_value: f64}`.

### Group D — Correlation tests (exprs/correlation.py)

```python
ps.pearson(x, y, conf_level=0.95)
ps.spearman(x, y, conf_level=0.95)
ps.kendall(x, y, variant="b")
ps.distance_cor(x, y, n_permutations=999, seed=None)
ps.partial_cor(x, y, covariates: list)
ps.semi_partial_cor(x, y, covariates: list)
ps.icc(*rater_cols, icc_type="icc2")
```
[VERIFIED: python/polars_statistics/exprs/correlation.py:14-449]

- `pearson`/`spearman` return `struct{estimate, statistic, p_value, ci_lower, ci_upper, n}`.
- `kendall` `variant` options: `"a"`, `"b"` (default, adjusted for ties), `"c"` (Stuart's tau-c).
- `icc` `icc_type` options: `"icc1"`, `"icc2"` (default), `"icc3"`, `"icc1k"`, `"icc2k"`, `"icc3k"`. Positional args are rater columns (rows = subjects). Returns `struct{icc, f_value, df1, df2, p_value, ci_lower, ci_upper, n_subjects, n_raters}`. [VERIFIED: python/polars_statistics/exprs/correlation.py:377-449]
- `covariates` for `partial_cor`/`semi_partial_cor` is a `list` of column names or expressions.
- **ICC gotcha (from STATE.md):** The matrix-input contract changed. Current implementation: pass each rater as a separate column argument (not a matrix), rows are subjects. [VERIFIED: python/polars_statistics/exprs/correlation.py:378-393]

### Group E — Categorical tests (exprs/categorical.py)

```python
ps.binom_test(successes: int, n: int, p0=0.5, alternative="two-sided")
ps.prop_test_one(successes: int, n: int, p0=0.5, alternative="two-sided")
ps.prop_test_two(successes1: int, n1: int, successes2: int, n2: int, alternative="two-sided", correction=False)
ps.chisq_test(data, n_rows=2, n_cols=2, correction=False)
ps.chisq_goodness_of_fit(observed, expected=None)
ps.g_test(data, n_rows=2, n_cols=2)
ps.fisher_exact(a: int, b: int, c: int, d: int, alternative="two-sided")
ps.mcnemar_test(a: int, b: int, c: int, d: int, correction=False)
ps.mcnemar_exact(a: int, b: int, c: int, d: int)
ps.cohen_kappa(data, n_categories=2, weighted=False)
ps.cramers_v(data, n_rows=2, n_cols=2)
ps.phi_coefficient(a: int, b: int, c: int, d: int)
ps.contingency_coef(data, n_rows=2, n_cols=2)
```
[VERIFIED: python/polars_statistics/exprs/categorical.py:14-592]

- `binom_test`, `prop_test_one`, `prop_test_two`, `fisher_exact`, `mcnemar_test`, `mcnemar_exact`, `phi_coefficient` take scalar integer arguments (not column references) — these are called without a DataFrame context (e.g., `ps.binom_test(7, 10)` directly).
- `chisq_test`, `g_test`, `cohen_kappa`, `cramers_v`, `contingency_coef` take a flattened contingency table column (`data`).
- `chisq_goodness_of_fit(observed, expected=None)`: if `expected` is None, assumes uniform distribution.

### Group F — Modern tests (exprs/modern.py)

```python
ps.energy_distance(x, y, n_permutations=999, seed=None)
ps.energy_distance_nd(x_cols: list, y_cols: list, n_permutations=999, seed=None)
ps.mmd_test(x, y, n_permutations=999, seed=None)
```
[VERIFIED: python/polars_statistics/exprs/modern.py:22-202]

- `energy_distance_nd`: `x_cols` and `y_cols` must have equal length (the dimension d). Raises `ValueError` if lengths differ or if either list is empty.
- All return `struct{statistic, p_value}`.
- `seed` is `int | None` (default `None`). For reproducibility pass any integer.

### Group G — TOST equivalence tests (exprs/tost.py)

```python
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
[VERIFIED: python/polars_statistics/exprs/tost.py:14-645]

- `bounds_type` options: `"symmetric"` (uses `delta` as ±bound), `"raw"` (uses `lower`/`upper` directly). `tost_t_test_*` also supports `"cohen_d"`.
- `tost_correlation` `method` options: `"pearson"`, `"spearman"`.
- All TOST functions return a struct with: `estimate, ci_lower, ci_upper, bound_lower, bound_upper, tost_p_value, equivalent, alpha, n`.

### Group H — Forecast comparison tests (exprs/forecast.py)

**These are in `__all__` but their family affiliation is "forecast comparison", not a standard hypothesis test family.** They are documented under this skill for discoverability since they are not ANOVA or regression. An agent may encounter `diebold_mariano`, `permutation_t_test`, `clark_west`, `spa_test`, `model_confidence_set`, `mspe_adjusted` and need routing.

```python
ps.diebold_mariano(e1, e2, loss="squared", horizon=1, alternative="two-sided", varestimator="acf")
ps.permutation_t_test(x, y, alternative="two-sided", n_permutations=999, seed=None)
ps.clark_west(e1, e2, f1, f2, horizon=1, alternative="two-sided")
ps.spa_test(losses_matrix, n_bootstrap=1000, seed=None)
ps.model_confidence_set(losses_matrix, alpha=0.1, n_bootstrap=1000, seed=None)
ps.mspe_adjusted(e1, e2, f1, f2)
```
[VERIFIED: python/polars_statistics/__init__.py:339-347 (these are in `__all__`); signatures ASSUMED from forecast.py — author must read remaining forecast.py lines before writing the skill file]

**Action for planner:** The research reading stopped at line 80 of forecast.py. Before the author writes the full signatures for `clark_west`, `spa_test`, `model_confidence_set`, `mspe_adjusted`, they must read the rest of `python/polars_statistics/exprs/forecast.py` during authoring.

### PyModel test classes (from `_polars_statistics.pyi`)

These parallel the expression forms. They take numpy arrays and are for direct imperative use (not group_by). Include them in a brief "PyModel alternative" subsection of hypothesis-tests:

```python
TTestInd(alternative="two-sided", equal_var=False, mu=0.0, conf_level=0.95)
  .fit(x, y) -> TTestInd
  .statistic, .p_value, .df, .conf_int, .mean_x, .mean_y, .null_value, .is_fitted()
  .summary() -> str, .to_dict() -> dict

TTestPaired(alternative="two-sided", mu=0.0, conf_level=0.95)
  .fit(x, y) -> TTestPaired
  .statistic, .p_value, .df, .conf_int, .mean_x, .mean_y, .null_value, .is_fitted()

BrownForsythe()
  .fit(x, y) -> BrownForsythe
  .statistic, .p_value

YuenTest(trim=0.2, alternative="two-sided", conf_level=0.95)
  .fit(x, y) -> YuenTest
  .statistic, .p_value, .df, .diff, .trimmed_mean_x, .trimmed_mean_y, .conf_int

MannWhitneyU(alternative="two-sided", continuity_correction=True, exact=False, conf_level=None, mu=None)
  .fit(x, y) -> MannWhitneyU
  .statistic, .p_value, .estimate, .conf_int, .null_value

WilcoxonSignedRank(alternative="two-sided", continuity_correction=True, exact=False, conf_level=None, mu=None)
  .fit(x, y) -> WilcoxonSignedRank
  .statistic, .p_value, .estimate, .conf_int, .null_value

KruskalWallis()
  .fit(*groups) -> KruskalWallis   # note: *groups varargs
  .statistic, .p_value

BrunnerMunzel(alternative="two-sided", alpha=None)
  .fit(x, y) -> BrunnerMunzel
  .statistic, .p_value, .df, .estimate, .conf_int

ShapiroWilk()
  .fit(x) -> ShapiroWilk           # note: single array, not two
  .statistic, .p_value

DAgostino()
  .fit(x) -> DAgostino             # note: single array
  .statistic, .p_value
```
[VERIFIED: python/polars_statistics/_polars_statistics.pyi:788-931]

---

## Skill 2: polars-statistics-anova

**Description field:** `Use polars-statistics ANOVA — run one-way ANOVA (Fisher or Welch), two-way ANOVA, repeated-measures ANOVA, or intraclass correlation coefficient (ICC) as Polars expressions.`

**Skill file:** `plugins/polars-statistics/skills/polars-statistics-anova/SKILL.md`

**Symbol count: 4 expression functions**

### ANOVA expressions (exprs/parametric.py)

```python
ps.one_way_anova(*groups, kind="fisher")
ps.two_way_anova(value, factor_a, factor_b)
ps.repeated_measures_anova(value, subject, condition, compute_sphericity=True)
```
[VERIFIED: python/polars_statistics/exprs/parametric.py:14-237]

And from correlation.py:

```python
ps.icc(*rater_cols, icc_type="icc2")
```
[VERIFIED: python/polars_statistics/exprs/correlation.py:377-449]

### one_way_anova

```python
ps.one_way_anova(*groups, kind="fisher")
# groups: 2+ column names or expressions, each holding one group's values
# kind: "fisher" (default) or "welch"
```

Returns struct: `{statistic, df_between, df_within, p_value, ss_between, ss_within, ms_between, ms_within, eta_squared, n_groups}`.

**Critical gotcha:** Do NOT use inside `group_by(...).agg()` unless each Polars group contains ALL ANOVA groups — this expression already aggregates across the named group columns. The input is wide (one column per group). [VERIFIED: python/polars_statistics/exprs/parametric.py:24-30]

Welch variant returns NaN for `ss_between`, `ss_within`, `ms_between`, `ms_within`, `eta_squared`. [VERIFIED: python/polars_statistics/exprs/parametric.py:33-38]

### two_way_anova

```python
ps.two_way_anova(value, factor_a, factor_b)
# value: continuous outcome column
# factor_a, factor_b: categorical factor columns (str or int — auto label-encoded)
```

Returns struct (flattened AnovaTableRow): `{a_ss, a_df, a_ms, a_f, a_p_value, b_ss, b_df, b_ms, b_f, b_p_value, ab_ss, ab_df, ab_ms, ab_f, ab_p_value, residual_ss, residual_df, residual_ms, grand_mean, n}`.

**Gotcha:** String factors are automatically label-encoded via `rank(method="dense")` after null-filtering. Do not pre-encode; doing so may break dense codes. [VERIFIED: python/polars_statistics/exprs/parametric.py:121-143]

### repeated_measures_anova

```python
ps.repeated_measures_anova(value, subject, condition, compute_sphericity=True)
# value: observed values (long format)
# subject: subject identifier column (str or int)
# condition: condition/time-point column (str or int)
# compute_sphericity: bool, default True (requires k >= 3 conditions)
```

Returns struct: `{ws_f, ws_df, ws_ss, ws_ms, ws_p_value, error_df, error_ss, error_ms, mauchly_w, mauchly_p_value, gg_epsilon, gg_p_value, hf_epsilon, hf_p_value, grand_mean}`.

**Requires balanced design:** every subject must have exactly the same set of conditions. Unbalanced input returns all-NaN struct without panicking. [VERIFIED: python/polars_statistics/exprs/parametric.py:159-164]

**Gotcha:** Subject and condition are encoded with `rank("dense") - 1` after filtering. Do not pre-encode; the same issue with categorical catalog applies as for two_way_anova. [VERIFIED: python/polars_statistics/exprs/parametric.py:218-225]

### icc (Intraclass Correlation)

```python
ps.icc(*rater_cols, icc_type="icc2")
# *rater_cols: one column per rater (rows = subjects), 2+ required
# icc_type: "icc1", "icc2" (default), "icc3", "icc1k", "icc2k", "icc3k"
```

Returns struct: `{icc, f_value, df1, df2, p_value, ci_lower, ci_upper, n_subjects, n_raters}`.

**Historical gotcha (documented in STATE.md/CONTEXT.md):** The matrix-input contract changed. The current API takes each rater as a separate positional column argument — it does NOT accept a single matrix column. Passing a matrix column will give wrong results. [VERIFIED: python/polars_statistics/exprs/correlation.py:378-394]

ICC type lookup:
- `icc1`: One-way random, single rater
- `icc2`: Two-way random, single rater (most common; default)
- `icc3`: Two-way mixed, single rater
- `icc1k/2k/3k`: Same models but for average of k raters (higher reliability)

---

## Skill 3: polars-statistics-regression

**Description field:** `Use polars-statistics regression — fit OLS, Ridge, Lasso, Elastic Net, WLS, RLS, BLS, NNLS, Quantile, Huber, PLS, ALM, or any GLM (logistic, Poisson, Negative Binomial, Tweedie, Probit, Cloglog) as Polars expressions or as PyModel classes; parse R-style formulas.`

**Skill file:** `plugins/polars-statistics/skills/polars-statistics-regression/SKILL.md`

**Symbol count: 43 PyModel classes + 56 expression functions** (fit/predict/summary/formula variants) + formula parser

### PyModel Classes (from `_polars_statistics.pyi`)

All PyModel classes share this common interface pattern:
- Constructor `__init__(...)` with hyperparameters
- `.fit(x, y)` → returns self (chainable), where `x` is numpy array shape `(n, p)` and `y` is shape `(n,)`
- `.predict(x)` → `numpy.ndarray`
- `.score(x, y)` → `float` (R²)
- `.is_fitted()` → `bool`
- `.summary()` → `str`
- `.to_dict()` → `dict`

#### Group 1 — Linear Models

```python
OLS(add_intercept=None, with_intercept=None, compute_inference=True, confidence_level=0.95, solve_method=None)
  .fit(x, y) -> OLS
  .fit_from_accumulator(acc) -> OLS      # acc is MomentAccumulator
  .hc_inference(x, hc_type="hc1") -> dict
  .coefficients, .intercept, .adj_r_squared, .r_squared, .mse, .rmse
  .f_statistic, .f_pvalue, .aic, .bic, .log_likelihood
  .p_values, .std_errors, .t_statistics
  .n_observations, .n_parameters
  .fitted_values, .residuals

Ridge(lambda_=1.0, add_intercept=None, with_intercept=None, compute_inference=True, confidence_level=0.95)
  .fit(x, y) -> Ridge
  .fit_from_accumulator(acc) -> Ridge
  .hc_inference(x, hc_type="hc1") -> dict
  .coefficients, .intercept, .r_squared, .adj_r_squared, .lambda_value

ElasticNet(lambda_=1.0, alpha=0.5, add_intercept=None, with_intercept=None, max_iter=1000, tol=1e-4)
  .fit(x, y) -> ElasticNet
  .coefficients, .intercept, .r_squared, .lambda_value, .alpha_value

WLS(add_intercept=None, with_intercept=None, compute_inference=True, confidence_level=0.95)
  .fit(x, y, weights) -> WLS            # note: 3-arg fit (weights required)
  .hc_inference(x, hc_type="hc1") -> dict
  .coefficients, .intercept, .r_squared

RLS(forgetting_factor=1.0, add_intercept=None, with_intercept=None)
  .fit(x, y) -> RLS
  .coefficients, .intercept, .r_squared, .forgetting_factor_value

BLS(lower_bound=None, upper_bound=None, add_intercept=None, with_intercept=None, max_iter=1000, tol=1e-6)
  .fit(x, y) -> BLS
  .nnls(with_intercept=True, max_iter=1000, tol=1e-6) -> BLS  # convenience for lower_bound=0
  .coefficients, .intercept, .r_squared
```
[VERIFIED: python/polars_statistics/_polars_statistics.pyi:524-769]

#### Group 2 — Robust & Sklearn-style Solvers

```python
Quantile(tau=0.5, add_intercept=None, with_intercept=None, max_iterations=100, tolerance=1e-6)
  # note: max_iterations (not max_iter), tolerance (not tol)
  .fit(x, y) -> Quantile
  .coefficients, .intercept, .tau, .pseudo_r_squared, .check_loss, .fitted_values, .residuals

Isotonic(increasing=True, out_of_bounds="clip")
  .fit(x, y) -> Isotonic
  .fit_2d(x, y) -> Isotonic
  .predict(x), .predict_2d(x)
  .fitted_values, .residuals, .r_squared, .is_increasing, .x_thresholds, .y_values

Huber(epsilon=1.35, alpha=0.0001, add_intercept=None, with_intercept=None, max_iter=100, tol=1e-5)
  .fit(x, y) -> Huber
  .coefficients, .intercept, .r_squared, .mse, .rmse, .scale
  .n_outliers, .outliers, .epsilon

PLS(n_components=2, add_intercept=None, with_intercept=None, tol=1e-6, scale=True)
  .fit(x, y) -> PLS
  .transform(x) -> numpy.ndarray
  .coefficients, .intercept, .r_squared, .n_components, .explained_variance_ratio

TheilSen(add_intercept=None, with_intercept=None, max_subpopulation=10000, n_subsamples=None, max_iter=300, tol=1e-3, random_state=0)
  .fit(x, y) -> TheilSen
  .coefficients, .intercept, .r_squared, .mse, .rmse, .residuals

RANSAC(add_intercept=None, with_intercept=None, min_samples=None, residual_threshold=None, max_trials=100, stop_probability=0.99, stop_n_inliers=None, random_state=0)
  .fit(x, y) -> RANSAC
  .coefficients, .intercept, .r_squared, .inlier_mask, .n_inliers, .n_trials, .residuals

BayesianRidge(fit_intercept=True, max_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6, alpha_init=None, lambda_init=None)
  .fit(x, y) -> BayesianRidge
  .coefficients, .intercept, .r_squared, .residuals, .alpha_, .lambda_, .sigma_diag

ARD(fit_intercept=True, max_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6, threshold_lambda=10000.0)
  .fit(x, y) -> ARD
  .coefficients, .intercept, .r_squared, .residuals, .alpha_, .lambdas
```
[VERIFIED: python/polars_statistics/_polars_statistics.pyi:300-730]

#### Group 3 — Streaming & Online

```python
LARS(method="lar", fit_intercept=True, n_nonzero_coefs=None, alpha=0.0, standardize=False)
  .fit(x, y) -> LARS
  .coefficients, .intercept, .r_squared, .alphas

PassiveAggressive(c=1.0, epsilon=0.1, add_intercept=None, with_intercept=None, max_iter=1000, tol=1e-3, shuffle=True, loss="epsilon_insensitive", random_state=0)
  .fit(x, y) -> PassiveAggressive
  .partial_fit(x_row, y_value) -> None    # online learning
  .predict_from_state(x) -> numpy.ndarray
  .coefficients, .intercept, .r_squared, .n_iter

MomentAccumulator(n_features)
  .push_row(x_row, y) -> None
  .merge(other) -> None
  .clear() -> None
  .n, .n_features, .sum_x, .sum_y, .xtx, .xty
  # Used with OLS.fit_from_accumulator(acc) / Ridge.fit_from_accumulator(acc)
```
[VERIFIED: python/polars_statistics/_polars_statistics.pyi:365-500]

#### Group 4 — Smoothers

```python
PSpline(n_basis=0, penalty_order=2, lambda_=None)
  .fit(x, y) -> PSpline
  .predict(x) -> numpy.ndarray
  .coefficients, .r_squared, .rmse, .edf, .sigma2
```
[VERIFIED: python/polars_statistics/_polars_statistics.pyi:593-610]

#### Group 5 — GLM Models

```python
Logistic(add_intercept=None, with_intercept=None, compute_inference=True, confidence_level=0.95, max_iter=25, tol=1e-8, lambda_=0.0)
  .fit(x, y) -> Logistic
  .predict(x, threshold=0.5) -> numpy.ndarray
  .predict_proba(x) -> numpy.ndarray
  .predict_linear(x) -> numpy.ndarray
  .score(x, y, threshold=0.5) -> float
  .coefficients, .intercept, .aic, .bic, .p_values, .std_errors
  .deviance_residuals, .pearson_residuals

LogisticRegression(penalty=..., C=1.0, threshold=0.5, add_intercept=None, with_intercept=None, max_iter=100, tol=1e-8, compute_inference=True, confidence_level=0.95)
  # sklearn-style: inverse-strength C, explicit penalty
  .fit(x, y) -> LogisticRegression
  .decision_function(x) -> numpy.ndarray
  .predict(x) -> numpy.ndarray
  .predict_proba(x) -> numpy.ndarray
  .score(x, y) -> float
  .n_iter

Poisson(add_intercept=None, with_intercept=None, compute_inference=True, confidence_level=0.95, max_iter=25, tol=1e-8, lambda_=0.0)
  .fit(x, y) -> Poisson
  .predict(x) -> numpy.ndarray
  .predict_linear(x) -> numpy.ndarray
  .coefficients, .intercept, .aic, .bic, .p_values, .std_errors
  .deviance_residuals, .pearson_residuals

NegativeBinomial(theta=None, estimate_theta=True, add_intercept=None, with_intercept=None, max_iter=100, tol=1e-6, lambda_=0.0)
  .fit(x, y) -> NegativeBinomial
  .predict(x) -> numpy.ndarray
  .coefficients, .intercept, .aic, .bic, .p_values, .std_errors, .theta_estimated

Tweedie(var_power=1.5, link_power=None, add_intercept=None, with_intercept=None, max_iter=100, tol=1e-6, lambda_=0.0)
  .fit(x, y) -> Tweedie
  .gaussian(with_intercept=True, max_iter=100, tol=1e-6, lambda_=0.0) -> Tweedie   # factory
  .gamma(with_intercept=True, ...) -> Tweedie
  .inverse_gaussian(with_intercept=True, ...) -> Tweedie
  .predict(x) -> numpy.ndarray
  .coefficients, .intercept, .aic, .bic, .p_values, .std_errors, .var_power_value

Probit(add_intercept=None, with_intercept=None, max_iter=100, tol=1e-6, lambda_=0.0)
  .fit(x, y) -> Probit
  .predict(x) -> numpy.ndarray
  .predict_proba(x) -> numpy.ndarray
  .score(x, y, threshold=0.5) -> float
  .coefficients, .intercept, .aic, .bic, .p_values, .std_errors

Cloglog(add_intercept=None, with_intercept=None, max_iter=100, tol=1e-6, lambda_=0.0)
  .fit(x, y) -> Cloglog
  .predict(x) -> numpy.ndarray
  .predict_proba(x) -> numpy.ndarray
  .score(x, y, threshold=0.5) -> float
  .coefficients, .intercept, .aic, .bic, .p_values, .std_errors

Gamma(add_intercept=None, with_intercept=None, max_iter=25, tol=1e-8, lambda_=0.0)
  .fit(x, y) -> Gamma
  .predict(x) -> numpy.ndarray
  .predict_eta(x) -> numpy.ndarray    # linear predictor
  .coefficients, .intercept, .aic, .bic, .p_values, .std_errors, .converged
```
[VERIFIED: python/polars_statistics/_polars_statistics.pyi:187-298, 421-678]

#### Group 6 — Mixed & Augmented Models

```python
GLMM()
  # No __init__ constructor params — configure via builder methods:
  .gaussian(add_intercept=None, with_intercept=None, reml=True, max_iter=100, tol=1e-8) -> GLMM
  .binomial(add_intercept=None, with_intercept=None, reml=False, max_iter=100, tol=1e-8) -> GLMM
  .poisson(add_intercept=None, with_intercept=None, reml=False, max_iter=100, tol=1e-8) -> GLMM
  .fit(x, y, group) -> GLMM          # group: Vec<u64> (integer group IDs)
  .fit_crossed(x, y, groups) -> GLMM # multiple crossed random effects
  .predict_fixed(x) -> numpy.ndarray
  .fixed_effects, .intercept, .random_effects, .random_sd, .slopes
  .sigma, .theta, .deviance, .log_likelihood
  .n_groups, .iterations, .converged, .std_errors
  .factors() -> list[dict]

ALM(distribution="normal", link=None, loss="likelihood", role_trim=None, add_intercept=None, with_intercept=None, compute_inference=True, confidence_level=0.95, max_iter=100, tol=1e-8, extra_parameter=None)
  # 25 supported distributions (see alm() expression for full list)
  .fit(x, y) -> ALM
  .predict(x) -> numpy.ndarray
  .score(x, y) -> float
  .coefficients, .intercept, .aic, .bic, .scale
  .p_values, .std_errors, .fitted_values, .residuals, .n_observations
  .distribution, .link_function
  # Builder methods for common distributions:
  .normal(add_intercept=None, with_intercept=None, compute_inference=True) -> ALM
  .gamma(add_intercept=None, with_intercept=None, compute_inference=True) -> ALM
  .laplace(add_intercept=None, with_intercept=None, compute_inference=True) -> ALM
  .poisson(add_intercept=None, with_intercept=None, compute_inference=True) -> ALM
  .student_t(df=5.0, add_intercept=None, with_intercept=None, compute_inference=True) -> ALM
  # beta(), normal() pattern
```
[VERIFIED: python/polars_statistics/_polars_statistics.pyi:9-46, 239-276]

#### Group 7 — Dynamic / Demand Models

```python
LmDynamic(ic="aicc", distribution="normal", lowess_span=..., max_models=..., add_intercept=None, with_intercept=None, confidence_level=0.95)
  .fit(x, y) -> LmDynamic
  .predict(x) -> numpy.ndarray
  .score(x, y) -> float
  .coefficients, .intercept, .r_squared, .adj_r_squared, .mse, .rmse
  .fitted_values, .residuals, .dynamic_coefficients, .model_weights, .pointwise_ic
  .smoothed_weights, .n_observations
  .coefficient_at(obs_index, coef_index) -> float | None
  .coefficients_at(obs_index) -> numpy.ndarray | None

Aid(intermittent_threshold=0.3, detect_anomalies=True, ic="aicc")
  .classify(y) -> AidResult          # not .fit() — uses .classify()
  .summary() -> str, .to_dict() -> dict

AidResult
  .demand_type, .distribution, .mean, .variance, .scale, .shape, .zero_prob, .zero_proportion
  .is_fractional, .n_observations, .ic_values, .anomalies
  .is_intermittent(), .is_regular(), .is_new_product(), .is_obsolete_product()
  .anomaly_counts() -> dict, .has_stockouts() -> bool, .summary(), .to_dict()
```
[VERIFIED: python/polars_statistics/_polars_statistics.pyi:71-109, 384-419]

#### Group 8 — Bootstrap

```python
StationaryBootstrap(expected_block_length, seed=None)
  .sample(data, length=None) -> numpy.ndarray
  .samples(data, n_samples, length=None) -> numpy.ndarray

CircularBlockBootstrap(block_length, seed=None)
  .sample(data, length=None) -> numpy.ndarray
  .samples(data, n_samples, length=None) -> numpy.ndarray
```
[VERIFIED: python/polars_statistics/_polars_statistics.pyi:180-185, 781-786]

### Regression Expression Functions

#### Core fit expressions

```python
ps.ols(y, *x, add_intercept=None, with_intercept=None, solve_method=None)
ps.ridge(y, *x, lambda_=1.0, add_intercept=None, with_intercept=None, solve_method=None)
ps.elastic_net(y, *x, lambda_=1.0, alpha=0.5, add_intercept=None, with_intercept=None)
ps.lasso(y, *x, lambda_=1.0, add_intercept=None, with_intercept=None)       # → elastic_net(alpha=1.0)
ps.wls(y, weights, *x, add_intercept=None, with_intercept=None, solve_method=None)  # weights arg BEFORE *x
ps.rls(y, *x, forgetting_factor=0.99, add_intercept=None, with_intercept=None)
ps.expanding_ols(y, *x, add_intercept=None, with_intercept=None)             # → rls(forgetting_factor=1.0)
ps.bls(y, *x, lower_bound=None, upper_bound=None, add_intercept=None, with_intercept=None)
ps.nnls(y, *x, add_intercept=None, with_intercept=None)                      # → bls(lower_bound=0.0)
ps.quantile(y, *x, tau=0.5, add_intercept=None, with_intercept=None)
ps.pls(y, *x, n_components=2, add_intercept=None, with_intercept=None, tol=1e-6, scale=True)
ps.huber(y, *x, epsilon=1.35, alpha=0.0001, add_intercept=None, with_intercept=None, max_iter=100, tol=1e-5)
ps.isotonic(y, x, increasing=True)  # note: single x (not *x)
# GLMs
ps.logistic(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.logistic_regression(y, *x, penalty="l2", C=1.0, threshold=0.5, add_intercept=None, with_intercept=None, max_iter=100, tol=1e-8)
ps.poisson(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.negative_binomial(y, *x, theta=None, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.tweedie(y, *x, var_power=1.5, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.probit(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.cloglog(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.alm(y, *x, distribution="normal", link=None, loss="likelihood", role_trim=None, extra_parameter=None, add_intercept=None, with_intercept=None)
# Streaming
ps.lm_dynamic(y, *x, ic="aicc", distribution="normal", lowess_span=None, max_models=None, add_intercept=None, with_intercept=None, confidence_level=0.95)
ps.aid(y, intermittent_threshold=0.3, detect_anomalies=True, ic="aicc")
ps.aid_anomalies(y, intermittent_threshold=0.3, detect_anomalies=True, ic="aicc")
```
[VERIFIED: python/polars_statistics/exprs/regression.py:51-2153]

**Intercept deprecation:** `with_intercept` is deprecated — use `add_intercept`. Both accepted for backward compatibility; the FutureWarning fires from `_resolve_intercept`. [VERIFIED: python/polars_statistics/exprs/regression.py:18-43]

#### Summary expressions (tidy coefficient output)

All `*_summary` variants return `List[Struct]` with fields: `{term, estimate, std_error, statistic, p_value}`. Use `.explode(col).unnest(col)` to get a tidy DataFrame.

```python
ps.ols_summary(y, *x, add_intercept=None, with_intercept=None, solve_method=None, hc_type=None)
ps.ridge_summary(y, *x, lambda_=1.0, add_intercept=None, with_intercept=None, solve_method=None)
ps.elastic_net_summary(y, *x, lambda_=1.0, alpha=0.5, add_intercept=None, with_intercept=None)
ps.lasso_summary(y, *x, lambda_=1.0, add_intercept=None, with_intercept=None)
ps.wls_summary(y, weights, *x, add_intercept=None, with_intercept=None, solve_method=None)
ps.rls_summary(y, *x, forgetting_factor=0.99, add_intercept=None, with_intercept=None)
ps.bls_summary(y, *x, lower_bound=None, upper_bound=None, add_intercept=None, with_intercept=None)
ps.logistic_summary(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.poisson_summary(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.negative_binomial_summary(y, *x, theta=None, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.tweedie_summary(y, *x, var_power=1.5, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.probit_summary(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.cloglog_summary(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.alm_summary(y, *x, distribution="normal", add_intercept=None, with_intercept=None)
ps.quantile_summary(y, *x, tau=0.5, add_intercept=None, with_intercept=None)  # in __all__
```
[VERIFIED: python/polars_statistics/exprs/regression.py:2161-2638]

#### Prediction expressions

All `*_predict` variants return per-row struct: `{<name>_prediction, <name>_lower, <name>_upper}`. The `name` kwarg customizes the prefix.

```python
ps.ols_predict(y, *x, add_intercept=True, solve_method=None, interval=None, level=0.95, null_policy="drop", name=None)
ps.ridge_predict(y, *x, lambda_=1.0, add_intercept=True, solve_method=None, interval=None, level=0.95, null_policy="drop", name=None)
ps.elastic_net_predict(y, *x, lambda_=1.0, alpha=0.5, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.lasso_predict(y, *x, lambda_=1.0, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.wls_predict(y, weights, *x, add_intercept=True, solve_method=None, interval=None, level=0.95, null_policy="drop", name=None)
ps.rls_predict(y, *x, forgetting_factor=0.99, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.bls_predict(y, *x, lower_bound=None, upper_bound=None, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.nnls_predict(y, *x, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.logistic_predict(y, *x, lambda_=0.0, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.poisson_predict(y, *x, lambda_=0.0, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
# and: negative_binomial_predict, tweedie_predict, probit_predict, cloglog_predict, alm_predict, quantile_predict
ps.isotonic_predict(y, x, increasing=True)   # no intervals; single x
ps.lm_dynamic_predict(y, *x, ...)
```
[VERIFIED: python/polars_statistics/exprs/regression.py:2646-3250+ (predict series)]

**`null_policy` options:** `"drop"` (default — drops nulls for fitting, NaN predictions for null rows), `"drop_y_zero_x"` (drop null-target rows for fitting, zero-fill null features, predict on all rows). [VERIFIED: python/polars_statistics/exprs/regression.py:2678-2682]

**`interval` options:** `None` (no intervals), `"confidence"`, `"prediction"`.

#### Formula-based expressions

The formula API parses R-style formula strings and constructs the appropriate Polars expression. Syntax: `"y ~ x1 + x2"` (main effects), `"y ~ x1 * x2"` (crossing = main + interaction), `"y ~ x1:x2"` (interaction only), `"y ~ poly(x, 2)"`, `"y ~ I(x^2)"`.

```python
ps.ols_formula(formula: str, add_intercept=True)
ps.ridge_formula(formula: str, lambda_=1.0, add_intercept=True)
ps.elastic_net_formula(formula: str, lambda_=1.0, alpha=0.5, add_intercept=True)
ps.lasso_formula(formula: str, lambda_=1.0, add_intercept=True)
ps.wls_formula(formula: str, add_intercept=True)      # note: WLS formula needs a weights column reference
ps.rls_formula(formula: str, forgetting_factor=0.99, add_intercept=True)
ps.bls_formula(formula: str, lower_bound=None, upper_bound=None, add_intercept=True)
ps.nnls_formula(formula: str, add_intercept=True)
ps.logistic_formula(formula: str, lambda_=0.0, add_intercept=True)
ps.poisson_formula(formula: str, lambda_=0.0, add_intercept=True)
ps.negative_binomial_formula(formula: str, add_intercept=True)   # [ASSUMED - verify in source]
ps.tweedie_formula(formula: str, var_power=1.5, lambda_=0.0, add_intercept=True)
ps.probit_formula(formula: str, lambda_=0.0, add_intercept=True)
ps.cloglog_formula(formula: str, lambda_=0.0, add_intercept=True)
ps.alm_formula(formula: str, distribution="normal", add_intercept=True)
# Formula summary/predict variants:
ps.ols_formula_summary(formula: str, ...)
ps.ols_formula_predict(formula: str, ...)
# Similarly for ridge, elastic_net, lasso, logistic, poisson, alm
```
[VERIFIED: python/polars_statistics/__init__.py:467-497 (formula summary/predict in `__all__`)]

**Action for planner:** The exact signatures for formula expressions are in `exprs/regression.py` lines ~2750-4760 (not fully read). The author must read the remaining formula expression definitions during authoring, particularly the `*_formula_predict` signatures.

#### Diagnostics expressions

```python
ps.condition_number(*x, add_intercept=None, with_intercept=None)
ps.vif(*x)
ps.leverage(*x, add_intercept=None, with_intercept=None)
ps.cooks_distance(y, *x, add_intercept=None, with_intercept=None)
ps.dffits(y, *x, add_intercept=None, with_intercept=None)
ps.standardized_residuals(y, *x, add_intercept=None, with_intercept=None)
ps.studentized_residuals(y, *x, add_intercept=None, with_intercept=None)
ps.externally_studentized_residuals(y, *x, add_intercept=None, with_intercept=None)
ps.residual_outliers(y, *x, threshold=2.0, add_intercept=None, with_intercept=None)
ps.influential_cooks(y, *x, threshold=None, add_intercept=None, with_intercept=None)
ps.influential_dffits(y, *x, threshold=None, add_intercept=None, with_intercept=None)
ps.high_leverage_points(*x, threshold=None, add_intercept=True)  # no y arg
ps.high_vif_predictors(*x, threshold=10.0)
ps.generalized_vif(*x, group_sizes: list[int])
ps.check_binary_separation(y, *x)
ps.check_count_sparsity(y, *x)
# GLM residual diagnostics:
ps.logistic_pearson_residuals(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.logistic_deviance_residuals(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.logistic_working_residuals(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.poisson_pearson_residuals(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.poisson_deviance_residuals(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.poisson_working_residuals(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.pearson_chi_squared_logistic(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.pearson_chi_squared_poisson(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.gamma_dispersion_deviance(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.gamma_dispersion_pearson(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.gamma_pearson_chi_squared(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.gamma_standardized_pearson_residuals(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.gamma_standardized_deviance_residuals(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
```
[VERIFIED: python/polars_statistics/exprs/regression.py:724-1688]

### R-style Formula Parser

```python
from polars_statistics.formula.parser import FormulaParser
from polars_statistics.formula.expander import expand_terms_to_expressions

parser = FormulaParser()
parsed = parser.parse("y ~ x1 + x2 + x1:x2")
# parsed.response -> "y"
# parsed.terms -> list of Term objects
exprs, names = expand_terms_to_expressions(parsed.terms)
```
[VERIFIED: python/polars_statistics/formula/parser.py:19-60; python/polars_statistics/formula/expander.py:18-60]

Formula syntax:
- `y ~ x1 + x2` — main effects
- `y ~ x1 * x2` — full crossing (expands to `x1 + x2 + x1:x2`)
- `y ~ x1:x2` — interaction only
- `y ~ poly(x, 2)` — polynomial (Legendre orthogonal, raw=False by default)
- `y ~ poly(x, 2, raw=True)` — raw polynomial powers
- `y ~ I(x^2)` — explicit transform

End users typically use formula expressions (`ps.ols_formula("y ~ x1 + x2")`) rather than calling the parser directly.

---

## Skill 4: polars-statistics-results

**Description field:** `Consume polars-statistics expression results — extract struct fields, unnest result columns, call to_dict() and summary() on PyModel classes, handle PyRuntimeError for unfitted models, or work with the typed .pyi API.`

**Skill file:** `plugins/polars-statistics/skills/polars-statistics-results/SKILL.md`

**Symbol count: 2 helper functions + universal PyModel methods**

### results.py helpers

```python
ps.unnest(
    df: pl.DataFrame,
    column: str | None = None,
    *,
    prefix: str | None = None,
    keep_others: bool = True,
) -> pl.DataFrame
```
[VERIFIED: python/polars_statistics/results.py:67-122]

```python
ps.struct_to_dict(
    df: pl.DataFrame,
    column: str | None = None,
) -> dict[str, Any] | list[dict[str, Any]]
```
[VERIFIED: python/polars_statistics/results.py:125-163]

**Behavior notes:**
- If `column=None` and the DataFrame has exactly one Struct column, it is auto-detected.
- If `column=None` and there are 0 or 2+ Struct columns, raises `ValueError` with a helpful message listing available columns.
- `struct_to_dict` returns a single `dict` for 1-row DataFrames (the common case for whole-frame tests), or `list[dict]` for multi-row (grouped results).
- `unnest(prefix=...)` renames produced fields to `f"{prefix}{field}"` — use this to disambiguate two result structs unnested into the same frame.

### Native Polars struct access (without helpers)

```python
# Access individual field without unnesting:
df.select(ps.ttest_ind("a", "b").alias("t"))["t"].struct.field("p_value")

# Unnest all fields:
df.select(ps.ttest_ind("a", "b").alias("t")).unnest("t")

# Grouped result — explode then unnest:
df.group_by("g").agg(ps.ols_summary("y", "x").alias("coef")).explode("coef").unnest("coef")
```

### PyModel universal methods

Every PyModel class exposes:
```python
.summary() -> str          # human-readable coefficient table
.to_dict() -> dict         # machine-readable {field: value}
.is_fitted() -> bool       # check before accessing fitted properties
```

**Error pattern:** Accessing a fitted property (`.coefficients`, `.r_squared`, etc.) on an unfitted model raises `PyRuntimeError("Model not fitted")`. Always guard with `is_fitted()` or wrap in try/except if calling conditionally.

### Output struct field reference

For reference during cheat-sheet authoring:

| Expression | Key output fields |
|------------|-------------------|
| `ttest_ind`, `ttest_paired` | `statistic`, `p_value` |
| `pearson`, `spearman`, `kendall` | `estimate`, `statistic`, `p_value`, `ci_lower`, `ci_upper`, `n` |
| `icc` | `icc`, `f_value`, `df1`, `df2`, `p_value`, `ci_lower`, `ci_upper`, `n_subjects`, `n_raters` |
| `one_way_anova` | `statistic`, `df_between`, `df_within`, `p_value`, `ss_between`, `ss_within`, `ms_between`, `ms_within`, `eta_squared`, `n_groups` |
| `repeated_measures_anova` | `ws_f`, `ws_df`, `ws_ss`, `ws_ms`, `ws_p_value`, `error_df/ss/ms`, `mauchly_w`, `mauchly_p_value`, `gg_epsilon`, `gg_p_value`, `hf_epsilon`, `hf_p_value`, `grand_mean` |
| `ols` | `intercept`, `coefficients`, `r_squared`, `adj_r_squared`, `mse`, `rmse`, `f_statistic`, `f_pvalue`, `aic`, `bic`, `n_observations` |
| `ols_predict` | `ols_prediction`, `ols_lower`, `ols_upper` (customizable with `name=`) |
| `ols_summary` | `List[{term, estimate, std_error, statistic, p_value}]` |
| TOST functions | `estimate`, `ci_lower`, `ci_upper`, `bound_lower`, `bound_upper`, `tost_p_value`, `equivalent`, `alpha`, `n` |

### Typed `.pyi` API

The stub file `python/polars_statistics/_polars_statistics.pyi` provides full type information for all 43 PyModel classes. It is authoritative for property return types, method signatures, and optional vs. required parameters.

---

## Skill 5: polars-statistics-getting-started

**Description field:** `Get started with polars-statistics — install the package, understand the two API styles (expressions vs. PyModel classes), choose the right model with the decision matrix, and migrate from sklearn.`

**Skill file:** `plugins/polars-statistics/skills/polars-statistics-getting-started/SKILL.md`

### Install

```bash
pip install polars-statistics
```

Requires Python 3.9+ (all platforms). Wheels available for Linux x86_64/aarch64, macOS x86_64/ARM64, Windows x86_64. [VERIFIED: python/polars_statistics/__init__.py:258 — __version__ = "0.7.0"; 0.8.0 is the target release of this milestone]

```python
import polars_statistics as ps   # conventional alias
import polars as pl
print(ps.__version__)             # "0.8.0"
```

### Two API styles

**Expression API** — for group_by / over operations, lazy DataFrame pipelines:
```python
# Whole-frame test:
df.select(ps.ttest_ind("a", "b"))
# Per-group test:
df.group_by("cohort").agg(ps.ttest_ind("a", "b").alias("t"))
# Per-row predictions with .over():
df.with_columns(ps.ols_predict("y", "x1", "x2").over("group").alias("pred"))
```

**PyModel API** — for imperative sklearn-style workflows, fine-grained inspection:
```python
from polars_statistics import OLS
import numpy as np
X = df.select("x1", "x2").to_numpy()
y = df["y"].to_numpy()
m = OLS().fit(X, y)
print(m.r_squared, m.coefficients)
yhat = m.predict(X)
```

### Model Selection Matrix

| Task | Recommended | Notes |
|------|-------------|-------|
| Compare two group means | `ps.ttest_ind` or `TTestInd` | `equal_var=False` (Welch's) by default |
| Non-parametric two groups | `ps.mann_whitney_u` or `MannWhitneyU` | — |
| Multiple groups | `ps.one_way_anova` / `ps.kruskal_wallis` | ANOVA=parametric; KW=non-parametric |
| Two-factor design | `ps.two_way_anova` | auto-encodes string factors |
| Repeated measures | `ps.repeated_measures_anova` | balanced design required |
| Normality check | `ps.shapiro_wilk` / `ps.dagostino` | SW for n<5000; DA for n>=20 |
| Correlation | `ps.pearson` / `ps.spearman` / `ps.kendall` | kendall for small n or many ties |
| Non-linear association | `ps.distance_cor` | detects non-linear dependence |
| Equivalence testing | `ps.tost_*` | replaces significance-only testing |
| Distribution comparison | `ps.mmd_test` / `ps.energy_distance` | detects location + scale + shape |
| Linear regression (group_by) | `ps.ols` | expression form |
| Linear regression (imperative) | `OLS` | PyModel form |
| Regularized regression | `ps.ridge` / `ps.elastic_net` / `ps.lasso` | lambda_ controls strength |
| Robust regression | `ps.huber` / `ps.quantile` / `TheilSen` | outlier-resistant |
| Count data | `ps.poisson` / `ps.negative_binomial` | NB for overdispersion |
| Binary outcome | `ps.logistic` / `Logistic` / `LogisticRegression` | logistic=lower-level; LR=sklearn-style |
| Flexible GLM (25 families) | `ps.alm` / `ALM` | Augmented Linear Model |
| Mixed effects | `GLMM` | not available as expression |

### sklearn migration

```python
# sklearn                           polars-statistics equivalent
from sklearn.linear_model import (
    LinearRegression                 → OLS(add_intercept=True)
    Ridge                            → Ridge(lambda_=alpha)
    Lasso                            → ElasticNet(lambda_=alpha, alpha=1.0)
    ElasticNet                       → ElasticNet(lambda_=alpha, alpha=l1_ratio)
    HuberRegressor                   → Huber(epsilon=epsilon)
    QuantileRegressor                → Quantile(tau=quantile)
    TheilSenRegressor                → TheilSen()
    RANSACRegressor                  → RANSAC()
    BayesianRidge                    → BayesianRidge()
    ARDRegression                    → ARD()
    Lars                             → LARS(method="lar")
    LassoLars                        → LARS(method="lasso")
    PLSRegression                    → PLS(n_components=n_components)
    IsotonicRegression               → Isotonic()
    PassiveAggressiveRegressor       → PassiveAggressive()
    LogisticRegression               → LogisticRegression(C=C, penalty=penalty)
)
# Key differences:
# - Use add_intercept= instead of fit_intercept=
# - Use lambda_= instead of alpha= for regularization strength
# - .score() returns R² (same as sklearn)
# - No normalize= parameter; use StandardScaler equivalent or scale=True in PLS
```

---

## Architecture Patterns

### Expression API — group_by pattern

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({"group": ["A"]*50 + ["B"]*50, "y": ..., "x1": ..., "x2": ...})

# Per-group OLS
result = df.group_by("group").agg(
    ps.ols("y", "x1", "x2").alias("model")
)
# Extract R²:
result.with_columns(
    pl.col("model").struct.field("r_squared")
)
# Or use unnest helper:
ps.unnest(result.select("model"), "model")
```

### PyModel API — sklearn-style pattern

```python
import polars_statistics as ps
import polars as pl
import numpy as np

df = pl.read_csv("data.csv")
X = df.select("x1", "x2").to_numpy()
y = df["y"].to_numpy()

m = ps.OLS(add_intercept=True).fit(X, y)
if m.is_fitted():
    print(m.summary())
    yhat = m.predict(X)
```

### Anti-Patterns to Avoid

- **Inventing kwargs:** Never pass `normalize=True`, `fit_intercept=True` (use `add_intercept=`), or `alpha=` for regularization strength (use `lambda_=`). These do not exist and will raise `TypeError`.
- **with_intercept (deprecated):** Use `add_intercept=` in all new code. `with_intercept` still works but emits a `FutureWarning`.
- **one_way_anova inside group_by incorrectly:** Do not use `group_by(...).agg(ps.one_way_anova(...))` unless each Polars group already contains all ANOVA groups. The expression aggregates across its named column arguments.
- **icc with a matrix column:** Pass each rater as a separate positional argument. Passing a matrix in a single column gives wrong results.
- **Missing .alias():** All expression results are unnamed by default; use `.alias("name")` to name the output column before `unnest()`.
- **Accessing unfitted properties:** Raises `PyRuntimeError("Model not fitted")`. Always check `model.is_fitted()` first.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| HC-robust standard errors | custom bootstrap | `OLS.hc_inference(x, hc_type="hc1")` | Exact HC0/HC1/HC2/HC3 |
| Heteroskedastic SEs in group_by | — | `ps.ols_summary(..., hc_type="hc3")` | Built into expression |
| Tidy coefficient table | manual struct extraction | `ps.ols_summary(...)` | Returns `List[Struct]` ready for `.explode().unnest()` |
| Mauchly's sphericity test | — | `ps.repeated_measures_anova(..., compute_sphericity=True)` | Included with RM-ANOVA |
| Greenhouse-Geisser correction | — | Same `repeated_measures_anova` result struct | Included in output |
| Cook's distance / leverage | manual hat-matrix code | `ps.cooks_distance()`, `ps.leverage()`, `ps.influential_cooks()` | Built-in diagnostics |
| VIF multicollinearity check | custom regression loop | `ps.vif()`, `ps.high_vif_predictors()`, `ps.generalized_vif()` | Handles grouped predictors (GVIF) |
| Formula expansion | manual column selection | `ps.ols_formula("y ~ x1 * x2")` | Handles interactions, poly(), I() |
| Struct extraction | `.struct.field()` loop | `ps.unnest()`, `ps.struct_to_dict()` | Auto-detects struct column |
| Block bootstrap for time series | manual resampling | `StationaryBootstrap`, `CircularBlockBootstrap` | Proper block resampling |
| Online/streaming regression | incremental fit loop | `MomentAccumulator` + `OLS.fit_from_accumulator()` | Exact accumulation |

---

## Common Pitfalls

### Pitfall 1: one_way_anova used inside group_by with stacked data

**What goes wrong:** User has a "long" DataFrame (value column, group column) and tries `df.group_by("group").agg(ps.one_way_anova("value"))`. This is wrong — each Polars group has only one ANOVA group's values.

**Why it happens:** Users familiar with other statistical packages expect a "group" column pattern.

**How to avoid:** `one_way_anova` and `kruskal_wallis` take one column per ANOVA group (wide format). Pivot or pass columns directly: `df_wide.select(ps.one_way_anova("ctrl", "treat_a", "treat_b"))`.

### Pitfall 2: ICC matrix-input contract

**What goes wrong:** Passing a single matrix column to `ps.icc()` instead of separate rater columns produces wrong results without error.

**How to avoid:** Pass each rater as a positional argument: `ps.icc("rater1", "rater2", "rater3", icc_type="icc2")`.

### Pitfall 3: Welch ANOVA NaN fields

**What goes wrong:** User calls `ps.one_way_anova(..., kind="welch")` and finds NaN in `ss_between`, `eta_squared` etc.

**Why it happens:** Welch's one-way ANOVA does not compute SS components.

**How to avoid:** Expected behavior — documented in the output struct. Only `statistic`, `df_between`, `df_within`, `p_value`, `n_groups` are populated for Welch.

### Pitfall 4: Deprecated with_intercept kwarg

**What goes wrong:** `FutureWarning: Parameter 'with_intercept' is deprecated` at runtime.

**How to avoid:** Use `add_intercept=True/False` in all new code.

### Pitfall 5: categorical factor encoding in two_way_anova / repeated_measures_anova

**What goes wrong:** Pre-encoding factors with `cast(Categorical).to_physical()` before calling the expression gives wrong factor codes inside `group_by`, because the shared Polars Categorical catalog assigns non-zero-based codes across groups.

**How to avoid:** Pass raw string or integer columns. The expression internally uses `rank("dense") - 1` after filtering, which always produces contiguous 0-based codes per group.

### Pitfall 6: struct column name required after group_by

**What goes wrong:** `ps.unnest(result)` raises `ValueError: Multiple Struct columns found`.

**How to avoid:** After `group_by(...).agg(ps.ttest_ind("a","b").alias("t"), ps.ols("y","x").alias("m"))`, pass the column name explicitly: `ps.unnest(result, "t")`.

---

## Validation Architecture

> `workflow.nyquist_validation` defaults to enabled. Include.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | `pyproject.toml` (lines 103-105) |
| Quick run command | `python -c "import polars_statistics as ps; print(ps.__version__)"` |
| Full suite command | `pytest tests/ -v --tb=short` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SKILL-01 | hypothesis-tests SKILL.md accurate signatures | manual-review + spot-check | `python -c "import polars as pl, polars_statistics as ps; df=pl.DataFrame({'a':[1.,2.,3.],'b':[2.,3.,4.]}); print(ps.struct_to_dict(df.select(ps.ttest_ind('a','b').alias('t'))))"` | ❌ Wave 0: author spot-checks during writing |
| SKILL-02 | anova SKILL.md accurate signatures | manual-review + spot-check | `python -c "import polars as pl, polars_statistics as ps; df=pl.DataFrame({'g1':[1.,2.,3.],'g2':[4.,5.,6.]}); print(df.select(ps.one_way_anova('g1','g2')))"` | ❌ Wave 0 |
| SKILL-03 | regression SKILL.md accurate signatures | manual-review + spot-check | `python -c "import polars as pl, polars_statistics as ps; df=pl.DataFrame({'y':[1.,2.,3.,4.],'x':[1.,2.,3.,4.]}); print(df.select(ps.ols('y','x')))"` | ❌ Wave 0 |
| SKILL-04 | results SKILL.md accurate | manual-review + spot-check | `python -c "import polars as pl, polars_statistics as ps; df=pl.DataFrame({'y':[1.,2.,3.,4.],'x':[1.,2.,3.,4.]}); res=df.select(ps.ttest_ind('y','x').alias('t')); print(ps.struct_to_dict(res,'t'))"` | ❌ Wave 0 |
| SKILL-05 | getting-started SKILL.md install+patterns correct | manual-review | `pip show polars-statistics` | ❌ Wave 0 |
| SKILL-06 | all frontmatter valid YAML with correct fields | lint/parse | `python -c "import yaml; [yaml.safe_load(open(f)) for f in [...]]"` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** Import smoke check for the skill just written.
- **Per wave merge:** Read all five SKILL.md files and verify frontmatter parses as valid YAML.
- **Phase gate:** All five SKILL.md files authored and spot-checked; frontmatter validated; stub content replaced. Full example accuracy gate is Phase 15.

### Wave 0 Gaps

- [ ] No new test files needed — this phase is documentation-only. The "tests" are spot-checks run by the author inline during writing.
- [ ] Frontmatter YAML validation can be run as a one-liner in the plan verification step.

---

## Security Domain

> This phase writes documentation files only. No authentication, session management, input validation, cryptography, or external services involved. `security_enforcement` is not applicable.

ASVS categories: None applicable for documentation authoring.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python + polars-statistics installed | Spot-checks during authoring | See note | 0.7.0 | Read .pyi stubs for signature checks without running |
| pytest | Phase 15 full gate | Not needed here | — | — |

**Note:** The author must have the polars-statistics wheel installed (`pip install polars-statistics` or `maturin develop`) for spot-checks. If the wheel is not installed, signature accuracy must be verified solely against `python/polars_statistics/` source files and `_polars_statistics.pyi`.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `with_intercept=` | `add_intercept=` | v0.7.0 | Old name still works with FutureWarning |
| Matrix-column input for ICC | Separate-column positional args | v0.7.0 | Breaking change documented in correlation.md |
| Stub SKILL.md placeholders | Full cheat-sheet content | Phase 14 | Phase 15 verifies all examples run |

**Deprecated:**
- `with_intercept=` parameter: replaced by `add_intercept=`. Will be removed in a future release.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Forecast expression signatures for `clark_west`, `spa_test`, `model_confidence_set`, `mspe_adjusted` (research stopped at line 80 of forecast.py) | Skill 1, Group H | Wrong signatures in SKILL.md — author must read rest of forecast.py before writing |
| A2 | Formula expression signatures match the pattern of the base expressions (same kwargs, with `formula: str` as first positional arg) | Skill 3, Formula section | Minor signature differences possible — author must read formula expression definitions in regression.py ~2750-4760 |
| A3 | `negative_binomial_formula` follows same pattern as other formula variants | Skill 3 | May not exist or have different signature |

**If this table has items:** Assumptions A1, A2, A3 require the author to read specific source sections during the writing task (not before planning). The planner should include a task step "read source before writing" for each skill that covers these sections.

---

## Open Questions

1. **forecast.py remaining signatures**
   - What we know: `diebold_mariano`, `permutation_t_test` are confirmed. Four more functions are in `__all__`.
   - What's unclear: exact signatures for `clark_west`, `spa_test`, `model_confidence_set`, `mspe_adjusted`.
   - Recommendation: Assign reading `forecast.py` lines 80-end as part of the SKILL-01 authoring task.

2. **ols_formula_predict and formula predict signatures**
   - What we know: They appear in `__all__` at `python/polars_statistics/__init__.py` lines 491-497.
   - What's unclear: Whether formula_predict takes the same `interval`/`level`/`null_policy`/`name` kwargs as the non-formula predict.
   - Recommendation: Assign reading `regression.py` lines 2750+ as part of SKILL-03 authoring task.

---

## Sources

### Primary (HIGH confidence)
- `python/polars_statistics/__init__.py` — complete `__all__` export list; `__version__` [VERIFIED: lines 1-512]
- `python/polars_statistics/_polars_statistics.pyi` — all 43 PyModel class signatures [VERIFIED: lines 1-931]
- `python/polars_statistics/exprs/parametric.py` — ANOVA + parametric test signatures [VERIFIED: lines 1-481]
- `python/polars_statistics/exprs/nonparametric.py` — non-parametric test signatures [VERIFIED: lines 1-261]
- `python/polars_statistics/exprs/distributional.py` — normality test signatures [VERIFIED: lines 1-105]
- `python/polars_statistics/exprs/correlation.py` — correlation + ICC signatures [VERIFIED: lines 1-449]
- `python/polars_statistics/exprs/categorical.py` — categorical test signatures [VERIFIED: lines 1-592]
- `python/polars_statistics/exprs/modern.py` — MMD/energy test signatures [VERIFIED: lines 1-202]
- `python/polars_statistics/exprs/tost.py` — TOST equivalence signatures [VERIFIED: lines 1-645]
- `python/polars_statistics/exprs/regression.py` — regression fit/summary/predict/diagnostic signatures [VERIFIED: lines 1-3250 (partial; formula_predict section unread)]
- `python/polars_statistics/results.py` — unnest/struct_to_dict helpers [VERIFIED: lines 1-163]
- `python/polars_statistics/formula/parser.py` — FormulaParser.parse() [VERIFIED: lines 1-60]
- `python/polars_statistics/formula/expander.py` — expand_terms_to_expressions() [VERIFIED: lines 1-60]
- `plugins/polars-statistics/.claude-plugin/plugin.json` — version: "0.8.0" [VERIFIED: line 4]
- `.planning/phases/14-skill-cheat-sheet-authoring/14-CONTEXT.md` — locked decisions
- `.planning/STATE.md` — ANOVA/ICC nuance, blockers note

### Secondary (MEDIUM confidence)
- `python/polars_statistics/exprs/forecast.py` lines 1-80 — first two forecast functions
- fdars SKILL.md at `/home/simonm/.claude/plugins/cache/pyfda/fdars/0.13.0/skills/fdars-advisor/SKILL.md` — structural reference for SKILL.md conventions

### Tertiary (LOW confidence)
- Forecast functions `clark_west`, `spa_test`, `model_confidence_set`, `mspe_adjusted` signatures: [ASSUMED] — research stopped at forecast.py line 80

---

## Metadata

**Confidence breakdown:**
- Symbol inventories (Skills 1-5): HIGH — every symbol read from source files this session
- Signatures for parametric/nonparametric/distributional/correlation/categorical/modern/tost/results: HIGH — read from source
- PyModel class signatures: HIGH — read from `_polars_statistics.pyi`
- Regression expression signatures (fit/summary/predict non-formula): HIGH — read from regression.py lines 1-3250
- Formula expression predict signatures: MEDIUM — exists in `__all__` but definitions not read (lines 2750+)
- Forecast functions (4 of 6): LOW/ASSUMED — not read this session

**Research date:** 2026-09-21
**Valid until:** Stable API — valid until next API-breaking version change (v0.9.0+)
