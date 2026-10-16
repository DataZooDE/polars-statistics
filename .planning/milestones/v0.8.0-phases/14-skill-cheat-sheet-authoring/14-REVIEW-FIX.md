---
phase: 14-skill-cheat-sheet-authoring
fixed_at: 2026-09-21T00:00:00Z
review_path: .planning/phases/14-skill-cheat-sheet-authoring/14-REVIEW.md
iteration: 1
findings_in_scope: 7
fixed: 7
skipped: 0
status: all_fixed
---

# Phase 14: Code Review Fix Report

**Fixed at:** 2026-09-21
**Source review:** .planning/phases/14-skill-cheat-sheet-authoring/14-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 7
- Fixed: 7
- Skipped: 0

## Fixed Issues

### CR-01: `distance_cor` output struct fields wrong in hypothesis-tests and results skills

**Files modified:** `plugins/polars-statistics/skills/polars-statistics-hypothesis-tests/SKILL.md`, `plugins/polars-statistics/skills/polars-statistics-results/SKILL.md`
**Commits:** e97852d (hypothesis-tests), 47b9300 (results)
**Applied fix:** Changed `distance_cor` output from `statistic, p_value` to `estimate, statistic, p_value, n` in both skills. Verified against `exprs/correlation.py` docstring line 206.

---

### CR-02: `kendall` output struct fields wrong in hypothesis-tests skill

**Files modified:** `plugins/polars-statistics/skills/polars-statistics-hypothesis-tests/SKILL.md`
**Commit:** e97852d
**Applied fix:** Added `ci_lower`, `ci_upper` to `kendall` output fields in the Group D table. Full field list now `estimate, statistic, p_value, ci_lower, ci_upper, n`. Verified against `exprs/correlation.py` lines 143-145. The results skill already had this correct.

---

### CR-03: `binom_test` / `prop_test_one` / `prop_test_two` output struct fields wrong

**Files modified:** `plugins/polars-statistics/skills/polars-statistics-hypothesis-tests/SKILL.md`, `plugins/polars-statistics/skills/polars-statistics-results/SKILL.md`
**Commits:** 1381353 (hypothesis-tests), 47b9300 (results)
**Applied fix:** Replaced the blanket "statistic, p_value for all Group E functions" note with a per-function output table. `binom_test`, `prop_test_one`, `prop_test_two` now show `estimate, statistic, p_value, ci_lower, ci_upper, n`. Verified against `exprs/categorical.py` docstrings lines 40-41, 89-90, 136-137.

---

### CR-04: `chisq_test` / `g_test` / `mcnemar_test` output struct fields wrong

**Files modified:** `plugins/polars-statistics/skills/polars-statistics-hypothesis-tests/SKILL.md`, `plugins/polars-statistics/skills/polars-statistics-results/SKILL.md`
**Commits:** 1381353 (hypothesis-tests), 47b9300 (results)
**Applied fix:** Per-function table in hypothesis-tests skill now shows `chisq_test, g_test, mcnemar_test` → `statistic, p_value, df, n`; `fisher_exact` remains `statistic (odds ratio), p_value`; `mcnemar_exact` remains `statistic, p_value`. In results skill the combined `chisq_test, fisher_exact, mcnemar_test` row was split: `chisq_test, g_test, mcnemar_test` → `statistic, p_value, df, n` and `fisher_exact` → `statistic (odds ratio), p_value`. Also fixed Example 5 comment in hypothesis-tests to show `# statistic, p_value, df, n` for chisq_test. Verified against `exprs/categorical.py` docstrings lines 177, 281, 374.

---

### WR-01: `n_groups` field type documented as `f64` but source declares `u32`

**Files modified:** `plugins/polars-statistics/skills/polars-statistics-anova/SKILL.md`
**Commit:** 4dcbf6c
**Applied fix:** Changed type column entry for `n_groups` from `f64` to `u32` in the `one_way_anova` output struct table. Verified against `exprs/parametric.py` line 39 docstring: `n_groups (u32)`.

---

### WR-02: `Tweedie` factory methods use `with_intercept=` not `add_intercept=`

**Files modified:** `plugins/polars-statistics/skills/polars-statistics-regression/SKILL.md`
**Commit:** f101aa6
**Applied fix:** Added an inline comment block after the three Tweedie factory method signatures explaining that `.gaussian`, `.gamma`, `.inverse_gaussian` accept only `with_intercept=` (not `add_intercept=`) because they have not yet been migrated to the new parameter name. Verified against `python/polars_statistics/_polars_statistics.pyi` lines 862-866 which show `def gaussian(self, with_intercept=True, ...)`.

---

### WR-03: `negative_binomial_predict` missing `theta=` asymmetry note

**Files modified:** `plugins/polars-statistics/skills/polars-statistics-regression/SKILL.md`
**Commit:** f101aa6
**Applied fix:** Added an inline comment after `ps.negative_binomial_predict(...)` noting that the predict variant does not expose `theta=` (unlike the fit expression which accepts `theta=`); theta is estimated internally. Verified against `exprs/regression.py` — the predict function does not expose `theta=`.

---

_Fixed: 2026-09-21_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
_Verification: spot-checks run via `uv run python` in the main checkout; `distance_cor` docstring confirmed `struct{estimate, statistic, p_value, n}`; `chisq_test` docstring confirmed `struct{statistic, p_value, df, n}`; `binom_test` docstring confirmed `struct{estimate, statistic, p_value, ci_lower, ci_upper, n}`. Frontmatter parse verified on all 4 edited SKILL.md files (name/version 0.8.0/user-invocable false all OK)._
