---
phase: 14-skill-cheat-sheet-authoring
reviewed: 2026-09-21T00:00:00Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - plugins/polars-statistics/skills/polars-statistics-getting-started/SKILL.md
  - plugins/polars-statistics/skills/polars-statistics-hypothesis-tests/SKILL.md
  - plugins/polars-statistics/skills/polars-statistics-anova/SKILL.md
  - plugins/polars-statistics/skills/polars-statistics-regression/SKILL.md
  - plugins/polars-statistics/skills/polars-statistics-results/SKILL.md
findings:
  critical: 4
  warning: 3
  info: 1
  total: 8
status: resolved
# All 8 findings (4 CRITICAL + 3 WARNING + 1 INFO) fixed against source and spot-checked;
# see 14-REVIEW-FIX.md. Commits e97852d, 1381353, 47b9300, 4dcbf6c, f101aa6.
---

# Phase 14: Code Review Report

**Reviewed:** 2026-09-21
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

All five SKILL.md cheat-sheets were reviewed against the authoritative sources
(`python/polars_statistics/exprs/*.py`, `python/polars_statistics/_polars_statistics.pyi`,
`python/polars_statistics/results.py`). Mandated gotchas (ICC rater columns,
`add_intercept=`, one_way_anova wide format, two_way_anova/repeated_measures_anova raw
factors, WLS 3-arg fit) are stated correctly in all five skills.

The principal defect class is **wrong output-struct field lists** — an agent reading the
hypothesis-tests or results skill will construct incorrect `.struct.field("...")` accesses
or `ps.unnest(...)` downstream calls. Four BLOCKERs share this pattern; they span two
skills but describe the same underlying inaccuracy. Three WARNINGs cover narrower
inconsistencies.

No invented functions or nonexistent kwargs were found. No forecast functions appear in
any skill.

---

## Critical Issues

### CR-01: `distance_cor` output struct fields wrong in hypothesis-tests and results skills

**File:** `plugins/polars-statistics/skills/polars-statistics-hypothesis-tests/SKILL.md:167`
**Also:** `plugins/polars-statistics/skills/polars-statistics-results/SKILL.md:189`

**Issue:** Both skills list the `distance_cor` output as `statistic, p_value`. The
authoritative source (`exprs/correlation.py`, `distance_cor` docstring, line 205-206)
declares the return struct as `{estimate, statistic, p_value, n}`. An agent reading the
cheat-sheet and writing `.struct.field("estimate")` or checking `d["n"]` will get a
`FieldNotFound` / `KeyError` that would never have occurred if the struct had only two
fields.

**Fix:**
```
# hypothesis-tests SKILL, output struct table (line 167):
| `distance_cor` | `estimate`, `statistic`, `p_value`, `n` |

# results SKILL, output struct field reference (line 189):
| `distance_cor` | `estimate`, `statistic`, `p_value`, `n` |
```

---

### CR-02: `kendall` output struct fields wrong in hypothesis-tests skill

**File:** `plugins/polars-statistics/skills/polars-statistics-hypothesis-tests/SKILL.md:166`

**Issue:** The output table lists `kendall` fields as `estimate, statistic, p_value, n`,
omitting `ci_lower` and `ci_upper`. The authoritative source (`exprs/correlation.py` line
143-145) declares `struct{estimate, statistic, p_value, ci_lower, ci_upper, n}`. The
results skill (line 188) has this correct, so the inconsistency is isolated to the
hypothesis-tests skill. An agent generating `.struct.field("ci_lower")` from the results
skill reference but failing to find it documented in the hypothesis-tests reference will
produce confusing answers.

**Fix:**
```
# hypothesis-tests SKILL, output struct table (line 166):
| `kendall` | `estimate`, `statistic`, `p_value`, `ci_lower`, `ci_upper`, `n` |
```

---

### CR-03: `binom_test` / `prop_test_one` / `prop_test_two` output struct fields wrong

**File:** `plugins/polars-statistics/skills/polars-statistics-hypothesis-tests/SKILL.md:237`
(blanket note "Output struct fields: statistic, p_value for all Group E test functions")
**Also:** `plugins/polars-statistics/skills/polars-statistics-results/SKILL.md:191`

**Issue:** Both skills claim `binom_test`, `prop_test_one`, and `prop_test_two` return
`{statistic, p_value}`. The authoritative source (`exprs/categorical.py`, docstrings at
lines 40-41, 89-90, 136-137) declares all three return
`{estimate, statistic, p_value, ci_lower, ci_upper, n}`. An agent following the cheat-sheet
will omit `estimate` (the observed proportion/difference) and the CI bounds from any
downstream code using these functions.

**Fix:**
```
# hypothesis-tests SKILL — replace blanket note for Group E with per-function table:
| Function | Fields |
|----------|--------|
| `binom_test`, `prop_test_one`, `prop_test_two` | `estimate`, `statistic`, `p_value`, `ci_lower`, `ci_upper`, `n` |
| `chisq_test`, `g_test`, `mcnemar_test` | `statistic`, `p_value`, `df`, `n` |
| `fisher_exact` | `statistic` (odds ratio), `p_value` |
| `mcnemar_exact` | `statistic`, `p_value` |
| `phi_coefficient`, `cramers_v`, `cohen_kappa`, `contingency_coef` | `estimate`, `statistic`, `p_value` |

# results SKILL line 191 — update to match above
```

---

### CR-04: `chisq_test` / `fisher_exact` / `mcnemar_test` output struct fields wrong

**File:** `plugins/polars-statistics/skills/polars-statistics-hypothesis-tests/SKILL.md:237`
**Also:** `plugins/polars-statistics/skills/polars-statistics-results/SKILL.md:192`

**Issue:** Both skills say these three functions return `{statistic, p_value}`. The source
(`exprs/categorical.py`) documents:

- `chisq_test` → `{statistic, p_value, df, n}` (line 177)
- `g_test` → `{statistic, p_value, df, n}` (line 281)
- `mcnemar_test` → `{statistic, p_value, df, n}` (line 374)
- `fisher_exact` → `{statistic (odds ratio), p_value}` — this one is actually correct but
  worth noting that `statistic` is the odds ratio, not a test statistic in the usual sense.

Missing `df` and `n` means an agent that tries to display or check `d["df"]` will receive a
`KeyError`. The fix is subsumed in the per-function table above (CR-03 fix block).

---

## Warnings

### WR-01: `n_groups` field type documented as `f64` but source declares `u32`

**File:** `plugins/polars-statistics/skills/polars-statistics-anova/SKILL.md:71`

**Issue:** The output struct field table for `one_way_anova` lists `n_groups` with type
`f64`. The authoritative source (`exprs/parametric.py` line 39) declares `n_groups (u32)`.
An agent or user writing type-sensitive code (e.g., casting back to Python `int` without
going through `float`) may be misled. Whether Polars promotes this to Float64 in the final
struct is a runtime detail, but the source docstring is the reference the skill should
mirror.

**Fix:** Change type column entry from `f64` to `u32` for `n_groups` in the anova skill
output struct table.

---

### WR-02: `Tweedie` factory methods documented with `add_intercept=` but `.pyi` uses `with_intercept=`

**File:** `plugins/polars-statistics/skills/polars-statistics-regression/SKILL.md:360-362`

**Issue:** The skill documents the three `Tweedie` factory methods as taking
`with_intercept: bool = True` (lines 360-362), consistent with the `.pyi` stub (which shows
`def gaussian(self, with_intercept=True, ...)` for Tweedie). This is internally consistent.
However, the description of these factory methods is inconsistent with the general
skill-wide stance that `add_intercept=` is preferred and `with_intercept=` is deprecated.

The `.pyi` stub is authoritative: `Tweedie`'s factory methods (`gaussian`, `gamma`,
`inverse_gaussian`) genuinely accept only `with_intercept=` — not `add_intercept=` — in
the current shipped stub. This means the skill-wide gotcha ("use `add_intercept=`, not
`with_intercept=`") does not apply to these three methods. The skill should call this
exception out explicitly to prevent an agent from substituting `add_intercept=` in a
`Tweedie().gaussian(add_intercept=True)` call, which would pass an unexpected keyword
argument at runtime.

**Fix:** Add a note to the `Tweedie` block:
```
# NOTE: Tweedie factory methods (.gaussian, .gamma, .inverse_gaussian) currently
# accept only with_intercept= (not add_intercept=) because they have not yet been
# migrated to the new parameter name. Use with_intercept=True/False for these methods.
```

---

### WR-03: `negative_binomial_predict` expression missing `theta=` kwarg compared to `negative_binomial`

**File:** `plugins/polars-statistics/skills/polars-statistics-regression/SKILL.md:625`

**Issue:** The skill documents `ps.negative_binomial_predict(y, *x, lambda_=0.0, ...)` but
omits `theta=`. The `negative_binomial` fit expression (`exprs/regression.py` line
1857-1864) has `theta: float | None = None`. The predict function (`regression.py` line
3307-3315) does not expose `theta=` — it estimates theta internally. The cheat-sheet is
therefore correct in not listing `theta=` for the predict variant, but this asymmetry is
silent and may confuse an agent that attempts to pass the same kwargs to both fit and
predict. A brief note would prevent silent wrong usage.

**Fix:**
```
ps.negative_binomial_predict(y, *x, lambda_=0.0, add_intercept=True, interval=None,
                             level=0.95, null_policy="drop", name=None)
# Note: theta= is not exposed on the predict variant; theta is estimated internally.
```

---

## Info

### IN-01: Getting-started SKILL distance_cor claim propagates from hypothesis-tests skill

**File:** `plugins/polars-statistics/skills/polars-statistics-getting-started/SKILL.md:159`

**Issue:** The model-selection matrix row for `ps.distance_cor` contains no output-struct
claim, so the getting-started skill does not independently repeat the error in CR-01.
However, in the context where CR-01 and CR-03/CR-04 are fixed, the getting-started skill
should remain current — no immediate change needed here unless the hypothesis-tests and
results skills are updated.

**Fix:** No action required in isolation; revisit after fixing CR-01 through CR-04.

---

_Reviewed: 2026-09-21_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
