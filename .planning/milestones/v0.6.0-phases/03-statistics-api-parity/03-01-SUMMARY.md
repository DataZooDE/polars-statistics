---
phase: 03-statistics-api-parity
plan: 01
subsystem: expressions/parametric
tags: [anova, statistics, polars-expression, rust, pyo3, tdd]
status: complete

dependency_graph:
  requires:
    - anofox-statistics 0.4.2 (one_way_anova, two_way_anova, repeated_measures_anova at crate root)
  provides:
    - ps.one_way_anova(*groups, kind) — STAT-01
    - ps.two_way_anova(value, factor_a, factor_b) — STAT-02
    - ps.repeated_measures_anova(value, subject, condition, compute_sphericity) — STAT-03
    - one_way_anova_output_dtype, two_way_anova_output_dtype, repeated_measures_anova_output_dtype, icc_output_dtype in output_types.rs
  affects:
    - python/polars_statistics/exprs/__init__.py (3 new exports)
    - python/polars_statistics/__init__.py (3 new exports)

tech_stack:
  added:
    - anofox_statistics::{one_way_anova, two_way_anova, repeated_measures_anova, AnovaKind} (crate-root imports)
  patterns:
    - Separate-group-Series input (kruskal_wallis analog) for one_way_anova
    - rank(dense)-1 encoding for factor/subject/condition columns (replaces Categorical.to_physical which shares global catalog)
    - StructChunked::from_series with length=1 for all ANOVA output structs
    - NaN-on-error helper pattern (one_way_anova_error_output etc.)
    - Long-format pivot in Rust (subject×condition matrix from value/subject/condition Series)

key_files:
  created:
    - tests/test_statistics_parity.py (11 smoke tests: TestOneWayAnova 4, TestTwoWayAnova 3, TestRmAnova 4)
  modified:
    - src/expressions/output_types.rs (added one_way_anova_output_dtype, two_way_anova_output_dtype, repeated_measures_anova_output_dtype, icc_output_dtype)
    - src/expressions/parametric.rs (added parse_anova_kind, *_error_output helpers, one_way_anova_fit, two_way_anova_fit, repeated_measures_anova_fit, pl_* shims)
    - python/polars_statistics/exprs/parametric.py (added one_way_anova, two_way_anova, repeated_measures_anova builders)
    - python/polars_statistics/exprs/__init__.py (added 3 ANOVA imports + __all__ entries)
    - python/polars_statistics/__init__.py (added 3 ANOVA imports + __all__ entries)

decisions:
  - key: one_way_anova input encoding
    choice: Separate group Series (kruskal_wallis pattern) with kind literal as last input
    rationale: Context note confirmed kruskal_wallis uses separate-group-Series; avoids value+label partitioning; cleaner Rust code
  - key: Factor encoding for two_way_anova and repeated_measures_anova
    choice: rank(method="dense") - 1 in Python builder for per-column 0-indexed codes
    rationale: Categorical.to_physical() shares global catalog across columns; fb=[p,q] encoded as [2,3] not [0,1] when fa=[x,y] was encoded first. rank(dense)-1 gives independent 0-indexed codes per column.
  - key: RM-ANOVA balanced design validation
    choice: Validate in Rust wrapper before calling crate; return NaN struct on mismatch
    rationale: Pitfall 3 from RESEARCH.md; crate returns StatError on unbalanced; NaN struct is safer than propagating the error
  - key: icc_output_dtype added in Task 1
    choice: Added all four output dtypes (including icc_output_dtype) in Task 1 for completeness
    rationale: All four were planned together in output_types.rs; adding them all at once avoids a separate commit; icc_output_dtype is needed by the ICC stub-replacement plan (03-03)

metrics:
  duration_seconds: 5400
  completed: 2026-08-12
  tasks_completed: 4
  commits: 5

actuals:
  tokens: 42000
  tasks: 4
  commits: 5
---

# Phase 03 Plan 01: ANOVA Family (one-way, two-way, RM-ANOVA) — Summary

**One-liner:** Full ANOVA family wired end-to-end via Rust fit functions + output structs + Python builders, tracer-first: `ps.one_way_anova` proven callable before two-way and RM-ANOVA expansion.

## What Was Built

Three new Polars expressions expose the ANOVA family from anofox-statistics 0.4.2:

- **`ps.one_way_anova(*groups, kind="fisher"|"welch")`** — Fisher (SS/MS/eta_squared fields populated) or Welch (SS/MS/eta_squared all NaN); n_groups UInt32. Takes separate group columns, kind literal as last arg.
- **`ps.two_way_anova(value, factor_a, factor_b)`** — Flattened a_/b_/ab_/residual_ AnovaTableRow fields plus grand_mean and n. Factor columns are automatically label-encoded by the Python builder.
- **`ps.repeated_measures_anova(value, subject, condition, compute_sphericity=True)`** — Within-subjects/error/sphericity/correction fields. Long-format pivoted to subject×condition matrix in Rust; unbalanced input returns NaN struct.

Four output dtype functions added to `output_types.rs`: `one_way_anova_output_dtype`, `two_way_anova_output_dtype`, `repeated_measures_anova_output_dtype`, `icc_output_dtype` (the last for the future ICC plan 03-03).

## TDD Gate Compliance

| Gate | Commit | Status |
|------|--------|--------|
| RED  | a0fd2e2 test(03-01): add failing tests... | Tests failed as expected (AttributeError: no attribute 'one_way_anova') |
| GREEN | f0a4e71 feat(03-01): wire one_way_anova... | TestOneWayAnova 4/4 pass |
| GREEN (Task 2) | 51b9b73 feat(03-01): add two_way_anova... | TestTwoWayAnova 3/3 pass |
| GREEN (Task 3) | 2e31465 feat(03-01): add repeated_measures_anova... | TestRmAnova 4/4 pass |
| REFACTOR | fa1efa2 chore(03-01): apply cargo fmt... | cargo fmt clean |

## Tracer Verification

`ps.one_way_anova` was verified callable from Python before expansion to two_way/rm_anova:

```
one_way_anova: PASS statistic=27.0, p_value=0.001
two_way_anova: PASS n=12
repeated_measures_anova: PASS grand_mean=3.5
```

All 11 smoke tests pass. cargo clippy --all-targets --features python -D warnings: clean. cargo fmt --check: clean.

## Commits

| # | Hash | Message |
|---|------|---------|
| 1 | a0fd2e2 | test(03-01): add failing tests for one_way_anova, two_way_anova, repeated_measures_anova |
| 2 | f0a4e71 | feat(03-01): wire one_way_anova end-to-end (STAT-01 tracer) |
| 3 | 51b9b73 | feat(03-01): add two_way_anova expression + builder (STAT-02) |
| 4 | 2e31465 | feat(03-01): add repeated_measures_anova expression + builder (STAT-03) |
| 5 | fa1efa2 | chore(03-01): apply cargo fmt to ANOVA additions |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Categorical.to_physical() gives non-0-indexed codes for multi-column factors**
- **Found during:** Task 2 (two_way_anova)
- **Issue:** The planned `cast(pl.Categorical).to_physical().cast(pl.UInt32)` pattern gives codes from the global Polars Categorical catalog. When factor_b=[p,q] is encoded after factor_a=[x,y], it gets physical codes [2,3] instead of [0,1], causing the crate to see 4 factor levels with empty cells and return an error.
- **Fix:** Use `rank(method="dense").cast(pl.UInt32) - pl.lit(1, dtype=pl.UInt32)` for per-column 0-indexed encoding. Applied to both `two_way_anova` and `repeated_measures_anova` builders.
- **Files modified:** `python/polars_statistics/exprs/parametric.py`

**2. [Rule 1 - Bug] Test data for RM-ANOVA too perfect (error_ss=0, ws_f=inf)**
- **Found during:** Task 3 (repeated_measures_anova test)
- **Issue:** The balanced test data [1,2,3,2,3,4,3,4,5,4,5,6] has perfectly correlated subject profiles — error SS=0, making ws_f=inf. The test assertion `math.isfinite(ws_f)` failed.
- **Fix:** Added jitter to test data values (e.g., 2.1, 3.2 instead of 2.0, 3.0) to create non-zero error SS. `inf` is a legitimate crate output for degenerate data but not useful for a smoke test.
- **Files modified:** `tests/test_statistics_parity.py`

### Notes on Ruff Violations

UP007 violations (`Union[X, Y]` instead of `X | Y`) exist throughout `exprs/parametric.py` including in the pre-existing `ttest_ind`, `ttest_paired`, `brown_forsythe`, `yuen_test` functions. The project targets Python 3.9 and `Union[]` is correct for 3.9 compatibility. These are pre-existing violations (confirmed by checking `git show HEAD~4`). The CI workflow does not include a ruff step; only `cargo fmt` and `cargo clippy` are CI-gated. My new ANOVA builders use the same `Union[]` pattern as the rest of the file.

## Self-Check

Files created/modified:
- [x] `tests/test_statistics_parity.py` — created, 11 tests
- [x] `src/expressions/output_types.rs` — 4 new output_dtype fns
- [x] `src/expressions/parametric.rs` — 3 fit fns + 3 shims + parse helper + 3 error helpers
- [x] `python/polars_statistics/exprs/parametric.py` — 3 new builders
- [x] `python/polars_statistics/exprs/__init__.py` — 3 new exports
- [x] `python/polars_statistics/__init__.py` — 3 new exports

Commits exist:
- [x] a0fd2e2 (test RED)
- [x] f0a4e71 (feat STAT-01 tracer)
- [x] 51b9b73 (feat STAT-02)
- [x] 2e31465 (feat STAT-03)
- [x] fa1efa2 (chore fmt)

## Self-Check: PASSED
