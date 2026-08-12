---
phase: 03-statistics-api-parity
status: all_fixed
fix_scope: critical_warning_plus_info_test
findings_in_scope: 6
fixed: 6
skipped: 0
iteration: 1
applied_by: orchestrator (inline)
fixed_at: 2026-08-12
commit: 19abd90
---

# Phase 3 Code Review — Fix Report

All 3 critical + 3 warning findings from `03-REVIEW.md` fixed and verified. The 3 criticals
were silently-wrong-result bugs in the ANOVA factor-encoding path — directly at odds with the
library's "validated against R" promise — so they were treated as must-fix before the phase seals.

## Fixes

| ID | Severity | Fix |
|----|----------|-----|
| CR-01 | Critical | `two_way_anova` (parametric.py): moved factor `rank("dense")` encoding to AFTER the row filter, so dense codes stay contiguous from 0 even when a whole factor level is dropped (no phantom level). |
| CR-02 | Critical | `two_way_anova` + `repeated_measures_anova` (parametric.py): the drop-mask now includes `factor/subject/condition.is_not_null()`, not just `value.is_finite()`, so a null in any aligned column removes the whole row before the Rust `into_no_null_iter()` can shorten one column and misalign the arrays. |
| CR-03 | Critical | `repeated_measures_anova` (parametric.py): replaced `cast(Categorical).to_physical()` (global-catalog-shared codes) with per-column `rank("dense") - 1`, so encoding is independent per group_by group — fixes the all-NaN result on 2nd+ groups. |
| WR-01 | Warning | `icc_fit` (correlation.rs): added a rectangular/complete-matrix guard before the `matrix[r][s]` transpose (returns the NaN error struct instead of panicking on a ragged/short matrix). |
| WR-02 | Warning | `one_way_anova_fit` (parametric.rs): eta² now falls back to `ss_between / (ss_between + ss_within)` when the crate leaves `ss_total` unpopulated (Fisher fits), instead of silently NaN. |
| WR-03 | Warning | `energy_distance_nd` (modern.py): added a `d == 0` guard that raises `ValueError` on empty column lists (previously returned a silent NaN struct). |
| IN-01 | Info→fixed | Added two regression tests (`TestTwoWayAnova::test_nulls_dropped_and_aligned`, `TestRmAnova::test_group_by_multiple_groups`) that fail on the pre-fix code — the suite previously had no null-factor or multi-group fixtures. |

## Verification

- `cargo clippy --all-targets --features python -- -D warnings` — CLEAN
- `cargo fmt --check` — CLEAN
- `cargo test --no-default-features --test rust_api` — 15 passed
- `maturin develop` + `pytest tests/` — **479 passed** (477 + 2 new regression tests)
- The 2 new regression tests pass on the fixed code (and encode the exact CR-01/02/03 failure modes).
