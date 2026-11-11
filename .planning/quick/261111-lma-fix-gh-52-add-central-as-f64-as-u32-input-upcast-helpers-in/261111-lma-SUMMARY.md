---
phase: quick-261111-lma
plan: 01
subsystem: api
tags: [polars-expr, pyo3, rust, dtype-coercion, correlation, regression]

requires: []
provides:
  - "src/expressions/inputs.rs: crate-internal as_f64/as_u32 input upcast helpers"
  - "Every *_fit function in src/expressions reads through as_f64/as_u32 instead of strict typed accessors"
  - "tests/rust_api.rs::input_dtype_upcast: rlib-level dtype parity + strict-cast + error-preservation coverage"
  - "tests/test_dtype_upcast.py: Python dtype-parity coverage (53 tests)"
affects: [python-api, rust-api, regression-diagnostics, correlation]

actuals:
  tokens: 37707
  tasks: 3
  commits: 3
  plan_head_before: e026384f4ffca85b79c485fe67e1bc6463be1893
  plan_head_after: 8cb1e7d471530611f73cf46ceeda962f65be86c4

tech-stack:
  added: []
  patterns:
    - "Crate-internal pub(crate) coercion helpers (as_f64/as_u32) as the single input-dtype entry point for all expression *_fit functions"
    - "Hoist per-column casts out of Mat::from_fn per-cell closures (one as_f64 call per design-matrix column, not per cell)"

key-files:
  created:
    - src/expressions/inputs.rs
    - tests/test_dtype_upcast.py
  modified:
    - src/expressions/mod.rs
    - src/expressions/aft.rs
    - src/expressions/categorical.rs
    - src/expressions/correlation.rs
    - src/expressions/distributional.rs
    - src/expressions/forecast.rs
    - src/expressions/modern.rs
    - src/expressions/nonparametric.rs
    - src/expressions/parametric.rs
    - src/expressions/penalized_glm.rs
    - src/expressions/posthoc.rs
    - src/expressions/regression.rs
    - src/expressions/resampling.rs
    - src/expressions/shrinkage.rs
    - src/expressions/tost.rs
    - tests/rust_api.rs
    - CHANGELOG.md

key-decisions:
  - "Hoisted design-matrix casts to one as_f64 call per column (not per cell) in regression.rs's nine Mat::from_fn builders, matching the plan's DoS-mitigation requirement without changing per-function fill behaviour (NaN vs 0.0 preserved per site)."
  - "Non-numeric X columns and partial/semi-partial correlation covariates now return Err instead of being silently NaN/zero-filled or dropped — matches threat register T-261111-lma-05/07's mitigation plan exactly, no fallback needed."
  - "build_xy_with_null_policy's drop-policy x_fit keeps its .unwrap() but it is now sound: the hoisted as_f64(..)? on all x_cols already proves every column convertible before the null scan runs."

requirements-completed: [GH-52]

coverage:
  - id: D1
    description: "ps.spearman on two UInt32 columns (GH #52 repro) returns the same struct as the Float64-cast equivalent, instead of raising the dtype error."
    requirement: "GH-52"
    verification:
      - kind: unit
        ref: "tests/test_dtype_upcast.py#test_issue_52_spearman_uint32_repro"
        status: pass
      - kind: integration
        ref: "tests/rust_api.rs#input_dtype_upcast"
        status: pass
    human_judgment: false
  - id: D2
    description: "as_f64/as_u32 crate-internal helpers upcast every primitive-numeric, Boolean, and all-null dtype to Float64/UInt32 respectively; as_u32 strict-casts so negative/out-of-range values error instead of being nulled; non-numeric dtypes keep the original dtype error."
    requirement: "GH-52"
    verification:
      - kind: unit
        ref: "src/expressions/inputs.rs#cfg(test) mod tests (10 cases)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Every *_fit input read in src/expressions (332 original call sites) routes through as_f64/as_u32; regression design-matrix builders and partial/semi-partial correlation covariates no longer silently corrupt or drop non-Float64 columns."
    requirement: "GH-52"
    verification:
      - kind: integration
        ref: "tests/rust_api.rs#input_dtype_upcast (ols_fit/ols_predict_fit/condition_number_fit/partial_cor_fit/kruskal_wallis_fit/two_way_anova_fit dtype-parity cases)"
        status: pass
      - kind: unit
        ref: "cargo test --no-default-features --lib && cargo test --no-default-features --test rust_api (34 tests total)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Python dtype-parity surface: 53 tests across pearson/spearman/kendall/ttest_ind/mann_whitney_u/kruskal_wallis/one_way_anova/shapiro_wilk/icc/partial_cor/ols x Int32/UInt32/Int64/Float32, plus integer-dtype chisq_test and group_by parity."
    requirement: "GH-52"
    verification:
      - kind: unit
        ref: "uv run pytest tests/test_dtype_upcast.py -q (53 passed)"
        status: pass
      - kind: integration
        ref: "uv run pytest tests/ -q (1864 passed, 6 pre-existing skips)"
        status: pass
    human_judgment: false

duration: ~2h
completed: 2026-11-11
status: complete
---

# Quick Task 261111-lma: Fix GH #52 — add as_f64/as_u32 input upcast helpers Summary

**Added `as_f64`/`as_u32` crate-internal upcast helpers in `src/expressions/inputs.rs` and routed all ~332 `*_fit` input reads in `src/expressions` through them, fixing `ps.spearman` (and every other expression) raising `"expected Float64, got u32"` on integer-dtype columns.**

## Performance

- **Duration:** ~2h (dominated by first-time Rust rebuilds of the full dependency tree per cargo invocation)
- **Tasks:** 3/3 completed
- **Files modified:** 17 (2 created, 15 modified)
- **Commits:** 3 (ledger-measured, `e026384..8cb1e7d`)

## Accomplishments

- `src/expressions/inputs.rs`: `pub(crate) as_f64(&Series) -> PolarsResult<Float64Chunked>` and `as_u32(&Series) -> PolarsResult<UInt32Chunked>`, each with a `#[cfg(test)]` unit-test module (10 tests). `as_f64` upcasts any primitive-numeric/Boolean/Null dtype to Float64 (Float64 fast path is a zero-copy clone); `as_u32` strict-casts any integer/Null dtype to UInt32, rejecting floats and out-of-range/negative values with an error rather than silently nulling them.
- Every single-index typed accessor on an `inputs` element across the 14 expression files (baseline 332 occurrences) now reads through `super::inputs::as_f64`/`as_u32`. The only `.f64()`/`.u32()` reads left outside `inputs.rs` are `aft.rs`'s two already-Float64-cast event-column reads (by design, per the plan's Step E).
- `regression.rs`'s nine `Mat::from_fn`-based design-matrix builders (`build_xy_data`, `condition_number_fit`, `check_binary_separation_fit`, `check_count_sparsity_fit`, `vif_fit`, `leverage_fit`, `high_leverage_points_fit`, `high_vif_predictors_fit`, `generalized_vif_fit`, `augment_fit`, `build_xy_with_null_policy`) now hoist one `as_f64` cast per column before the closure runs, instead of re-attempting (and silently failing) the cast per cell.
- `correlation.rs`'s `partial_cor_fit`/`semi_partial_cor_fit` covariate loops now propagate a cast failure with `?` instead of silently skipping the covariate (`if let Ok(cov) = ...`).
- `tests/rust_api.rs::input_dtype_upcast`: 10 sub-cases covering the GH #52 repro (UInt32/Int64 spearman vs Float64), a String-input error-preservation check, integer-dtype `chisq_test_fit` parity and its strict-cast negative-value error, plus `ols_fit`/`ols_predict_fit`/`condition_number_fit`/`partial_cor_fit`/`kruskal_wallis_fit`/`two_way_anova_fit` dtype-parity cases from the plan's `<behavior>` block.
- `tests/test_dtype_upcast.py`: 53 tests — the literal GH #52 repro, an `Int32`/`UInt32`/`Int64`/`Float32` x 11-function parity matrix (pearson, spearman, kendall, ttest_ind, mann_whitney_u, kruskal_wallis, one_way_anova, shapiro_wilk, icc, partial_cor, ols), an integer-covariate `partial_cor` "covariate actually used" check, integer-dtype `chisq_test`, and `group_by` dtype parity for `spearman`.
- `CHANGELOG.md` `[Unreleased] ### Fixed` entry referencing #52 (no version bump).

## Task Commits

1. **Task 1: Tracer — as_f64/as_u32 helpers wired end-to-end through spearman_fit and chisq_test_fit** - `5eb0ae5` (feat)
2. **Task 2: Route every remaining src/expressions input read through the helpers** - `8a6c5d0` (feat)
3. **Task 3: Python dtype-parity tests + CHANGELOG entry + full gates** - `8cb1e7d` (test)

_No separate plan-metadata commit was made — per the execution context, `.planning/`/`.gsd/` docs artifacts (PLAN.md, SUMMARY.md) are intentionally left uncommitted on this branch._

## Files Created/Modified

- `src/expressions/inputs.rs` - New: `as_f64`/`as_u32` crate-internal upcast helpers + unit tests
- `src/expressions/mod.rs` - `mod inputs;` registered privately (no re-export)
- `src/expressions/{aft,categorical,correlation,distributional,forecast,modern,nonparametric,parametric,penalized_glm,posthoc,regression,resampling,shrinkage,tost}.rs` - All `*_fit` input reads routed through `as_f64`/`as_u32`; `regression.rs` additionally hoists per-column casts out of `Mat::from_fn` closures
- `tests/rust_api.rs` - `input_dtype_upcast` test (10 sub-cases) + two new test helpers (`struct_f64_field`, `struct_first_coef`)
- `tests/test_dtype_upcast.py` - New: 53 Python dtype-parity tests
- `CHANGELOG.md` - `[Unreleased] ### Fixed` entry referencing #52

## Decisions Made

- Hoisted design-matrix casts to one `as_f64` call per column (not per cell) across all nine `regression.rs` `Mat::from_fn` builders — matches the plan's DoS-mitigation requirement (T-261111-lma-06) and fixes a real pre-existing bug: non-Float64 X columns used to be silently NaN/zero-filled per cell, or (in `build_xy_with_null_policy`'s drop path) could reach a `.unwrap()` panic on a failed cast.
- Non-numeric X columns and partial/semi-partial correlation covariates now return `Err` instead of being silently corrupted or dropped, matching the plan's threat-register mitigation for T-261111-lma-05/07. No fallback to the old silent behaviour was needed anywhere — the full test suites (Rust and Python) stayed green without exceptions.
- `build_xy_with_null_policy`'s drop-policy `x_fit` keeps its `.unwrap()` call, now provably sound: the hoisted `as_f64(..)?` over every x column already proves convertibility before the null scan and `Mat::from_fn` build run, so a failed cast short-circuits the whole function earlier instead of surfacing at the old per-cell site.
- Picked integer-exact, non-collinear synthetic fixtures for the new `condition_number_fit` and `partial_cor_fit` Rust test cases after the first attempt (reusing existing fractional/perfectly-correlated fixtures from other tests) produced false failures from lossy i64 truncation and degenerate collinearity — not a dtype-upcast bug, just bad test data.

## Deviations from Plan

None — plan executed exactly as written, including the "leave aft.rs's already-cast event column alone" (Step E) and "no new `use` statements in the expression files" constraint (verified: the only new `use` line across the whole diff is in the brand-new `inputs.rs` file itself).

## Issues Encountered

- Two Rust integration-test fixtures I added for Task 2's `<behavior>` coverage initially failed: `condition_number_fit`'s int64 x2 column used `0.5*i + 1.0` fractional values that truncate under `as i64`, and `partial_cor_fit`'s reused `two_samples()` fixture made x/y perfectly correlated, degenerating the partial-correlation comparison. Both were test-data bugs, not implementation bugs; fixed by switching to integer-exact, non-degenerate fixtures and re-running to green.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The fix is fully self-contained on `fix/52-upcast-numeric-inputs`; it does not depend on, and was explicitly isolated from, the sibling item 261111-lm9 (GH #51, on branch `fix/51-optional-pyo3`) per the execution context's instructions.
- All required local gates pass: `cargo fmt --all -- --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test --no-default-features --lib` (11 passed), `cargo test --no-default-features --test rust_api` (23 passed, including `input_dtype_upcast`), `uv run maturin develop --release`, `uv run pytest tests/ -q` (1864 passed, 6 pre-existing skips, 0 failures), `uvx ruff@0.16.8 check .` (all checks passed).
- `cargo clippy --no-default-features --lib -- -D warnings` was also run informationally (sibling lm9's CI gate line is absent from `.github/workflows/ci.yml` on this branch) and is clean — this item added no new dead-code/unused warnings for lm9 to inherit when the branches merge.
- Ready for PR / merge once the sibling branch `fix/51-optional-pyo3` is also ready; the two diffs are merge-disjoint on `src/expressions/*.rs` `use` blocks by construction.

---
*Quick task: 261111-lma*
*Completed: 2026-11-11*

## Self-Check: PASSED

- FOUND: src/expressions/inputs.rs
- FOUND: tests/test_dtype_upcast.py
- FOUND: src/expressions/mod.rs
- FOUND: tests/rust_api.rs
- FOUND: CHANGELOG.md
- FOUND commit: 5eb0ae5 (ancestor of HEAD)
- FOUND commit: 8a6c5d0 (ancestor of HEAD)
- FOUND commit: 8cb1e7d (ancestor of HEAD)
