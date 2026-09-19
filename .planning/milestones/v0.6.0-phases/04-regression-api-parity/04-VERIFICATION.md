---
phase: 04-regression-api-parity
verified: 2026-08-12
status: passed
score: 6/6
behavior_unverified: 0
overrides_applied: 0
verified_by: orchestrator (executed full gate inline; code review + fixes applied)
---

# Phase 4: Regression API Parity — Verification Report

**Phase Goal:** Every unexposed `anofox-regression` capability identified in the audit is callable via expressions and/or PyModel classes, following the existing PyModel and expression patterns.
**Verified:** 2026-08-12
**Status:** passed
**Requirements:** REGR-01, REGR-02, REGR-03, REGR-04, REGR-05, REGR-06

## Goal Achievement — Observable Truths

| # | Requirement / Truth | Status | Evidence |
|---|---------------------|--------|----------|
| 1 | REGR-03: `Gamma` GLM PyModel fits/predicts | VERIFIED | `src/pymodels/py_gamma.rs`; registered; `tests/test_gamma.py` passes |
| 2 | REGR-01: `GLMM` (fit/fit_crossed/factors) + REGR-01/02: `PSpline` | VERIFIED | `py_glmm.rs`, `py_pspline.rs`; `test_glmm.py`, `test_pspline.py` pass |
| 3 | REGR-04: HC robust SEs for regression fits | VERIFIED | `Ridge.hc_inference` returns correct HC dict (uses `self.confidence_level`); OLS HC pre-existing. WLS HC intentionally raises `NotImplementedError` (CR-03) — the crate cannot express a weighted HC sandwich and returning the unweighted one would be statistically wrong; documented as a deferred limitation rather than shipping incorrect statistics. REGR-04 delivered for OLS + Ridge. |
| 4 | REGR-05: GLM diagnostics — dispersion + standardized residuals | VERIFIED | 5 `#[polars_expr]` in `regression.rs` (gamma_dispersion_deviance/pearson, gamma_pearson_chi_squared, gamma_standardized_pearson/deviance_residuals) + Python builders; registered; `test_glm_diagnostics.py` passes |
| 5 | REGR-06: all remaining solvers + MomentAccumulator | VERIFIED | `TheilSen`, `RANSAC`, `BayesianRidge`, `ARD`, `LARS`, `PassiveAggressive` (with working `partial_fit`/`predict_from_state`), `MomentAccumulator` + `OLS/Ridge.fit_from_accumulator`; all registered; per-model smoke tests pass |
| 6 | Output schemas + callability follow existing PyModel/expression conventions | VERIFIED | All mirror `py_tweedie.rs`/`py_huber.rs`; all 10 classes + 5 expressions importable + callable from top-level `polars_statistics` |

**Score:** 6/6 requirements delivered.

## Executed Gate (orchestrator, 2026-08-12)

- `cargo clippy --all-targets --features python -- -D warnings` — CLEAN
- `cargo fmt --check` — CLEAN
- `cargo test --no-default-features --test rust_api` — 15 passed
- `maturin develop` + `pytest tests/` — **582 passed, 0 failed**
- ruff F-rules (real-bug category) on all new/changed Python files — passed
- All 10 new PyModels + 5 new expressions importable + callable from top-level `polars_statistics`

## Code Review

A standard-depth review of all 15 new source files found 5 critical + 6 warning findings. All 10 in-scope findings were fixed and re-validated (see `04-REVIEW.md`, `04-REVIEW-FIX.md`): non-contiguous-array panics (CR-01), broken online-learning predict path (CR-02), WLS weighted-HC correctness (CR-03 → NotImplementedError), missing input guards (CR-04/WR-03), GLMM group-length validation (CR-05), hardcoded confidence level (WR-01), double `_resolve_intercept` backward-compat break (WR-02), PA feature-count re-init (WR-04), dead LARS `eps` + duplicate helper (WR-06/IN-02). Two pre-existing findings (WR-05, IN-01) were out of Phase 4 scope and left unchanged.

## Deferred (recorded, not blocking)

- Weighted HC sandwich for WLS (REGR-04, WLS only) — requires the backing crate to expose a weighted-HC path; currently raises `NotImplementedError`.
- Comprehensive R-validation of all new regressors — Phase 6 (TEST-03).
- Full CI-matrix ruff/style pass — Phase 6 (TEST-05).
