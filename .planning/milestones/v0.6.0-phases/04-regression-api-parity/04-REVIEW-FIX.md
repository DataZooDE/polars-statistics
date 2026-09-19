---
phase: 04-regression-api-parity
fixed_at: 2026-08-12T00:00:00Z
review_path: .planning/phases/04-regression-api-parity/04-REVIEW.md
iteration: 1
findings_in_scope: 10
fixed: 10
skipped: 0
status: all_fixed
---

# Phase 4: Code Review Fix Report

**Fixed at:** 2026-08-12T00:00:00Z
**Source review:** `.planning/phases/04-regression-api-parity/04-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 10 (5 Critical, 4 Warning, 1 Info)
- Fixed: 10
- Skipped: 0

**Verification ran in:** isolated git worktree (`rf-04-1123395-1786538976`) — not the main checkout. Gates were `cargo build --features python`, `cargo clippy --all-targets --features python -- -D warnings`, `cargo fmt --check` — all three passed clean before any commits.

## Fixed Issues

### CR-01: `as_slice().unwrap()` panics on non-contiguous numpy arrays

**Files modified:** `src/pymodels/py_moment_accumulator.rs`, `src/pymodels/py_passive_aggressive.rs`
**Commit:** `5a2a36f`
**Applied fix:** Replaced `.as_slice().unwrap()` with `.as_slice().map_err(|_| PyErr::new::<PyValueError, _>("x_row must be a contiguous (C-order) 1-D float64 array; try passing np.ascontiguousarray(x_row) if it is a slice"))?` in both `push_row` (MomentAccumulator) and `partial_fit` (PassiveAggressive). No process crash on non-contiguous input — callers receive a descriptive `ValueError` instead.

---

### CR-02: `partial_fit` leaves `self.fitted = None`; `predict` permanently broken after online-only use

**Files modified:** `src/pymodels/py_passive_aggressive.rs`
**Commit:** `5a2a36f` (bundled with CR-01 since the same file was modified)
**Applied fix:** `FittedPassiveAggressive` has private fields and no public constructor (`fitted_from_state` does not exist in the crate). Applied the minimum-correct alternative documented in the review: added a `predict_from_state` method that reads `self.state.weights` and `self.state.intercept` directly for streaming predictions after `partial_fit`. Also updated `is_fitted()` to return `true` when `self.state.is_some()` so callers can detect a usable online-learned model. The `predict` method continues to require `fit` to have been called; `predict_from_state` serves the `partial_fit`-only path.

---

### CR-03: WLS `hc_inference` ignores observation weights — produces incorrect sandwich estimates

**Files modified:** `src/pymodels/py_wls.rs`
**Commit:** `434b7a6`
**Applied fix (requires human verification):** Investigation confirmed that `FittedWls` (which exposes `weights()`) cannot be retrieved through the `Box<dyn FittedRegressor>` stored in `PyWLS::fitted` — the trait object provides no downcast path. Returning the unweighted OLS sandwich is silently wrong. Instead of returning incorrect results, the method now raises `PyNotImplementedError` with a clear explanation and the documented workaround (scale X and y by `sqrt(w_i)`, fit OLS on scaled data, call `OLS.hc_inference`). The hardcoded `0.95` was simultaneously replaced with `self.confidence_level` in `py_ridge.rs` (WR-01, same commit). Also removed the now-unused `compute_hc_inference` and `HcType` imports from `py_wls.rs`.

---

### CR-04: `gamma_dispersion_deviance_fit` and `gamma_dispersion_pearson_fit` missing minimum-input guard

**Files modified:** `src/expressions/regression.rs`
**Commit:** `bb50fd1`
**Applied fix:** Added `if inputs.len() < 4 { return dispersion_nan_output(); }` at the top of both `gamma_dispersion_deviance_fit` and `gamma_dispersion_pearson_fit`, matching the guard pattern already present in `gamma_pearson_chi_squared_fit`. Prevents out-of-bounds slice indexing when called with degenerate Polars plugin inputs.

---

### CR-05: GLMM `fit` and `fit_crossed` have no group-length vs. n_rows validation

**Files modified:** `src/pymodels/py_glmm.rs`
**Commit:** `a675b3e`
**Applied fix:** Added `if group.len() != n_rows { return Err(PyValueError) }` after `x.to_faer()` in `fit`. In `fit_crossed`, added a loop over each inner group slice validating `g.len() == n_rows`. Both raise `PyValueError` with a descriptive message including the mismatched lengths. Consistent with the validation pattern used in the expression layer.

---

### WR-01: Ridge `hc_inference` hardcodes confidence level at 0.95

**Files modified:** `src/pymodels/py_ridge.rs`
**Commit:** `434b7a6`
**Applied fix:** Replaced literal `0.95` with `self.confidence_level` in the `compute_hc_inference` call. `hc_inference` takes `&self` so `self.confidence_level` is directly accessible. Bundled with CR-03 as both are HC-inference confidence-level fixes in the same commit.

---

### WR-02: `_resolve_intercept` called twice in 14+ Python wrapper functions

**Files modified:** `python/polars_statistics/exprs/regression.py`
**Commit:** `777205b`
**Applied fix:** Used a Python regex to remove all consecutive duplicate `add_intercept = _resolve_intercept(add_intercept, with_intercept)` lines globally. Removed 18 duplicate calls. Verified no consecutive duplicates remain via a post-pass scan. The `_resolve_intercept` function is now called exactly once per function, restoring backward compatibility for the deprecated `with_intercept` kwarg path.

---

### WR-03: `gamma_standardized_pearson_residuals_fit` and `gamma_standardized_deviance_residuals_fit` missing minimum-input guard

**Files modified:** `src/expressions/regression.rs`
**Commit:** `bb50fd1`
**Applied fix:** Added `if inputs.len() < 4 { return residual_diag_nan_output(); }` at the top of both `gamma_standardized_pearson_residuals_fit` and `gamma_standardized_deviance_residuals_fit`. Bundled with CR-04 as they share the same file and guard pattern.

---

### WR-04: `partial_fit` silently re-initializes PaState on feature-count change

**Files modified:** `src/pymodels/py_passive_aggressive.rs`
**Commit:** `5a2a36f`
**Applied fix:** Added a pre-check before `self.state.get_or_insert_with(...)`: if `self.state` is already `Some(ref state)` and `state.weights.len() != n_features`, returns `PyValueError` with a message stating the mismatch. Prior learning is preserved; the error forces the caller to use consistent feature counts. Bundled with CR-01 and CR-02.

---

### WR-06: LARS `eps` field is stored but silently discarded — API contract broken

**Files modified:** `src/pymodels/py_lars.rs`
**Commit:** `ce20623`
**Applied fix:** Removed the `eps` field, the `#[allow(dead_code)]` annotation, the `eps: f64::EPSILON` default in the `#[pyo3(signature)]` attribute, the `eps` parameter in `fn new`, and the docstring entry for `eps`. The crate builder has no `eps()` setter, so removing it from the public API is the correct fix. Callers who passed `eps` explicitly will now receive a Python `TypeError` (unexpected keyword argument) rather than silently having the value discarded.

---

### IN-02: `col_into_vec` duplicates `col_to_vec` in `regression.rs`

**Files modified:** `src/expressions/regression.rs`
**Commit:** `ce20623`
**Applied fix:** Deleted the `col_into_vec` function (8 call sites) and replaced all call sites with `col_to_vec` using a `replace_all` edit. The two functions were identical (`(0..c.nrows()).map(|i| c[i]).collect()`). Bundled with WR-06 as they affect the same files.

---

_Fixed: 2026-08-12T00:00:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
