---
phase: 04-regression-api-parity
reviewed: 2026-08-12T00:00:00Z
depth: standard
files_reviewed: 15
files_reviewed_list:
  - src/pymodels/py_gamma.rs
  - src/pymodels/py_glmm.rs
  - src/pymodels/py_pspline.rs
  - src/pymodels/py_theil_sen.rs
  - src/pymodels/py_ransac.rs
  - src/pymodels/py_bayesian_ridge.rs
  - src/pymodels/py_ard.rs
  - src/pymodels/py_lars.rs
  - src/pymodels/py_passive_aggressive.rs
  - src/pymodels/py_moment_accumulator.rs
  - src/pymodels/py_ols.rs
  - src/pymodels/py_ridge.rs
  - src/pymodels/py_wls.rs
  - src/expressions/regression.rs
  - python/polars_statistics/exprs/regression.py
findings:
  critical: 5
  warning: 6
  info: 2
  total: 13
status: issues_found
---

# Phase 4: Code Review Report

**Reviewed:** 2026-08-12T00:00:00Z
**Depth:** standard
**Files Reviewed:** 15
**Status:** issues_found

## Summary

Reviewed the 10 new PyModel wrappers (Gamma, GLMM, PSpline, TheilSen, RANSAC, BayesianRidge, ARD,
LARS, PassiveAggressive, MomentAccumulator), the HC-inference additions on Ridge/WLS, the
fit_from_accumulator path on OLS/Ridge, and the 5 new Gamma GLM diagnostic expressions in
regression.rs plus their Python builders.

The new model wrappers follow established patterns well and the MomentAccumulator correctly uses
`as_slice()` rather than `.to_faer()` per the design notes. However, five correctness issues were
found: two panics on user-controlled input, a statistical incorrectness in WLS HC inference
(weights ignored in the sandwich), a behavioral gap that makes `partial_fit`-only usage completely
broken (no `predict` available, no error), and a missing minimum-length guard in two of the five
new Gamma diagnostic expressions. Thirteen findings total.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: `as_slice().unwrap()` panics on non-contiguous numpy arrays

**File:** `src/pymodels/py_moment_accumulator.rs:71`, `src/pymodels/py_passive_aggressive.rs:157`

**Issue:** Both `push_row` (MomentAccumulator) and `partial_fit` (PassiveAggressive) call
`x_row.as_slice().unwrap()`. `PyReadonlyArray1::as_slice()` returns
`Result<&[T], NotContiguousError>` — it returns `Err` when the backing memory is not
C-contiguous. Calling `.unwrap()` then panics unconditionally, crossing the FFI boundary into
Python as an unrecoverable hard crash rather than a `ValueError`. Non-contiguous 1D arrays arise
routinely from row/column slices of 2D arrays, e.g. `X[0, :]` when `X` is Fortran-order, or
`arr[::2]`. Because these are public entry points for streaming/online learning, user-hostile
panics here are a significant correctness risk.

**Fix:**
```rust
// Replace the unwrap with a proper error conversion in both files:
let slice = x_row
    .as_slice()
    .map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x_row must be a contiguous (C-order) 1-D float64 array; \
             try passing np.ascontiguousarray(x_row) if it is a slice",
        )
    })?;
```

---

### CR-02: `partial_fit` updates `self.state` but leaves `self.fitted = None`; `predict` is permanently broken after online-only use

**File:** `src/pymodels/py_passive_aggressive.rs:156-164`, `src/pymodels/py_passive_aggressive.rs:175-179`

**Issue:** `partial_fit` accumulates streaming updates into `self.state` but never populates
`self.fitted`. After one or more `partial_fit` calls without a prior `fit` call, `is_fitted()`
returns `False` and `predict()` raises `RuntimeError("Model not fitted")`. There is no way for
the user to obtain predictions from an online-learned model — the entire online-learning path is
silently useless.

The docstring explicitly promises:

> Particularly useful for streaming data where the full design matrix cannot be materialised up front.

...and the example shows `partial_fit` followed by (implicitly) `predict`.

`PaState` contains the current weight vector; the fix is to extract a `FittedPassiveAggressive`
from it after each update (or provide a `predict_from_state` method that reads `self.state`
directly).

**Fix:**
```rust
fn partial_fit(&mut self, x_row: PyReadonlyArray1<'_, f64>, y_value: f64) -> PyResult<()> {
    let slice = x_row.as_slice().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("x_row must be contiguous")
    })?;
    let n_features = slice.len();
    let model = self.build_model();
    let state = self.state.get_or_insert_with(|| PaState::new(n_features));
    model
        .partial_fit(state, slice, y_value)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    // Expose predictions from online state:
    if let Ok(fitted) = model.fitted_from_state(state) {
        self.fitted = Some(fitted);
    }
    Ok(())
}
```
If `fitted_from_state` does not exist in the crate, the minimum acceptable fix is to add a
`predict_from_state` method that reads `self.state` directly, and document that `predict`
requires `fit` to have been called first (removing the misleading docstring).

---

### CR-03: WLS `hc_inference` ignores observation weights — produces incorrect sandwich estimates

**File:** `src/pymodels/py_wls.rs:147-191`

**Issue:** `PyWLS::hc_inference` calls `compute_hc_inference` with the raw (unweighted) `X`
matrix and the WLS residuals. HC standard errors require the sandwich matrix
`(X'WX)^{-1} (X' diag(w·e²) X) (X'WX)^{-1}`, but this implementation passes the unweighted `X`
producing `(X'X)^{-1} (X' diag(e²) X) (X'X)^{-1}`, which is the OLS sandwich — wrong for WLS.
For heterogeneous weights (the primary use case of WLS), the resulting standard errors, t-statistics
and p-values are all incorrect and not conservative in any predictable direction.

By contrast, `PyOLS::hc_inference` correctly delegates to `fitted.hc_inference(&x_mat, hc)`,
which is the crate-internal path that already accounts for the model's own internals.

Additionally, the confidence level is hardcoded to `0.95` at line 170, ignoring `self.confidence_level`.

**Fix:**
```rust
// Option A (preferred): delegate to the crate if FittedRegressor trait exposes hc_inference
// let result = fitted.hc_inference(&x_mat, hc)  // mirrors PyOLS

// Option B (interim): document the limitation and raise NotImplementedError
return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
    "WLS hc_inference requires weight-aware sandwich; use OLS.hc_inference on the \
     sqrt-weight-scaled data instead"
));
```
At minimum: replace hardcoded `0.95` with `self.confidence_level`.

---

### CR-04: `gamma_dispersion_deviance_fit` and `gamma_dispersion_pearson_fit` missing minimum-input guard

**File:** `src/expressions/regression.rs:2354-2376`, `src/expressions/regression.rs:2386-2408`

**Issue:** Both Gamma dispersion functions index directly into `inputs[1]` and `inputs[2]` at the
top of the function body without first checking `inputs.len() >= 4`. Compare with the analogous
logistic/Poisson chi-squared functions at lines 2030 and 2066, which each have:

```rust
if inputs.len() < 4 {
    return chi_squared_nan_output();
}
```

If called with fewer than 3 inputs (which Polars plugin machinery can do with degenerate
expressions), `inputs[1]` and `inputs[2]` index out-of-bounds on the slice, producing a panic
that crashes the worker thread.

**Fix:**
```rust
pub fn gamma_dispersion_deviance_fit(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() < 4 {          // <-- add this guard
        return dispersion_nan_output();
    }
    let lambda = inputs[1].f64()?.get(0).unwrap_or(0.0);
    // ...
}
```
Apply the same guard to `gamma_dispersion_pearson_fit`.

---

### CR-05: GLMM `fit` and `fit_crossed` have no group-length vs. n_rows validation

**File:** `src/pymodels/py_glmm.rs:122-152`, `src/pymodels/py_glmm.rs:171-204`

**Issue:** Both `fit(x, y, group)` and `fit_crossed(x, y, groups)` silently pass to the crate
without checking that `group.len() == x.nrows()`. If a user passes a group array of the wrong
length, the crate receives misaligned data; depending on crate internals this may silently compute
wrong BLUPs, panic, or give incorrect random-effect estimates. All other methods in this codebase
(e.g., the expression layer) validate shape consistency before delegating to the solver. The
`groups` inner list length is also not validated.

**Fix:**
```rust
fn fit<'py>(
    mut slf: PyRefMut<'py, Self>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    group: Vec<u64>,
) -> PyResult<PyRefMut<'py, Self>> {
    let x_mat = x.to_faer();
    let y_col = y.to_faer();
    let n_rows = x_mat.nrows();
    if group.len() != n_rows {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "group length ({}) must equal number of samples ({})",
            group.len(), n_rows
        )));
    }
    // ...
}
```
For `fit_crossed`, validate each inner group slice length equals `n_rows`.

---

## Warnings

### WR-01: Ridge `hc_inference` hardcodes confidence level at 0.95, ignores `self.confidence_level`

**File:** `src/pymodels/py_ridge.rs:182`

**Issue:** `compute_hc_inference` is called with a literal `0.95` as the confidence level. The
struct stores `self.confidence_level` (which may be different if the user passed e.g.
`confidence_level=0.90` at construction). This means `conf_interval_lower` and
`conf_interval_upper` in the returned dict are silently computed at 95% regardless of the user's
preference.

**Fix:**
```rust
// Line 182 — replace 0.95 with self.confidence_level:
let result = compute_hc_inference(
    &x_mat, &coef, intercept, &residuals, &aliased, true, hc, result_data.confidence_level,
)
```
Note: `result_data` does not expose confidence_level directly — use `fitted.confidence_level()`
if available, or read `slf.confidence_level`.

---

### WR-02: `_resolve_intercept` called twice in 14+ Python wrapper functions, causes spurious `ValueError` for deprecated `with_intercept` callers

**File:** `python/polars_statistics/exprs/regression.py:356-357`, `:475-476`, `:2272-2273`, `:2308-2309`, `:2358-2359`, `:2391-2392`, `:2432-2433`, `:2464-2465`, `:2497-2498`, `:2533-2534`, `:2566-2567`, `:2598-2599`, `:2630-2631`

**Issue:** At least 14 functions call `_resolve_intercept(add_intercept, with_intercept)` twice
on consecutive lines. Affected functions include `expanding_ols`, `lasso`, `elastic_net_summary`,
`lasso_summary`, `rls_summary`, `bls_summary`, `logistic_summary`, `poisson_summary`,
`negative_binomial_summary`, `tweedie_summary`, `probit_summary`, `cloglog_summary`, `alm_summary`,
and several formula variants.

The first call resolves `add_intercept` from `None` to a `bool`. The second call then sees
`add_intercept` is a non-`None` bool while `with_intercept` may also be non-`None` (when the
user passes the deprecated kwarg), which unconditionally raises:

```
ValueError: Cannot specify both 'add_intercept' and 'with_intercept'.
```

This is a regression against the backward-compatibility contract: users who call
`expanding_ols(y, x, with_intercept=True)` get a `ValueError` instead of a `FutureWarning`.

**Fix:** Remove the duplicate call. Each function needs exactly one `_resolve_intercept` call:
```python
# WRONG (current):
add_intercept = _resolve_intercept(add_intercept, with_intercept)
add_intercept = _resolve_intercept(add_intercept, with_intercept)

# CORRECT:
add_intercept = _resolve_intercept(add_intercept, with_intercept)
```

---

### WR-03: `gamma_standardized_pearson_residuals_fit` and `gamma_standardized_deviance_residuals_fit` missing minimum-input guard

**File:** `src/expressions/regression.rs:2457-2481`, `src/expressions/regression.rs:2493-2517`

**Issue:** Same pattern as CR-04. These two functions do not guard against `inputs.len() < 4`
before indexing `inputs[1]` and `inputs[2]`. The risk is lower than CR-04 because the output is
a residual vector (the NaN fallback is more informative), but the panic risk is identical.

**Fix:** Add `if inputs.len() < 4 { return residual_diag_nan_output(); }` at the top of both
functions, matching the pattern used in `pearson_chi_squared_logistic_fit` (line 2030).

---

### WR-04: `partial_fit` rebuilds a new `PassiveAggressiveRegressor` on every call — hyperparameters baked into `PaState` at first call are then overridden

**File:** `src/pymodels/py_passive_aggressive.rs:160-164`

**Issue:** `build_model()` constructs a brand-new `PassiveAggressiveRegressor` from the current
field values every time `partial_fit` is called. If the user mutates any hyperparameter between
calls (which is possible via direct field access if `#[pyclass]` exposes setters), the model used
for the update diverges from the model that initialized `PaState`. Even without mutation, this
design rebuilds a non-trivial struct on every sample, which is wasteful in a streaming loop.
More critically, `PaState::new(n_features)` is called with the length of the current `x_row`
slice — if two calls have different feature lengths, the state is silently re-initialized to the
second length on the first mismatch, discarding all prior learning.

**Fix:** Validate that `n_features == state.n_features()` before calling `get_or_insert_with`,
and raise `ValueError` on mismatch instead of silently re-initializing:
```rust
fn partial_fit(&mut self, x_row: PyReadonlyArray1<'_, f64>, y_value: f64) -> PyResult<()> {
    let slice = /* ... as above ... */;
    let n_features = slice.len();
    // Validate consistency if state already exists
    if let Some(ref state) = self.state {
        if state.n_features() != n_features {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "x_row has {} features but model was initialized with {}",
                n_features, state.n_features()
            )));
        }
    }
    let model = self.build_model();
    let state = self.state.get_or_insert_with(|| PaState::new(n_features));
    // ...
}
```

---

### WR-05: `pearson_chi_squared_logistic_fit` and `pearson_chi_squared_poisson_fit` compute chi-squared as the sum of squared residuals, not using the crate `pearson_chi_squared` function

**File:** `src/expressions/regression.rs:2047`, `src/expressions/regression.rs:2083`

**Issue:** The logistic and Poisson chi-squared functions compute the statistic as:
```rust
let chi2: f64 = (0..pr.nrows()).map(|i| pr[i] * pr[i]).sum();
```
This is `Σ r_P²` — equivalent to Pearson chi-squared only if `r_P` is already the Pearson
residual defined as `(y - μ) / sqrt(V(μ))`. The crate provides a dedicated `pearson_chi_squared`
function (imported, used in `gamma_pearson_chi_squared_fit` at line 2440). Using this function
for consistency would be more reliable and is the established pattern for Gamma. If `pr` from
`f.pearson_residuals()` is defined identically (`(y-μ)/sqrt(V(μ))`), then `Σ r_P²` and
`pearson_chi_squared(y, mu, family)` are equivalent and this is only a code-quality concern.
But if they differ in edge cases (e.g., saturated Binomial cells), the raw-sum approach will
silently diverge.

**Fix:** Use the imported `pearson_chi_squared` function consistently:
```rust
// In pearson_chi_squared_logistic_fit and pearson_chi_squared_poisson_fit:
let chi2 = pearson_chi_squared(&y_vec, &mu_vec, &family);
```
where `family` is `TweedieFamily::binomial()` / `TweedieFamily::poisson()` respectively.

---

### WR-06: LARS `eps` field is stored but silently discarded — API contract broken

**File:** `src/pymodels/py_lars.rs:55-56`, `src/pymodels/py_lars.rs:63`

**Issue:** The docstring documents `eps` as:

> Accepted for API compatibility; the underlying crate builder uses its own fixed default value.

The struct stores `eps` with `#[allow(dead_code)]` and the builder never passes it anywhere.
This is accepted by the author as a known limitation, but the Python signature claims `eps`
influences the computation. If users pass a custom `eps` expecting precision control (as in
scikit-learn's `LassoLars`), they silently receive a different numerical result with no warning.

**Fix:** Either:
1. Emit a `warnings.warn` from the Python side when `eps != f64::EPSILON` is detected, or
2. Plumb `eps` into the builder if the crate supports it, or
3. Remove `eps` from the signature entirely and document the break.

---

## Info

### IN-01: `PyOLS::summary()` uses generic `x1, x2, ...` labels regardless of actual column names

**File:** `src/pymodels/py_ols.rs:470-484`

**Issue:** The `summary()` method prints coefficient labels as `x1`, `x2`, etc. The user's actual
column names are lost because the PyO3 model API takes raw numpy arrays without column metadata.
This is a fundamental constraint of the design, but the docstring does not mention it, which could
confuse users who expect column names to appear in the summary output.

**Fix:** Document in the docstring that coefficient labels are positional (`x1`, `x2`, ...) and
that column names must be tracked by the caller.

---

### IN-02: Dead code — `col_into_vec` duplicates `col_to_vec` in `regression.rs`

**File:** `src/expressions/regression.rs:2210-2212`

**Issue:** `col_into_vec` (defined at line 2210) and `col_to_vec` (defined at line 477) are
identical functions with different names. The Gamma diagnostic section introduces `col_into_vec`
without consolidating with the existing helper.

**Fix:** Remove `col_into_vec` and replace its call sites with `col_to_vec`.

---

_Reviewed: 2026-08-12T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
