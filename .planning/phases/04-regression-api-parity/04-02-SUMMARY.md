---
phase: 04-regression-api-parity
plan: "02"
subsystem: pymodels
tags: [glmm, pspline, pymodel, regression, rust, pyo3, wave-2]
status: complete

dependencies:
  requires: [04-01]
  provides: [GLMM-pymodel, PSpline-pymodel]
  affects: [src/pymodels/, src/lib.rs, python/polars_statistics/__init__.py]

tech_stack:
  added:
    - "PyGLMM: GlmmRegressor factory-staticmethod pattern with gaussian/poisson/binomial families"
    - "PyPSpline: PSplineRegressor builder chain with GCV-selected smoothing"
  patterns:
    - "Factory staticmethod construction (GLMM family dispatch via string → GlmmRegressor::gaussian/poisson/binomial)"
    - "Vec<u64> → Vec<usize> group conversion for 32/64-bit portability"
    - "ncols guard after to_faer() (matching py_isotonic.rs pattern)"
    - "FactorSummary exposed as list[dict] (no separate PyClass)"

key_files:
  created:
    - src/pymodels/py_glmm.rs
    - src/pymodels/py_pspline.rs
    - tests/test_glmm.py
    - tests/test_pspline.py
  modified:
    - src/pymodels/mod.rs
    - src/lib.rs
    - python/polars_statistics/__init__.py

decisions:
  - "group validation happens before fit() call per threat model T-04-03 (crate returns Err, mapped to PyValueError)"
  - "ncols guard placed after to_faer() to reuse faer matrix API (matching existing py_isotonic.rs pattern)"
  - "random_sd() returns Vec<f64> so PyArray1::from_slice used (not IntoNumpy which requires Col<f64>)"
  - "Regressor trait import added to py_pspline.rs to access PSplineRegressor::fit (trait method)"
  - "PyGLMM and PyPSpline added to lib.rs during their respective implementation tasks to keep clippy -D warnings green"

metrics:
  duration: "56 minutes"
  completed: "2026-08-12"
  tasks_completed: 3
  tasks_planned: 3
  commits: 3
  files_created: 4
  files_modified: 3

actuals:
  tokens: 6342
  tasks: 3
  commits: 3
---

# Phase 04 Plan 02: GLMM + PSpline PyModels Summary

GLMM (GlmmRegressor with crossed-factor support and FactorSummary list-of-dicts output) and PSpline (GCV-selected B-spline smoother with ncols validation) implemented as PyO3 PyModel classes, registered in all three shared files, and covered by smoke-test files for the plan-06 phase gate.

## What Was Built

### Task 1: PyGLMM (src/pymodels/py_glmm.rs)

`PyGLMM` with `#[pyclass(name = "GLMM")]` implementing:

- Three `#[staticmethod]` constructors: `gaussian()`, `poisson()`, `binomial()`, each accepting `(with_intercept=true, reml=true/false, max_iter=100, tol=1e-8)`.
- `fit(x, y, group: Vec<u64>)` — accepts Python `list[int]` or numpy int array; converts portably via `.iter().map(|&g| g as usize)` before calling `GlmmRegressor::fit`.
- `fit_crossed(x, y, groups: Vec<Vec<u64>>)` — converts each inner list to `Vec<usize>`, builds `Vec<&[usize]>` slices for `GlmmRegressor::fit_crossed`.
- `factors()` — returns `Vec<Bound<PyDict>>` with keys `n_levels`, `sd`, `blups` (as `PyArray1::from_slice`). No separate `PyFactorSummary` class.
- `predict_fixed(x)` — marginal predictions using fixed effects only.
- Getters: `fixed_effects`, `std_errors`, `intercept`, `slopes`, `random_effects`, `random_sd`, `theta`, `sigma`, `deviance`, `log_likelihood`, `n_groups`, `converged`, `iterations`.
- `is_fitted()` guard on all getters.

### Task 2: PyPSpline (src/pymodels/py_pspline.rs)

`PyPSpline` with `#[pyclass(name = "PSpline")]` implementing:

- `#[new]` with `(n_basis=0, penalty_order=2, lambda_=None)`.
- `fit(x, y)` — converts to faer first, then checks `x_mat.ncols() != 1` and raises `PyValueError("PSpline requires a single-column predictor matrix (n, 1).")`.
- Builder chain: `PSplineRegressor::new().with_n_basis().with_penalty_order()` then conditionally `.with_lambda(l)` when `Some`.
- `predict(x)` via `FittedRegressor::predict`.
- Getters: `edf` (FittedPSpline.edf()), `sigma2` (FittedPSpline.sigma2()), `coefficients` (B-spline basis coefs via FittedRegressor::coefficients()), `r_squared`, `rmse` (from result()). AIC/BIC intentionally not exposed per RESEARCH.

### Task 3: Registration

- `src/pymodels/mod.rs`: `mod py_glmm;` (after py_gamma) + `mod py_pspline;` (alphabetical after py_probit); `pub use py_glmm::PyGLMM;` + `pub use py_pspline::PyPSpline;`.
- `src/lib.rs`: `PyGLMM` under `// GLM Models`; `PyPSpline` under new `// Smoothers` comment.
- `python/polars_statistics/__init__.py`: `GLMM` in import block (under `# GLM models`) + `"GLMM"` in `__all__`; `PSpline` in import block (under `# Smoothers`) + `"PSpline"` in `__all__`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] random_sd() returns Vec<f64>, not Col<f64>**
- **Found during:** Task 1 build verification
- **Issue:** `FittedGlmm::random_sd()` returns `Vec<f64>`, not `Col<f64>`. Calling `.into_numpy(py)` (which is only implemented for `Col<f64>`) caused a compile error.
- **Fix:** Changed to `PyArray1::from_slice(py, &fitted.random_sd())`.
- **Files modified:** `src/pymodels/py_glmm.rs`
- **Commit:** 4becf5d

**2. [Rule 1 - Bug] PSpline x.shape()[1] - wrong numpy API**
- **Found during:** Task 2 build verification
- **Issue:** `PyReadonlyArray2` has no `.shape()` method; tried to check column count before `to_faer()`. The established pattern (from `py_isotonic.rs`) checks `x_mat.ncols()` after converting to faer.
- **Fix:** Moved the ncols guard after `x.to_faer()`, consistent with existing `py_isotonic.rs` `fit_2d` pattern.
- **Files modified:** `src/pymodels/py_pspline.rs`
- **Commit:** 95d781f

**3. [Rule 1 - Bug] PSplineRegressor::fit requires Regressor trait in scope**
- **Found during:** Task 2 build verification
- **Issue:** `GlmmRegressor` has `fit`/`fit_crossed` defined directly on its impl; `PSplineRegressor` implements the `Regressor` trait and needs `use anofox_regression::solvers::Regressor` in scope.
- **Fix:** Added `Regressor` to the import line.
- **Files modified:** `src/pymodels/py_pspline.rs`
- **Commit:** 95d781f

**4. [Rule 3 - Blocking] Unused pub use warnings with clippy -D warnings**
- **Found during:** Task 1 build verification
- **Issue:** Adding `pub use py_glmm::PyGLMM` in mod.rs before PyGLMM is registered in lib.rs causes an "unused import" error under `clippy -D warnings`. The plan's Task 3 sequencing assumes builds stay green between tasks.
- **Fix:** Added `m.add_class::<pymodels::PyGLMM>()?;` to lib.rs as part of Task 1; added `m.add_class::<pymodels::PyPSpline>()?;` as part of Task 2. Task 3 completed the remaining `__init__.py` registration.
- **Impact:** lib.rs was modified in Tasks 1 and 2 in addition to Task 3. This is still a single-task-per-shared-file principle for `__init__.py`; lib.rs was necessarily updated alongside each implementation.

**5. [Rule 1 - Bug] mod.rs alphabetical ordering**
- **Found during:** Task 2 fmt check
- **Issue:** `mod py_pspline;` was placed before `mod py_probit;` in mod.rs, violating alphabetical order (ps > pr alphabetically but the formatter detected a diff). Fixed ordering: py_probit before py_pspline. Same for pub use exports.
- **Fix:** Reordered mod declarations and pub use exports.
- **Files modified:** `src/pymodels/mod.rs`
- **Commit:** 95d781f

## Cargo Gate Results

All three gates green after Task 3:
- `cargo build --features python`: exit 0
- `cargo clippy --all-targets --features python -- -D warnings`: exit 0 (no warnings)
- `cargo fmt --check`: exit 0

## Known Stubs

None — all getters call real crate methods. `factors()` returns an empty list for single-factor fits (correct crate behavior per RESEARCH: `factors()` is empty for `fit()`-based fits, non-empty only for `fit_crossed()`-based fits).

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced. Threat mitigations T-04-03 and T-04-04 implemented as designed:
- T-04-03 (GLMM group array length): crate's `GlmmRegressor::fit` validates internally and returns `Err` mapped to `PyValueError`.
- T-04-04 (PSpline multi-column X): explicit `ncols == 1` guard raises `PyValueError` before `predict()` could silently ignore extra columns.

## Self-Check: PASSED

**Created files:**
- FOUND: src/pymodels/py_glmm.rs
- FOUND: src/pymodels/py_pspline.rs
- FOUND: tests/test_glmm.py
- FOUND: tests/test_pspline.py

**Commits:**
- FOUND: 4becf5d (feat(04-02): implement PyGLMM)
- FOUND: 95d781f (feat(04-02): implement PyPSpline)
- FOUND: 6c8e8e4 (chore(04-02): register GLMM and PSpline)
