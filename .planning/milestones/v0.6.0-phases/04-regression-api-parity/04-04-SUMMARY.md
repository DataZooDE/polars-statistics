---
phase: 04-regression-api-parity
plan: "04"
subsystem: pymodels
tags: [regression, pymodel, robust, bayesian, pyo3, regr-06]
status: complete

dependency_graph:
  requires: [04-03]
  provides: [TheilSen, RANSAC, BayesianRidge, ARD PyModels]
  affects: [src/pymodels/, src/lib.rs, python/polars_statistics/__init__.py]

tech_stack:
  added:
    - TheilSenRegressor / FittedTheilSen from anofox-regression 0.5.13
    - RansacRegressor / FittedRansac from anofox-regression 0.5.13
    - BayesianRidge / FittedBayesianRidge from anofox-regression 0.5.13
    - ArdRegression / FittedArd from anofox-regression 0.5.13
  patterns:
    - PyO3 PyModel with FittedRegressor trait (fit/predict/is_fitted/getters)
    - Optional builder parameter pattern (if let Some(v) { b = b.setter(v); })
    - PyArray1::from_slice for bool/f64 slice getters (inlier_mask, lambdas, sigma_diag)
    - "#[allow(clippy::too_many_arguments)] on constructors with >= 8 params"

key_files:
  created:
    - src/pymodels/py_theil_sen.rs
    - src/pymodels/py_ransac.rs
    - src/pymodels/py_bayesian_ridge.rs
    - src/pymodels/py_ard.rs
    - tests/test_theil_sen.py
    - tests/test_ransac.py
    - tests/test_bayesian.py
  modified:
    - src/pymodels/mod.rs
    - src/lib.rs
    - python/polars_statistics/__init__.py

decisions:
  - "Registrations added to mod.rs and lib.rs during Tasks 1 and 2 (not deferred to Task 3) to keep clippy -D warnings clean throughout; Task 3 only added __init__.py"
  - "Maintained alphabetical ordering in mod.rs as enforced by cargo fmt"
  - "fit_intercept (BayesianRidge/ARD) vs with_intercept (TheilSen/RANSAC) preserved exactly per RESEARCH anti-pattern note"
  - "No alpha_init/lambda_init optional params for ARD (crate does not expose them for ArdRegressionBuilder)"

metrics:
  duration_minutes: 8
  completed_date: "2026-08-12"
  tasks_completed: 3
  tasks_total: 3
  commits: 3

estimate:
  tokens: 88000
  tasks: 3

actuals:
  tokens: 44000
  tasks: 3
  commits: 3
---

# Phase 04 Plan 04: TheilSen, RANSAC, BayesianRidge, ARD PyModels — REGR-06

Implemented four sklearn-style regression PyModel classes (TheilSen, RANSAC, BayesianRidge, ARD) following the established py_huber.rs pattern, registered them in all three shared files, and wrote smoke tests.

## One-liner

PyModel wrappers for TheilSen (spatial-median robust), RANSAC (consensus-set robust), BayesianRidge (SVD-based empirical Bayes), and ARD (per-feature automatic relevance determination) exposing their unique getters (inlier_mask, lambdas, alpha_, lambda_, sigma_diag).

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | PyTheilSen and PyRANSAC | 24858f8 | py_theil_sen.rs, py_ransac.rs, test_theil_sen.py, test_ransac.py |
| 2 | PyBayesianRidge and PyARD | b1cb02f | py_bayesian_ridge.rs, py_ard.rs, test_bayesian.py |
| 3 | Register in three shared files | 983a30e | __init__.py |

## What Was Built

### PyTheilSen (TheilSen class)

- Constructor: `with_intercept=True, max_subpopulation=10000, n_subsamples=None, max_iter=300, tol=1e-3, random_state=0`
- Getters: `coefficients`, `intercept`, `r_squared`, `mse`, `rmse`, `residuals`, `n_observations`
- Optional builder param `n_subsamples` applied only when `Some` per PATTERNS optional-param pattern
- No inference table (f_statistic/aic are NaN by design; exposed as-is via result())

### PyRANSAC (RANSAC class)

- Constructor: `with_intercept=True, min_samples=None, residual_threshold=None, max_trials=100, stop_probability=0.99, stop_n_inliers=None, random_state=0`
- Unique getter: `inlier_mask` (PyArray1<bool> via PyArray1::from_slice)
- Additional getters: `n_inliers`, `n_trials`, `residual_threshold`, `residuals`, `r_squared`, `n_observations`
- Three optional builder params applied only when Some
- ConvergenceFailed error mapped to PyValueError (T-04-09 mitigation)

### PyBayesianRidge (BayesianRidge class)

- Constructor: `fit_intercept=True, max_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6, alpha_init=None, lambda_init=None`
- Uses `fit_intercept` (not `with_intercept`) per REGR-06c anti-pattern note
- Hyperparameter getters: `alpha_` (noise precision), `lambda_` (weight precision), `sigma_diag` (posterior covariance diagonal)
- SingularMatrix error mapped to PyValueError (T-04-10 mitigation)

### PyARD (ARD class)

- Constructor: `fit_intercept=True, max_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6, threshold_lambda=10000.0`
- Uses `fit_intercept` (not `with_intercept`) per REGR-06c anti-pattern note
- Unique getter: `lambdas` (PyArray1<f64>, length n_features; via PyArray1::from_slice)
- `alpha_` getter for noise precision
- Features pruned to zero when `lambda_j > threshold_lambda` (crate handles internally)

### Registration

- `src/pymodels/mod.rs`: Added `mod py_ard`, `mod py_bayesian_ridge`, `mod py_ransac`, `mod py_theil_sen` (alphabetical) + 4 `pub use` exports
- `src/lib.rs`: Added 4 `m.add_class` lines under `// Robust & Sklearn-Style Solvers` comment
- `python/polars_statistics/__init__.py`: Added 4 imports + 4 `__all__` entries under `# Robust & sklearn-style solvers`

### Smoke Tests

- `tests/test_theil_sen.py`: 9 tests (fit, predict shape, not-fitted raises, intercept, r_squared, residuals shape, outlier robustness, no-intercept, n_observations)
- `tests/test_ransac.py`: 10 tests (fit, predict shape, not-fitted raises, inlier_mask length, dtype bool, n_inliers consistency, n_trials positive, residual_threshold positive, outlier robustness, custom threshold, n_observations)
- `tests/test_bayesian.py`: 10+10 tests — TestBayesianRidge (alpha_, lambda_, sigma_diag shape/positive) and TestARD (lambdas length, lambdas positive, alpha_, sparse recovery)

## Deviations from Plan

### Deviation 1 (Auto-fix — Rule 3)

**Registration in Tasks 1 and 2 (not deferred fully to Task 3):** The plan put all registration work in Task 3. However, clippy `-D warnings` with `unused_imports` would fail on `pub use py_theil_sen::PyTheilSen` etc. without a corresponding `m.add_class` reference in lib.rs. To keep each task's cargo checks green, the mod.rs and lib.rs registrations were added at the end of each implementation task. Task 3 was then left to handle only `__init__.py`. This is the minimal necessary deviation to satisfy the `cargo clippy --all-targets -- -D warnings` gate after each task commit.

### Deviation 2 (Research finding)

**ARD has no alpha_init/lambda_init in ArdRegressionBuilder:** The BayesianRidge plan mentions both parameters. The ARD crate source confirms `ArdRegressionBuilder` does not expose `alpha_init` or `lambda_init` setters (unlike BayesianRidgeBuilder which does). The PyARD constructor omits them accordingly.

## Cargo Gate Results

- `cargo build --features python`: PASSED
- `cargo clippy --all-targets --features python -- -D warnings`: PASSED
- `cargo fmt --check`: PASSED

## Known Stubs

None. All four PyModel classes are fully implemented with correct getters and builder wiring. Smoke tests are written but not executed (deferred to Plan 06 phase gate per plan spec).

## Threat Surface Scan

No new network endpoints, auth paths, or schema changes introduced. All four PyModels use the same PyO3 trust boundary pattern as existing wrappers:

- T-04-09 (RANSAC ConvergenceFailed) mitigated: mapped to PyValueError
- T-04-10 (BayesianRidge/ARD SingularMatrix) mitigated: mapped to PyValueError
- T-04-11 (TheilSen max_subpopulation compute) accepted: bounded by user-tunable default

## Self-Check: PASSED

All created files found on disk. All three task commits verified in git log (24858f8, b1cb02f, 983a30e).
