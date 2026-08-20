---
phase: 04-regression-api-parity
plan: "05"
subsystem: pymodels
tags: [LARS, PassiveAggressive, MomentAccumulator, streaming, online, path-regression, PyO3]
status: complete
completed: 2026-08-12
requirements: [REGR-06]
estimate:
  tokens: 90000
  tasks: 4
  confidence: med
actuals:
  tokens: 74000
  tasks: 3
  commits: 3

dependency_graph:
  requires: ["04-04"]
  provides: ["LARS PyModel", "PassiveAggressive PyModel with partial_fit", "MomentAccumulator PyModel", "OLS.fit_from_accumulator", "Ridge.fit_from_accumulator"]
  affects: ["src/pymodels/mod.rs", "src/lib.rs", "python/polars_statistics/__init__.py", "src/pymodels/py_ols.rs", "src/pymodels/py_ridge.rs"]

tech_stack:
  added: []
  patterns:
    - "Plain impl block for non-Python helper methods (build_model pattern for PassiveAggressive)"
    - "as_slice().unwrap() for 1-D numpy → &[f64] (NOT to_faer()) for push_row"
    - "pub(crate) inner field on PyMomentAccumulator for cross-module access in OLS/Ridge fit_from_accumulator"
    - "LarsMethod / PaLoss string dispatch — enums not registered as #[pyclass]"

key_files:
  created:
    - src/pymodels/py_lars.rs
    - src/pymodels/py_passive_aggressive.rs
    - src/pymodels/py_moment_accumulator.rs
    - tests/test_lars.py
    - tests/test_pa.py
    - tests/test_moments.py
  modified:
    - src/pymodels/py_ols.rs
    - src/pymodels/py_ridge.rs
    - src/pymodels/mod.rs
    - src/lib.rs
    - python/polars_statistics/__init__.py

decisions:
  - "fit_from_accumulator confirmed as exact method name (vs fit_from_moments) by reading anofox-regression-0.5.13/src/solvers/ols.rs — resolves Research Open Question 1"
  - "LarsMethod has no eps builder setter — eps accepted as PyLARS field for API compatibility but not passed through; #[allow(dead_code)] suppresses warning"
  - "build_model helper on PyPassiveAggressive placed in plain impl block (not #[pymethods]) to avoid PyO3 trying to expose PassiveAggressiveRegressor as a Python object"
  - "x_row.as_slice().unwrap().len() used instead of x_row.shape()[0] to avoid PyUntypedArrayMethods import"
  - "PyMomentAccumulator.inner made pub(crate) so fit_from_accumulator in py_ols.rs and py_ridge.rs can access it via crate-qualified path"
---

# Phase 04 Plan 05: LARS, PassiveAggressive, MomentAccumulator + fit_from_accumulator Summary

## One-liner

LARS path regression (alphas getter), PassiveAggressive online learner (partial_fit + PaState), MomentAccumulator streaming accumulator, and OLS/Ridge fit_from_accumulator — all three REGR-06 streaming/path gap models implemented, registered, and tested.

## Tasks Completed

| # | Name | Commit | Files |
|---|------|--------|-------|
| 1 | PyLARS + PyPassiveAggressive (tests) | 485609a | py_lars.rs, py_passive_aggressive.rs, test_lars.py, test_pa.py |
| 2 | PyMomentAccumulator + fit_from_accumulator on OLS/Ridge (tests) | 3b95f54 | py_moment_accumulator.rs, py_ols.rs, py_ridge.rs, test_moments.py |
| 3 | Register all three in mod.rs + lib.rs + __init__.py | 3f87699 | mod.rs, lib.rs, __init__.py |

## Cargo Gates

- `cargo build --features python`: PASSED (clean, no warnings)
- `cargo clippy --all-targets --features python -- -D warnings`: PASSED (clean)
- `cargo fmt --check`: PASSED

## Implementation Notes

### PyLARS

`#[pyclass(name = "LARS")]` wraps `LarsRegressor` with string-dispatch for `LarsMethod` ("lar" / "lasso"). Fields: `method`, `fit_intercept`, `n_nonzero_coefs: Option<usize>`, `alpha`, `standardize`, `eps` (compatibility field, unused). Builder has no `eps` setter so the crate default is used. Key getter: `alphas` returns `PyArray1::from_slice(py, fitted.alphas())`. The `coefs_path` field is private in the crate and not exposed.

### PyPassiveAggressive

`#[pyclass(name = "PassiveAggressive")]` wraps `PassiveAggressiveRegressor`. Contains both `fitted: Option<FittedPassiveAggressive>` and `state: Option<PaState>` for the streaming path. The `build_model` helper is in a plain `impl PyPassiveAggressive` block (NOT `#[pymethods]`) to avoid PyO3 attempting to expose `PassiveAggressiveRegressor` as a Python object. `partial_fit` calls `as_slice().unwrap()` on the 1-D numpy row — correct anti-pattern avoidance.

### PyMomentAccumulator

`#[pyclass(name = "MomentAccumulator")]` wraps `MomentAccumulator` with `inner` field marked `pub(crate)` so the OLS/Ridge `fit_from_accumulator` methods can reach it via the crate-qualified path `crate::pymodels::py_moment_accumulator::PyMomentAccumulator`. `push_row` validates arity before calling the crate method (T-04-12 threat mitigation), uses `as_slice().unwrap()` (NOT `to_faer()` anti-pattern). `xtx` / `sum_x` / `xty` returned via `IntoNumpy`.

### fit_from_accumulator on OLS/Ridge

Added to the existing `#[pymethods]` blocks of `PyOLS` and `PyRidge`. Both call the verified crate method `fit_from_accumulator(&acc.inner)` which is a convenience wrapper around `fit_from_moments`. Ridge stores result as `Box::new(fitted_ridge)` to match the existing `Box<dyn FittedRegressor + Send + Sync>` storage pattern.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `x_row.len()` returns `Result<usize, PyErr>` not `usize`**
- **Found during:** Task 2 build (first `cargo build` after implementing)
- **Issue:** `PyReadonlyArray1::len()` is from `PyUntypedArrayMethods` trait, returns `Result<usize, PyErr>`; using it directly in a comparison fails to compile
- **Fix:** Used `x_row.as_slice().unwrap().len()` which returns `usize` directly and avoids importing `PyUntypedArrayMethods`
- **Files modified:** py_moment_accumulator.rs, py_passive_aggressive.rs
- **Commit:** included in 485609a / 3b95f54

**2. [Rule 1 - Bug] `build_model` method in `#[pymethods]` block rejected by PyO3**
- **Found during:** Task 1 build
- **Issue:** PyO3 tries to wrap every method in `#[pymethods]` as a Python-callable; `PassiveAggressiveRegressor` doesn't implement `IntoPyObject` so PyO3 errors
- **Fix:** Moved `build_model` to a separate plain `impl PyPassiveAggressive` block outside `#[pymethods]`
- **Files modified:** py_passive_aggressive.rs
- **Commit:** 485609a

**3. [Rule 1 - Bug] Borrow conflict in `partial_fit` — mutable `&mut self.state` overlaps immutable `&self` for `build_model`**
- **Found during:** Task 1 build
- **Issue:** `get_or_insert_with` holds `&mut self.state` while `build_model` needs `&self`; Rust borrow checker rejects
- **Fix:** Build model before calling `get_or_insert_with` so immutable borrow ends before mutable borrow begins
- **Files modified:** py_passive_aggressive.rs
- **Commit:** 485609a

**4. [Rule 2 - Missing functionality] `eps` field has no builder setter in LarsRegressorBuilder**
- **Found during:** Task 1 implementation (reading crate source)
- **Issue:** RESEARCH.md listed `eps` as a parameter but `LarsRegressorBuilder` only exposes method/fit_intercept/n_nonzero_coefs/alpha/standardize setters — no `eps` setter
- **Fix:** Kept `eps` as an accepted parameter for API compatibility with `#[allow(dead_code)]`; documented that the crate uses its own fixed default
- **Files modified:** py_lars.rs
- **Commit:** 485609a

**5. [Rule 1 - Formatting] cargo fmt moved module declarations to alphabetical order**
- **Found during:** Task 3 (cargo fmt --check failed after manual edits)
- **Fix:** Applied `cargo fmt` which reordered `mod py_lars;`, `mod py_moment_accumulator;`, `mod py_passive_aggressive;` to alphabetical position
- **Files modified:** mod.rs, py_lars.rs, py_passive_aggressive.rs
- **Commit:** included in 3f87699

## Known Stubs

None. All three models have real implementations backed by crate solvers.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced. The new PyO3 classes only accept numpy arrays and primitive Python values at trust boundaries — covered by the existing T-04-12 mitigation (push_row arity check) and T-04-13/T-04-14 accepted risks documented in the plan.

## Self-Check: PASSED

- src/pymodels/py_lars.rs: FOUND
- src/pymodels/py_passive_aggressive.rs: FOUND
- src/pymodels/py_moment_accumulator.rs: FOUND
- tests/test_lars.py: FOUND
- tests/test_pa.py: FOUND
- tests/test_moments.py: FOUND
- Commit 485609a: FOUND
- Commit 3b95f54: FOUND
- Commit 3f87699: FOUND
- grep 'name = "LARS"' py_lars.rs: FOUND
- grep 'fn alphas' py_lars.rs: FOUND
- grep 'name = "PassiveAggressive"' py_passive_aggressive.rs: FOUND
- grep 'fn partial_fit' py_passive_aggressive.rs: FOUND
- grep 'PaState' py_passive_aggressive.rs: FOUND
- grep 'as_slice' py_passive_aggressive.rs: FOUND
- grep 'name = "MomentAccumulator"' py_moment_accumulator.rs: FOUND
- grep 'fn push_row' py_moment_accumulator.rs: FOUND
- grep 'as_slice' py_moment_accumulator.rs: FOUND
- grep 'fn fit_from_accumulator' py_ols.rs: FOUND
- grep 'fn fit_from_accumulator' py_ridge.rs: FOUND
- cargo build --features python: PASSED
- cargo clippy --all-targets --features python -- -D warnings: PASSED
- cargo fmt --check: PASSED
