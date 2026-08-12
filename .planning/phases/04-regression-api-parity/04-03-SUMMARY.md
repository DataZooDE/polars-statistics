---
phase: "04-regression-api-parity"
plan: "03"
subsystem: "GLM diagnostics + HC inference extension"
tags: ["regression", "diagnostics", "glm", "gamma", "hc-inference", "ridge", "wls", "polars-expressions"]
status: complete

dependency_graph:
  requires: ["04-02"]
  provides: ["gamma-glm-diagnostics", "ridge-hc-inference", "wls-hc-inference"]
  affects: ["src/expressions/regression.rs", "src/pymodels/py_ridge.rs", "src/pymodels/py_wls.rs", "python/polars_statistics/exprs/regression.py", "python/polars_statistics/__init__.py"]

tech_stack:
  added:
    - "TweedieFamily::gamma() — Gamma GLM family trait object for diagnostic dispatch"
    - "dispersion_output_dtype / dispersion_output — new scalar struct output for φ̂ values"
  patterns:
    - "#[polars_expr] two-fn pattern (internal *_fit + public pl_* wrapper) for all 5 diagnostics"
    - "compute_hc_inference direct call for PyRidge/PyWLS (no FittedRidge.hc_inference — not available)"

key_files:
  created:
    - tests/test_glm_diagnostics.py
  modified:
    - src/expressions/regression.rs
    - src/pymodels/py_ridge.rs
    - src/pymodels/py_wls.rs
    - python/polars_statistics/exprs/regression.py
    - python/polars_statistics/exprs/__init__.py
    - python/polars_statistics/__init__.py

decisions:
  - "dispersion_output_dtype: single-field struct (not bare Float64) to stay consistent with the existing chi_squared_output_dtype/residual_diag_output_dtype pattern — all diagnostic outputs are structs"
  - "compute_hc_inference arg order corrected from RESEARCH.md draft: actual crate signature is (x, coef, intercept, residuals, aliased, with_intercept, hc_type, confidence_level) — no df param"
  - "redundant_closure lint: map_err(|e| PyErr::new(e)) -> map_err(PyErr::new) as required by clippy -D warnings"

metrics:
  duration_seconds: 5896
  completed: "2026-08-12"
  tasks_completed: 3
  commits: 1

actuals:
  tokens: 45000
  tasks: 3
  commits: 1
---

# Phase 04 Plan 03: GLM Diagnostics + HC Inference Extension Summary

Closed REGR-05 (5 Gamma GLM diagnostic Polars expressions) and REGR-04 (HC robust standard errors for Ridge and WLS), reusing the existing OLS HC path.

## What Was Built

### Task 1 + 3: 5 Gamma GLM Diagnostic Expressions (REGR-05)

Added to `src/expressions/regression.rs` following the `logistic_pearson_residuals_fit` / `pl_*` two-function pattern:

| Expression | Output dtype | Computes |
|---|---|---|
| `gamma_dispersion_deviance` | `dispersion_output_dtype` | φ̂ = D / (n − p) |
| `gamma_dispersion_pearson` | `dispersion_output_dtype` | φ̂ = X² / (n − p) |
| `gamma_pearson_chi_squared` | `chi_squared_output_dtype` | X² with df_resid, n_obs |
| `gamma_standardized_pearson_residuals` | `residual_diag_output_dtype` | r_P / √(φ·(1−h_ii)) |
| `gamma_standardized_deviance_residuals` | `residual_diag_output_dtype` | r_D / √(φ·(1−h_ii)) |

Each expression fits a Gamma GLM internally via `build_gamma_log()`, extracts fitted values from `f.inner().result().fitted_values`, constructs a `TweedieFamily::gamma()` family object, and calls the corresponding `anofox_regression::diagnostics` function. The standardized residual expressions additionally call `compute_leverage(&x, with_intercept)`.

A new `dispersion_output_dtype` / `dispersion_output` helper pair was added (struct with single `dispersion: Float64` field) alongside the existing `chi_squared_output` family.

All 5 expression names were added to:
- `python/polars_statistics/exprs/regression.py` (builder functions)
- `python/polars_statistics/exprs/__init__.py` (import + `__all__`)
- `python/polars_statistics/__init__.py` (import + `__all__`)

### Task 2: hc_inference for PyRidge and PyWLS (REGR-04)

Added `hc_inference(&self, py, x, hc_type="hc1")` method to both `PyRidge` and `PyWLS`, mirroring `PyOLS.hc_inference`. Both call `anofox_regression::inference::compute_hc_inference` directly (the `FittedRidge`/`FittedWLS` types do not expose their own `hc_inference` method). The method returns a Python dict with keys: `std_errors`, `t_statistics`, `p_values`, `conf_interval_lower`, `conf_interval_upper`, and optionally `intercept_std_error`, `intercept_t_statistic`, `intercept_p_value`.

### Task 1 + 2 (test file): `tests/test_glm_diagnostics.py`

Smoke tests for all 5 expressions (finite output, positive dispersion, correct shape) and Ridge/WLS `hc_inference` (finite std_errors, required keys, shape matches features, unfitted guard). Executed by Plan 06 phase gate (maturin develop + pytest).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] compute_hc_inference argument order mismatch**
- **Found during:** Task 2 first cargo build
- **Issue:** RESEARCH.md documented signature `(x, residuals, aliased, with_intercept, hc_type, coef, intercept, confidence_level, df)`. Actual crate signature: `(x, coefficients, intercept, residuals, aliased, with_intercept, hc_type, confidence_level)` — different argument order AND no `df` parameter (df is computed internally).
- **Fix:** Corrected argument order to match actual crate; removed `df` argument; removed `n`/`p` variables that were only needed to compute df.
- **Files modified:** `src/pymodels/py_ridge.rs`, `src/pymodels/py_wls.rs`
- **Commit:** 5f8a82c

**2. [Rule 1 - Clippy] Redundant closure lint in hc_inference**
- **Found during:** Task 2 clippy run
- **Issue:** `.map_err(|e| PyErr::new::<...>(e))` triggers `clippy::redundant_closure` under `-D warnings`.
- **Fix:** Changed to `.map_err(PyErr::new::<...>)`.
- **Files modified:** `src/pymodels/py_ridge.rs`, `src/pymodels/py_wls.rs`
- **Commit:** 5f8a82c (same commit — fix applied before commit)

## Verification

- `cargo build --features python`: EXIT 0
- `cargo clippy --all-targets --features python -- -D warnings`: EXIT 0 (no unused imports, no redundant closures)
- `cargo fmt --check`: EXIT 0
- `python -c "import ast; ast.parse(...)"`: EXIT 0 (both __init__.py files valid)
- All acceptance criteria grep checks: PASSED

## Known Stubs

None. All 5 expression builders call real Rust plugin functions with production paths. The HC inference methods call the real `compute_hc_inference` crate function.

## Self-Check

All committed files verified present:
- `src/expressions/regression.rs` — 5 pl_gamma_ functions: CONFIRMED (grep -c = 5)
- `src/pymodels/py_ridge.rs` — fn hc_inference: CONFIRMED
- `src/pymodels/py_wls.rs` — fn hc_inference: CONFIRMED
- `tests/test_glm_diagnostics.py` — EXISTS
- Commit 5f8a82c: CONFIRMED (git log)

## Self-Check: PASSED
