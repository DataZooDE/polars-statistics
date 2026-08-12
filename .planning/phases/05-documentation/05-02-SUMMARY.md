---
phase: 05-documentation
plan: "02"
subsystem: documentation
status: complete
tags: [docs, rust-doc, pymodel, docstrings, DOCS-01, DOCS-03]
completed: 2026-08-12

dependency_graph:
  requires:
    - 05-01 (check_docstring.py checker, tracer pattern)
  provides:
    - All new statistics expressions documented (DOCS-01)
    - All 10 new regression PyModel classes carry /// class doc + runnable >>> example (DOCS-03)
    - All new *_fit expression wrappers carry /// doc comments (DOCS-03)
    - WLS hc_inference NotImplementedError limitation documented
    - ICC matrix-input contract documented in icc() docstring
    - check_rust_docs.py checker for DOCS-03 gate (shared by future plans)
  affects:
    - src/pymodels/py_theil_sen.rs
    - src/pymodels/py_ransac.rs
    - src/pymodels/py_bayesian_ridge.rs
    - src/pymodels/py_ard.rs
    - src/pymodels/py_lars.rs
    - src/pymodels/py_passive_aggressive.rs
    - src/pymodels/py_ridge.rs
    - src/pymodels/py_wls.rs
    - .planning/phases/05-documentation/check_rust_docs.py

tech_stack:
  added: []
  patterns:
    - Rust /// class doc with runnable >>> example as Python docstring (via PyO3)
    - grep-then-walk-back algorithm for contiguous /// block extraction (check_rust_docs.py)

key_files:
  created:
    - .planning/phases/05-documentation/check_rust_docs.py
  modified:
    - src/pymodels/py_theil_sen.rs
    - src/pymodels/py_ransac.rs
    - src/pymodels/py_bayesian_ridge.rs
    - src/pymodels/py_ard.rs
    - src/pymodels/py_lars.rs
    - src/pymodels/py_passive_aggressive.rs
    - src/pymodels/py_ridge.rs
    - src/pymodels/py_wls.rs

decisions:
  - Python statistics expression docstrings (two_way_anova, repeated_measures_anova,
    energy_distance_nd, icc) were already complete from Phase 3 implementation —
    audit confirmed no gap to fill for DOCS-01 on these four symbols.
  - Gamma/GLMM/PSpline PyModel /// docs were already complete from Phase 4 —
    audit confirmed no gap; used py_gamma.rs as the completeness bar for the audit.
  - TheilSen/RANSAC/BayesianRidge/ARD were missing class >>> examples and had
    undocumented getters — filled all gaps; each now matches the py_gamma.rs bar.
  - LARS and PassiveAggressive were nearly complete — only is_fitted() was missing ///.
  - Ridge and WLS had no class example, no /// on fit/predict/is_fitted/coefficients/
    intercept/r_squared — filled all gaps.
  - WLS hc_inference: expanded /// doc block to explicitly state NotImplementedError,
    include the reason (weight vector not accessible via trait), and document the
    workaround (scale X/y by sqrt(w_i), use OLS.hc_inference). The constraint was
    already in an implementation comment; it is now in the user-visible PyO3 ///.
  - check_rust_docs.py uses a walk-backwards algorithm instead of a single regex
    to handle /// blocks with interleaved #[pyo3(...)] attribute lines between
    the closing /// line and the fn declaration.
  - All 5 gamma_* diagnostic expression wrappers (gamma_dispersion_deviance_fit,
    gamma_dispersion_pearson_fit, gamma_pearson_chi_squared_fit,
    gamma_standardized_pearson_residuals_fit, gamma_standardized_deviance_residuals_fit)
    and all statistics expression wrappers (two_way_anova_fit,
    repeated_measures_anova_fit, energy_distance_nd_fit, icc_fit) already carried
    /// doc comments from Phase 3/4 implementation — checker confirms this.

metrics:
  duration_minutes: ~8
  tasks_completed: 3
  tasks_total: 3
  commits: 2

actuals:
  tokens: 11000
  tasks: 3
  commits: 2
---

# Phase 05 Plan 02: DOCS-01/03 Audit-and-Fill — All Remaining New Symbols Summary

One-liner: Audited and filled Python docstrings (DOCS-01) and Rust /// comments (DOCS-03) for every remaining new symbol from Phases 3-4; authored check_rust_docs.py as the shared DOCS-03 gate.

## What Was Built

### Task 1 — Fill Python docstrings for remaining statistics expressions (DOCS-01)

Audit result: **all four statistics expressions already had complete docstrings** from Phase 3 implementation — no edits required.

| Symbol | File | Status | Notes |
|--------|------|--------|-------|
| `two_way_anova` | `parametric.py` | Already complete | Parameters + >>> example present |
| `repeated_measures_anova` | `parametric.py` | Already complete | Parameters + >>> example present |
| `energy_distance_nd` | `modern.py` | Already complete | Parameters + >>> + Raises sections present |
| `icc` | `correlation.py` | Already complete | Parameters + >>> example; matrix-input contract documented (n_raters as *args) |

STAT coverage: STAT-02/03/04/05 (two_way_anova, repeated_measures_anova, energy_distance_nd, icc).

### Task 2 — Audit/fill PyO3 /// docstrings on 10 new regression PyModels (DOCS-01 + DOCS-03)

Audit result: Gamma/GLMM/PSpline were already complete. Seven others needed gaps filled.

| Class | File | Changes Made |
|-------|------|-------------|
| `PyGamma` | `py_gamma.rs` | Already complete — audit only |
| `PyGLMM` | `py_glmm.rs` | Already complete — audit only |
| `PyPSpline` | `py_pspline.rs` | Already complete — audit only |
| `PyTheilSen` | `py_theil_sen.rs` | Added class >>> example; /// on fit, predict, is_fitted, coefficients, intercept, r_squared, mse, rmse, residuals, n_observations |
| `PyRANSAC` | `py_ransac.rs` | Added class >>> example; /// on predict, is_fitted, coefficients, intercept, r_squared, residuals, n_observations |
| `PyBayesianRidge` | `py_bayesian_ridge.rs` | Added class >>> example; /// on predict, is_fitted, coefficients, intercept, r_squared, residuals, n_observations |
| `PyARD` | `py_ard.rs` | Added class >>> example; /// on predict, is_fitted, coefficients, intercept, r_squared, residuals, n_observations |
| `PyLARS` | `py_lars.rs` | Added /// on is_fitted |
| `PyPassiveAggressive` | `py_passive_aggressive.rs` | Added /// on is_fitted |
| `PyMomentAccumulator` | `py_moment_accumulator.rs` | Already complete — audit only |

**OLS/Ridge fit_from_accumulator and hc_inference** (py_ols.rs, py_ridge.rs): already had complete /// docs from Phase 4 — no edits required.

**Ridge** (py_ridge.rs): Added class >>> example and /// on fit, predict, is_fitted, coefficients, intercept, r_squared, adj_r_squared, lambda_value.

**WLS** (py_wls.rs):
- Added class >>> example with the NotImplementedError limitation note
- Added /// on predict, is_fitted, coefficients, intercept, r_squared
- **Expanded hc_inference /// block** to explicitly state it raises `NotImplementedError`, explain why (weight vector inaccessible via FittedRegressor trait), and document the workaround

### Task 3 — Author check_rust_docs.py + confirm Rust expression wrapper /// docs (DOCS-03)

**`check_rust_docs.py`** created at `.planning/phases/05-documentation/`:
- Checks all 10 new PyModel structs for `///` class doc with `>>>` runnable example
- Checks all new `pub fn *_fit` wrappers have `///` immediately preceding them
- Checks py_wls.rs `hc_inference` doc mentions `NotImplementedError`
- Uses a walk-backwards algorithm to handle `#[pyo3(...)]` attributes between /// and fn

**All expression wrappers already carried /// docs:**
- `two_way_anova_fit`, `repeated_measures_anova_fit` (parametric.rs) — confirmed
- `energy_distance_nd_fit` (modern.rs) — confirmed
- `icc_fit` (correlation.rs) — confirmed
- `gamma_dispersion_deviance_fit`, `gamma_dispersion_pearson_fit`, `gamma_pearson_chi_squared_fit`, `gamma_standardized_pearson_residuals_fit`, `gamma_standardized_deviance_residuals_fit` (regression.rs) — all confirmed

## Verification Results

```
$ python .planning/phases/05-documentation/check_docstring.py two_way_anova python/polars_statistics/exprs/parametric.py
OK: 'two_way_anova' has Parameters section and runnable examples

$ python .planning/phases/05-documentation/check_docstring.py repeated_measures_anova python/polars_statistics/exprs/parametric.py
OK: 'repeated_measures_anova' has Parameters section and runnable examples

$ python .planning/phases/05-documentation/check_docstring.py energy_distance_nd python/polars_statistics/exprs/modern.py
OK: 'energy_distance_nd' has Parameters section and runnable examples

$ python .planning/phases/05-documentation/check_docstring.py icc python/polars_statistics/exprs/correlation.py
OK: 'icc' has Parameters section and runnable examples

$ python .planning/phases/05-documentation/check_rust_docs.py
OK: 'PyGamma' has /// class doc with runnable example
OK: 'PyGLMM' has /// class doc with runnable example
OK: 'PyPSpline' has /// class doc with runnable example
OK: 'PyTheilSen' has /// class doc with runnable example
OK: 'PyRANSAC' has /// class doc with runnable example
OK: 'PyBayesianRidge' has /// class doc with runnable example
OK: 'PyARD' has /// class doc with runnable example
OK: 'PyLARS' has /// class doc with runnable example
OK: 'PyPassiveAggressive' has /// class doc with runnable example
OK: 'PyMomentAccumulator' has /// class doc with runnable example
OK: 'pub fn two_way_anova_fit' has a /// doc comment
OK: 'pub fn repeated_measures_anova_fit' has a /// doc comment
OK: 'pub fn energy_distance_nd_fit' has a /// doc comment
OK: 'pub fn icc_fit' has a /// doc comment
OK: 'pub fn gamma_dispersion_deviance_fit' has a /// doc comment
OK: 'pub fn gamma_dispersion_pearson_fit' has a /// doc comment
OK: 'pub fn gamma_pearson_chi_squared_fit' has a /// doc comment
OK: 'pub fn gamma_standardized_pearson_residuals_fit' has a /// doc comment
OK: 'pub fn gamma_standardized_deviance_residuals_fit' has a /// doc comment
OK: py_wls.rs hc_inference doc mentions NotImplementedError

All Rust doc checks PASSED.
```

## Documented Caveats

**WLS hc_inference NotImplementedError** — documented in the `///` block of `py_wls.rs::hc_inference` and in the WLS class `///` block. The reason (weight vector inaccessible via `FittedRegressor` trait after fitting) and the workaround (scale X/y by `sqrt(w_i)`, use `OLS.hc_inference`) are both stated.

**ICC matrix-input contract** — the `icc()` docstring (correlation.py) already correctly describes the new contract: N rater columns passed as positional `*rater_cols` args, with rows as subjects. The former all-NaN stub is replaced; the contract change is visible in the Parameters section and the >>> example.

## Deviations from Plan

**[Audit-confirmed no-ops]** Four statistics expressions (two_way_anova, repeated_measures_anova, energy_distance_nd, icc) and three PyModels (Gamma, GLMM, PSpline) were already complete from their Phase 3/4 implementation plans — no edits required. This is the expected outcome of an audit-and-fill plan; it does not reduce coverage.

**[check_rust_docs.py walk-back algorithm]** The initial regex for checking the WLS hc_inference /// block failed because a `#[pyo3(signature = ...)]` attribute sits between the last `///` line and `fn hc_inference`. Fixed by using a backwards line-walk that skips attribute lines and blank lines, matching the actual Rust source layout.

No architectural deviations. No behavior changes.

## Known Stubs

None — all new docstrings contain real, runnable inline examples that require no external data.

## Self-Check

- [x] check_docstring.py passes for two_way_anova, repeated_measures_anova, energy_distance_nd, icc
- [x] check_rust_docs.py exits 0 (all 10 new PyModels + all new *_fit wrappers carry ///)
- [x] py_wls.rs hc_inference doc mentions NotImplementedError
- [x] icc docstring notes the matrix-input contract (n_raters as *rater_cols args)
- [x] Commit a205170 exists (PyModel /// fills)
- [x] Commit cdb6d77 exists (check_rust_docs.py)
