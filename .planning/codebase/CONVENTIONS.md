# Coding Conventions

**Analysis Date:** 2026-08-11

## Naming Patterns

**Files:**
- Snake case for module files: `py_ols.rs`, `py_elastic_net.rs`, `py_mann_whitney.rs`
- Prefixes by category:
  - `py_*.rs` for PyO3 wrapper modules in `src/pymodels/`
  - No prefix for core expression modules in `src/expressions/`

**Functions:**
- Snake case: `ols_fit`, `ridge_fit`, `quantile_fit`, `binom_test_fit`
- Suffix patterns:
  - `*_fit` for main model/test fit functions
  - `*_summary_fit` for summary variants (tidy coefficient output)
  - `*_predict_fit` for prediction variants
  - `*_residuals_fit` for residual diagnostics
- Helper functions: `parse_solver_type`, `parse_alternative`, `parse_hc_type`
- Internal/private: prefix with underscore when needed (e.g., `_build_xy_with_null_policy`)

**Variables:**
- Snake case throughout: `with_intercept`, `conf_level`, `solve_method`, `n_observations`
- Abbreviations in type names: `xy` (features-target pair), `glm` (generalized linear model), `ols` (ordinary least squares), `vif` (variance inflation factor)
- Field names in structs: snake case with underscore: `fitted`, `solve_method`, `with_intercept`

**Types:**
- Upper camel case for structs: `PyOLS`, `PyElasticNet`, `PyQuantile`, `PyLogisticRegression`
- Type aliases lowercase: `XyNullPolicyResult`, `OlsResidualContext`
- Generic params: uppercase single letter or descriptive (e.g., `T`, `'py` for Python lifetime)

## Code Style

**Formatting:**
- Edition 2021 (see `Cargo.toml`)
- Line length: observe ~100 char soft limit in most modules
- Indentation: 4 spaces

**Linting:**
- No explicit `.clippy.toml` or `clippy.toml` detected
- Follows standard Rust conventions and Polars ecosystem patterns
- Common allow attributes: `#[allow(unused_imports)]` for FFI re-exports

## Import Organization

**Order:**
1. External crates (polars, pyo3, numpy, anofox-*): grouped by purpose
2. Module-level imports: `use crate::expressions::*`, `use crate::utils::*`, `use crate::pymodels::*`
3. Internal submodule imports within same module

**Examples from `src/pymodels/py_ols.rs`:**
```rust
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::prelude::*;
use anofox_regression::solvers::{FittedOls, OlsRegressor, Regressor};
use anofox_regression::{HcType, SolverType};

use crate::utils::{IntoNumpy, ToFaer};
```

**Path Aliases:**
- None detected; full paths used throughout

## Error Handling

**Patterns:**
- Primary return type: `PolarsResult<T>` for Polars expressions (wraps `Result<T, PolarsError>`)
- PyO3 return type: `PyResult<T>` for Python methods (wraps `Result<T, PyErr>`)
- Error conversion: `.map_err()` for converting errors between types
  - Example: `.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?`
- Null-coalescing for optional values: `.get(0).unwrap_or(default)` in scalar extraction
- Model state checks: `.ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?`

**Examples:**
```rust
// From py_ols.rs line 107
.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?

// From py_ols.rs lines 129-132
let fitted = self
    .fitted
    .as_ref()
    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

// From categorical.rs line 94
let successes = inputs[0].u32()?.get(0).unwrap_or(0) as usize;
```

## Logging

**Framework:** No explicit logging framework detected (log, tracing, etc.)

**Patterns:**
- Errors reported via `PolarsResult` and `PyResult` return types
- No debug/info logging in observed code
- Error messages propagated to caller (Python or Polars expression system)

## Comments

**When to Comment:**
- Module-level `//!` doc comments on all public modules and files
- Function-level `///` doc comments on all public functions
- Type-level `///` doc comments on struct/enum definitions
- Inline comments rare; code assumed self-documenting

**JSDoc/TSDoc:**
- Uses Rust-style doc comments (`///` for items, `//!` for modules)
- PyO3 modules document Python API with docstring sections: Parameters, Returns, Examples
- Example format from `py_ols.rs`:
  ```rust
  /// Ordinary Least Squares regression model.
  ///
  /// Fits a linear model with coefficients w = (w1, ..., wp) to minimize
  /// the residual sum of squares between the observed targets and the
  /// predictions.
  ///
  /// Parameters
  /// ----------
  /// with_intercept : bool, default True
  ///     Whether to include an intercept term in the model.
  ///
  /// Examples
  /// --------
  /// >>> import numpy as np
  /// >>> from polars_statistics import OLS
  ```

## Function Design

**Size:** Most functions 20-50 lines; regression fit functions larger (~100-150 lines due to parameter extraction and error handling)

**Parameters:**
- PyO3 methods: prefer `#[pyo3(signature = (...))]` for default parameters
- Polars expression functions: accept `&[Series]` slice, extract by index
- Helper functions: typed parameters with no reliance on positional index

**Return Values:**
- Polars expressions: `PolarsResult<Series>` (returns Struct series with named fields)
- PyO3 methods: `PyResult<T>` where T is numpy array, dict, String, or bool
- Helpers: explicit `PolarsResult` for error propagation

## Module Design

**Exports:**
- `src/lib.rs`: gate all modules behind `#[cfg(feature = "python")]` except `pub mod expressions`
- `src/expressions/mod.rs`: re-exports all submodule contents with `#[allow(unused_imports)]`
- `src/pymodels/mod.rs`: selective re-exports of public struct types only

**Barrel Files:**
- `src/expressions/mod.rs` (line 6-37): re-exports all expression submodules
  ```rust
  mod categorical;
  mod correlation;
  // ...
  pub use categorical::*;
  pub use correlation::*;
  ```
- `src/pymodels/mod.rs` (line 39-71): re-exports PyO3 classes for Python module registration

**Struct Fields:**
- Private fields by default in PyO3 structs (e.g., `struct PyOLS { with_intercept: bool, fitted: Option<FittedOls> }`)
- Getters via `#[getter]` for property-style access from Python
- Methods use `#[pymethods]` for Python-callable methods

## Feature Gates

**Pattern:**
- `#[cfg(feature = "python")]` gates Python-specific code at module and statement level
- `src/lib.rs` (lines 13-25): conditional compilation of Python bindings
- Default feature: `default = ["python"]`
- Crate usable as rlib without `python` feature for non-Python consumers

## Polars Expression Macro Usage

**Pattern:**
- `#[polars_expr]` macro from `pyo3_polars::derive` decorates both public and private fit functions
- Public fit function: exposed via FFI for Polars
- Private `pl_*` variant: internal implementation (may not always exist)
- Example from `categorical.rs`:
  ```rust
  pub fn binom_test_fit(inputs: &[Series]) -> PolarsResult<Series>
  
  #[polars_expr]
  fn pl_binom_test(inputs: &[Series]) -> PolarsResult<Series>
  ```

---

*Convention analysis: 2026-08-11*
