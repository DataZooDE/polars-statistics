<!-- refreshed: 2026-08-11 -->
# Architecture

**Analysis Date:** 2026-08-11

## System Overview

```text
┌────────────────────────────────────────────────────────────────────┐
│                    Python/Rust Bridge (PyO3)                       │
│              `src/lib.rs` — #[pymodule] _polars_statistics        │
├──────────────┬───────────────────────┬──────────────┬──────────────┤
│ PyModels     │  Polars Expressions   │  Utilities   │              │
│ `src/pymodels` │  `src/expressions` │ `src/utils`  │              │
│  (40 model    │   (7 test/regression │ (array       │              │
│   wrappers)   │    module groups)    │  conversion) │              │
└────────┬──────┴───────────────────┬──┴──────────────┴──────────────┘
         │                          │
         └──────────────┬───────────┘
                        │
         ┌──────────────▼──────────────┐
         │   Rust Backend              │
         │   (anofox-regression        │
         │    anofox-statistics)       │
         │   (faer linear algebra)     │
         │   (statrs distributions)    │
         └─────────────────────────────┘
                        │
         ┌──────────────▼──────────────────────┐
         │  Python Wrapper Layer                │
         │  `python/polars_statistics/`         │
         │  - exprs (expression builders)       │
         │  - models (class bindings)           │
         │  - formula (R-style syntax parser)   │
         └──────────────────────────────────────┘
                        │
         ┌──────────────▼──────────────────────┐
         │  User API (polars-statistics wheel) │
         │  Python 3.9+ with Polars installed  │
         └──────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| **Polars Expressions** | Define `#[polars_expr]` decorated functions for group_by/over operations | `src/expressions/*.rs` |
| **PyO3 Model Wrappers** | Expose 40+ regression and test models as Python classes with fit/predict methods | `src/pymodels/*.rs` |
| **Output Types** | Define struct-based output schemas (regression results, test stats) | `src/expressions/output_types.rs` |
| **Array Conversion** | Bridge numpy ↔ faer matrices for zero-copy data transfer | `src/utils/array_conversion.rs` |
| **Python Expression API** | Wrap Rust expressions as Python callables for `df.select(ps.ttest_ind(...))` | `python/polars_statistics/exprs/*.py` |
| **Formula Parser** | Expand R-style formulas (`y ~ x1 + x2`) to Polars select statements | `python/polars_statistics/formula/*.py` |

## Pattern Overview

**Overall:** Hybrid dual-build pattern (cdylib + rlib) with language-level FFI between Rust and Python.

**Key Characteristics:**
- **Expression-first design**: All statistical operations expose via Polars `#[polars_expr]` macro for native DataFrame integration
- **PyO3 bindings**: High-level models wrapped as Python classes; low-level operations as lazy expressions
- **Zero-copy data transfer**: numpy arrays → faer matrices via read-only borrows
- **Layered error handling**: Rust `Result` → `PolarsResult` → Python exceptions
- **Feature-gated Python**: Core library works as pure Rust rlib; Python (PyO3) is optional `python` feature

## Layers

**Expression Layer (Polars Plugin):**
- Purpose: Provide lazy-evaluable statistical functions that integrate with group_by, over, and other Polars operations
- Location: `src/expressions/`
- Contains: 
  - Test expressions (parametric, non-parametric, distributional, modern, correlation, categorical, forecast, TOST)
  - Regression expressions (linear models, GLMs, robust regression)
  - Output type definitions (struct schemas for all result types)
- Depends on: `anofox-regression`, `anofox-statistics`, Polars, faer
- Used by: Python expression API (`python/polars_statistics/exprs/`), lazy DataFrame operations

**PyModel Layer (Direct Python Classes):**
- Purpose: Expose regression and test models as imperative classes for direct fit/predict workflows
- Location: `src/pymodels/`
- Contains:
  - Linear model classes: OLS, Ridge, ElasticNet, WLS, RLS, BLS, PLS
  - Robust regression: Quantile, Isotonic, Huber
  - GLM models: Logistic, LogisticRegression, Poisson, NegativeBinomial, Tweedie, Probit, Cloglog
  - Specialized: ALM, LmDynamic, Aid
  - Bootstrap: StationaryBootstrap, CircularBlockBootstrap
  - Test models (20+ parametric, non-parametric, distributional tests)
- Depends on: `anofox-regression` solvers, numpy, pyo3
- Used by: Python programs importing `from polars_statistics import OLS` etc.

**Utility Layer:**
- Purpose: Provide data marshaling between NumPy and faer/Polars formats
- Location: `src/utils/`
- Contains: Type traits (`ToFaer`, `IntoNumpy`) for bidirectional array conversion
- Depends on: numpy, faer, pyo3
- Used by: All PyModel wrappers for array handling

**Python Expression Wrapper Layer:**
- Purpose: Make Rust expressions callable as Python functions that return Polars expression objects
- Location: `python/polars_statistics/exprs/`
- Contains: One .py file per category (parametric.py, categorical.py, regression.py, etc.), each exporting builder functions
- Depends on: `_polars_statistics` (Rust cdylib), polars
- Used by: End users in `df.select(ps.ttest_ind("x", "y"))`

**Formula Parser Layer:**
- Purpose: Parse R-style formulas and expand to Polars column selections
- Location: `python/polars_statistics/formula/`
- Contains: Lexer/parser (formula.parser.py), expander (formula.expander.py), term representation (formula.terms.py)
- Depends on: (none)
- Used by: Regression expression builders to support `df.ols("y ~ x1 + x2 + x1:x2")`

## Data Flow

### Primary Request Path: Expression via group_by

1. User calls `df.group_by("group").agg(ps.ols("y", "x1", "x2"))` (`python/polars_statistics/exprs/regression.py:ols()`)
2. Expression builder calls Polars plugin entry point: `pl.Expr.register_plugin("ols_fit")` with input columns
3. Polars scheduler invokes `#[polars_expr]` function at `src/expressions/regression.rs:pl_ols_fit()`
4. Expression handler parses kwargs, calls public `ols_fit([y_series, x_columns...])` with per-group data
5. Handler builds faer matrices from Series data, calls `OlsRegressor::fit()` from `anofox_regression`
6. Linear algebra (faer + BLAS) computes: intercept, coefficients, R², AIC, BIC, F-test
7. Output type handler (`linear_regression_output_dtype()`) marshals results into Polars struct schema
8. Result struct flows back through group_by aggregation

### PyModel Direct API Path

1. User imports: `from polars_statistics import OLS`
2. Instantiates: `model = OLS(with_intercept=True, compute_inference=True)`
3. Calls: `model.fit(X_numpy, y_numpy)` — routes to `src/pymodels/py_ols.rs:PyOLS::fit()`
4. PyOLS converts numpy arrays to faer via `src/utils/array_conversion.rs`
5. Calls `OlsRegressor::fit()` from `anofox_regression`
6. Stores fitted regressor as `self.fitted: FittedOls`
7. User calls: `model.coefficients`, `model.predict(X_test)`, `model.summary()` — dispatches to stored regressor

### Formula Expansion Path

1. User provides formula string: `"y ~ x1 + x2 + x1:x2"`
2. `python/polars_statistics/formula/parser.py` tokenizes and builds AST of terms
3. `formula.expander.py` expands interaction terms: `x1:x2` → cross-product
4. Expression builder passes expanded column names to Rust layer
5. Rust layer performs standard regression on provided columns

**State Management:**
- **Expressions**: Stateless. Per-group data materialized at call time.
- **PyModels**: Stateful. Fitted regressor object stored in `self.fitted` field. Reused across multiple calls (predict, residuals, diagnostics).
- **Global**: Polars allocator (pyo3_polars::PolarsAllocator) installed in `src/lib.rs` for compatibility

## Key Abstractions

**Regressor Trait:**
- Purpose: Unified interface for fit/predict across all model types (OLS, Ridge, GLM, etc.)
- Examples: `OlsRegressor`, `RidgeRegressor`, `LogisticRegression` in `anofox_regression`
- Pattern: All implement `Regressor` trait; expressions dispatch via concrete type selection based on model_type string parameter

**PolarsExpr Macro:**
- Purpose: Codegen for Polars plugin FFI — handles serialization, type checking, group_by integration
- Examples: `#[polars_expr(output_type_func=linear_regression_output_dtype)]`
- Pattern: Macro automatically registers function name in plugin namespace; expression name becomes kebab-case

**Output Struct Schema:**
- Purpose: Type-safe result marshaling as nested structs (e.g., struct with fields: intercept, coefficients, r_squared)
- Examples: `linear_regression_output_dtype()`, `stats_output_dtype()`, `glm_output_dtype()`
- Pattern: Each function returns `Field::new(..., DataType::Struct(vec![...]))` defining nested schema

## Entry Points

**Rust Library Entry (rlib):**
- Location: `src/lib.rs:pub mod expressions`
- Triggers: `use polars_statistics::expressions::*;` in downstream Rust crates
- Responsibilities: Re-export all public expression functions (ols_fit, ttest_ind_fit, etc.)

**Python Extension Entry (cdylib):**
- Location: `src/lib.rs:#[pymodule]` block
- Triggers: `import _polars_statistics` or `from polars_statistics import OLS`
- Responsibilities: Register 40+ PyO3 classes and expression builders in Python module namespace

**Python Expression API Entry:**
- Location: `python/polars_statistics/exprs/*.py` module functions
- Triggers: `df.select(ps.ttest_ind("x", "y"))` or `df.group_by(...).agg(ps.ols(...))`
- Responsibilities: Build Polars expression from kwargs; invoke Rust backend via plugin

**Polars Plugin FFI Entry:**
- Location: `src/expressions/*.rs:#[polars_expr]` functions
- Triggers: Polars evaluates lazy expression with group_by/over
- Responsibilities: Per-group data vectorization and model fitting

## Architectural Constraints

- **Threading:** Single-threaded per group (Polars manages thread pool). Regression solvers use rayon internally (faer feature).
- **Global state:** `pyo3_polars::PolarsAllocator` installed in `src/lib.rs` — ensures memory pooling compatibility with Polars.
- **Circular imports:** None detected. Clear hierarchy: expressions → PyModels → utilities → anofox crates.
- **Feature coupling:** Python feature gates entire PyModel and PyO3 dependency tree. Rlib consumers need no Python stack.
- **Array memory:** NumPy arrays borrowed as read-only in ToFaer trait; no mutable aliases to Polars series.

## Error Handling

**Strategy:** Rust `Result`/`PolarsResult` → Python exceptions via PyO3 automatic conversion.

**Patterns:**
- Expression layer: Return `PolarsResult::Err` on fitting failure; caller (group_by) propagates or marks group as null
- PyModel layer: Use `#[pyo3(...)] fn` to automatically marshal `PyErr` to Python exceptions
- Validation: Input shape checks (n_samples >= n_params) in anofox solvers; null/NaN handling in expression pre-processing

**Error Examples:**
- Rank deficiency: `anofox_regression` returns `Err`; expression converts to NaN coefficients in output struct
- Shape mismatch: `ToFaer` panics (debug) or silently truncates; caught by Polars plugin error wrapper
- GLM non-convergence: Stored in result struct as divergence flag (count_status field in supported models)

## Cross-Cutting Concerns

**Logging:** None built-in. Diagnostics (residuals, influence masks) exported as output struct fields for inspection.

**Validation:**
- Input shapes: Checked in faer matrix construction (panics on rank > row count)
- Data types: Polars cast_to_f64 enforced in expression wrappers
- Parameter bounds: alpha in [0,1], tau in (0,1) for quantile regression

**Authentication:** Not applicable (statistical library).

---

*Architecture analysis: 2026-08-11*
