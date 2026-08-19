<!-- GSD:project-start source:PROJECT.md -->

## Project

**polars-statistics**

A Rust + Python (PyO3) library that exposes comprehensive statistical hypothesis
testing and regression modeling as native Polars expressions and Python classes.
Users write `df.group_by(...).agg(ps.ols("y", "x1", "x2"))` or
`from polars_statistics import OLS` and get R-validated statistics with zero-copy
numpy↔faer data transfer. It is distributed as a PyPI wheel for Python 3.9+.

This milestone modernizes the two backing engine crates — `anofox-statistics`
(0.4.1 → 0.4.2) and `anofox-regression` (0.5.4 → 0.5.13) — closes the API-exposure
gaps those crates now expose, documents and tests the new surface, and ships a new
release to PyPI.

**Core Value:** Every public statistical capability in the backing `anofox-*` crates is exposed
through the Polars/Python API — correctly (post-bugfix), documented, and tested —
and shipped to users via PyPI.

### Constraints

- **Tech stack**: Rust 2021 + PyO3 0.26 + Polars 0.52; new API must follow the existing `#[polars_expr]` / PyModel patterns and output-struct schemas — Why: consistency and native Polars integration.
- **Validation**: New tests validated against R where the crates supply references — Why: the library's core promise is "validated against R."
- **CI gates**: clippy `-D warnings`, cargo fmt, ruff (line length 100, py39 target) must stay green — Why: existing merge gates.
- **Compatibility**: Maintain Python 3.9–3.12 support across Linux/macOS/Windows wheels — Why: existing distribution matrix.
- **Release**: Production PyPI publish via the existing GH Actions pipeline (OIDC, no hardcoded tokens) — Why: established, secure release path.

<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->

## Technology Stack

## Languages

- Rust 2021 Edition - Core library and statistical implementations in `src/`
- Python 3.9+ - PyO3 bindings and user-facing API in `python/polars_statistics/`
- YAML - Configuration for CI/CD and documentation builds
- Markdown - Documentation and examples

## Runtime

- Rust: Stable toolchain with rustfmt and clippy for linting
- Python: CPython 3.9, 3.10, 3.11, 3.12 (tested across multiple versions)
- Operating Systems: Linux (Ubuntu), macOS (x86_64 and ARM64), Windows
- Rust: Cargo with Cargo.lock for deterministic builds
- Python: pip with uv package manager (as seen in uv.lock)
- Build: maturin for building Python wheels from Rust source

## Frameworks

- Polars 0.52 - DataFrame processing and lazy evaluation (`Cargo.toml` line 28)
- pyo3 0.26 - Python bindings with abi3-py39 and extension-module features (optional, gated behind `python` feature)
- pyo3-polars 0.25 - Polars-to-Python bridge with derive and lazy features
- pytest 7.0+ - Python test runner (configured in `pyproject.toml`)
- pytest-cov - Code coverage reporting (used in CI)
- maturin 1.7.4+ - Build Python wheels from Rust source (`pyproject.toml` line 2)
- cargo-fmt - Rust code formatting (checked in CI `.github/workflows/ci.yml` line 28)
- cargo-clippy - Rust linter with warnings-as-errors in CI (line 31)

## Key Dependencies

- faer 0.23.2 - Linear algebra with Rayon parallelization (`Cargo.toml` line 48) - powers matrix operations for regression
- statrs 0.18 - Statistics library for distributions and statistical functions
- polars-arrow 0.52 - Arrow integration for Polars data manipulation
- anofox-regression 0.5.4 - Regression models (OLS, Ridge, Elastic Net, GLM, Quantile, Huber, PLS, ALM) validated against R
- anofox-statistics 0.4.1 - Statistical tests (t-tests, ANOVA, Mann-Whitney, Kruskal-Wallis, normality tests, correlation tests)
- rayon (via faer) - Data-level parallelization for linear algebra operations
- rand 0.8 - Random number generation
- rand_chacha 0.3 - ChaCha RNG for deterministic randomness
- serde 1.0.228 - Serialization framework with derive macros
- thiserror 2.0.17 - Structured error types with derive macros
- `lazy` - Lazy evaluation
- `dtype-struct`, `dtype-array` - Complex data types
- `dtype-i8`, `dtype-i16`, `dtype-u8`, `dtype-u16` - Integer types
- `zip_with`, `abs`, `round_series`, `log`, `is_in`, `cross_join`, `diff`, `partition_by` - Expression operations

## Configuration

- Build feature flags in `Cargo.toml` (lines 17-19):
- Python version constraint: `requires-python = ">=3.9"` (`pyproject.toml` line 11)
- Maturin build configuration: `python-source = "python"`, module-name `polars_statistics._polars_statistics` (lines 97-101)
- Release profile optimization (`Cargo.toml` lines 69-72):
- pytest: `testpaths = ["tests"]`, `addopts = "-v --tb=short"` (`pyproject.toml` lines 103-105)
- Code coverage: Codecov integration (`.github/workflows/ci.yml` lines 136-141)
- Ruff (Python): Line length 100, target Python 3.9, rules: E, F, W, I, UP, B, C4 (ignore E501)
- Cargo fmt: Rust formatting validation in CI
- Clippy: Rust linting with deny-warnings (`-D warnings`)

## Platform Requirements

- Rust stable toolchain with rustfmt and clippy components
- Python 3.9+ with venv support
- C compiler and OpenSSL development libraries (for Linux wheels)
- Cross-compilation targets for wheel building: x86_64, aarch64 (ARM64)
- Deployment: PyPI (Python Package Index) for wheels and source distributions
- Distribution: Platform-specific wheels for Linux (manylinux2014+), macOS (10.13+, ARM64), Windows
- PyPI distribution: Two-tier with TestPyPI for pre-release testing
- Trusted Publishing: Uses OIDC for PyPI authentication (no hardcoded tokens)

## CI/CD & Dependencies

- GitHub Actions workflows in `.github/workflows/`:
- Wheels built for 5 platform combinations: Linux x86_64, Linux aarch64, macOS x86_64, macOS aarch64, Windows x86_64
- Source distribution (sdist) for pip install --no-binary
- mkdocs with Material theme (`mkdocs.yml`)
- Python-based markdown extensions: admonition, superfences, tabbed, toc, syntax highlighting
- Published to GitHub Pages at `https://datazoode.github.io/polars-statistics/`

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

## Naming Patterns

- Snake case for module files: `py_ols.rs`, `py_elastic_net.rs`, `py_mann_whitney.rs`
- Prefixes by category:
- Snake case: `ols_fit`, `ridge_fit`, `quantile_fit`, `binom_test_fit`
- Suffix patterns:
- Helper functions: `parse_solver_type`, `parse_alternative`, `parse_hc_type`
- Internal/private: prefix with underscore when needed (e.g., `_build_xy_with_null_policy`)
- Snake case throughout: `with_intercept`, `conf_level`, `solve_method`, `n_observations`
- Abbreviations in type names: `xy` (features-target pair), `glm` (generalized linear model), `ols` (ordinary least squares), `vif` (variance inflation factor)
- Field names in structs: snake case with underscore: `fitted`, `solve_method`, `with_intercept`
- Upper camel case for structs: `PyOLS`, `PyElasticNet`, `PyQuantile`, `PyLogisticRegression`
- Type aliases lowercase: `XyNullPolicyResult`, `OlsResidualContext`
- Generic params: uppercase single letter or descriptive (e.g., `T`, `'py` for Python lifetime)

## Code Style

- Edition 2021 (see `Cargo.toml`)
- Line length: observe ~100 char soft limit in most modules
- Indentation: 4 spaces
- No explicit `.clippy.toml` or `clippy.toml` detected
- Follows standard Rust conventions and Polars ecosystem patterns
- Common allow attributes: `#[allow(unused_imports)]` for FFI re-exports

## Import Organization

- None detected; full paths used throughout

## Error Handling

- Primary return type: `PolarsResult<T>` for Polars expressions (wraps `Result<T, PolarsError>`)
- PyO3 return type: `PyResult<T>` for Python methods (wraps `Result<T, PyErr>`)
- Error conversion: `.map_err()` for converting errors between types
- Null-coalescing for optional values: `.get(0).unwrap_or(default)` in scalar extraction
- Model state checks: `.ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?`

## Logging

- Errors reported via `PolarsResult` and `PyResult` return types
- No debug/info logging in observed code
- Error messages propagated to caller (Python or Polars expression system)

## Comments

- Module-level `//!` doc comments on all public modules and files
- Function-level `///` doc comments on all public functions
- Type-level `///` doc comments on struct/enum definitions
- Inline comments rare; code assumed self-documenting
- Uses Rust-style doc comments (`///` for items, `//!` for modules)
- PyO3 modules document Python API with docstring sections: Parameters, Returns, Examples
- Example format from `py_ols.rs`:

## Function Design

- PyO3 methods: prefer `#[pyo3(signature = (...))]` for default parameters
- Polars expression functions: accept `&[Series]` slice, extract by index
- Helper functions: typed parameters with no reliance on positional index
- Polars expressions: `PolarsResult<Series>` (returns Struct series with named fields)
- PyO3 methods: `PyResult<T>` where T is numpy array, dict, String, or bool
- Helpers: explicit `PolarsResult` for error propagation

## Module Design

- `src/lib.rs`: gate all modules behind `#[cfg(feature = "python")]` except `pub mod expressions`
- `src/expressions/mod.rs`: re-exports all submodule contents with `#[allow(unused_imports)]`
- `src/pymodels/mod.rs`: selective re-exports of public struct types only
- `src/expressions/mod.rs` (line 6-37): re-exports all expression submodules
- `src/pymodels/mod.rs` (line 39-71): re-exports PyO3 classes for Python module registration
- Private fields by default in PyO3 structs (e.g., `struct PyOLS { with_intercept: bool, fitted: Option<FittedOls> }`)
- Getters via `#[getter]` for property-style access from Python
- Methods use `#[pymethods]` for Python-callable methods

## Feature Gates

- `#[cfg(feature = "python")]` gates Python-specific code at module and statement level
- `src/lib.rs` (lines 13-25): conditional compilation of Python bindings
- Default feature: `default = ["python"]`
- Crate usable as rlib without `python` feature for non-Python consumers

## Polars Expression Macro Usage

- `#[polars_expr]` macro from `pyo3_polars::derive` decorates both public and private fit functions
- Public fit function: exposed via FFI for Polars
- Private `pl_*` variant: internal implementation (may not always exist)
- Example from `categorical.rs`:

<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

## System Overview

```text

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

- **Expression-first design**: All statistical operations expose via Polars `#[polars_expr]` macro for native DataFrame integration
- **PyO3 bindings**: High-level models wrapped as Python classes; low-level operations as lazy expressions
- **Zero-copy data transfer**: numpy arrays → faer matrices via read-only borrows
- **Layered error handling**: Rust `Result` → `PolarsResult` → Python exceptions
- **Feature-gated Python**: Core library works as pure Rust rlib; Python (PyO3) is optional `python` feature

## Layers

- Purpose: Provide lazy-evaluable statistical functions that integrate with group_by, over, and other Polars operations
- Location: `src/expressions/`
- Contains: 
- Depends on: `anofox-regression`, `anofox-statistics`, Polars, faer
- Used by: Python expression API (`python/polars_statistics/exprs/`), lazy DataFrame operations
- Purpose: Expose regression and test models as imperative classes for direct fit/predict workflows
- Location: `src/pymodels/`
- Contains:
- Depends on: `anofox-regression` solvers, numpy, pyo3
- Used by: Python programs importing `from polars_statistics import OLS` etc.
- Purpose: Provide data marshaling between NumPy and faer/Polars formats
- Location: `src/utils/`
- Contains: Type traits (`ToFaer`, `IntoNumpy`) for bidirectional array conversion
- Depends on: numpy, faer, pyo3
- Used by: All PyModel wrappers for array handling
- Purpose: Make Rust expressions callable as Python functions that return Polars expression objects
- Location: `python/polars_statistics/exprs/`
- Contains: One .py file per category (parametric.py, categorical.py, regression.py, etc.), each exporting builder functions
- Depends on: `_polars_statistics` (Rust cdylib), polars
- Used by: End users in `df.select(ps.ttest_ind("x", "y"))`
- Purpose: Parse R-style formulas and expand to Polars column selections
- Location: `python/polars_statistics/formula/`
- Contains: Lexer/parser (formula.parser.py), expander (formula.expander.py), term representation (formula.terms.py)
- Depends on: (none)
- Used by: Regression expression builders to support `df.ols("y ~ x1 + x2 + x1:x2")`

## Data Flow

### Primary Request Path: Expression via group_by

### PyModel Direct API Path

### Formula Expansion Path

- **Expressions**: Stateless. Per-group data materialized at call time.
- **PyModels**: Stateful. Fitted regressor object stored in `self.fitted` field. Reused across multiple calls (predict, residuals, diagnostics).
- **Global**: Polars allocator (pyo3_polars::PolarsAllocator) installed in `src/lib.rs` for compatibility

## Key Abstractions

- Purpose: Unified interface for fit/predict across all model types (OLS, Ridge, GLM, etc.)
- Examples: `OlsRegressor`, `RidgeRegressor`, `LogisticRegression` in `anofox_regression`
- Pattern: All implement `Regressor` trait; expressions dispatch via concrete type selection based on model_type string parameter
- Purpose: Codegen for Polars plugin FFI — handles serialization, type checking, group_by integration
- Examples: `#[polars_expr(output_type_func=linear_regression_output_dtype)]`
- Pattern: Macro automatically registers function name in plugin namespace; expression name becomes kebab-case
- Purpose: Type-safe result marshaling as nested structs (e.g., struct with fields: intercept, coefficients, r_squared)
- Examples: `linear_regression_output_dtype()`, `stats_output_dtype()`, `glm_output_dtype()`
- Pattern: Each function returns `Field::new(..., DataType::Struct(vec![...]))` defining nested schema

## Entry Points

- Location: `src/lib.rs:pub mod expressions`
- Triggers: `use polars_statistics::expressions::*;` in downstream Rust crates
- Responsibilities: Re-export all public expression functions (ols_fit, ttest_ind_fit, etc.)
- Location: `src/lib.rs:#[pymodule]` block
- Triggers: `import _polars_statistics` or `from polars_statistics import OLS`
- Responsibilities: Register 40+ PyO3 classes and expression builders in Python module namespace
- Location: `python/polars_statistics/exprs/*.py` module functions
- Triggers: `df.select(ps.ttest_ind("x", "y"))` or `df.group_by(...).agg(ps.ols(...))`
- Responsibilities: Build Polars expression from kwargs; invoke Rust backend via plugin
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

- Expression layer: Return `PolarsResult::Err` on fitting failure; caller (group_by) propagates or marks group as null
- PyModel layer: Use `#[pyo3(...)] fn` to automatically marshal `PyErr` to Python exceptions
- Validation: Input shape checks (n_samples >= n_params) in anofox solvers; null/NaN handling in expression pre-processing
- Rank deficiency: `anofox_regression` returns `Err`; expression converts to NaN coefficients in output struct
- Shape mismatch: `ToFaer` panics (debug) or silently truncates; caught by Polars plugin error wrapper
- GLM non-convergence: Stored in result struct as divergence flag (count_status field in supported models)

## Cross-Cutting Concerns

- Input shapes: Checked in faer matrix construction (panics on rank > row count)
- Data types: Polars cast_to_f64 enforced in expression wrappers
- Parameter bounds: alpha in [0,1], tau in (0,1) for quantile regression

<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
