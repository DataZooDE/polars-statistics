# Codebase Structure

**Analysis Date:** 2026-08-11

## Directory Layout

```
polars-statistics/
├── src/                           # Rust source (compiled to cdylib + rlib)
│   ├── lib.rs                     # Root: PyO3 module def + feature gating
│   ├── expressions/               # Polars plugin expressions (~1800 lines)
│   │   ├── mod.rs                 # Module re-exports
│   │   ├── regression.rs          # Linear/GLM/robust regression expressions
│   │   ├── parametric.rs          # t-test, Brown-Forsythe, Yuen
│   │   ├── nonparametric.rs       # Mann-Whitney, Wilcoxon, Kruskal-Wallis
│   │   ├── distributional.rs      # Shapiro-Wilk, D'Agostino
│   │   ├── correlation.rs         # Pearson, Spearman, Kendall, distance, ICC
│   │   ├── categorical.rs         # Binomial, proportion, chi-sq tests
│   │   ├── forecast.rs            # Diebold-Mariano, permutation t, Clark-West, SPA, MCS
│   │   ├── modern.rs              # Energy distance, MMD test
│   │   ├── tost.rs                # TOST equivalence tests
│   │   └── output_types.rs        # Struct field schemas for all outputs
│   ├── pymodels/                  # PyO3 wrappers for models (~3000 lines)
│   │   ├── mod.rs                 # Module re-exports (40 classes)
│   │   ├── py_*.rs                # One file per model (e.g., py_ols.rs, py_ttest_ind.rs)
│   │   └── (25 regression + 20 test models)
│   └── utils/                     # Data marshaling (200 lines)
│       ├── mod.rs                 # Module re-exports
│       └── array_conversion.rs    # ToFaer, IntoNumpy traits
├── python/                        # Python package (wheel build via maturin)
│   └── polars_statistics/
│       ├── __init__.py            # Public API: import OLS, Ridge, ..., ps.ttest_ind, ...
│       ├── models/                # Empty stub (models come from Rust PyO3)
│       ├── exprs/                 # Expression builders (~1000 lines)
│       │   ├── __init__.py        # Re-exports from all submodules
│       │   ├── regression.py      # ols(), ridge(), elastic_net(), etc. (140KB)
│       │   ├── parametric.py      # ttest_ind(), ttest_paired(), brown_forsythe()
│       │   ├── nonparametric.py   # mann_whitney_u(), wilcoxon_signed_rank(), etc.
│       │   ├── correlation.py     # pearson(), spearman(), kendall(), icc()
│       │   ├── categorical.py     # binom_test(), prop_test_one(), etc.
│       │   ├── distributional.py  # shapiro_wilk(), dagostino()
│       │   ├── forecast.py        # diebold_mariano(), permutation_t_test(), etc.
│       │   ├── modern.py          # energy_distance(), mmd_test()
│       │   └── tost.py            # TOST equivalence test suite
│       └── formula/               # R-style formula parser
│           ├── __init__.py        # Public: parse_formula()
│           ├── parser.py          # Tokenizer & AST builder
│           ├── expander.py        # Expand interaction terms & polynomials
│           └── terms.py           # Term representation classes
├── tests/                         # Python + Rust test suite (24 files)
│   ├── conftest.py                # pytest fixtures: sample data, Polars DF
│   ├── rust_api.rs                # Rust API test (no default features)
│   ├── test_expressions.py        # Basic expression functionality
│   ├── test_models.py             # PyModel fit/predict/summary
│   ├── test_group_by_over.py      # group_by/over integration
│   ├── test_formula.py            # Formula parsing & expansion
│   ├── test_*_parity.py           # Validation against R/statsmodels
│   └── ... (16 more specialized tests)
├── examples/                      # Minimal example scripts
│   ├── rust_wls.rs                # Per-group WLS via rlib (no Python feature)
│   └── performance_1m_groups/     # Large-scale groupby benchmark
├── docs/                          # User documentation (markdown)
│   ├── api/                       # Auto-generated API reference stubs
│   ├── examples/                  # Cookbook examples (10+ scenarios)
│   └── validation/                # Parity reports (R validation)
├── .planning/                     # GSD milestone tracking (post-scaffold)
│   ├── PROJECT.md                 # Project charter & version info
│   └── codebase/                  # Architecture docs (this location)
├── .github/                       # GitHub workflows (CI/CD)
├── Cargo.toml                     # Rust dependency manifest
│   ├── [package] name="polars-statistics"
│   ├── [lib] crate-type=["cdylib", "rlib"]
│   ├── [features] default=["python"]; python=["pyo3", "numpy"]
│   └── [dependencies] faer, polars, anofox-regression, anofox-statistics, ...
├── pyproject.toml                 # Python package config (maturin build)
├── Cargo.lock                     # Locked dependency versions
├── uv.lock                        # Python (uv) lock file
├── README.md                      # High-level overview
├── CHANGELOG.md                   # Release notes
└── mkdocs.yml                     # Documentation site config
```

## Directory Purposes

**src/expressions/:**
- Purpose: Polars plugin expressions — lazy evaluable operations integrated with group_by/over
- Contains: 80+ `#[polars_expr]` functions split by statistical domain
- Key files: `regression.rs` (largest), `parametric.rs`, `correlation.rs`, `output_types.rs`
- Pattern: Public functions named `{operation}_fit()` + private `pl_{operation}()` decorator

**src/pymodels/:**
- Purpose: PyO3 model wrappers — direct Python class API for imperative fit/predict workflows
- Contains: 40 `#[pyclass]` structures; each wraps a solver from anofox-regression
- Pattern: `PyOLS`, `PyRidge`, `PyTTestInd`, etc. — one class per model type

**src/utils/:**
- Purpose: Data marshaling between numpy/Polars and faer matrices
- Contains: Conversion trait implementations (`ToFaer`, `IntoNumpy`)
- Imported by: All PyModel wrappers for array handling

**python/polars_statistics/exprs/:**
- Purpose: Expression factory functions — build lazy Polars expressions from high-level API
- Contains: One .py per category; each exports builder functions (e.g., `ols()`, `ttest_ind()`)
- Pattern: Each function constructs `pl.expr.call_plugin()` with "function_name" and kwargs

**python/polars_statistics/formula/:**
- Purpose: Parse R-style formulas and expand to column selections
- Contains: Lexer (tokenizer), parser (AST builder), expander (interaction/polynomial handling)
- Example: `"y ~ x1 + x2 + x1:x2"` → `("y", ["x1", "x2", "x1*x2"])`

**tests/:**
- Purpose: Comprehensive test coverage (unit, integration, parity)
- Contains: 24 test files covering expressions, models, group_by, formula, parity with R
- Pattern: Python tests use pytest + fixtures; Rust test uses `#[cfg(test)]`

**examples/:**
- Purpose: Demonstrate key usage patterns
- Contains: Standalone scripts (rust_wls.rs for rlib, performance benchmarks)

**docs/:**
- Purpose: User documentation and API reference
- Contains: Markdown pages, API reference stubs, validation reports

## Key File Locations

**Entry Points:**
- `src/lib.rs` — Root crate: PyO3 module definition + feature gates
- `python/polars_statistics/__init__.py` — Python package public API
- `examples/rust_wls.rs` — Rlib-only example (no Python feature)

**Configuration:**
- `Cargo.toml` — Rust dependencies, crate type (cdylib + rlib), feature flags
- `pyproject.toml` — Python build config (maturin, version, entry point)
- `mkdocs.yml` — Documentation site config

**Core Logic:**
- `src/expressions/regression.rs` — Regression model expressions (1400+ lines)
- `src/expressions/parametric.rs` — Parametric test expressions
- `src/pymodels/py_ols.rs` — Example PyModel wrapper (OLS)
- `python/polars_statistics/exprs/regression.py` — Regression expression builders (140KB)
- `python/polars_statistics/formula/parser.py` — Formula parser

**Testing:**
- `tests/conftest.py` — Pytest fixtures (sample DataFrames, constants)
- `tests/test_expressions.py` — Expression API tests
- `tests/test_models.py` — PyModel tests
- `tests/test_group_by_over.py` — Group_by integration tests
- `tests/test_formula.py` — Formula parser tests

## Naming Conventions

**Files:**
- Rust: Snake case (`py_ols.rs`, `output_types.rs`)
- Python: Snake case (`regression.py`, `parser.py`)
- Test files: `test_{feature}.py` or `test_{component}_parity.py`

**Functions:**
- Rust: Snake case (`ols_fit()`, `linear_regression_output_dtype()`)
- Python: Snake case (`ols()`, `ttest_ind()`)
- Expression decorators: Prefixed with `pl_` (`pl_ols`, `pl_ttest_ind`)

**Variables:**
- Polars columns: Snake case, lowercase (e.g., "r_squared", "p_value")
- Output struct fields: Snake case (e.g., `intercept`, `coefficients`, `n_observations`)
- Model parameters: Lowercase (e.g., `with_intercept`, `compute_inference`, `confidence_level`)

**Types:**
- PyO3 classes: PascalCase with leading `Py` prefix (`PyOLS`, `PyTTestInd`)
- Regressor types: From anofox crates (`OlsRegressor`, `RidgeRegressor`)
- Enums: PascalCase (`Alternative`, `TTestKind`, `HcType`)

## Where to Add New Code

**New Statistical Test (e.g., Levene's test):**
1. **Rust implementation**: 
   - Add to `src/expressions/parametric.rs` (if parametric) or `nonparametric.rs`
   - Define public `levene_fit()` + private `pl_levene()` with `#[polars_expr]` decorator
   - Define output type in `src/expressions/output_types.rs` if schema is new
2. **Python wrapper**:
   - Add to `python/polars_statistics/exprs/parametric.py`
   - Function `levene(x: IntoExpr, y: IntoExpr, ...) -> Expr` calls plugin
3. **Tests**:
   - Add to `tests/test_parametric.py` (or `test_expressions.py` if cross-category)

**New Regression Model (e.g., Elastic Net variant):**
1. **Rust PyModel**:
   - Create `src/pymodels/py_elastic_net_v2.rs` with `#[pyclass]` struct
   - Implement `fit()`, `predict()`, `summary()`, `get_coefficients()` etc.
   - Register in `src/pymodels/mod.rs` (pub use + add_class in lib.rs)
2. **Rust Expression** (optional):
   - Add `src/expressions/regression.rs:elastic_net_v2_fit()` if group_by-friendly
3. **Python Expression Wrapper**:
   - Add `python/polars_statistics/exprs/regression.py:elastic_net_v2()`
4. **Tests**:
   - Add to `tests/test_models.py` (PyModel) + `tests/test_expressions.py` (lazy)

**Utilities/Helpers:**
- Shared array conversion logic: `src/utils/array_conversion.rs`
- Output struct builders: `src/expressions/output_types.rs` (add new dtype function)
- Formula expansion logic: `python/polars_statistics/formula/expander.py`

**Documentation:**
- API reference: Auto-generated from docstrings in Python + Rust (PyO3 docs)
- User guide: Add markdown to `docs/examples/` with narrative + code blocks
- Validation: Add parity test to `tests/test_{model}_parity.py`

## Special Directories

**site/:**
- Purpose: Built documentation (generated by mkdocs build)
- Generated: Yes (from docs/ markdown via mkdocs)
- Committed: No (git-ignored, CI-generated)

**python/polars_statistics/__pycache__/:**
- Purpose: Python bytecode cache
- Generated: Yes (by Python interpreter)
- Committed: No

**target/:**
- Purpose: Compiled Rust artifacts (not shown in listing, git-ignored)
- Generated: Yes (cargo build)
- Committed: No

**.pytest_cache/:**
- Purpose: Pytest fixture and test result cache
- Generated: Yes (by pytest)
- Committed: No

---

*Structure analysis: 2026-08-11*
