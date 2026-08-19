# Technology Stack

**Analysis Date:** 2026-08-11

## Languages

**Primary:**
- Rust 2021 Edition - Core library and statistical implementations in `src/`
- Python 3.9+ - PyO3 bindings and user-facing API in `python/polars_statistics/`

**Secondary:**
- YAML - Configuration for CI/CD and documentation builds
- Markdown - Documentation and examples

## Runtime

**Environment:**
- Rust: Stable toolchain with rustfmt and clippy for linting
- Python: CPython 3.9, 3.10, 3.11, 3.12 (tested across multiple versions)
- Operating Systems: Linux (Ubuntu), macOS (x86_64 and ARM64), Windows

**Package Manager:**
- Rust: Cargo with Cargo.lock for deterministic builds
- Python: pip with uv package manager (as seen in uv.lock)
- Build: maturin for building Python wheels from Rust source

## Frameworks

**Core:**
- Polars 0.52 - DataFrame processing and lazy evaluation (`Cargo.toml` line 28)
- pyo3 0.26 - Python bindings with abi3-py39 and extension-module features (optional, gated behind `python` feature)
- pyo3-polars 0.25 - Polars-to-Python bridge with derive and lazy features

**Testing:**
- pytest 7.0+ - Python test runner (configured in `pyproject.toml`)
- pytest-cov - Code coverage reporting (used in CI)

**Build/Dev:**
- maturin 1.7.4+ - Build Python wheels from Rust source (`pyproject.toml` line 2)
- cargo-fmt - Rust code formatting (checked in CI `.github/workflows/ci.yml` line 28)
- cargo-clippy - Rust linter with warnings-as-errors in CI (line 31)

## Key Dependencies

**Critical:**
- faer 0.23.2 - Linear algebra with Rayon parallelization (`Cargo.toml` line 48) - powers matrix operations for regression
- statrs 0.18 - Statistics library for distributions and statistical functions
- polars-arrow 0.52 - Arrow integration for Polars data manipulation
- anofox-regression 0.5.4 - Regression models (OLS, Ridge, Elastic Net, GLM, Quantile, Huber, PLS, ALM) validated against R
- anofox-statistics 0.4.1 - Statistical tests (t-tests, ANOVA, Mann-Whitney, Kruskal-Wallis, normality tests, correlation tests)

**Parallel/Performance:**
- rayon (via faer) - Data-level parallelization for linear algebra operations
- rand 0.8 - Random number generation
- rand_chacha 0.3 - ChaCha RNG for deterministic randomness

**Serialization:**
- serde 1.0.228 - Serialization framework with derive macros

**Error Handling:**
- thiserror 2.0.17 - Structured error types with derive macros

**Data Type Support:**
Polars 0.52 is configured with specific dtype features enabled (line 28-44):
- `lazy` - Lazy evaluation
- `dtype-struct`, `dtype-array` - Complex data types
- `dtype-i8`, `dtype-i16`, `dtype-u8`, `dtype-u16` - Integer types
- `zip_with`, `abs`, `round_series`, `log`, `is_in`, `cross_join`, `diff`, `partition_by` - Expression operations

## Configuration

**Environment:**
- Build feature flags in `Cargo.toml` (lines 17-19):
  - `default = ["python"]` - Enables PyO3 bindings by default
  - `python` - Optional feature gate for pyo3 and numpy dependencies
- Python version constraint: `requires-python = ">=3.9"` (`pyproject.toml` line 11)
- Maturin build configuration: `python-source = "python"`, module-name `polars_statistics._polars_statistics` (lines 97-101)

**Build:**
- Release profile optimization (`Cargo.toml` lines 69-72):
  - `codegen-units = 1` - Single codegen unit for better optimization
  - `lto = "fat"` - Full Link Time Optimization
  - `strip = "symbols"` - Strip debug symbols from binary

**Testing Configuration:**
- pytest: `testpaths = ["tests"]`, `addopts = "-v --tb=short"` (`pyproject.toml` lines 103-105)
- Code coverage: Codecov integration (`.github/workflows/ci.yml` lines 136-141)

**Linting & Formatting:**
- Ruff (Python): Line length 100, target Python 3.9, rules: E, F, W, I, UP, B, C4 (ignore E501)
- Cargo fmt: Rust formatting validation in CI
- Clippy: Rust linting with deny-warnings (`-D warnings`)

## Platform Requirements

**Development:**
- Rust stable toolchain with rustfmt and clippy components
- Python 3.9+ with venv support
- C compiler and OpenSSL development libraries (for Linux wheels)
- Cross-compilation targets for wheel building: x86_64, aarch64 (ARM64)

**Production:**
- Deployment: PyPI (Python Package Index) for wheels and source distributions
- Distribution: Platform-specific wheels for Linux (manylinux2014+), macOS (10.13+, ARM64), Windows
- PyPI distribution: Two-tier with TestPyPI for pre-release testing
- Trusted Publishing: Uses OIDC for PyPI authentication (no hardcoded tokens)

## CI/CD & Dependencies

**Version Control:**
- GitHub Actions workflows in `.github/workflows/`:
  - `ci.yml` - Lint, test (matrix: Python 3.9-3.12 × Ubuntu/macOS/Windows), coverage
  - `publish.yml` - Build wheels and publish to PyPI/TestPyPI
  - `docs.yml` - Documentation building (not shown but referenced)

**Artifact Management:**
- Wheels built for 5 platform combinations: Linux x86_64, Linux aarch64, macOS x86_64, macOS aarch64, Windows x86_64
- Source distribution (sdist) for pip install --no-binary

**Documentation:**
- mkdocs with Material theme (`mkdocs.yml`)
- Python-based markdown extensions: admonition, superfences, tabbed, toc, syntax highlighting
- Published to GitHub Pages at `https://datazoode.github.io/polars-statistics/`

---

*Stack analysis: 2026-08-11*
