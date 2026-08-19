# External Integrations

**Analysis Date:** 2026-08-11

## APIs & External Services

**Statistical Libraries (Rust):**
- anofox-regression - Custom regression models (OLS, Ridge, Elastic Net, GLM, Quantile, Huber, PLS, ALM)
  - SDK/Client: `anofox-regression = "0.5.4"` in Cargo.toml
  - No authentication required - direct library dependency
  - Integration: Core regression engine wrapped by `src/expressions/regression.rs`

- anofox-statistics - Statistical tests and distributions
  - SDK/Client: `anofox-statistics = "0.4.1"` in Cargo.toml
  - No authentication required - direct library dependency
  - Integration: Powers parametric, non-parametric, and distributional tests in `src/expressions/`

**Linear Algebra:**
- faer - Matrix operations with Rayon parallelization
  - SDK/Client: `faer = { version = "0.23.2", features = ["rayon"] }` in Cargo.toml
  - No authentication required - direct library dependency
  - Integration: Core matrix operations for regression models

**Data Processing:**
- Polars - DataFrame processing engine
  - SDK/Client: `polars = "0.52"` with specific feature flags in Cargo.toml
  - No authentication required - direct library dependency
  - Integration: Primary data structure for all statistical operations via expressions API

**Python Bridge:**
- PyO3 - Python bindings (optional feature)
  - SDK/Client: `pyo3 = { version = "0.26", features = ["abi3-py39", "extension-module"], optional = true }`
  - No authentication required - direct library dependency
  - Integration: Enables `_polars_statistics` extension module built by maturin
  - Allocator: Uses `PolarsAllocator` from pyo3-polars for memory management

- NumPy (optional)
  - SDK/Client: `numpy = { version = "0.26", optional = true }`
  - No authentication required - direct library dependency
  - Integration: Optional dependency for NumPy array interop (gated behind `python` feature)

## Data Storage

**Databases:**
- None required - operates on in-memory DataFrames

**File Storage:**
- Local filesystem only
  - Examples stored in `examples/` directory
  - Documentation in `docs/` directory
  - Generated site in `site/` directory

**Caching:**
- None - all computations are stateless expressions

## Authentication & Identity

**Auth Provider:**
- Custom/None required
  - PyPI uses Trusted Publishing (OIDC) for automated releases (`.github/workflows/publish.yml` line 125)
  - No API keys or tokens stored in code

**Build Secrets:**
- TestPyPI: `secrets.TEST_PYPI_API_TOKEN` (line 115 in publish.yml)
- PyPI: Trusted Publishing via OIDC (no token needed)
- Codecov: `secrets.CODECOV_TOKEN` (line 141 in ci.yml)

## Monitoring & Observability

**Error Tracking:**
- None - no error tracking service integrated
- Returns structured error types via thiserror (`src/` modules)

**Logs:**
- Standard output/stderr only
- No centralized logging service
- CI output captured in GitHub Actions logs

**Testing Validation:**
- Comparison against R statistical functions (documented in `validation/` directory)
- Test cases include validation datasets from scipy and statsmodels

## CI/CD & Deployment

**Hosting:**
- PyPI (https://pypi.org/p/polars-statistics/) - Primary distribution
- TestPyPI (https://test.pypi.org/p/polars-statistics/) - Pre-release testing
- GitHub Releases - Version tags and CHANGELOG
- GitHub Pages - Documentation site (https://datazoode.github.io/polars-statistics/)

**CI Pipeline:**
- GitHub Actions - Automated testing and building
  - Lint job: Cargo fmt and clippy checks
  - Test job: Python tests across 3 versions × 3 OS platforms (12 matrix combinations)
  - Coverage job: Pytest with coverage reporting to Codecov
  - Build wheels: PyO3/maturin cross-compilation for 5 platform targets

**Workflow Triggers:**
- Push to main/master branches
- Pull requests to main/master
- Release creation (auto-publish to PyPI)
- Manual workflow dispatch (for TestPyPI testing)

## Environment Configuration

**Required env vars:**
- `CARGO_TERM_COLOR` - Set to `always` in CI for colored output
- `CODECOV_TOKEN` - Required for uploading coverage to Codecov (GitHub secret)
- `TEST_PYPI_API_TOKEN` - Required for TestPyPI publishing (GitHub secret)
- Build targets configured in publish.yml: x86_64, aarch64

**Secrets location:**
- GitHub repository secrets (not in code)
- `.env` files: None required for typical usage

**Build Configuration:**
- Maturin: `python-source = "python"`, bindings = "pyo3", features = ["pyo3/extension-module", "python"]
- Release optimization: LTO, single codegen unit, symbol stripping

## Webhooks & Callbacks

**Incoming:**
- None - no webhook endpoints

**Outgoing:**
- GitHub Actions workflow: Triggered by push/pull_request events
- Codecov: Coverage report upload from CI
- PyPI: Automatic publishing on GitHub Release creation via Trusted Publishing

## External Data/Test Validation

**Reference Implementations:**
- R statistical functions - Validation dataset for testing accuracy
- scipy.stats - Python reference for comparison
- statsmodels - Python reference implementations
- Used in CI for regression validation against known outputs

**Documentation References:**
- GitHub repository README (in-project docs)
- Published documentation on GitHub Pages (auto-deployed)
- mkdocs site generation from markdown in `docs/` directory

---

*Integration audit: 2026-08-11*
