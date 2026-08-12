# Requirements: polars-statistics

**Defined:** 2026-08-11
**Core Value:** Every public statistical capability in the backing `anofox-*` crates is exposed through the Polars/Python API — correctly, documented, and tested — and shipped to PyPI.

## v1 Requirements

Requirements for this milestone (`anofox-*` upgrade + API parity + 0.6.0 release). Each maps to roadmap phases.

### Dependency Upgrade

- [x] **DEP-01**: `anofox-statistics` is bumped 0.4.1 → 0.4.2 in `Cargo.toml`/`Cargo.lock` and the workspace builds with the `python` feature
- [x] **DEP-02**: `anofox-regression` is bumped 0.5.4 → 0.5.13 in `Cargo.toml`/`Cargo.lock` and the workspace builds
- [x] **DEP-03**: Any breaking API changes introduced by the bumps are reconciled in the wrapper layer so it compiles and existing behavior is preserved
- [x] **DEP-04**: All pre-existing tests (Rust + pytest) pass against the upgraded crates — no regressions from the bump

### API Audit

- [x] **AUDIT-01**: An authoritative, documented gap list enumerates every public `anofox-statistics` function not yet exposed via the Polars/Python API
- [x] **AUDIT-02**: An authoritative, documented gap list enumerates every public `anofox-regression` capability not yet exposed via expressions or PyModel classes

### Statistics API Parity

- [x] **STAT-01**: User can compute a one-way ANOVA via the Polars expression API
- [x] **STAT-02**: User can compute a two-way ANOVA via the Polars expression API
- [x] **STAT-03**: User can compute a repeated-measures ANOVA via the Polars expression API
- [x] **STAT-04**: User can compute the energy distance test via the Polars expression API
- [x] **STAT-05**: Every remaining unexposed `anofox-statistics` function identified in AUDIT-01 is callable via the Polars/Python API

### Regression API Parity

- [x] **REGR-01**: User can fit a generalized linear mixed model (`GlmmRegressor`) via a PyModel class
- [x] **REGR-02**: User can fit a penalized B-spline (P-spline) smoother via the API
- [x] **REGR-03**: User can fit a Gamma GLM via the API
- [ ] **REGR-04**: User can obtain HC (heteroskedasticity-consistent) robust standard errors (`HcInference`/`HcType`) for regression fits
- [ ] **REGR-05**: User can compute regression diagnostics — Cook's distance, VIF, leverage, residual variants, and condition diagnostics — via the API
- [ ] **REGR-06**: Every remaining unexposed `anofox-regression` capability identified in AUDIT-02 is callable via expressions and/or PyModel classes

### Documentation

- [ ] **DOCS-01**: Every newly exposed function/class has a Python docstring with signature, parameters, and a runnable example
- [ ] **DOCS-02**: The mkdocs API reference pages are updated to list all newly exposed API
- [ ] **DOCS-03**: Rust doc comments are added for all new public wrapper functions
- [ ] **DOCS-04**: The CHANGELOG / release notes document the new API and the 0.6.0 release

### Testing

- [ ] **TEST-01**: Each newly exposed statistics function has a pytest test asserting correct output shape and values
- [ ] **TEST-02**: Each newly exposed regression capability has a pytest test asserting correct output shape and values
- [ ] **TEST-03**: New results are validated against R reference values where the crates provide them
- [ ] **TEST-04**: Rust-side tests cover the new expression wrappers and output-type schemas
- [ ] **TEST-05**: The full CI matrix (Python 3.9–3.12 × Linux/macOS/Windows, clippy/fmt/ruff) passes green

### Release

- [ ] **REL-01**: `polars-statistics` version is bumped 0.5.0 → 0.6.0 in both `Cargo.toml` and `pyproject.toml`
- [ ] **REL-02**: A 0.6.0 git tag and release notes are prepared
- [ ] **REL-03**: Wheels (sdist + platform wheels) are built and published to production PyPI via the GitHub Actions pipeline
- [ ] **REL-04**: The published 0.6.0 wheel installs and imports cleanly as a post-release smoke check

## v2 Requirements

Deferred to future releases.

### Forecast Integration

- **FCST-01**: Evaluate exposing `anofox-forecast` capabilities (separate crate, not in this milestone)

## Out of Scope

| Feature | Reason |
|---------|--------|
| New statistical methods not in the `anofox-*` crates | This milestone is exposure/parity, not novel algorithm work |
| `anofox-forecast` integration | Separate crate, not requested for this milestone |
| FFI / numpy↔faer architecture changes | Existing bridge is sound and stays as-is |
| Reworking already-exposed API beyond bump requirements | Out of scope; avoids churn on working surface |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DEP-01 | Phase 1 | Complete |
| DEP-02 | Phase 1 | Complete |
| DEP-03 | Phase 1 | Complete |
| DEP-04 | Phase 1 | Complete |
| AUDIT-01 | Phase 2 | Complete |
| AUDIT-02 | Phase 2 | Complete |
| STAT-01 | Phase 3 | Complete |
| STAT-02 | Phase 3 | Complete |
| STAT-03 | Phase 3 | Complete |
| STAT-04 | Phase 3 | Complete |
| STAT-05 | Phase 3 | Complete |
| REGR-01 | Phase 4 | Complete |
| REGR-02 | Phase 4 | Complete |
| REGR-03 | Phase 4 | Complete |
| REGR-04 | Phase 4 | Pending |
| REGR-05 | Phase 4 | Pending |
| REGR-06 | Phase 4 | Pending |
| DOCS-01 | Phase 5 | Pending |
| DOCS-02 | Phase 5 | Pending |
| DOCS-03 | Phase 5 | Pending |
| DOCS-04 | Phase 5 | Pending |
| TEST-01 | Phase 6 | Pending |
| TEST-02 | Phase 6 | Pending |
| TEST-03 | Phase 6 | Pending |
| TEST-04 | Phase 6 | Pending |
| TEST-05 | Phase 6 | Pending |
| REL-01 | Phase 7 | Pending |
| REL-02 | Phase 7 | Pending |
| REL-03 | Phase 7 | Pending |
| REL-04 | Phase 7 | Pending |

**Coverage:**

- v1 requirements: 28 total
- Mapped to phases: 28
- Unmapped: 0 ✓

---
*Requirements defined: 2026-08-11*
*Last updated: 2026-08-11 after roadmap creation (traceability populated)*
