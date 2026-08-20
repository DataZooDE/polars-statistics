# Requirements: polars-statistics

**Defined:** 2026-08-20
**Milestone:** v0.7.0 — Ergonomics, Adoption & Documentation
**Core Value:** Every public statistical capability in the backing `anofox-*` crates is exposed through the Polars/Python API — correctly, documented, and tested — and shipped to PyPI.

> Milestone focus: v0.6.0 delivered full API parity. v0.7.0 makes that surface
> best-in-class to adopt — type-safe, ergonomic to consume, clear on errors,
> clean, and fully documented — shipped as 0.7.0 on PyPI. Requirements for the
> completed v0.6.0 milestone are preserved in git history and
> `.planning/milestones/v0.6.0-phases/`.

## v1 Requirements

Requirements for this milestone. Each maps to exactly one roadmap phase.

### Type Safety

- [ ] **TYPE-01**: Package ships a `py.typed` marker and `.pyi` stubs so IDEs and type checkers resolve every Rust-bound model/test class (OLS, Ridge, Logistic, GLMM, …) with accurate constructor, method, and getter signatures
- [ ] **TYPE-02**: Type stubs cover the Python expression builders (`ps.ols(...)`, `ps.ttest_ind(...)`, …) and formula helpers with accurate signatures and return types
- [ ] **TYPE-03**: A CI/type check verifies the stubs stay in sync with the runtime API so they cannot silently rot

### Result Ergonomics

- [ ] **ERGO-01**: User can convert any result struct to a Python dict (e.g. `.to_dict()`) without manual `.struct.field()` extraction
- [ ] **ERGO-02**: Fitted models and test results provide a readable `.summary()` and an informative `__repr__`
- [ ] **ERGO-03**: A documented helper unnests result structs into flat DataFrame columns in one call

### Error Messages

- [ ] **ERR-01**: Calling a method on an unfitted model raises an error that names the model and the required `.fit(...)` call
- [ ] **ERR-02**: Shape mismatches and degenerate inputs (e.g. perfect separation, rank deficiency) raise actionable, contextual errors instead of panics or opaque failures

### API Consistency

- [ ] **API-01**: `with_intercept` is deprecated in favor of a single `add_intercept` path with a back-compatible `FutureWarning`, consistently across expressions and classes
- [ ] **API-02**: sklearn-style `fit`/`predict`/`score` behaves consistently across the regressor classes (uniform signatures and return conventions)

### Documentation

- [ ] **DOCS-05**: Runnable cookbook + `examples/` scripts cover the robust/sparse regressors (TheilSen, RANSAC, BayesianRidge, ARD, LARS, PassiveAggressive) with real-world scenarios and output interpretation
- [ ] **DOCS-06**: Runnable cookbook + examples cover the GLM/smoother/streaming models (Gamma, GLMM, PSpline, MomentAccumulator)
- [ ] **DOCS-07**: A cookbook entry covers the new ANOVA functions (one-way, two-way, repeated-measures) with worked examples
- [ ] **DOCS-08**: A model-selection decision matrix helps users choose among comparable models (use case, robustness, interpretability, speed, formula support)
- [ ] **DOCS-09**: A migration guide documents the `icc` single-column→matrix contract change (old→new side by side) and the `with_intercept` deprecation timeline
- [ ] **DOCS-10**: An sklearn-migration page maps common sklearn workflows to polars-statistics equivalents
- [ ] **DOCS-11**: The README is refreshed with an adoption-focused quickstart and narrative (typed API, sklearn-compat, Polars-native advantages)

### Release

- [ ] **REL-05**: `polars-statistics` is bumped 0.6.0 → 0.7.0 across all version sources (`Cargo.toml`, `pyproject.toml`, `python/polars_statistics/__init__.py` `__version__`) and they stay aligned
- [ ] **REL-06**: 0.7.0 is published to production PyPI via the GitHub Actions OIDC pipeline and passes a post-release install/import smoke check

## v2 Requirements

Deferred to future releases.

### Forecast Integration

- **FCST-01**: Evaluate exposing `anofox-forecast` capabilities (separate crate, not in this milestone)

## Out of Scope

| Feature | Reason |
|---------|--------|
| New statistical methods not in the `anofox-*` crates | This milestone is adoption/polish, not novel algorithm work |
| Removing the deprecated `with_intercept` kwarg | Kept back-compatible this milestone (warning only); removal is a future major-version concern |
| `anofox-forecast` integration | Separate crate, deferred (FCST-01) |
| FFI / numpy↔faer architecture changes | Existing bridge is sound and stays as-is |
| Breaking changes to already-exposed API | v0.7.0 is additive under semver; no breaking changes |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| _(populated by roadmapper)_ | | |

**Coverage:**

- v1 requirements: 19 total
- Mapped to phases: _(pending roadmap)_
- Unmapped: _(pending roadmap)_

---
*Requirements defined: 2026-08-20*
