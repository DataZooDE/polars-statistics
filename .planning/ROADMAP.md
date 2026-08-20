# Roadmap: polars-statistics

## Milestones

- ✅ **v0.6.0 API Parity & Release** - Phases 1-7 (shipped 2026-08-19)
- 🚧 **v0.7.0 Ergonomics, Adoption & Documentation** - Phases 8-12 (in progress)
- 📋 **v2.0 Forecast Integration** - deferred (FCST-01, not yet planned)

## Overview

v0.6.0 delivered full API parity with the backing `anofox-*` crates and shipped to
production PyPI. v0.7.0 shifts from capability to **adoption**: it makes that surface
best-in-class to pick up and use. The work moves code-first, docs-second, release-last.
First the API is made type-safe (stubs + `py.typed` + a CI drift guard), then results
become ergonomic to consume (dict/summary/repr + unnest helper), then errors are made
contextual and the API is cleaned up (deprecation + uniform sklearn-style `fit/predict/score`).
Only once those user-visible changes exist are they documented in a single adoption-focused
documentation phase (cookbook, decision matrix, migration guide, sklearn page, README), and
finally 0.7.0 is bumped across all version sources and published to PyPI.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.
Phase numbering is continuous across milestones — v0.7.0 continues from Phase 8.

<details>
<summary>✅ v0.6.0 API Parity & Release (Phases 1-7) — SHIPPED 2026-08-19</summary>

- [x] **Phase 1: Dependency Modernization** - Bump both anofox crates, reconcile breakage, keep existing tests green (completed 2026-08-11)
- [x] **Phase 2: Public API Audit** - Enumerate every unexposed function in both crates as an authoritative gap list (completed 2026-08-11)
- [x] **Phase 3: Statistics API Parity** - Expose ANOVA family, energy distance, and all remaining statistics gaps (completed 2026-08-12)
- [x] **Phase 4: Regression API Parity** - Expose GLMM, P-spline, Gamma GLM, HC inference, diagnostics, and remaining regression gaps (completed 2026-08-12)
- [x] **Phase 5: Documentation** - Document all newly exposed API across docstrings, mkdocs, Rust doc comments, and CHANGELOG (completed 2026-08-12)
- [x] **Phase 6: Testing & Validation** - Cover all new API with pytest + Rust tests, validated against R, CI matrix green (completed 2026-08-12)
- [x] **Phase 7: Release 0.6.0** - Bump version, publish wheels to production PyPI, smoke-check the release (completed 2026-08-19)

Full v0.6.0 phase detail is preserved in git history and `.planning/milestones/v0.6.0-phases/`.

</details>

### 🚧 v0.7.0 Ergonomics, Adoption & Documentation (In Progress)

**Milestone Goal:** Make polars-statistics best-in-class to adopt and use — type-safe,
ergonomic results, clear errors, a clean API, and complete docs for the full v0.6.0
surface — shipped as 0.7.0 on PyPI.

- [ ] **Phase 8: Type Safety** - Ship `.pyi` stubs + `py.typed` for all Rust-bound classes, expressions, and formula helpers, with a CI drift guard
- [ ] **Phase 9: Result Ergonomics** - Friendly result access: `.to_dict()`, `.summary()`/`__repr__`, and a one-call struct-unnest helper
- [ ] **Phase 10: Errors & API Consistency** - Contextual errors (not-fitted, shape/separation) plus `with_intercept`→`add_intercept` deprecation and uniform sklearn-style `fit/predict/score`
- [ ] **Phase 11: Documentation & Adoption** - Cookbook + examples for the new models/ANOVA, decision matrix, migration guide, sklearn-migration page, and refreshed README
- [ ] **Phase 12: Release 0.7.0** - Bump 0.6.0→0.7.0 across all version sources and publish to production PyPI via the OIDC pipeline

## Phase Details

### Phase 8: Type Safety
**Goal**: The whole public surface is statically typed — IDEs and type checkers resolve every Rust-bound class, expression builder, and formula helper with accurate signatures, and the stubs cannot silently rot.
**Depends on**: Phase 7 (v0.6.0 surface complete)
**Requirements**: TYPE-01, TYPE-02, TYPE-03
**Success Criteria** (what must be TRUE):
  1. Installing the package exposes a `py.typed` marker and `.pyi` stubs so a type checker (mypy/pyright) resolves every Rust-bound model/test class with accurate constructor, method, and getter signatures
  2. Autocomplete and type checking work for the Python expression builders (`ps.ols(...)`, `ps.ttest_ind(...)`, …) and formula helpers with accurate argument and return types
  3. A CI check fails when a stub drifts out of sync with the runtime API, so stubs stay accurate over time
**Plans**: TBD

### Phase 9: Result Ergonomics
**Goal**: Consuming a result no longer requires manual `.struct.field()` boilerplate — every result is inspectable, dict-convertible, and flattenable in one call.
**Depends on**: Phase 8
**Requirements**: ERGO-01, ERGO-02, ERGO-03
**Success Criteria** (what must be TRUE):
  1. User can call `.to_dict()` on any result struct and get a plain Python dict without manual `.struct.field()` extraction
  2. User can call `.summary()` on a fitted model or test result and read a formatted summary, and printing the object shows an informative `__repr__`
  3. User can unnest a result struct into flat DataFrame columns with a single documented helper call
**Plans**: TBD

### Phase 10: Errors & API Consistency
**Goal**: Failures are self-explanatory and the API is uniform — misuse produces actionable guidance, the intercept option has one canonical name (with a back-compatible warning), and every regressor follows the same sklearn-style contract.
**Depends on**: Phase 9
**Requirements**: ERR-01, ERR-02, API-01, API-02
**Success Criteria** (what must be TRUE):
  1. Calling a method on an unfitted model raises an error that names the model and points to the required `.fit(...)` call
  2. Shape mismatches and degenerate inputs (perfect separation, rank deficiency) raise actionable, contextual errors instead of panics or opaque failures
  3. Passing `with_intercept` still works but emits a `FutureWarning` steering the user to `add_intercept`, consistently across expressions and classes
  4. `fit`/`predict`/`score` have uniform signatures and return conventions across the regressor classes
**Plans**: TBD

### Phase 11: Documentation & Adoption
**Goal**: A new user can discover, choose, migrate to, and correctly use the full v0.6.0 surface — every new model and ANOVA function has runnable examples, comparable models have a decision matrix, breaking/deprecated changes have a migration guide, and the sklearn story and README tell the adoption narrative.
**Depends on**: Phase 10 (documents the deprecation from API-01 and the sklearn-style contract from API-02)
**Requirements**: DOCS-05, DOCS-06, DOCS-07, DOCS-08, DOCS-09, DOCS-10, DOCS-11
**Success Criteria** (what must be TRUE):
  1. Runnable cookbook + `examples/` scripts cover the robust/sparse regressors (TheilSen, RANSAC, BayesianRidge, ARD, LARS, PassiveAggressive) with real-world scenarios and output interpretation
  2. Runnable cookbook + examples cover the GLM/smoother/streaming models (Gamma, GLMM, PSpline, MomentAccumulator) and the new ANOVA functions (one-way, two-way, repeated-measures)
  3. A model-selection decision matrix helps users choose among comparable models across use case, robustness, interpretability, speed, and formula support
  4. A migration guide documents the `icc` single-column→matrix contract change (old→new side by side) and the `with_intercept` deprecation timeline
  5. An sklearn-migration page maps common sklearn workflows to polars-statistics equivalents, and the README quickstart/narrative presents the typed, sklearn-compatible, Polars-native adoption story
**Plans**: TBD

### Phase 12: Release 0.7.0
**Goal**: polars-statistics 0.7.0 — carrying all the ergonomics, error, API, and documentation work — is built, published to production PyPI via the existing OIDC pipeline, and confirmed installable.
**Depends on**: Phase 11 (release captures all prior milestone work)
**Requirements**: REL-05, REL-06
**Success Criteria** (what must be TRUE):
  1. Version reads 0.7.0 in `Cargo.toml`, `pyproject.toml`, and `python/polars_statistics/__init__.py` `__version__`, and all three stay aligned (the v0.6.0 stuck-release failure mode from misaligned version sources does not recur)
  2. sdist and platform wheels are built and published to production PyPI via the GitHub Actions OIDC pipeline (no hardcoded tokens)
  3. The published 0.7.0 wheel installs and imports cleanly as a post-release smoke check
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 8 → 9 → 10 → 11 → 12

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Dependency Modernization | v0.6.0 | 2/2 | Complete | 2026-08-11 |
| 2. Public API Audit | v0.6.0 | 1/1 | Complete | 2026-08-11 |
| 3. Statistics API Parity | v0.6.0 | 4/4 | Complete | 2026-08-12 |
| 4. Regression API Parity | v0.6.0 | 6/6 | Complete | 2026-08-12 |
| 5. Documentation | v0.6.0 | 4/4 | Complete | 2026-08-12 |
| 6. Testing & Validation | v0.6.0 | 5/5 | Complete | 2026-08-12 |
| 7. Release 0.6.0 | v0.6.0 | 3/3 | Complete | 2026-08-19 |
| 8. Type Safety | v0.7.0 | 0/TBD | Not started | - |
| 9. Result Ergonomics | v0.7.0 | 0/TBD | Not started | - |
| 10. Errors & API Consistency | v0.7.0 | 0/TBD | Not started | - |
| 11. Documentation & Adoption | v0.7.0 | 0/TBD | Not started | - |
| 12. Release 0.7.0 | v0.7.0 | 0/TBD | Not started | - |
