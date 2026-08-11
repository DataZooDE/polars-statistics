# Roadmap: polars-statistics

## Overview

This milestone modernizes the two backing engine crates, closes the resulting API-exposure
gaps, documents and tests the new surface, and ships 0.6.0 to production PyPI. The work follows
a layered, sequential technical progression driven by a hard dependency chain: the dependency
bump must land first (it unblocks every new API and carries a critical upstream correctness fix),
an authoritative API audit then defines the exact parity scope, statistics and regression APIs are
wrapped to reach parity, the new surface is documented and tested against R references, and finally
the 0.6.0 wheel is published and smoke-checked. Each phase completes a coherent technical stage and
unblocks the next.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Dependency Modernization** - Bump both anofox crates, reconcile breakage, keep existing tests green (completed 2026-08-11)
- [x] **Phase 2: Public API Audit** - Enumerate every unexposed function in both crates as an authoritative gap list (completed 2026-08-11)
- [ ] **Phase 3: Statistics API Parity** - Expose ANOVA family, energy distance, and all remaining statistics gaps
- [ ] **Phase 4: Regression API Parity** - Expose GLMM, P-spline, Gamma GLM, HC inference, diagnostics, and remaining regression gaps
- [ ] **Phase 5: Documentation** - Document all newly exposed API across docstrings, mkdocs, Rust doc comments, and CHANGELOG
- [ ] **Phase 6: Testing & Validation** - Cover all new API with pytest + Rust tests, validated against R, CI matrix green
- [ ] **Phase 7: Release 0.6.0** - Bump version, publish wheels to production PyPI, smoke-check the release

## Phase Details

### Phase 1: Dependency Modernization

**Goal**: Both backing crates are upgraded, the wrapper compiles and behaves as before, and the critical column-pivot correctness fix is inherited — establishing the foundation every later phase builds on.
**Depends on**: Nothing (first phase)
**Requirements**: DEP-01, DEP-02, DEP-03, DEP-04
**Success Criteria** (what must be TRUE):

  1. `anofox-statistics` reads 0.4.2 and `anofox-regression` reads 0.5.13 in `Cargo.toml`/`Cargo.lock`, and the workspace builds with the `python` feature
  2. Any breaking API changes from the bumps are reconciled so the wrapper compiles clean under clippy `-D warnings` with existing behavior preserved
  3. The full pre-existing test suite (Rust `rust_api` + pytest) passes with no regressions against the upgraded crates
  4. OLS/WLS/NNLS fits on differently-scaled designs now return correct coefficients (the 0.5.13 column-pivot fix is confirmed active)

**Plans**: 2/2 plans executed
**Wave 1**

- [x] 01-01-PLAN.md — Bump both anofox crate pins, regenerate Cargo.lock, prove wrapper compiles clean under clippy (tracer)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 01-02-PLAN.md — Add three column-pivot correctness tests, run full Rust + pytest suites (no regressions)

### Phase 2: Public API Audit

**Goal**: An authoritative, documented gap list enumerates exactly which public functions in each upgraded crate are not yet exposed, defining the concrete scope for the two parity phases.
**Depends on**: Phase 1
**Requirements**: AUDIT-01, AUDIT-02
**Success Criteria** (what must be TRUE):

  1. A documented gap list enumerates every public `anofox-statistics` function not yet reachable via the Polars/Python API
  2. A documented gap list enumerates every public `anofox-regression` capability not yet reachable via expressions or PyModel classes
  3. Each gap entry records the target exposure surface (Polars expression, PyModel class, or both) so parity work is unambiguous
  4. The known candidates (ANOVA family, energy distance, GLMM, P-spline, Gamma GLM, HC inference, diagnostics) are confirmed present-or-absent against the actual upgraded crate surface

**Plans**: 1/1 plans executed
**Wave 1**

- [x] 02-01-PLAN.md — Synthesize RESEARCH.md enumeration into the authoritative 02-API-AUDIT.md gap list (both crates, verdict + target surface per item)

### Phase 3: Statistics API Parity

**Goal**: Every unexposed `anofox-statistics` function identified in the audit is callable from Python via the Polars expression API, following the existing `#[polars_expr]` and output-struct patterns.
**Depends on**: Phase 2
**Requirements**: STAT-01, STAT-02, STAT-03, STAT-04, STAT-05
**Success Criteria** (what must be TRUE):

  1. User can compute a one-way ANOVA via the Polars expression API and receives an F/p output struct
  2. User can compute two-way and repeated-measures ANOVA via the Polars expression API
  3. User can compute the energy distance test via the Polars expression API
  4. Every remaining statistics function from the AUDIT-01 gap list is callable via the Polars/Python API
  5. All new statistics expressions define output-struct schemas consistent with existing expression conventions

**Plans**: 1/4 plans executed

**Wave 1** *(parallel — no shared files)*

- [x] 03-01-PLAN.md — ANOVA family tracer: one/two-way + RM-ANOVA expressions, output schemas, Python builders, ANOVA smoke tests (STAT-01/02/03)
- [ ] 03-02-PLAN.md — energy_distance_nd expression + Python builder (STAT-04)

**Wave 2** *(blocked on 03-01 — shares output_types.rs)*

- [ ] 03-03-PLAN.md — real matrix-input ICC replacing the all-NaN stub + icc_output_dtype + rewritten icc builder (STAT-05)

**Wave 3** *(blocked on 03-01/02/03 — registration + integration)*

- [ ] 03-04-PLAN.md — register energy_distance_nd, full smoke-test suite (5 classes) + Rust smoke tests, clippy/fmt/ruff + full-suite phase gate (STAT-01..05)

### Phase 4: Regression API Parity

**Goal**: Every unexposed `anofox-regression` capability identified in the audit is callable via expressions and/or PyModel classes, following the existing PyModel and expression patterns.
**Depends on**: Phase 2
**Requirements**: REGR-01, REGR-02, REGR-03, REGR-04, REGR-05, REGR-06
**Success Criteria** (what must be TRUE):

  1. User can fit a generalized linear mixed model (`GlmmRegressor`) and a penalized B-spline (P-spline) smoother via the API
  2. User can fit a Gamma GLM via the API
  3. User can obtain HC (heteroskedasticity-consistent) robust standard errors (`HcInference`/`HcType`) for regression fits
  4. User can compute regression diagnostics — Cook's distance, VIF, leverage, residual variants, and condition diagnostics — via the API
  5. Every remaining regression capability from the AUDIT-02 gap list is callable via expressions and/or PyModel classes

**Plans**: TBD

### Phase 5: Documentation

**Goal**: The entire newly exposed API surface is documented for users and Rust consumers, and the release is described in the changelog.
**Depends on**: Phase 3, Phase 4
**Requirements**: DOCS-01, DOCS-02, DOCS-03, DOCS-04
**Success Criteria** (what must be TRUE):

  1. Every newly exposed function/class has a Python docstring with signature, parameters, and a runnable example
  2. The mkdocs API reference pages list all newly exposed API
  3. Rust doc comments are present for all new public wrapper functions
  4. The CHANGELOG documents the new API and the 0.6.0 release

**Plans**: TBD

### Phase 6: Testing & Validation

**Goal**: The newly exposed API is verified for correct shape and values, validated against R references where available, and the full CI matrix passes green.
**Depends on**: Phase 3, Phase 4
**Requirements**: TEST-01, TEST-02, TEST-03, TEST-04, TEST-05
**Success Criteria** (what must be TRUE):

  1. Each newly exposed statistics function has a pytest test asserting correct output shape and values
  2. Each newly exposed regression capability has a pytest test asserting correct output shape and values
  3. New results are validated against R reference values wherever the crates supply them
  4. Rust-side tests cover the new expression wrappers and output-type schemas
  5. The full CI matrix (Python 3.9–3.12 × Linux/macOS/Windows, clippy/fmt/ruff) passes green

**Plans**: TBD

### Phase 7: Release 0.6.0

**Goal**: polars-statistics 0.6.0 is built, published to production PyPI via the existing OIDC pipeline, and confirmed installable — delivering the full modernized, parity-complete surface to users.
**Depends on**: Phase 5, Phase 6
**Requirements**: REL-01, REL-02, REL-03, REL-04
**Success Criteria** (what must be TRUE):

  1. Version reads 0.6.0 in both `Cargo.toml` and `pyproject.toml`
  2. A 0.6.0 git tag and release notes are prepared
  3. sdist and platform wheels are built and published to production PyPI via the GitHub Actions pipeline
  4. The published 0.6.0 wheel installs and imports cleanly as a post-release smoke check

**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Dependency Modernization | 2/2 | Complete    | 2026-08-11 |
| 2. Public API Audit | 1/1 | Complete    | 2026-08-11 |
| 3. Statistics API Parity | 1/4 | In Progress|  |
| 4. Regression API Parity | 0/TBD | Not started | - |
| 5. Documentation | 0/TBD | Not started | - |
| 6. Testing & Validation | 0/TBD | Not started | - |
| 7. Release 0.6.0 | 0/TBD | Not started | - |
