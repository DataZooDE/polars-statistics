# polars-statistics

## What This Is

A Rust + Python (PyO3) library that exposes comprehensive statistical hypothesis
testing and regression modeling as native Polars expressions and Python classes.
Users write `df.group_by(...).agg(ps.ols("y", "x1", "x2"))` or
`from polars_statistics import OLS` and get R-validated statistics with zero-copy
numpy↔faer data transfer. It is distributed as a PyPI wheel for Python 3.9+.

As of v0.6.0 the library exposes full parity with the backing `anofox-*` crates —
every public statistical capability is reachable, documented, tested, and shipped
on PyPI. This milestone (v0.7.0) shifts from capability to **adoption**: making the
existing surface type-safe, ergonomic to consume, clear on errors, and thoroughly
documented so the library is best-in-class to pick up and use.

## Core Value

Every public statistical capability in the backing `anofox-*` crates is exposed
through the Polars/Python API — correctly (post-bugfix), documented, and tested —
and shipped to users via PyPI.

## Current Milestone: v0.7.0 Ergonomics, Adoption & Documentation

**Goal:** Make polars-statistics best-in-class to adopt and use — type-safe,
ergonomic results, clear errors, a clean API, and complete docs for the full
v0.6.0 surface — shipped as 0.7.0 on PyPI.

**Target features:**
- Type stubs (`.pyi`) + `py.typed` marker for all Rust-bound classes & expressions
- Friendlier result access — `.to_dict()`/`.summary()`/repr + struct-unnest helpers
- Contextual error messages (not-fitted, shape mismatch, separation)
- Deprecation cleanup (`with_intercept`→`add_intercept`) + consistent sklearn-style `fit/predict/score`
- Cookbook + runnable examples for the 10 new v0.6.0 models and the new ANOVA functions
- Model-selection decision matrix
- Migration guide (icc contract change + deprecation timeline)
- sklearn-migration page + README adoption narrative
- Publish 0.7.0 to production PyPI via the existing OIDC pipeline

## Requirements

### Validated

<!-- Inferred from the existing codebase (.planning/codebase/) — shipped in v0.5.0. -->

- ✓ Polars expression API for statistical tests across parametric, nonparametric, distributional, correlation, categorical, modern (MMD/energy), forecast, and TOST/equivalence categories — existing
- ✓ 40+ regression & test PyModel classes (OLS, Ridge, ElasticNet, WLS, RLS, BLS, PLS, Quantile, Isotonic, Huber, Logistic, Poisson, NegativeBinomial, Tweedie, Probit, Cloglog, ALM, LmDynamic, Aid, StationaryBootstrap, CircularBlockBootstrap) — existing
- ✓ R-style formula parser (`y ~ x1 + x2 + x1:x2`) with interaction expansion — existing
- ✓ Zero-copy numpy↔faer array conversion (`ToFaer`/`IntoNumpy` traits) — existing
- ✓ Dual-build crate (cdylib for Python + rlib for Rust consumers), `python` feature gate — existing
- ✓ CI matrix (Python 3.9–3.12 × Linux/macOS/Windows), clippy/fmt/ruff gates, coverage — existing
- ✓ GitHub Actions PyPI + TestPyPI publish pipeline with OIDC trusted publishing — existing
- ✓ mkdocs (Material) documentation published to GitHub Pages — existing
- ✓ `anofox-statistics` bumped 0.4.1 → 0.4.2 — Phase 1 (clean build, existing tests green)
- ✓ `anofox-regression` bumped 0.5.4 → 0.5.13, inheriting the column-pivot correctness fix — Phase 1 (zero wrapper reconciliation; 3 new column-pivot tests confirm the fix is active; Rust 15 + pytest 457 passing)
- ✓ Authoritative public-API gap list for both crates (`02-API-AUDIT.md`) — Phase 2 (statistics: 3 ANOVA fns + energy nD + ICC stub; regression: Gamma/GLMM/PSpline/FactorSummary + HC-extend + 6 solvers + minor diagnostics; all 21 known candidates resolved; HC confirmed already-PARTIAL for OLS)
- ✓ All unexposed `anofox-statistics` functions exposed as Polars expressions — Phase 3 (one/two-way + repeated-measures ANOVA, energy_distance_nd, real matrix-input ICC replacing the NaN stub; 479 pytest + 15 rust_api green; 3 critical ANOVA factor-encoding bugs caught in review and fixed with regression tests)
- ✓ All unexposed `anofox-regression` capabilities exposed (full parity) — Phase 4 (10 new PyModels: Gamma, GLMM, PSpline, TheilSen, RANSAC, BayesianRidge, ARD, LARS, PassiveAggressive, MomentAccumulator; Ridge/WLS HC; 5 GLM-diagnostic expressions; OLS/Ridge fit_from_accumulator. 582 pytest + 15 rust_api green; 5 critical + 6 warning review findings fixed. Known limitation: weighted HC for WLS raises NotImplementedError pending crate support.)
- ✓ Full API documentation for the new surface — Phase 5 (Python docstrings w/ runnable examples, mkdocs reference pages, Rust /// comments, CHANGELOG 0.6.0 with known limitations)
- ✓ Value + shape tests for all new API, validated vs scipy/statsmodels/analytic references — Phase 6 (602 pytest + 17 rust_api green; ICC(3,1)=0.715 vs Shrout&Fleiss, GLMM slope vs truth, Gamma vs statsmodels, fit_from_accumulator==batch; full OS/py matrix delegated to GH Actions)
- ✓ Bumped `polars-statistics` 0.5.0 → 0.6.0 and published to production PyPI — Phase 7 + quick task 260819-jkj (0.6.0 live on PyPI; publish had failed on a stale 0.5.0-named wheel because the version bump had only reached Cargo.lock — closed by aligning Cargo.toml/pyproject.toml/`__version__`, re-pointing the tag, and re-running the OIDC pipeline)

### Active

<!-- This milestone (v0.7.0). Hypotheses until shipped & validated. -->

- [ ] Package ships type stubs (`.pyi`) and a `py.typed` marker so IDEs and type checkers see all Rust-bound classes and expressions
- [ ] Results are ergonomic to consume — friendly `.to_dict()`/`.summary()`/repr and struct-unnest helpers replace manual `.struct.field()` boilerplate
- [ ] Errors are contextual and actionable (not-fitted names the model + next call; shape mismatch and separation are caught with guidance)
- [ ] API is clean — `with_intercept`→`add_intercept` deprecation resolved (back-compat warning) and sklearn-style `fit/predict/score` consistent across regressors
- [ ] Every v0.6.0 model and the new ANOVA functions have runnable cookbook + `examples/` coverage
- [ ] A model-selection decision matrix helps users choose between comparable models
- [ ] A migration guide documents the icc contract change and the deprecation timeline
- [ ] An sklearn-migration page + refreshed README present the adoption story
- [ ] `polars-statistics` 0.7.0 is published to production PyPI via the existing OIDC pipeline

### Out of Scope

- New statistical methods not present in the `anofox-*` crates — this milestone is exposure/parity, not novel algorithm work
- `anofox-forecast` integration — separate crate, not requested
- Changes to the FFI architecture or the numpy↔faer transfer strategy — the existing bridge stays
- Reworking already-exposed API surface beyond what the version bump requires

## Context

- **Backing crates:** `anofox-regression` and `anofox-statistics` (from github.com/sipemu), both registry deps in `Cargo.toml` (lines 64, 67). The regression jump spans 0.5.5–0.5.13 and includes GLMM (0.5.12), P-spline (0.5.10), streaming moments (0.5.9), and the 0.5.13 correctness fixes.
- **Correctness driver:** anofox-regression 0.5.13 fixes a column-pivot unpermute bug that silently scrambled OLS/WLS/NNLS coefficients on differently-scaled designs. The wrapper currently pins 0.5.4, so it is exposed to this bug today — the bump is a correctness fix, not just a feature add.
- **Codebase map:** Full map exists at `.planning/codebase/` (STACK, ARCHITECTURE, STRUCTURE, CONVENTIONS, TESTING, INTEGRATIONS, CONCERNS).
- **Two exposure surfaces per capability:** most statistics functions are wrapped as Polars expressions (`src/expressions/*.rs` → `python/polars_statistics/exprs/*.py`); regression models are additionally wrapped as PyModel classes (`src/pymodels/*.rs`). New API may need both.
- **Release mechanics:** version lives in both `Cargo.toml` and `pyproject.toml` (both 0.5.0); publish is tag-triggered via `.github/workflows/publish.yml` using OIDC trusted publishing (no tokens).

## Constraints

- **Tech stack**: Rust 2021 + PyO3 0.26 + Polars 0.52; new API must follow the existing `#[polars_expr]` / PyModel patterns and output-struct schemas — Why: consistency and native Polars integration.
- **Validation**: New tests validated against R where the crates supply references — Why: the library's core promise is "validated against R."
- **CI gates**: clippy `-D warnings`, cargo fmt, ruff (line length 100, py39 target) must stay green — Why: existing merge gates.
- **Compatibility**: Maintain Python 3.9–3.12 support across Linux/macOS/Windows wheels — Why: existing distribution matrix.
- **Release**: Production PyPI publish via the existing GH Actions pipeline (OIDC, no hardcoded tokens) — Why: established, secure release path.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Bump both `anofox-statistics` and `anofox-regression` (not statistics only) | User chose full modernization; regression 0.5.13 also carries a correctness fix and major new API | ✓ Done — Phase 1 (zero reconciliation; column-pivot fix confirmed active by 3 new tests) |
| Expose all unexposed functions (full API parity) | User chose completeness over a targeted subset | ✓ Done — Phases 3+4 (all statistics + regression gaps exposed) |
| Production PyPI publish (v0.6.0) | User wanted a real release, not TestPyPI/prepare-only | ✓ Done — 0.6.0 live on PyPI |
| Target release version 0.6.0 | New API surface was additive → minor bump under semver | ✓ Done |
| v0.7.0 milestone = ergonomics + adoption + docs (not new algorithms) | v0.6.0 delivered full capability parity; the remaining friction is typing, result ergonomics, errors, and docs coverage | — Pending |
| Target release version 0.7.0 (additive, no breaking changes) | Stubs/helpers/docs are additive; `with_intercept` deprecation stays back-compatible (warning, not removal) → minor bump | — Pending |
| Align GSD milestone label to package version (v0.7.0) | STATE previously carried placeholder `v1.0`, diverging from the real 0.x package version | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-08-20 — started milestone v0.7.0 (Ergonomics, Adoption & Documentation)*
