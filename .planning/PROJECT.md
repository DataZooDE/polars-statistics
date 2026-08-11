# polars-statistics

## What This Is

A Rust + Python (PyO3) library that exposes comprehensive statistical hypothesis
testing and regression modeling as native Polars expressions and Python classes.
Users write `df.group_by(...).agg(ps.ols("y", "x1", "x2"))` or
`from polars_statistics import OLS` and get R-validated statistics with zero-copy
numpy↔faer data transfer. It is distributed as a PyPI wheel for Python 3.9+.

This milestone modernizes the two backing engine crates — `anofox-statistics`
(0.4.1 → 0.4.2) and `anofox-regression` (0.5.4 → 0.5.13) — closes the API-exposure
gaps those crates now expose, documents and tests the new surface, and ships a new
release to PyPI.

## Core Value

Every public statistical capability in the backing `anofox-*` crates is exposed
through the Polars/Python API — correctly (post-bugfix), documented, and tested —
and shipped to users via PyPI.

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

### Active

<!-- This milestone. Hypotheses until shipped & validated. -->

- [ ] Expose all unexposed `anofox-statistics` functions (known: ANOVA family — `one_way_anova`, `two_way_anova`, `repeated_measures_anova`; `energy_distance_test`; plus any surfaced by the audit)
- [ ] Expose all unexposed `anofox-regression` functions (known candidates: `GlmmRegressor` mixed models, `PSplineRegressor` smoother, Gamma GLM, HC robust inference `HcInference`/`HcType`, diagnostics suite — Cook's distance, VIF, leverage, residual variants, condition diagnostics — streaming moment fits, and robust solvers e.g. Theil–Sen/RANSAC/LOWESS/Bayesian/passive-aggressive/LARS where applicable; final list from the audit)
- [ ] Update documentation (mkdocs API pages, Python docstrings, Rust doc comments) for all newly exposed API
- [ ] Create tests (pytest + Rust) covering all newly exposed API, validated against R where the crates provide reference values
- [ ] Bump `polars-statistics` 0.5.0 → 0.6.0 (Cargo.toml + pyproject.toml) and publish to production PyPI via the GitHub Actions pipeline

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
| Expose all unexposed functions (full API parity) | User chose completeness over a targeted subset | — Pending |
| Production PyPI publish this milestone | User wants a real release, not TestPyPI/prepare-only | — Pending |
| Target release version 0.6.0 | New API surface is additive → minor bump under semver | — Pending |

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
*Last updated: 2026-08-11 after Phase 2 (Public API Audit)*
