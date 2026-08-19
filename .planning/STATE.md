---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 7
current_phase_name: Release 0.6.0
status: planning
stopped_at: context exhaustion at 76% (2026-08-17)
last_updated: "2026-08-17T14:55:46.862Z"
last_activity: 2026-08-12
last_activity_desc: Phase 3 complete, transitioned to Phase 4
progress:
  total_phases: 6
  completed_phases: 6
  total_plans: 22
  completed_plans: 22
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-11)

**Core value:** Every public statistical capability in the backing `anofox-*` crates is exposed through the Polars/Python API — correctly, documented, tested, and shipped to PyPI.
**Current focus:** Phase 7 — Release 0.6.0

## Current Position

Phase: 7 — Release 0.6.0
Plan: Not started
Status: Ready to plan
Last activity: 2026-08-18 — Completed quick task 260818-rj2: add Examples section to README (fixes #36)

Progress: [████████░░] 86% (6/7 phases complete)

## Performance Metrics

**Velocity:**

- Total plans completed: 22
- Average duration: — min
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 2 | - | - |
| 2 | 1 | - | - |
| 3 | 4 | - | - |
| 4 | 6 | - | - |
| 5 | 4 | - | - |
| 6 | 5 | - | - |

**Recent Trend:**

- Last 5 plans: —
- Trend: —

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 01 P01 | 10 | 1 tasks | 1 files |
| Phase 01 P02 | 9 | 2 tasks | 1 files |
| Phase 02-public-api-audit P01 | 12 | 1 tasks | 1 files |
| Phase 03-statistics-api-parity P01 | 5400 | 4 tasks | 6 files |
| Phase 03-statistics-api-parity P02 | 10m | 2 tasks | 2 files |
| Phase 03-statistics-api-parity P03 | 7 | 2 tasks | 4 files |
| Phase 04-regression-api-parity P01 | 4 | 2 tasks | 4 files |
| Phase 04-regression-api-parity P02 | 56m | 3 tasks | 7 files |
| Phase 04-regression-api-parity P03 | 5896 | 3 tasks | 7 files |
| Phase 04-regression-api-parity P04 | 8 | 3 tasks | 10 files |
| Phase 04-regression-api-parity P05 | 10m | 3 tasks | 11 files |
| Phase 05-documentation P01 | 8 | 2 tasks | 5 files |
| Phase 05-documentation P02 | 8 | 3 tasks | 9 files |
| Phase 05-documentation P03 | 5 | 2 tasks | 9 files |
| Phase 05-documentation P04 | 3 | 1 tasks | 2 files |
| Phase 06 P02 | 10 | 2 tasks | 2 files |
| Phase 06 P03 | 20 | 3 tasks | 10 files |
| Phase 06 P04 | 15 | 2 tasks | 1 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Milestone: Bump both anofox crates (not statistics only) — regression 0.5.13 carries a correctness fix + major new API.
- Milestone: Expose all unexposed functions (full API parity) over a targeted subset.
- Milestone: Target 0.6.0 and publish to production PyPI this milestone.
- Roadmap: Layered (sequential technical-stage) phase structure — dep bump first, release last, audit before parity.
- [Phase ?]: Scoped cargo update (-p flags) prevents polars/pyo3 version churn from unrelated dep bumps
- [Phase ?]: No wrapper source reconciliation needed for anofox 0.5.4->0.5.13 bump — all call-site symbols stable
- [Phase ?]: Fixed rank-deficient test design (x2=100*x1) to periodic modular x2=((i%7)+1)*100 — maintains >=100:1 norm ratio with full rank
- [Phase ?]: Pre-existing .venv used for maturin develop + pytest — no new venv creation needed
- [Phase ?]: ICC stub classified as exposed-but-stubbed — icc_fit returns all-NaN with TODO; Phase 3 must implement real matrix-input ICC
- [Phase ?]: LOWESS confirmed internal-only — lowess_smooth_weights not in solvers/mod.rs pub use list; not a user-facing gap
- [Phase ?]: FactorSummary classified HIGH priority user output (FittedGlmm::factors()) — not internal despite naming
- [Phase ?]: one_way_anova uses separate-group-Series input (kruskal_wallis pattern) with kind literal as last arg
- [Phase ?]: Factor/subject/condition encoding uses rank(dense)-1 for per-column 0-indexed codes (Categorical.to_physical shares global catalog)
- [Phase ?]: Reuse stats_output_dtype for energy_distance_nd — no new output_dtype needed
- [Phase ?]: icc() crate-root import confirmed; anofox_statistics::icc re-exported at root, not correlation::icc
- [Phase ?]: Zero-rater guard: Python builder short-circuits before pl.all_horizontal([]) when no rater columns passed
- [Phase ?]: Used GammaRegressor::builder() pattern mirroring PyTweedie as the exact structural analog; predict_eta exposed as method not getter since it requires x input
- [Phase ?]: GLMM fit(x,y,group) accepts Vec<u64> for portability across 32/64-bit wheel targets; converts to Vec<usize> inside wrapper
- [Phase ?]: PSpline ncols guard runs after to_faer() to reuse faer matrix API, matching existing py_isotonic.rs pattern
- [Phase ?]: FactorSummary exposed as Python list[dict] (not a separate PyClass) per RESEARCH FactorSummary exposure strategy
- [Phase ?]: dispersion_output_dtype uses single-field struct (not bare Float64) for consistency with all other diagnostic output types
- [Phase ?]: compute_hc_inference arg order: actual crate signature differs from RESEARCH.md draft — corrected to (x, coef, intercept, residuals, aliased, with_intercept, hc_type, confidence_level) with no df param
- [Phase ?]: Registrations added in Tasks 1/2 not deferred to Task 3 to keep clippy -D warnings clean throughout
- [Phase ?]: fit_intercept (BayesianRidge/ARD) vs with_intercept (TheilSen/RANSAC) preserved per RESEARCH anti-pattern note
- [Phase ?]: fit_from_accumulator confirmed as exact method name in anofox-regression OLS/Ridge solvers (resolves Research Open Question 1)
- [Phase ?]: build_model helper on PyPassiveAggressive placed in plain impl block (not #[pymethods]) to prevent PyO3 from wrapping non-Python type
- [Phase ?]: PyMomentAccumulator.inner field marked pub(crate) for cross-module access in OLS/Ridge fit_from_accumulator methods
- [Phase ?]: check_docs_build.py uses structural nav-reference fallback when mkdocs not importable (all 39 nav files verified)
- [Phase ?]: CHANGELOG 0.6.0 section seeded with full API listing; plan 05-04 will finalize
- [Phase ?]: TheilSen/RANSAC/BayesianRidge/ARD needed class examples and getter docs; LARS/PA only needed is_fitted(); Ridge/WLS needed full doc lift
- [Phase ?]: check_rust_docs.py uses line-walk not regex to handle #[pyo3] attrs between /// and fn hc_inference
- [Phase ?]: energy_distance_nd placed in forecast.md Modern Distribution Tests (alongside energy_distance)
- [Phase ?]: icc updated to matrix-input contract with Breaking Change note in correlation.md
- [Phase ?]: Breaking Changes section added for icc matrix-input contract change (user-visible API change)
- [Phase ?]: [0.6.0] footer link uses HEAD since Phase 7 handles the actual release tag
- [Phase ?]: two_way_anova: validated vs statsmodels anova_lm with 15% tolerance
- [Phase ?]: icc: validated against Shrout & Fleiss (1979) ICC(2,1)=0.71 published value
- [Phase ?]: rust_api.rs: jittered data required for RM-ANOVA (ws_f=inf on exact monotone design) and Gamma diagnostics (dispersion=0 on exact exponential design)

### Pending Todos

None yet.

### Blockers/Concerns

- ✓ Resolved (Phase 1): dep bump landed, DEP-04 green (Rust 15 + pytest 457 passing), and the column-pivot correctness fix is confirmed active via 3 new tests. New API can now be wrapped in Phases 3–4.
- None currently open.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260818-rj2 | Add Examples section to README linking runnable examples/ scripts and docs/examples cookbook (fixes #36) | 2026-08-18 | b5e8e16 | [260818-rj2-add-examples-section-to-readme-linking-r](./quick/260818-rj2-add-examples-section-to-readme-linking-r/) |
| 260819-jkj | Fix stuck v0.6.0 PyPI release — PR #37 bumped Cargo.toml/pyproject.toml but missed python __version__ (still 0.5.0); bump it to 0.6.0. Publish still needs re-triggering (tag points at pre-fix commit) | 2026-08-19 | (pending) | [260819-jkj-fix-v0.6.0-pypi-release](./quick/260819-jkj-fix-v0.6.0-pypi-release/) |

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Forecast | Evaluate exposing `anofox-forecast` capabilities (FCST-01, v2) | Deferred | 2026-08-11 |

## Session Continuity

Last session: 2026-08-17T14:55:46.852Z
Stopped at: Phase 7 — v0.6.0 tag + GitHub Release published; publish.yml running (REL-03 in progress). REL-04 smoke check pending workflow completion (~25min) + PyPI propagation.
Resume file: None
