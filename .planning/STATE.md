---
gsd_state_version: "1.0"
milestone: v0.8.0
milestone_name: Claude Code Skills Plugin
current_phase: 14
current_phase_name: Skill Cheat-Sheet Authoring
status: executing
stopped_at: Phase 13 complete, ready to plan Phase 14
last_updated: "2026-09-21T18:37:27.730Z"
last_activity: 2026-09-21
last_activity_desc: Phase 14 execution started
state_head: c9eb378d7935b5a1a4f4c7aa973a2f359f7a9598
progress:
  total_phases: 5
  completed_phases: 1
  total_plans: 6
  completed_plans: 1
  percent: 17
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-20)

**Core value:** Every public statistical capability in the backing `anofox-*` crates is exposed through the Polars/Python API — correctly, documented, tested, and shipped to PyPI.
**Current focus:** Phase 14 — Skill Cheat-Sheet Authoring

## Current Position

Phase: 14 (Skill Cheat-Sheet Authoring) — EXECUTING
Plan: 1 of 5
Status: Executing Phase 14
Last activity: 2026-09-21 — Phase 14 execution started

## Milestone Roadmap (v0.8.0)

Execution order: 13 → 14 → 15 → 16 → 17

| Phase | Name | Requirements |
|-------|------|--------------|
| 13 | Plugin & Marketplace Scaffolding | PLUG-01, PLUG-02, PLUG-03 |
| 14 | Skill Cheat-Sheet Authoring | SKILL-01..06 |
| 15 | Accuracy Verification Pass | VERIFY-01, VERIFY-02 |
| 16 | Wheel Bundling | PKG-01 |
| 17 | Release 0.8.0 | REL-07, REL-08 |

## Performance Metrics

**Velocity:**

- Total plans completed: 23
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
| 13 | 1 | - | - |

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

- Milestone (v0.8.0): Package the full API as Claude Code skills — a plugin + marketplace mirroring anofox-forecast.
- Milestone (v0.8.0): 5 skills grouped by capability family (hypothesis-tests, anova, regression, results, getting-started).
- Milestone (v0.8.0): Plugin/marketplace layout mirrors anofox-forecast — root `.claude-plugin/marketplace.json` + `plugins/polars-statistics/` with `plugin.json` and `skills/`; `user-invocable: false` auto-surfaced skills.
- Milestone (v0.8.0): Bundle skills into the wheel (maturin package data) + publish 0.8.0 to PyPI; additive → minor bump.
- Roadmap (v0.8.0): Sequential technical-stage phases — scaffolding first (13), author skills (14), dedicated accuracy pass (15), wheel bundling (16), release last (17). Mirrors prior milestones' "release-last" pattern.
- Roadmap (v0.8.0): SKILL-01..06 grouped into a single authoring phase (14) since the five cheat-sheets share frontmatter/structure conventions and benefit from being written together with live example checks.
- Roadmap (v0.8.0): VERIFY-01/02 kept as a dedicated phase (15) so "every example runs" is a gate, not an afterthought inside authoring.

<details>
<summary>Prior-milestone decisions (v0.6.0 / v0.7.0) — reference</summary>

- Scoped cargo update (-p flags) prevents polars/pyo3 version churn from unrelated dep bumps
- No wrapper source reconciliation needed for anofox 0.5.4->0.5.13 bump — all call-site symbols stable
- ICC stub classified as exposed-but-stubbed; Phase 3 implemented real matrix-input ICC (validated vs Shrout & Fleiss ICC(2,1)=0.71)
- one_way_anova uses separate-group-Series input; factor/subject/condition encoding uses rank(dense)-1 per-column 0-indexed codes
- Reuse stats_output_dtype for energy_distance_nd; icc re-exported at crate root
- GammaRegressor::builder() pattern mirrors PyTweedie; GLMM fit(x,y,group) accepts Vec<u64> for 32/64-bit portability
- FactorSummary exposed as Python list[dict]; dispersion_output_dtype single-field struct for consistency
- fit_from_accumulator confirmed exact method name in OLS/Ridge solvers
- icc matrix-input contract change documented as Breaking Change (correlation.md)
- v0.6.0 stuck-release root cause: version bump reached only Cargo.lock; fixed by aligning Cargo.toml/pyproject.toml/`__version__` + re-pointing tag — carried into REL-07 as a hard requirement (all version sources aligned)

</details>

### Pending Todos

None yet.

### Blockers/Concerns

- None currently open.
- Note for Phase 14/15: examples must be verified against the *shipped* v0.7.0 surface (soon 0.8.0) — signatures live in `python/polars_statistics/` (exprs/*.py, regression.py, results.py, _polars_statistics.pyi). Do not invent kwargs.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260818-rj2 | Add Examples section to README linking runnable examples/ scripts and docs/examples cookbook (fixes #36) | 2026-08-18 | b5e8e16 | [260818-rj2-add-examples-section-to-readme-linking-r](./quick/260818-rj2-add-examples-section-to-readme-linking-r/) |
| 260819-jkj | Fix stuck v0.6.0 PyPI release — bumped python __version__ to 0.6.0, re-triggered publish | 2026-08-19 | (pending) | [260819-jkj-fix-v0.6.0-pypi-release](./quick/260819-jkj-fix-v0.6.0-pypi-release/) |

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Forecast | Evaluate exposing `anofox-forecast` capabilities (FCST-01, v2) | Deferred | 2026-08-11 |

## Session Continuity

Last session: 2026-09-20 — v0.8.0 roadmap created (Phases 13-17).
Stopped at: Phase 13 complete, ready to plan Phase 14
Resume file: None
</content>
