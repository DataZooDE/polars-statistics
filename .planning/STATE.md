---
gsd_state_version: "1.0"
milestone: v0.8.0
milestone_name: Claude Code Skills Plugin
current_phase: "16.1"
current_phase_name: Upstream Crate Refresh
current_plan: 6
status: executing
stopped_at: Completed 16.1-04-PLAN.md
last_updated: "2026-10-09T07:19:14.506Z"
last_activity: 2026-10-08
last_activity_desc: Phase 16.1 execution resumed (wave continue)
state_head: 47e02749413783dcc4c1a9d578cbe588966e1388
progress:
  total_phases: 8
  completed_phases: 3
  total_plans: 25
  completed_plans: 12
  percent: 38
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-20)

**Core value:** Every public statistical capability in the backing `anofox-*` crates is exposed through the Polars/Python API — correctly, documented, tested, and shipped to PyPI.
**Current focus:** Phase 16.1 — Upstream Crate Refresh

## Current Position

Phase: 16.1 (Upstream Crate Refresh) — EXECUTING
Current Plan: 6
Total Plans in Phase: 9
Status: Ready to execute
Last activity: 2026-10-08 — Phase 16.1 execution resumed (wave continue)

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

- Total plans completed: 29
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
| 14 | 5 | - | - |
| 15 | 1 | - | - |

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
| Phase 14 P01 | 4 | 1 tasks | 1 files |
| Phase 14 P02 | 9min | 1 tasks | 1 files |
| Phase 16.1 P01 | 34min | 3 tasks | 9 files |
| Phase 16.1 P02 | 25min | 3 tasks | 10 files |
| Phase 16.1 P03 | 55min | 3 tasks | 10 files |
| Phase 16.1 P09 | 10min | 2 tasks | 2 files |
| Phase 16.1 P04 | 42min | 3 tasks | 10 files |

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

- [Phase 14]: 14-01: Frontmatter contract locked — four fields only (name, description, version 0.8.0, user-invocable false); Wave-2 plans copy verbatim
- [Phase 14]: 14-01: Section structure locked — Overview / When-to-use / API-Reference / Worked-Examples / Gotchas / See-also for all five SKILL.md files
- [Phase 14]: Forecast functions excluded from hypothesis-tests skill; ICC gotcha documented inline under Group D; categorical split into scalar-arg vs contingency-table-column subgroups
- [Phase 16.1]: 16.1-01: Cargo.lock is gitignored (library-crate convention) — version pins verified on disk via grep, not committed to git
- [Phase 16.1]: 16.1-02: fisher_exact schema widening (statistic/p_value -> +ci_lower/ci_upper) required zero existing-test edits. — Backward-compatible additive struct field; membership-style assertions kept passing unmodified.
- [Phase 16.1]: 16.1-03: resumed stalled executor's uncommitted Task 1 work after reviewing it matched the plan exactly
- [Phase 16.1]: 16.1-03: Tasks 1/2 edits to shared files temporarily set aside/restored per-task (not git stash) to keep each commit's rebuild+verify scoped to only that task's code
- [Phase 16.1]: 16.1-09: Fixture copied byte-identically from upstream release commit 3ab3bec exactly as the plan specified; no fallback commit needed.
- [Phase 16.1]: 16.1-09: Split the plan's single test file into two commits aligned with the plan's two <task> blocks to keep per-task git history atomic.
- [Phase 16.1]: 16.1-04: bootstrap_mean_ci/bootstrap_ci Python builders cast x to Float64 before the non-finite filter (int64 columns otherwise fail the plugin's Float64 dtype check), matching exprs/modern.py's existing pattern.

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

### Roadmap Evolution

- Phase 16.1 inserted after Phase 16: Upstream crate refresh: anofox-statistics 0.4.4 + anofox-regression 0.5.15 — bump, adapt to R-matching behaviour changes, expose new API, update stubs/docs/CHANGELOG/skills (URGENT)
- Phase 16.2 inserted after Phase 16.1: Penalized GLM + AFT survival regression from anofox-regression 0.5.16; Mann-Whitney/Wilcoxon expression estimate+CI (URGENT)
- Phase 16.3 inserted after Phase 16.2: Upstream API wave 2: anofox-statistics 0.4.7 + anofox-regression 0.5.17–0.5.21; 16.2 retargeted to bump to 0.4.7/0.5.21 (URGENT)

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Forecast | Evaluate exposing `anofox-forecast` capabilities (FCST-01, v2) | Deferred | 2026-08-11 |

## Session Continuity

Last session: 2026-10-09T07:19:14.461Z
Stopped at: Completed 16.1-04-PLAN.md
Resume file: None
</content>
