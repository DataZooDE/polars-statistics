---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 2
current_phase_name: Public API Audit
status: planning
stopped_at: Completed 01-02-PLAN.md — full Rust (15 passed) and pytest (457 passed) suites green; DEP-04 satisfied
last_updated: "2026-08-11T19:48:34.110Z"
last_activity: 2026-08-11
last_activity_desc: Roadmap created (7 phases, layered structure, 28/28 requirements mapped)
progress:
  total_phases: 1
  completed_phases: 1
  total_plans: 2
  completed_plans: 2
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-11)

**Core value:** Every public statistical capability in the backing `anofox-*` crates is exposed through the Polars/Python API — correctly, documented, tested, and shipped to PyPI.
**Current focus:** Phase 1 — Dependency Modernization

## Current Position

Phase: 2 — Public API Audit
Plan: Not started
Status: Ready to plan
Last activity: 2026-08-11 — Phase 1 complete, transitioned to Phase 2

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**

- Total plans completed: 2
- Average duration: — min
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 2 | - | - |

**Recent Trend:**

- Last 5 plans: —
- Trend: —

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 01 P01 | 10 | 1 tasks | 1 files |
| Phase 01 P02 | 9 | 2 tasks | 1 files |

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

### Pending Todos

None yet.

### Blockers/Concerns

- Phase 1 hard prerequisite: no new API can be wrapped until the dep bump lands and DEP-04 (existing tests still pass) is green. The regression bump also inherits a critical column-pivot coefficient correctness fix — treat DEP-04 as a correctness gate, not just a build gate.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Forecast | Evaluate exposing `anofox-forecast` capabilities (FCST-01, v2) | Deferred | 2026-08-11 |

## Session Continuity

Last session: 2026-08-11T19:27:47.037Z
Stopped at: Completed 01-02-PLAN.md — full Rust (15 passed) and pytest (457 passed) suites green; DEP-04 satisfied
Resume file: None
