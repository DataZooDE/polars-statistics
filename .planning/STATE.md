---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 3
current_phase_name: Statistics API Parity
status: planning
stopped_at: Completed 02-01-PLAN.md — 02-API-AUDIT.md written; all 21 candidates resolved; ICC stub, LOWESS, FactorSummary flagged; 3 deferred questions recorded
last_updated: "2026-08-11T20:32:11.690Z"
last_activity: 2026-08-11
last_activity_desc: Roadmap created (7 phases, layered structure, 28/28 requirements mapped)
progress:
  total_phases: 2
  completed_phases: 2
  total_plans: 3
  completed_plans: 3
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-11)

**Core value:** Every public statistical capability in the backing `anofox-*` crates is exposed through the Polars/Python API — correctly, documented, tested, and shipped to PyPI.
**Current focus:** Phase 3 — Statistics API Parity

## Current Position

Phase: 3 — Statistics API Parity
Plan: Not started
Status: Ready to plan
Last activity: 2026-08-11 — Phase 2 complete, transitioned to Phase 3

Progress: [██░░░░░░░░] 29% (2/7 phases complete)

## Performance Metrics

**Velocity:**

- Total plans completed: 3
- Average duration: — min
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 2 | - | - |
| 2 | 1 | - | - |

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

### Pending Todos

None yet.

### Blockers/Concerns

- ✓ Resolved (Phase 1): dep bump landed, DEP-04 green (Rust 15 + pytest 457 passing), and the column-pivot correctness fix is confirmed active via 3 new tests. New API can now be wrapped in Phases 3–4.
- None currently open.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Forecast | Evaluate exposing `anofox-forecast` capabilities (FCST-01, v2) | Deferred | 2026-08-11 |

## Session Continuity

Last session: 2026-08-11
Stopped at: Phase 2 complete (audit doc verified passed after correcting HC-inference rows to PARTIAL/OLS); autonomous run advancing to Phase 3 (Statistics API Parity)
Resume file: None
