---
phase: 02-public-api-audit
plan: "01"
subsystem: planning
tags: [audit, documentation, gap-analysis, anofox-statistics, anofox-regression]
status: complete

dependency_graph:
  requires:
    - 01-dep-bump (anofox-statistics 0.4.2, anofox-regression 0.5.13 installed)
    - 02-RESEARCH.md (enumeration tables — primary input)
  provides:
    - 02-API-AUDIT.md (authoritative gap list consumed by Phase 3 and Phase 4)
  affects:
    - Phase 3 planner scope (statistics parity)
    - Phase 4 planner scope (regression parity)

tech_stack:
  added: []
  patterns:
    - Per-crate markdown audit tables with one row per public fn/type
    - Known-candidate checklist pattern for deferred scope confirmation
    - Cross-reference section mapping requirements to gap rows

key_files:
  created:
    - .planning/phases/02-public-api-audit/02-API-AUDIT.md
  modified: []

decisions:
  - "ICC stub classified as exposed-but-stubbed (not a missing-binding gap) — confirmed by reading icc_fit in src/expressions/correlation.rs which returns all NaN with TODO comment"
  - "LOWESS confirmed internal-only (lowess_smooth_weights is not in solvers/mod.rs pub use list) — not a user-facing gap"
  - "FactorSummary classified HIGH priority user-facing output (returned by FittedGlmm::factors()) — not internal despite naming"
  - "Three scope questions deferred to Phase 3/4 planners: STAT-04 nD sufficiency, PassiveAggressive/MomentAccumulator PyModel-only, ICC stub as Phase 3 scope"

metrics:
  duration_minutes: 12
  completed_date: "2026-08-11"
  tasks_completed: 1
  tasks_total: 1
  commits: 1

actuals:
  tokens: 8400
  tasks: 1
  commits: 1

requirements_satisfied:
  - AUDIT-01
  - AUDIT-02
---

# Phase 02 Plan 01: Write 02-API-AUDIT.md Summary

Synthesized the RESEARCH.md enumeration tables into the authoritative `02-API-AUDIT.md` gap list covering anofox-statistics 0.4.2 (102 pub items, 13 user-facing gaps) and anofox-regression 0.5.13 (7 HIGH + 13 MEDIUM gaps), with all 21 known candidates explicitly resolved, the ICC stub and FactorSummary special cases flagged, and 3 deferred scope questions recorded for Phase 3/4 planners.

## What Was Built

`02-API-AUDIT.md` — single planning artifact at
`.planning/phases/02-public-api-audit/02-API-AUDIT.md`. Sections:

1. **Overview** — crate versions, enumeration method, summary counts for both crates
2. **anofox-statistics 0.4.2 Gap List** — 9 categories with per-item tables; internal-only subsection
3. **anofox-regression 0.5.13 Gap List** — Core Types, HC Inference, Solvers (exposed + gaps), Diagnostics, Utility; internal-only subsection
4. **Known-Candidate Checklist** — all 21 candidates with explicit verdicts
5. **Special-Case Flags** — ICC stub and FactorSummary described with Phase 3/4 action notes
6. **Deferred Scope Questions** — 3 questions recorded with recommendations, each marked "deferred to Phase 3/4 planning"
7. **Cross-Reference: Requirement to Gap** — STAT-01..05 and REGR-01..06 mapped to specific gap rows

## Decisions Made

- **ICC stub treatment:** `icc_fit` body confirmed to return all-NaN with `// TODO: Implement proper ICC with matrix input`. Classified "exposed-but-stubbed" — distinct from a missing-binding gap. Phase 3 must implement real matrix-input ICC, not just re-register the expression.
- **LOWESS verdict:** `lowess_smooth_weights` is not present in `solvers/mod.rs` pub use list. No `LowessRegressor` or `FittedLowess` type exists. Confirmed internal-only helper.
- **FactorSummary:** `pub` struct returned by `FittedGlmm::factors()`. Classified HIGH priority user output — required for any GLMM multi-factor fit to be useful.
- **Deferred Q1 (STAT-04):** Whether existing 1D `energy_distance` expression satisfies STAT-04 or nD overload is needed — deferred to Phase 3 planner with recommendation to clarify.
- **Deferred Q2 (streaming):** `PassiveAggressiveRegressor` + `MomentAccumulator` as PyModel-only — deferred to Phase 4 with recommendation (PyModel only, stateful streaming does not map to per-group expressions).
- **Deferred Q3 (ICC scope):** Whether ICC stub fix counts as Phase 3 scope — deferred with recommendation to treat as unexposed capability.

## Key Gap Findings

### anofox-statistics 0.4.2

| Gap | Priority | Phase |
|-----|----------|-------|
| `one_way_anova` + `OneWayAnovaResult` | HIGH | 3 |
| `two_way_anova` + `TwoWayAnovaResult` | HIGH | 3 |
| `repeated_measures_anova` + `RmAnovaResult` + `SphericityResult` + `CorrectedResult` | HIGH | 3 |
| `energy_distance_test` (nD overload) | HIGH | 3 |
| ICC stub fix + `ICCType` + `ICCResult` wiring | MEDIUM | 3 |
| `mmd_test` nD + `Kernel` enum | MEDIUM | 3 |

### anofox-regression 0.5.13

| Gap | Priority | Phase |
|-----|----------|-------|
| `GlmmRegressor` / `FittedGlmm` / `FactorSummary` | HIGH | 4 |
| `PSplineRegressor` / `FittedPSpline` | HIGH | 4 |
| `GammaRegressor` / `FittedGamma` | HIGH | 4 |
| `HcInference` + `compute_hc_inference` + wiring | HIGH | 4 |
| GLM dispersion/residual variants (5 fns) | MEDIUM | 4 |
| TheilSen, RANSAC, BayesianRidge, ARD, LARS, PassiveAggressive, MomentAccumulator | MEDIUM | 4 |

## Deviations from Plan

None. Plan executed exactly as written — single task, documentation-only, no source changes.

The automated verify passed on first run:

```
test -f 02-API-AUDIT.md
  && grep -q "anofox-statistics 0.4.2" ...
  && grep -q "anofox-regression 0.5.13" ...
  && grep -q "Known-Candidate Checklist" ...
  && grep -q "Deferred Scope Questions" ...
  && grep -qi "one_way_anova" ...
  && grep -qi "GlmmRegressor" ...
  && grep -qi "HcInference" ...
→ PASS
```

`git status` confirmed no changes under `src/` or `python/` — only the audit document was created.

## Commits

| Hash | Message |
|------|---------|
| `3296e5d` | `docs(02-01): write 02-API-AUDIT.md — authoritative gap list for both crates` |

## Self-Check: PASSED

- `02-API-AUDIT.md` exists at `.planning/phases/02-public-api-audit/02-API-AUDIT.md` — FOUND
- Commit `3296e5d` exists — FOUND
- No `src/` or `python/` files modified — CONFIRMED
- All 21 known candidates carry verdicts — CONFIRMED
- ICC stub flagged as exposed-but-stubbed — CONFIRMED
- FactorSummary flagged HIGH priority user output — CONFIRMED
- LOWESS flagged internal-only — CONFIRMED
- 3 deferred scope questions recorded — CONFIRMED
- STAT-01..05 and REGR-01..06 cross-referenced — CONFIRMED
