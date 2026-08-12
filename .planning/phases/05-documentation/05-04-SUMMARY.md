---
phase: 05-documentation
plan: "04"
subsystem: documentation
status: complete
tags: [docs, changelog, release-notes, DOCS-04]
completed: 2026-08-12

dependency_graph:
  requires: [05-01]
  provides:
    - Complete CHANGELOG 0.6.0 section (DOCS-04 satisfied)
    - check_changelog.py: changelog completeness checker
  affects:
    - CHANGELOG.md
    - .planning/phases/05-documentation/check_changelog.py

tech_stack:
  added: []
  patterns:
    - Keep-a-Changelog format (Added / Changed / Breaking Changes / Known Limitations)
    - Simple text-search changelog checker (check_changelog.py)

key_files:
  created:
    - .planning/phases/05-documentation/check_changelog.py
  modified:
    - CHANGELOG.md

decisions:
  - Breaking Changes section added for the icc contract change (matrix-input replaces
    all-NaN stub) — this is user-visible and warranted explicit documentation.
  - [0.6.0] footer link uses HEAD (not a tag) since Phase 7 handles the actual tag/release.
  - Column-pivot correctness fix placed in Changed (not Added) since it is a behavior
    correction to an existing model, not new functionality.

metrics:
  duration_minutes: ~3
  tasks_completed: 1
  tasks_total: 1
  commits: 1

actuals:
  tokens: 9000
  tasks: 1
  commits: 1
---

# Phase 05 Plan 04: CHANGELOG 0.6.0 (DOCS-04) Summary

One-liner: CHANGELOG 0.6.0 section completed from 05-01 stub into a full Keep-a-Changelog entry covering all Phase 3+4 API, both crate bumps, the column-pivot correctness fix, and two known limitations; changelog checker created.

## What Was Built

### Task 1 — Complete CHANGELOG 0.6.0 section (DOCS-04)

The `## [0.6.0] - Unreleased` stub from plan 05-01 was expanded into a full
Keep-a-Changelog entry:

**### Added:**
- New statistics API (5 symbols): `one_way_anova`, `two_way_anova`,
  `repeated_measures_anova`, `energy_distance_nd`, and the new matrix-input `icc`.
- New regression classes (10 + methods): `Gamma`, `GLMM`, `PSpline`, `TheilSen`,
  `RANSAC`, `BayesianRidge`, `ARD`, `LARS`, `PassiveAggressive`, `MomentAccumulator`;
  `Ridge.hc_inference`, `OLS.fit_from_accumulator`, `Ridge.fit_from_accumulator`.
- 5 gamma_* GLM diagnostic expressions.

**### Changed:**
- `anofox-statistics` 0.4.1 → 0.4.2
- `anofox-regression` 0.5.4 → 0.5.13
- Column-pivot correctness fix: OLS/WLS/NNLS on differently-scaled designs now return
  correct coefficients (prior release assigned coefficients to wrong predictor).

**### Breaking Changes:**
- `icc` new matrix-input contract (old single-column all-NaN stub removed).

**### Known Limitations:**
- `WLS.hc_inference` raises `NotImplementedError`
- `icc` call sites using old single-column form must be updated

**Footer:** `[0.6.0]` compare-link added pointing to `v0.5.0...HEAD`.

**`check_changelog.py`:** Asserts all 20 required items present (0.6.0 heading + 5 statistics
symbols + 10 regression symbols + 2 crate version strings `0.4.2` and `0.5.13` + 2 limitation
keywords `NotImplementedError` and `ICC`). Exits 0.

## Verification Results

```
$ python .planning/phases/05-documentation/check_changelog.py
OK: CHANGELOG.md 0.6.0 section contains all 20 required items (5 statistics symbols,
10 regression symbols, 2 crate versions, 2 limitation keywords)
```

## Deviations from Plan

None — plan executed exactly as written. The 05-01 tracer's one_way_anova entry was
preserved and integrated into the new grouped structure.

## Known Stubs

None — CHANGELOG is prose documentation with no stubs.

## Self-Check

- [x] `CHANGELOG.md` contains `## [0.6.0]` heading
- [x] All 5 new statistics symbols present in CHANGELOG.md
- [x] All 10 new regression symbols present in CHANGELOG.md
- [x] Both crate versions `0.4.2` and `0.5.13` present
- [x] Both limitation keywords `NotImplementedError` and `ICC` present
- [x] `[0.6.0]` footer compare-link present
- [x] `check_changelog.py` exists and exits 0
- [x] Commit da68dae exists in git log
