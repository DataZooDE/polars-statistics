---
phase: 05-documentation
plan: "03"
subsystem: documentation
status: complete
tags: [docs, mkdocs, api-reference, regression, statistics, diagnostics]
completed: 2026-08-12

dependency_graph:
  requires: [05-01]
  provides:
    - All Phase 3+4 symbols documented on mkdocs API reference pages (DOCS-02)
    - check_docs_symbols.py: 19-symbol presence checker for all Phase 5 expansion plans
  affects:
    - docs/api/tests/parametric.md
    - docs/api/tests/correlation.md
    - docs/api/tests/forecast.md
    - docs/api/regression/diagnostics.md
    - docs/api/classes/linear.md
    - docs/api/classes/glm.md
    - docs/api/README.md
    - docs/API_REFERENCE.md
    - .planning/phases/05-documentation/check_docs_symbols.py

tech_stack:
  added: []
  patterns:
    - Symbol-presence grep checker (check_docs_symbols.py)
    - mkdocs structural nav-reference fallback (check_docs_build.py, existing)

key_files:
  created:
    - .planning/phases/05-documentation/check_docs_symbols.py
  modified:
    - docs/api/tests/parametric.md
    - docs/api/tests/correlation.md
    - docs/api/tests/forecast.md
    - docs/api/regression/diagnostics.md
    - docs/api/classes/linear.md
    - docs/api/classes/glm.md
    - docs/api/README.md
    - docs/API_REFERENCE.md

decisions:
  - energy_distance_nd placed in forecast.md Modern Distribution Tests section (alongside
    existing energy_distance) — no new page needed, consistent with plan do-not-restructure decision.
  - icc updated to new matrix-input contract in correlation.md with a Breaking Change note
    (0.6.0) explaining the old all-NaN stub is replaced by real implementation.
  - WLS.hc_inference NotImplementedError limitation added as callout in classes/linear.md
    WLS section.
  - Docs build used structural fallback (mkdocs not available in environment) — same path
    as 05-01 tracer; check_docs_build.py exits 0 with 39 nav-referenced files all present.

metrics:
  duration_minutes: ~5
  tasks_completed: 2
  tasks_total: 2
  commits: 2

actuals:
  tokens: 28000
  tasks: 2
  commits: 2
---

# Phase 05 Plan 03: mkdocs API Pages (DOCS-02) Summary

One-liner: All Phase 3+4 newly-exposed API documented on mkdocs reference pages — 4 statistics expressions, 10 regression PyModels, 5 gamma diagnostics, HC/accumulator additions; symbol checker script created; structural nav check clean.

## What Was Built

### Task 1 — New statistics expressions and gamma diagnostics (DOCS-02)

| Symbol | Page | Notes |
|--------|------|-------|
| `two_way_anova` | `docs/api/tests/parametric.md` | Added after `one_way_anova`; same section format |
| `repeated_measures_anova` | `docs/api/tests/parametric.md` | With Mauchly's + ε corrections noted |
| `energy_distance_nd` | `docs/api/tests/forecast.md` | Added to Modern Distribution Tests subsection |
| `icc` | `docs/api/tests/correlation.md` | Updated to new matrix-input contract + breaking change note |
| `gamma_dispersion_deviance` | `docs/api/regression/diagnostics.md` | New Gamma GLM Diagnostics section |
| `gamma_dispersion_pearson` | `docs/api/regression/diagnostics.md` | |
| `gamma_pearson_chi_squared` | `docs/api/regression/diagnostics.md` | |
| `gamma_standardized_pearson_residuals` | `docs/api/regression/diagnostics.md` | |
| `gamma_standardized_deviance_residuals` | `docs/api/regression/diagnostics.md` | |

### Task 2 — New regression PyModels + HC/accumulator + checker

**Linear model classes added (`docs/api/classes/linear.md`):**

| Class | Description |
|-------|-------------|
| `MomentAccumulator` | Online sufficient-statistic accumulator for OLS/Ridge |
| `TheilSen` | Median-of-slopes robust estimator |
| `RANSAC` | Random Sample Consensus with inlier mask |
| `BayesianRidge` | Bayesian Ridge with evidence maximisation |
| `ARD` | Automatic Relevance Determination with per-feature precision |
| `LARS` | Least Angle Regression with full path |
| `PassiveAggressive` | Online PA regression (PA/PA-I/PA-II) |

**Additions to existing classes:**
- `OLS.fit_from_accumulator` — fit from MomentAccumulator
- `Ridge.hc_inference` — HC0–HC3 robust inference + `fit_from_accumulator`
- `WLS` — NotImplementedError limitation callout for `hc_inference`

**GLM model classes added (`docs/api/classes/glm.md`):**
- `Gamma` — Gamma GLM (log link); dispersion property
- `GLMM` — Generalized Linear Mixed Model; random effects
- `PSpline` — Penalized spline with B-spline basis

**Index updates:** `docs/api/README.md` and `docs/API_REFERENCE.md` updated to list all new symbols.

**`check_docs_symbols.py`:** Checks all 19 new symbols (4 statistics + 10 PyModels + 5 gamma_*) are present in docs/. Exits 0 (all found).

## Verification Results

```
$ python .planning/phases/05-documentation/check_docs_symbols.py
OK: all 19 required symbols found under docs/

$ python .planning/phases/05-documentation/check_docs_build.py
INFO: mkdocs not importable in this environment — using structural fallback check
Structural check PASSED (fallback): all 39 nav-referenced files exist under docs/

CLASS_DOCS_OK
```

## Docs Build Path

**Fallback (structural nav-reference check) used** — same as 05-01 tracer. mkdocs not
installed; structural check confirms all 39 nav-referenced markdown files exist on disk.
No new nav entries were added (plan decision: do not restructure; all new symbols added
to existing pages only).

## Deviations from Plan

None — plan executed exactly as written. `energy_distance_nd` went to `forecast.md`
(confirmed by reading existing `energy_distance` location there). No new mkdocs nav
entries required.

## Known Stubs

None — all documented symbols are real implementations from Phase 3/4, not stubs.

## Self-Check

- [x] `docs/api/tests/parametric.md` contains `two_way_anova` and `repeated_measures_anova`
- [x] `docs/api/tests/forecast.md` contains `energy_distance_nd`
- [x] `docs/api/tests/correlation.md` contains updated `icc` with matrix-input contract
- [x] `docs/api/regression/diagnostics.md` contains all 5 `gamma_*` diagnostics
- [x] `docs/api/classes/linear.md` contains all 7 new linear model classes + HC/accumulator methods
- [x] `docs/api/classes/glm.md` contains `Gamma`, `GLMM`, `PSpline`
- [x] `check_docs_symbols.py` exists and exits 0 (all 19 symbols found)
- [x] `check_docs_build.py` exits 0 (structural fallback: 39 files present)
- [x] Commits dcc394c and e1a7da0 exist in git log
