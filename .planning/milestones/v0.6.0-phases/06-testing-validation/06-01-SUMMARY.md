---
phase: 06-testing-validation
plan: "01"
subsystem: tests
status: complete
tags: [testing, value-validation, scipy, anova]
dependency_graph:
  requires: []
  provides: [scipy-reference-pattern, require_scipy-fixture, require_statsmodels-fixture]
  affects: [06-02, 06-03]
tech_stack:
  added: []
  patterns: [pytest.importorskip guard, scipy reference validation, rtol/atol tolerance assertions]
key_files:
  created: []
  modified:
    - tests/conftest.py
    - tests/test_statistics_parity.py
decisions:
  - "Use pytest.importorskip in fixtures (not module-level) so missing scipy skips only the decorated tests, not the whole file"
  - "Assert F-statistic with rtol=1e-6 and p-value with atol=1e-9 — appropriate precision for floating-point ANOVA"
  - "Upgraded existing test_fisher_plausible_f_and_p in-place rather than adding a duplicate test"
metrics:
  duration: "~5 minutes"
  completed: "2026-08-12T14:14:38Z"
  tasks_completed: 1
  tasks_total: 1
  commits: 1
estimate:
  tokens: 45000
actuals:
  tokens: 8500
  tasks: 1
  commits: 1
---

# Phase 06 Plan 01: Reference-lib Skip Guard + scipy-Validated one_way_anova Summary

Established the scipy-reference value-validation pattern end-to-end: `require_scipy` and
`require_statsmodels` importorskip fixtures in conftest.py, and an upgraded `one_way_anova`
test asserting F-statistic (rtol=1e-6) and p-value (atol=1e-9) match `scipy.stats.f_oneway`.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (tracer) | Reference-lib skip guard + scipy-validated one_way_anova | 4eaba33 | tests/conftest.py, tests/test_statistics_parity.py |

## What Was Built

- `tests/conftest.py`: added `require_scipy` and `require_statsmodels` pytest fixtures using
  `pytest.importorskip`. Each fixture returns the imported module or skips the test automatically
  when the library is absent — keeping the runtime wheel dependency-light per 06-CONTEXT decision.

- `tests/test_statistics_parity.py`: upgraded `TestOneWayAnova.test_fisher_plausible_f_and_p`
  in-place to accept the `require_scipy` fixture and assert:
  - `abs(polars_statistics_F - scipy_F) / scipy_F < 1e-6` (relative tolerance on F-statistic)
  - `abs(polars_statistics_p - scipy_p) < 1e-9` (absolute tolerance on p-value)

  The original plausibility guards (F > 1, p < 0.05, n_groups == 3) are retained.

## Verification

- `python -c "import ast; ast.parse(...)"` — both files parse cleanly (Syntax OK)
- `grep -c 'f_oneway' test_statistics_parity.py` — returns 2 (reference used in assertion)
- Full maturin+pytest run deferred to phase-gate plan 06-05 per instructions

## Deviations from Plan

None — plan executed exactly as written.

## Reusable Pattern for 06-02 / 06-03

Plans 06-02 and 06-03 replicate this pattern:
1. Accept `require_scipy` (or `require_statsmodels`) as a fixture parameter
2. Call the reference library on the same input data
3. Assert field values within appropriate rtol/atol for the statistic type

## Self-Check: PASSED

- tests/conftest.py: modified, contains `require_scipy` and `require_statsmodels`
- tests/test_statistics_parity.py: modified, references `scipy_stats.f_oneway`
- Commit 4eaba33: present in git log
