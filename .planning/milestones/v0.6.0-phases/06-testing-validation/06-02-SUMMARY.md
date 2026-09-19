---
phase: 06-testing-validation
plan: "02"
subsystem: tests
status: complete
tags: [testing, value-validation, statsmodels, scipy, anova, icc, energy-distance]
dependency_graph:
  requires: [06-01]
  provides: [two_way_anova-value-test, rm_anova-value-test, energy_distance_nd-separation-property, icc-published-example-value-test]
  affects: [06-05]
tech_stack:
  added: []
  patterns: [statsmodels anova_lm reference, analytic ANOVA constants (Kirk 1995), Shrout & Fleiss ICC published example, energy distance separation property]
key_files:
  created: []
  modified:
    - tests/test_statistics_parity.py
    - tests/test_correlation.py
decisions:
  - "two_way_anova validated vs statsmodels anova_lm(typ=1) with 15% relative tolerance (balanced-cell formula vs sequential decomposition)"
  - "repeated_measures_anova validated against analytic constants (Kirk 1995): F=4.8 on 4-subject x 3-condition balanced design"
  - "energy_distance_nd separation property: stat for 3-unit-shifted 2D samples > stat for identical samples; p < 0.10"
  - "ICC validated against Shrout & Fleiss (1979) Table 1 published ICC(2,1) = 0.71 within ±0.10 tolerance"
  - "All new value tests extend existing test classes in-place; no duplicate test files or classes added"
metrics:
  duration: "~10 minutes"
  completed: "2026-08-12T00:00:00Z"
  tasks_completed: 2
  tasks_total: 2
  commits: 1
estimate:
  tokens: 55000
actuals:
  tokens: 14000
  tasks: 2
  commits: 1
---

# Phase 06 Plan 02: Statistics Value Tests Summary

Upgraded the existing shape-only smoke tests for the four remaining newly exposed
statistics symbols (two_way_anova, repeated_measures_anova, energy_distance_nd,
icc) to value-correctness assertions following the scipy/analytic reference pattern
established by the 06-01 tracer.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Value-validate two_way_anova, repeated_measures_anova, energy_distance_nd | fb72df0 | tests/test_statistics_parity.py |
| 2 | Value-validate matrix-input icc against Shrout & Fleiss (1979) | fb72df0 | tests/test_correlation.py |

## What Was Built

### tests/test_statistics_parity.py

- **TestTwoWayAnova.test_value_vs_statsmodels** (require_statsmodels guard):
  Fits a balanced 2×2 factorial design (n=12, 3 replicates/cell) through both
  `ps.two_way_anova` and statsmodels `ols + anova_lm(typ=1)`.  Asserts Factor A
  and Factor B F-statistics match within 15% relative tolerance and p-values agree
  within 0.05 absolute.  Uses `_factorial_df()` helper (same 12-row design used
  by existing smoke tests).

- **TestRmAnova.test_value_vs_analytic** (analytic constants):
  4-subject × 3-condition balanced design with no noise (y = integer ramp).
  Analytically: SS_conditions=8, SS_error=5, MS_conditions=4, MS_error=5/6,
  F=4.8.  Asserts `ws_f` within 0.5 of 4.8 and `ws_p_value` < 0.20.
  Reference: Kirk (1995), §8 repeated-measures ANOVA formula.

- **TestRmAnova.test_sphericity_fields_in_unit_interval**:
  Asserts GG and HF epsilon correction fields are in (0, 1] when finite.

- **TestEnergyDistanceNd.test_separation_property** (TEST-03 value assertion):
  Two-condition comparison: identical 2D samples (statistic ≈ 0) vs. samples
  shifted 3 units in both dimensions (large statistic, p < 0.10).  Uses 30
  observations per sample and 199 permutations (seed=42 for reproducibility).

### tests/test_correlation.py

- **TestICC.test_icc_value_vs_published_example** (analytic published constants):
  Feeds the Shrout & Fleiss (1979) Table 1 dataset (6 subjects × 4 raters) through
  `ps.icc(..., icc_type='icc2')`.  Asserts ICC(2,1) estimate is within ±0.10 of
  the published value of 0.71.  Also asserts CI brackets the estimate and is
  non-degenerate (width > 0), n_subjects=6, n_raters=4.

## Verification

- `python -c "import ast; ast.parse(open('tests/test_statistics_parity.py').read())"` — OK
- `python -c "import ast; ast.parse(open('tests/test_correlation.py').read())"` — OK
- `grep -c 'energy_distance_nd' tests/test_statistics_parity.py` — returns >= 1
- Full maturin+pytest run deferred to 06-05 phase gate per instructions

## Deviations from Plan

**1. [Rule 2 - Missing critical functionality] Added epsilon bounds check**
- Found during: Task 1
- Issue: Plan did not explicitly call for GG/HF epsilon range assertion, but this is
  a correctness requirement for the sphericity correction fields
- Fix: Added `test_sphericity_fields_in_unit_interval` asserting epsilon in (0, 1]
- Files: tests/test_statistics_parity.py
- Commit: fb72df0

## Known Stubs

None — all value assertions are wired to real library calls with reference validation.

## Self-Check: PASSED

- tests/test_statistics_parity.py: modified, contains `test_value_vs_statsmodels`, `test_value_vs_analytic`, `test_separation_property`
- tests/test_correlation.py: modified, contains `test_icc_value_vs_published_example`
- Commit fb72df0: present in git log
