---
phase: 06-testing-validation
plan: "04"
subsystem: tests
status: complete
tags: [testing, rust-api, anova, energy-distance, gamma-diagnostics, cargo-test]
dependency_graph:
  requires: []
  provides: [rust-anova-tests, rust-energy-distance-test, rust-gamma-diagnostic-tests]
  affects: [06-05]
tech_stack:
  added: []
  patterns: [Rust integration test, polars StructChunked field assertion, cargo test --no-default-features]
key_files:
  created: []
  modified:
    - tests/rust_api.rs
decisions:
  - "Used jittered data (multiplicative sine noise) for gamma diagnostics to ensure dispersion > 0 (exact exponential design gives dispersion ≈ 0)"
  - "Used jittered RM-ANOVA data to ensure error_ss > 0 and ws_f is finite (monotone exact design gives ws_f = inf)"
  - "Added let _ = ... pattern for field_by_name loop returns to silence clippy unused-value warning"
  - "anova_and_energy_fits tests one_way, two_way, rm_anova, and energy_nd in a single test fn"
  - "gamma_diagnostic_fits tests all 5 gamma_*_fit wrappers in a single test fn"
metrics:
  duration: "~15 minutes"
  completed: "2026-08-12T00:00:00Z"
  tasks_completed: 2
  tasks_total: 2
  commits: 1
estimate:
  tokens: 50000
actuals:
  tokens: 12000
  tasks: 2
  commits: 1
---

# Phase 06 Plan 04: Rust-side *_fit Wrapper Tests Summary

Added two new `#[test]` functions to `tests/rust_api.rs` covering the newly exposed
statistics and gamma GLM diagnostic expression `*_fit` wrappers that had zero prior
Rust-side test coverage.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Rust-side tests for ANOVA/energy *_fit wrappers | cd54079 | tests/rust_api.rs |
| 2 | Rust-side tests for 5 gamma_*_fit diagnostic wrappers | cd54079 | tests/rust_api.rs |

## What Was Built

### `#[test] fn anova_and_energy_fits`

**one_way_anova_fit** (3 groups, kind="fisher"):
  - Asserts all 10 struct fields present: statistic, df_between, df_within, p_value,
    ss_between, ss_within, ms_between, ms_within, eta_squared, n_groups.
  - Asserts `statistic.is_finite() && statistic > 0.0`.
  - Asserts `n_groups == 3`.

**two_way_anova_fit** (12-row 2×2 balanced design):
  - Asserts all 19 struct fields present (a_ss through grand_mean).
  - Asserts `n == 12`.
  - Asserts `a_f.is_finite() && a_f > 0.0` (Factor A has a real effect).

**repeated_measures_anova_fit** (4 subjects × 3 conditions, jittered):
  - Asserts all 15 struct fields present (ws_f through grand_mean).
  - Asserts `ws_f.is_finite() && ws_f > 0.0`.
  - Asserts `grand_mean.is_finite()`.
  - Note: jitter required because monotone-exact data yields `ws_f = inf`.

**energy_distance_nd_fit** (2D, X near origin, Y shifted by +3):
  - Uses input contract: [d_u32, n_perm_u32, seed_u64, x_cols..., y_cols...].
  - Asserts `statistic.is_finite() && statistic > 0.0`.
  - Asserts `p_value` field is present.

### `#[test] fn gamma_diagnostic_fits`

All five functions share the same input contract: `[y_f64, lambda_f64, with_intercept_bool, x_cols...]`.
The design uses multiplicative sine jitter so dispersion is non-trivially positive.

**gamma_dispersion_deviance_fit**: struct{dispersion} — `dispersion.is_finite() && dispersion > 0`.

**gamma_dispersion_pearson_fit**: struct{dispersion} — same assertion.

**gamma_pearson_chi_squared_fit**: struct{chi_squared, df_resid, n_observations} —
  `chi2 > 0`, `df_resid > 0`, `n_observations == 40`.

**gamma_standardized_pearson_residuals_fit**: struct{residuals (List<f64>), n_observations} —
  `len(residuals) == 40`, all values finite.

**gamma_standardized_deviance_residuals_fit**: same struct and assertions.

## Verification

- `cargo test --no-default-features --test rust_api anova_and_energy` — PASSED
- `cargo test --no-default-features --test rust_api gamma_diagnostic` — PASSED
- `cargo test --no-default-features --test rust_api` — all 17 tests PASSED
- `cargo clippy --no-default-features --tests` — no errors, no warnings

## Deviations from Plan

**1. [Rule 1 - Bug] Monotone RM-ANOVA design caused ws_f = inf**
- Found during: Task 1
- Issue: The initial test data (y = monotone ramp per subject) gave zero error_ss,
  causing ws_f = MS_conditions / 0 = inf, failing the `is_finite()` assertion.
- Fix: Replaced with jittered data (similar to the existing Python TestRmAnova fixture).
- Files: tests/rust_api.rs
- Commit: cd54079

**2. [Rule 1 - Bug] Exact-exponential Gamma design gave dispersion ≈ 0**
- Found during: Task 2
- Issue: y = exp(eta) with no noise means the model fits perfectly; deviance/Pearson
  dispersion = 0 (actually -0 floating-point), failing the `> 0.0` assertion.
- Fix: Added multiplicative sine jitter: `y_i *= (1 + 0.3 * sin(i * 1.7))`.
- Files: tests/rust_api.rs
- Commit: cd54079

## Known Stubs

None.

## Self-Check: PASSED

- tests/rust_api.rs: modified, contains `anova_and_energy_fits` and `gamma_diagnostic_fits`
- `grep -c 'one_way_anova_fit' tests/rust_api.rs` returns >= 1
- `grep -c 'gamma_standardized_deviance_residuals_fit' tests/rust_api.rs` returns >= 1
- Commit cd54079: present in git log
- All 17 cargo test --test rust_api tests pass
