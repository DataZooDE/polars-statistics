---
phase: "01"
plan: "02"
subsystem: "test"
tags: [regression-test, column-pivot, correctness-gate, dep-04]
status: complete

dependency_graph:
  requires: [anofox-regression@0.5.13, anofox-statistics@0.4.2]
  provides: [DEP-04-correctness-gate]
  affects: [tests/rust_api.rs]

tech_stack:
  added: []
  patterns:
    - Column-pivot correctness test using a non-collinear differently-scaled design (norm ratio ~360:1, Pearson r~0.24)
    - Periodic modular x2 to avoid perfect collinearity with monotone x1

key_files:
  created: []
  modified:
    - tests/rust_api.rs

decisions:
  - "Fixed plan-provided design (x2=100*x1, perfectly collinear) to x2=((i%7)+1)*100 (periodic, not proportional) — same >=100:1 norm ratio but full-rank"
  - "Design verification via numpy lstsq before Rust to confirm analytic recoverability"
  - "pytest run via .venv (pre-existing) — maturin develop rebuilt cdylib before test run"

metrics:
  duration: "~9 min"
  completed: "2026-08-11T19:27:00Z"
  tasks_completed: 2
  tasks_total: 2
  commits: 1
  files_changed: 1

actuals:
  tokens: 18000
  tasks: 2
  commits: 1
---

# Phase 01 Plan 02: Column-Pivot Correctness Tests — Summary

## One-liner

Added three `#[test]` functions to `tests/rust_api.rs` confirming the 0.5.13 column-pivot unpermute fix is active for OLS/WLS/BLS via a differently-scaled design (x1∈[0.1,2.0] / x2∈[100,700], norm ratio ~360:1, non-collinear); full Rust suite (15 tests) and pytest suite (457 tests) both pass with zero failures.

## What Was Built

This was the DEP-04 correctness expansion for Phase 1. Plan 01 bumped the crates and proved they compile; Plan 02 proves the 0.5.13 QR-pivot coefficient-unpermute fix is actually exercised by the expression layer.

### Changes Made

**`tests/rust_api.rs`** (1 file, 107 lines inserted):

- `ols_column_pivot_fix_differently_scaled`: OLS on `y = 1 + 2*x1 + 3*x2` with differently-scaled features, asserts `intercept~1.0`, `c1~2.0`, `c2~3.0` within `1e-6`.
- `wls_column_pivot_fix_differently_scaled`: Same design with unit weights (WLS → OLS), same assertions.
- `bls_column_pivot_fix_differently_scaled`: BLS (NNLS-bounded) with loose `[-10, 10]` bounds, same assertions.

**Design used:**
- `x1 = (i+1)*0.1` for `i` in `0..20` → range `[0.1, 2.0]`, norm ~5.8
- `x2 = ((i%7)+1)*100.0` → periodic values `[100, 200, 300, 400, 500, 600, 700, 100, ...]`, norm ~2084
- Column-norm ratio ~359:1, Pearson r~0.24 (not collinear, full rank)

## Verification Results

### Task 1: Three pivot tests

| Check | Result |
|-------|--------|
| `grep 'fn ols_column_pivot_fix_differently_scaled'` | PASS |
| `grep 'fn wls_column_pivot_fix_differently_scaled'` | PASS |
| `grep 'fn bls_column_pivot_fix_differently_scaled'` | PASS |
| `cargo test --no-default-features --test rust_api column_pivot_fix` | PASS — `test result: ok. 3 passed` |
| `cargo clippy --all-targets -- -D warnings` | PASS — Finished with no warnings |

### Task 2: Full suite regression

| Check | Result |
|-------|--------|
| `cargo test --no-default-features --test rust_api` | PASS — `test result: ok. 15 passed; 0 failed` |
| Three new pivot tests included in that run | PASS |
| `maturin develop` | PASS — 1m 56s, built `polars_statistics-0.5.0-cp39-abi3-linux_x86_64.whl` |
| `pytest tests/ -v` | PASS — `457 passed in 21.24s` |
| Numeric behavior changes from bump | NONE detected — all pre-existing assertions pass |

## DEP-04 Satisfaction

| Requirement | Status |
|-------------|--------|
| Three pivot tests present in `tests/rust_api.rs` | SATISFIED |
| OLS/WLS/BLS correct coefficients on differently-scaled design (>=100:1 norm ratio) | SATISFIED — returns `[1.0, 2.0, 3.0]` within `1e-6` |
| Full Rust `rust_api` suite passes (no regressions) | SATISFIED — 15/15 passed |
| Full pytest suite passes (no regressions) | SATISFIED — 457/457 passed |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed rank-deficient test design in 01-PATTERNS.md**

- **Found during:** Task 1 implementation — tests failed with `c1=NaN, c2=3.02`
- **Issue:** The design in 01-PATTERNS.md specified `x1 = i*0.5` and `x2 = i*50.0`, which gives `x2 = 100*x1` exactly (Pearson r = 1.0, perfectly collinear). A rank-deficient design matrix cannot uniquely identify both coefficients — the solver correctly returned NaN for the aliased predictor. This would have been the same with any 0.x version.
- **Root cause:** The PATTERNS.md design was intended to have differently-scaled features with a 100:1 NORM ratio, but the arithmetic progression `i*0.5` vs `i*50.0` is not just differently-scaled — it is PROPORTIONAL, making the design matrix rank-deficient.
- **Fix:** Changed to `x1 = (i+1)*0.1` and `x2 = ((i%7)+1)*100.0` — the periodic modular structure breaks the proportionality while maintaining the >=100:1 norm ratio requirement. Verified via numpy `lstsq` that the analytic solution recovers `[1.0, 2.0, 3.0]` exactly.
- **Files modified:** `tests/rust_api.rs` (all three test functions)
- **Commit:** 8a0f262

## Python Suite (DEP-04 Human-Check Outcome)

The `user_setup` Python venv was pre-existing at `.venv/` with maturin 1.13.3, pytest 9.0.3, numpy, polars, scipy, and statsmodels already installed. No venv creation step was needed.

`maturin develop` rebuilt the cdylib (1m 56s) before pytest. The pytest run exercised 457 tests across all modules against the freshly-built cdylib. **0 failures, 0 errors.**

## Requirements Satisfied

| Req ID | Description | Status |
|--------|-------------|--------|
| DEP-04 | All pre-existing tests (Rust + pytest) pass; pivot correctness confirmed | SATISFIED |

## Known Stubs

None. This plan adds test-only code. All assertions pass against real computed values.

## Threat Flags

None. Test-only additions with deterministic in-process inputs; no new runtime attack surface.

## Self-Check: PASSED

- `tests/rust_api.rs` with three pivot functions: FOUND — lines 1863, 1893, 1924
- Commit 8a0f262: FOUND — `git log --oneline -1` confirms
- `cargo test --no-default-features --test rust_api` result: `15 passed; 0 failed` — VERIFIED
- `pytest tests/ -v` result: `457 passed in 21.24s` — VERIFIED
