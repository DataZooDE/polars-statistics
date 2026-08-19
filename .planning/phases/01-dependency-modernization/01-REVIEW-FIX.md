---
phase: 01-dependency-modernization
fixed_at: 2026-08-11T00:00:00Z
review_path: .planning/phases/01-dependency-modernization/01-REVIEW.md
iteration: 1
findings_in_scope: 2
fixed: 2
skipped: 0
status: all_fixed
---

# Phase 01: Code Review Fix Report

**Fixed at:** 2026-08-11
**Source review:** `.planning/phases/01-dependency-modernization/01-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope (critical + warning): 2
- Fixed: 2
- Skipped: 0

## Fixed Issues

### WR-02: Unchecked `.unwrap()` on `get(1)` produces an uninformative panic message on failure

**Files modified:** `tests/rust_api.rs`
**Commit:** d569942
**Applied fix:** Replaced the bare `.get(1).unwrap()` calls with `.get(1).expect("expected coefficient[1] (x2) from the {OLS,WLS,BLS} pivot fit")` at all three coefficient-extraction sites (lines 1889, 1921, and 1953). The `get(0)` calls were left as `.unwrap()` — consistent with the reviewer guidance since `get(0)` on a non-empty slice cannot return `None`.

---

### WR-01: BLS pivot test may validate the wrong solver code path

**Files modified:** `tests/rust_api.rs`
**Commit:** 761ad34
**Applied fix:** Rewrote the BLS test function comment to accurately state what the solver does internally. Investigation of `/home/simonm/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/bls.rs` confirmed that `BlsRegressor::solve_passive_set` uses `col_piv_qr()` (faer QR with column pivoting) and explicitly applies the 0.5.13 unpermute fix (lines 295-303 of the upstream source, with a comment explaining the `perm.arrays().0` forward-scatter). With bounds [-10, 10] that are loose relative to the true coefficients [2.0, 3.0], the active set is empty throughout iteration, so BLS reduces to a pure unconstrained QR solve — directly exercising the same pivot fix as OLS/WLS. The original comment "BLS (NNLS-bounded) on the same positive-coefficient differently-scaled design" was replaced with a comment referencing `solve_passive_set`, `col_piv_qr`, and the active-set reduction rationale.

## Verification

All three `column_pivot_fix` integration tests were run in the isolated worktree after the WR-02 change. The WR-01 change is comment-only and was subsumed by the same build, so no separate test run was needed.

**Verification ran in:** isolated git worktree at `.claude/worktrees/rf-01-4183070-1786477284` (not the main checkout).

```
running 3 tests
test bls_column_pivot_fix_differently_scaled ... ok
test wls_column_pivot_fix_differently_scaled ... ok
test ols_column_pivot_fix_differently_scaled ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out; finished in 0.00s
```

## Skipped Issues

None — all in-scope findings were fixed.

## Out-of-scope (Info)

### IN-01: Block comment norm figure is slightly inaccurate

**File:** `tests/rust_api.rs:1858`
**Reason:** Skipped — out of scope for `fix_scope: critical_warning`. Info findings are excluded from this iteration.
**Original issue:** Block comment says "norm ~1800" but actual Euclidean norm of x2 over the 20-point design is ~1926 (~7% off).

---

_Fixed: 2026-08-11_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
