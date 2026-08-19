---
phase: 01-dependency-modernization
reviewed: 2026-08-11T00:00:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - Cargo.toml
  - tests/rust_api.rs
findings:
  critical: 0
  warning: 2
  info: 1
  total: 3
status: issues_found
---

# Phase 01: Code Review Report

**Reviewed:** 2026-08-11
**Depth:** standard
**Files Reviewed:** 2 (diff-scoped: 2 version pins in Cargo.toml + 3 new test functions appended to tests/rust_api.rs)
**Status:** issues_found

## Summary

This phase bumped `anofox-regression` from 0.5.4 to 0.5.13 and `anofox-statistics` from 0.4.1 to
0.4.2, then added three regression tests (`ols_column_pivot_fix_differently_scaled`,
`wls_column_pivot_fix_differently_scaled`, `bls_column_pivot_fix_differently_scaled`) that are
intended to guard against the QR column-pivot bug fixed in 0.5.13.

The version pins and Cargo.lock are internally consistent. The OLS and WLS pivot tests are
well-designed: noise-free data with ground truth 1e-6 tolerances, a two-feature design that
genuinely produces a norm ratio of ~360:1 (verified: 5.36 vs. 1926), and Pearson r ~ 0.24 (not
collinear). The design facts in the comment are slightly rounded ("norm ~1800" vs. the actual
~1926) but immaterially so.

Two substantive concerns require attention: the BLS test may be asserting properties of a solver
that was never subject to the QR pivot bug; and the OLS pivot test asserts coefficient recovery to
1e-6 on a design whose y-values span 301 to 2104, which carries a latent fragility if the solver
ever switches to a less numerically exact path.

---

## Warnings

### WR-01: BLS pivot test may validate the wrong solver code path

**File:** `tests/rust_api.rs:1929-1958`

**Issue:** The test comment states the intent is to verify that BLS "also applies the 0.5.13
unpermute fix." However, Bounded Least Squares (BLS) solvers canonically use an NNLS-style
algorithm (Lawson-Hanson or active-set, operating in the primal-variable space) rather than
QR-with-pivoting. If `anofox-regression`'s `BlsRegressor` does not internally use a QR pivot,
then:

1. BLS never exhibited the scrambled-coefficient bug this test purports to guard.
2. The test does not actually exercise the 0.5.13 unpermute fix at all.
3. A future regression specifically in the BLS solver (scrambled output through a different
   mechanism) would not be caught because the test structure cannot distinguish "BLS is correct"
   from "BLS was never broken this way."

The test is not *wrong* as a regression guard (it will catch accidental swap of the two
coefficient values regardless of cause), but the stated justification in the comment is
potentially incorrect and could mislead maintainers into thinking BLS's correctness on
scale-disparate designs is contingent on the same pivot fix as OLS/WLS.

**Fix:** One of:
- If `BlsRegressor` is confirmed to use QR internally (e.g., as an unconstrained phase before
  projection), add a code comment referencing that fact. Example:
  ```rust
  // BlsRegressor uses QR with column pivoting in its unconstrained initialisation
  // phase; 0.5.13 applies the same unpermute fix as OLS/WLS.
  ```
- If `BlsRegressor` uses NNLS and was never affected by the QR pivot bug, reframe the test
  comment to describe what it actually validates:
  ```rust
  // Verifies that BLS produces correct coefficients on a scale-disparate design
  // (norm ratio ~360:1), regardless of internal algorithm.
  ```

---

### WR-02: Unchecked `.unwrap()` on `get(1)` produces an uninformative panic message on failure

**File:** `tests/rust_api.rs:1889`, `1921`, `1953`

**Issue:** All three new tests extract two coefficients via:
```rust
let c1 = coefs_inner.f64().unwrap().get(0).unwrap();
let c2 = coefs_inner.f64().unwrap().get(1).unwrap();
```

`get(1)` returns `Option<f64>`. If the solver incorrectly returns only one coefficient (a
plausible failure mode when the pivot permutation is applied to the wrong length), the `.unwrap()`
on `None` produces a bare `called Option::unwrap() on a None value` panic with no diagnostic
context — making it very hard to distinguish from a data-layout issue or a type mismatch.

The existing single-feature tests in the same file use the same pattern for `get(0)`, but `get(0)`
on a non-empty list cannot return `None`, so this risk does not apply there. The two-feature case
here is genuinely susceptible.

**Fix:** Replace the unwrap with `unwrap_or_else` or an expect with a diagnostic message at the
three coefficient-extraction sites. Minimal change:
```rust
let c1 = coefs_inner.f64().unwrap()
    .get(0)
    .expect("expected coefficient[0] (x1) in pivot OLS result");
let c2 = coefs_inner.f64().unwrap()
    .get(1)
    .expect("expected coefficient[1] (x2) in pivot OLS result — may indicate wrong coefficient count");
```
Apply the same pattern to the WLS (`1921`) and BLS (`1953`) variants.

---

## Info

### IN-01: Block comment norm figure is slightly inaccurate

**File:** `tests/rust_api.rs:1858`

**Issue:** The section header comment reads:

> `x2 = ((i%7)+1)*100 (range 100..700, periodic, norm ~1800)`

The actual Euclidean norm of `x2` over the 20-point design is approximately **1926**, not 1800
(verified arithmetically: `sqrt(sum(((i%7+1)*100)^2, i=0..19)) ≈ 1926`). Similarly the norm
ratio is described as "~360:1" in the comment and the assertion sub-comment at line 1869 confirms
this figure, which is accurate (5.36 vs. 1926 ≈ 359:1). The "norm ~1800" figure alone is
misleading by ~7%.

**Fix:** Correct the comment:
```rust
// Design: x1 = (i+1)*0.1 (range 0.1..2.0, norm ~5), x2 = ((i%7)+1)*100 (range
// 100..700, periodic, norm ~1926). Norm ratio ~360:1, Pearson r ~0.24 (not
// collinear). Ground truth y = 1 + 2*x1 + 3*x2 recoverable uniquely.
```

---

_Reviewed: 2026-08-11_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
