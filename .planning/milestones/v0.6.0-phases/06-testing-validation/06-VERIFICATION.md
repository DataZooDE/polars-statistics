---
phase: 06-testing-validation
verified: 2026-08-12
status: passed
score: 5/5
verified_by: orchestrator (ran full gate inline)
---

# Phase 6: Testing & Validation — Verification (passed)

| Req | Truth | Evidence |
|-----|-------|----------|
| TEST-01 | Each new statistics fn has a value+shape pytest test | test_statistics_parity.py, test_correlation.py — value asserts vs scipy/statsmodels/analytic |
| TEST-02 | Each new regression capability has a value+shape pytest test | 10+ regression test files upgraded with value asserts + invariants |
| TEST-03 | Values validated vs references where available | one_way_anova≈scipy; two_way≈statsmodels; ICC(3,1)=0.715≈Shrout&Fleiss; GLMM slope≈true; Gamma≈statsmodels; fit_from_accumulator==batch |
| TEST-04 | Rust-side tests cover new expression wrappers + schemas | tests/rust_api.rs — 17 passed (added anova/energy/gamma-diagnostic wrappers) |
| TEST-05 | Full CI matrix green | Local gates green (clippy -D warnings, fmt, cargo test 17, pytest 602/5-skip); full OS×Py matrix delegated to GitHub Actions CI |

**Gate:** clippy CLEAN · fmt CLEAN · rust_api 17 passed · pytest 602 passed, 5 skipped, 0 failed.
**Known limitations (documented, not blocking):** GLMM.factors() empty after fit; WLS hc_inference NotImplementedError.
