---
plan: 06-05
phase: 06-testing-validation
status: complete
completed: 2026-08-12
requirements: [TEST-01, TEST-02, TEST-03, TEST-04, TEST-05]
completed_by: orchestrator (maturin+pytest gate run inline; stalls executors)
---

# Plan 06-05 Summary — Green Gate (Testing & Validation)

Definitive local gate, run by the orchestrator (maturin+pytest stalls executor subagents).

## Gate results (2026-08-12)
- `cargo clippy --all-targets --features python -- -D warnings` — CLEAN
- `cargo fmt --check` — CLEAN (fixed + committed new rust_api test formatting)
- `cargo test --no-default-features --test rust_api` — **17 passed** (added anova/energy/gamma-diagnostic wrapper+schema tests, TEST-04)
- `maturin develop` (foreground) + `pytest tests/` — **602 passed, 5 skipped, 0 failed**
  - 5 skips are optional-reference guards (scipy/statsmodels/pyarrow/sklearn `importorskip`)
- Value-validation confirmed correct against references: one_way_anova vs scipy.f_oneway; two_way vs statsmodels; ICC(3,1) = 0.715 vs Shrout & Fleiss (1979) 0.71; GLMM fixed-effect slope recovers true 1.5; Gamma vs statsmodels; MomentAccumulator fit_from_accumulator == batch OLS/Ridge (analytic identity).

## Value-test triage (first execution of the new value tests)
5 value tests failed on first run; all resolved (see 06-05-VALUE-FIXES.md): 4 were TEST bugs (statsmodels submodule import + pyarrow guard; RM-ANOVA degenerate perfectly-additive data → used jittered design; ICC needed `icc3` not `icc2`; GLMM slope is `fe[1]` not `fe[0]`). 1 genuine LIBRARY limitation documented: `GLMM.factors()` returns `[]` after `.fit()` — recorded as a known follow-up (does not block the phase; the GLMM fit itself is correct).

## Requirements
- TEST-01 (statistics value+shape), TEST-02 (regression value+shape), TEST-03 (validated vs references) — met.
- TEST-04 (Rust-side wrapper/schema tests) — met (rust_api 17 passed).
- TEST-05 (full CI matrix) — local gates green + authoritative; the OS/Python matrix + coverage is delegated to the existing GitHub Actions CI on push (cannot run in this sandbox).

## Known limitation carried forward
- `GLMM.factors()` empty after `.fit()` (documented in test_glmm.py) — follow-up.
- WLS `hc_inference` raises NotImplementedError (Phase 4 CR-03) — follow-up when the crate exposes weighted HC.
