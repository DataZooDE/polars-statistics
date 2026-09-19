---
phase: 6
slug: testing-validation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-12
---

# Phase 6 — Validation Strategy

> This phase IS the validation phase. Its own gate = the full local test suite green.

## Test Infrastructure
| Property | Value |
|----------|-------|
| Framework | pytest (value checks vs scipy/statsmodels + analytic) + cargo test rust_api |
| Quick run | `cargo clippy --all-targets --features python -- -D warnings` |
| Full suite | `maturin develop` (foreground) + `pytest tests/` + `cargo test --no-default-features --test rust_api` |

## Sampling Rate
- After each test-plan: the new value tests pass; suite count increases; no regressions.
- Before verify: clippy/fmt/ruff clean; cargo test green; pytest green (all new value tests included).

## Wave 0 Requirements
- Add scipy + statsmodels to pyproject test extras; guard with pytest.importorskip.

## Validation Sign-Off
- [ ] Each new statistics fn has a value+shape pytest test (TEST-01)
- [ ] Each new regression capability has a value+shape pytest test (TEST-02)
- [ ] Values validated vs scipy/statsmodels/analytic references (TEST-03)
- [ ] Rust-side tests cover new expression wrappers + output schemas (TEST-04)
- [ ] Local CI gates green; full matrix delegated to GH Actions (TEST-05)
- [ ] nyquist_compliant: true
