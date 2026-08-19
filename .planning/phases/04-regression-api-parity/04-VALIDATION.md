---
phase: 4
slug: regression-api-parity
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-12
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` (rust_api integration) + pytest (Python PyModel smoke) |
| **Config file** | Cargo.toml / pyproject.toml pytest config |
| **Quick run command** | `cargo build --features python && cargo clippy --all-targets --features python -- -D warnings` |
| **Full suite command** | `cargo test --no-default-features --test rust_api` + `maturin develop && pytest tests/` |
| **Estimated runtime** | ~3–5 min (compile-dominated; largest phase) |

---

## Sampling Rate

- **After every task commit:** `cargo build --features python` succeeds; the new PyModel/expression compiles.
- **After each new PyModel:** a smoke check — `fit` returns finite coefficients on a small known design; `predict` returns the right shape; a known-value sanity assertion where cheap.
- **Before `/gsd-verify-work`:** every new class/expression importable + callable from Python; `cargo clippy -D warnings` clean; existing suites green (no regressions).
- **Max feedback latency:** ~300 seconds (compile time; largest phase).

> Full R-validation of the new regressors is Phase 6 — this phase asserts fit/predict shape + a smoke-level correctness check only.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 4-*-* | * | * | REGR-01..06 | — | N/A (numeric models over user f64 arrays) | unit+smoke | `pytest tests/ -k <model>` / `cargo test --test rust_api` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Per-model smoke tests (pytest) for each new PyModel class (fit/predict/summary shape + finite coefficients)
- [ ] Rust smoke tests for new expression-surfaced diagnostics where added
- [ ] Existing test infra (cargo test + pytest) covers the rest

*No new framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Value correctness vs R | REGR-01..06 | Full R-validation is Phase 6 scope | Deferred to Phase 6 |

---

## Validation Sign-Off

- [ ] Each new PyModel/expression compiles and is callable from Python
- [ ] Each new model's fit returns finite coefficients on a known small design; predict shape correct
- [ ] HC extension returns HC SEs for Ridge/WLS
- [ ] `cargo clippy --all-targets -- -D warnings` clean; existing suites green
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
