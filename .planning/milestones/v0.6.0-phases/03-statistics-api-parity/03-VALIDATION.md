---
phase: 3
slug: statistics-api-parity
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-11
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` (rust_api integration) + pytest (Python smoke) |
| **Config file** | Cargo.toml `[[test]]` / pyproject.toml pytest config |
| **Quick run command** | `cargo build --features python && cargo clippy --all-targets -- -D warnings` |
| **Full suite command** | `cargo test --no-default-features --test rust_api` + `maturin develop && pytest tests/` |
| **Estimated runtime** | ~2–3 min (compile-dominated) |

---

## Sampling Rate

- **After every task commit:** `cargo build --features python` must succeed; new expression compiles.
- **After each new expression:** a smoke check — output struct has the expected fields; a known-value sanity assertion (shape + a hand-computed/analytic expected value).
- **Before `/gsd-verify-work`:** all new expressions callable from Python; `cargo clippy -D warnings` clean; existing suites still green (no regressions).
- **Max feedback latency:** ~180 seconds (compile time).

> Full R-validation of ANOVA/energy/ICC values is Phase 6 — this phase asserts shape + a smoke-level correctness check only.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 3-01-* | 01 | 1 | STAT-01/02/03 | — | N/A | unit+smoke | `cargo test --test rust_api anova` | ❌ W0 | ⬜ pending |
| 3-02-* | 02 | 1 | STAT-04 | — | N/A | unit+smoke | `cargo test --test rust_api energy` | ❌ W0 | ⬜ pending |
| 3-03-* | 03 | 1 | STAT-05 (ICC) | — | N/A | unit+smoke | `cargo test --test rust_api icc` | ❌ W0 | ⬜ pending |
| 3-04-* | 04 | 2 | STAT-01..05 | — | N/A | integration | `maturin develop && pytest` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] New smoke tests added to `tests/rust_api.rs` and/or `tests/` pytest for each new expression
- [ ] Existing test infrastructure (cargo test + pytest) covers the rest

*No new framework install needed — existing Rust + pytest infra covers this phase.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Value correctness vs R | STAT-01..05 | Full R-validation is Phase 6 scope | Deferred to Phase 6 |

---

## Validation Sign-Off

- [ ] Each new expression compiles and is callable from Python
- [ ] Each new expression's output struct schema matches the crate result type
- [ ] Smoke-level known-value/shape check passes for each capability
- [ ] `cargo clippy --all-targets -- -D warnings` clean; existing suites green
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
