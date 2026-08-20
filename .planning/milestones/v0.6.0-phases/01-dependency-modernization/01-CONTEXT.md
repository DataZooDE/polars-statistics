# Phase 1: Dependency Modernization - Context

**Gathered:** 2026-08-11
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure phase — discuss skipped)

<domain>
## Phase Boundary

Upgrade the two backing engine crates — `anofox-statistics` 0.4.1 → 0.4.2 and
`anofox-regression` 0.5.4 → 0.5.13 — in `Cargo.toml`/`Cargo.lock`, reconcile any
breaking API changes so the wrapper compiles clean under clippy `-D warnings` with
existing behavior preserved, keep the full pre-existing test suite (Rust `rust_api`
+ pytest) green, and confirm the 0.5.13 column-pivot correctness fix is active
(OLS/WLS/NNLS on differently-scaled designs return correct coefficients).

This phase is the foundation for all later work: no new API can be wrapped until
the bump lands and existing tests pass. It does NOT add or expose any new API —
that is Phases 3–4. Scope is strictly the version bump, compile reconciliation, and
regression-free validation.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure phase.
Use the ROADMAP phase goal, success criteria, and codebase conventions to guide
decisions. Specifically:
- Bump both crates to the exact target versions (statistics 0.4.2, regression 0.5.13).
- Reconcile breaking changes minimally — preserve existing wrapper behavior; do not
  refactor working surface or add new API in this phase.
- Treat DEP-04 as a correctness gate, not just a build gate (per STATE.md blocker note).
- Add a focused test confirming the column-pivot fix (OLS/WLS/NNLS correct coefficients
  on a differently-scaled design) if one does not already exist.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Full codebase map exists at `.planning/codebase/` (STACK, ARCHITECTURE, STRUCTURE,
  CONVENTIONS, TESTING, INTEGRATIONS, CONCERNS) — consult during planning.

### Established Patterns
- Version pins for the backing crates live in `Cargo.toml` (registry deps, ~lines 64/67).
- `polars-statistics` own version lives in both `Cargo.toml` and `pyproject.toml` (do NOT
  bump the wrapper version in this phase — that is Phase 7 / REL-01).
- `python` feature gates the PyO3/PyModel tree; the crate also builds as a pure rlib.
- Test suites: Rust `rust_api` tests + pytest under `tests/`.

### Integration Points
- Breaking changes from the regression jump (0.5.5–0.5.13: GLMM, P-spline, streaming
  moments, rank-detection + column-pivot fixes) may touch existing `src/pymodels/*.rs`
  and `src/expressions/*.rs` call sites — reconcile call signatures only, no behavior change.

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase. Refer to the ROADMAP phase
description, success criteria, and codebase conventions.

</specifics>

<deferred>
## Deferred Ideas

None — exposing the new API surface is deferred to Phases 3–4 (per the roadmap);
this phase is the bump + reconciliation + regression-free validation only.

</deferred>
