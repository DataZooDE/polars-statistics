---
phase: "01"
plan: "01"
subsystem: "build"
tags: [dependency-bump, cargo, rust, tracer]
status: complete

dependency_graph:
  requires: []
  provides: [anofox-statistics@0.4.2, anofox-regression@0.5.13]
  affects: [Cargo.toml, Cargo.lock]

tech_stack:
  added: []
  patterns:
    - Scoped `cargo update -p <pkg>` for targeted lock file regeneration

key_files:
  created: []
  modified:
    - Cargo.toml

decisions:
  - "Scoped update (-p flags) to prevent polars/pyo3 version churn"
  - "No wrapper source reconciliation needed — all imported symbols present and unchanged in bumped versions"
  - "Cargo.lock is gitignored (library crate); only Cargo.toml committed"

metrics:
  duration: "~10 min"
  completed: "2026-08-11T19:14:39Z"
  tasks_completed: 1
  tasks_total: 1
  commits: 1
  files_changed: 1

actuals:
  tokens: 1200
  tasks: 1
  commits: 1
---

# Phase 01 Plan 01: Dependency Version Bump — Summary

## One-liner

Bumped `anofox-statistics` 0.4.1→0.4.2 (survival-function p-value fix) and `anofox-regression` 0.5.4→0.5.13 (column-pivot coefficient-unpermute correctness fix + GLMM/PSpline/streaming-moments additions) with zero wrapper source changes; `cargo build --features python` and `cargo clippy --all-targets -- -D warnings` both pass clean.

## What Was Built

This was the tracer slice for Phase 1: the thinnest path that touches every layer the phase modifies — two version string edits in `Cargo.toml`, a scoped `cargo update`, and a full wrapper compile proving no call-site breakage.

### Changes Made

**`Cargo.toml`** (1 file, 2 lines):
- Line 64: `anofox-regression = "0.5.4"` → `"0.5.13"` (DEP-02)
- Line 67: `anofox-statistics = "0.4.1"` → `"0.4.2"` (DEP-01)

**`Cargo.lock`** (gitignored — regenerated but not committed):
- `anofox-regression` resolved to `0.5.13` (checksum `bd8869f0...`)
- `anofox-statistics` resolved to `0.4.2` (checksum `76af9b3e...`)
- Scoped update: exactly 2 packages changed (`cargo update` output: `Locking 2 packages to latest compatible versions`)

## Verification Results

| Check | Result |
|-------|--------|
| `grep 'anofox-regression = "0.5.13"' Cargo.toml` | PASS |
| `grep 'anofox-statistics = "0.4.2"' Cargo.toml` | PASS |
| Cargo.lock `version = "0.5.13"` under `anofox-regression` | PASS |
| Cargo.lock `version = "0.4.2"` under `anofox-statistics` | PASS |
| Scoped update (only 2 packages changed) | PASS — output explicitly states `Locking 2 packages` |
| `cargo build --features python` | PASS — 1m 37s clean compile |
| `cargo clippy --all-targets -- -D warnings` | PASS — `Finished` with no warnings |

## DEP-03 Reconciliation

**Expected: none. Actual: none.** The RESEARCH.md call-site inventory was accurate:
- All `anofox_regression` imports in `src/expressions/regression.rs` (diagnostics, solvers, types) are present unchanged in 0.5.13
- All `anofox_statistics` imports across `src/expressions/` are present unchanged in 0.4.2
- The `converged: bool` field added to `FittedBinomial/FittedPoisson/FittedTweedie/FittedNegativeBinomial` in 0.5.12 is backward-compatible (wrapper never uses struct-literal construction; `regressor.fit()` returns completed instances)
- Zero wrapper source files were modified

## Requirements Satisfied

| Req ID | Description | Status |
|--------|-------------|--------|
| DEP-01 | `anofox-statistics` bumped 0.4.1→0.4.2; workspace builds with `python` feature | SATISFIED |
| DEP-02 | `anofox-regression` bumped 0.5.4→0.5.13; workspace builds | SATISFIED |
| DEP-03 | Wrapper compiles clean under clippy `-D warnings`; existing behavior preserved | SATISFIED |

DEP-04 (pivot-correctness test and full pytest suite) is out of scope for Plan 01; it is addressed in Plan 02.

## Deviations from Plan

None — plan executed exactly as written. The single edit was applied in one operation, the scoped update produced exactly 2 package changes as expected, and no wrapper source reconciliation was needed as predicted by RESEARCH.md.

## Commits

| Hash | Message |
|------|---------|
| e2718ff | feat(01-01): bump anofox-statistics 0.4.1→0.4.2, anofox-regression 0.5.4→0.5.13 |

## Known Stubs

None. This plan edits only version strings and produces no code symbols, UI, or data-wiring.

## Threat Flags

None. This plan introduces no new network endpoints, auth paths, file access patterns, or schema changes. It bumps two trusted first-party crate versions already in the local registry cache.

## Self-Check: PASSED

- Cargo.toml version pins: FOUND — `anofox-regression = "0.5.13"` and `anofox-statistics = "0.4.2"` confirmed
- Cargo.lock resolved versions: FOUND — `version = "0.5.13"` and `version = "0.4.2"` under respective crate names
- Commit e2718ff: FOUND — `git log --oneline -1` confirms commit exists on main
