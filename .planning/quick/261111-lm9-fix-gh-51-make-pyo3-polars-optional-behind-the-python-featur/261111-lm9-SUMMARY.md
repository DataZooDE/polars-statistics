---
phase: quick-261111-lm9
plan: 01
subsystem: build-config
tags: [rust, cargo, feature-flags, pyo3, ci]
status: complete
dependency-graph:
  requires: []
  provides:
    - "pyo3-polars optional behind the `python` feature"
    - "no-default-features rlib build with zero pyo3 crates"
    - "CI no-default-features clippy gate + pyo3-absence guard"
  affects:
    - "Rust consumers depending on polars-statistics with default-features = false"
tech-stack:
  added: []
  patterns:
    - "cfg-gate `#[polars_expr]` shims and shim-only output-dtype helpers behind `feature = \"python\"` while keeping `pub` rlib API ungated"
key-files:
  created: []
  modified:
    - Cargo.toml
    - src/expressions/aft.rs
    - src/expressions/categorical.rs
    - src/expressions/correlation.rs
    - src/expressions/distributional.rs
    - src/expressions/forecast.rs
    - src/expressions/modern.rs
    - src/expressions/nonparametric.rs
    - src/expressions/parametric.rs
    - src/expressions/penalized_glm.rs
    - src/expressions/posthoc.rs
    - src/expressions/regression.rs
    - src/expressions/resampling.rs
    - src/expressions/shrinkage.rs
    - src/expressions/tost.rs
    - .github/workflows/ci.yml
    - README.md
    - docs/index.md
    - src/lib.rs
    - CHANGELOG.md
decisions:
  - "Gated the second-pass fallout (aft.rs/penalized_glm.rs importing `predict_output_dtype_with_prefix`/`PredictKwargs` from regression.rs) by splitting each `use` into an ungated statement for still-used names and a `#[cfg(feature = \"python\")]`-gated statement for the shim-only names, matching the plan's prescribed pattern exactly."
  - "Left `cargo fmt` free to reorder the new cfg-gated `use` statements alphabetically ahead of the ungated sibling statement (e.g. aft.rs) — semantically identical, no manual re-ordering needed."
metrics:
  duration: ~35min
  completed: 2026-11-11
actuals:
  tokens: 22744
  tasks: 2
  commits: 2
  plan_head_before: e026384
  plan_head_after: d4a5e8a
---

# Quick 261111-lm9: Fix GH #51 — make pyo3-polars optional behind the `python` feature Summary

Made `pyo3-polars` an optional dependency gated by the existing `python` feature, cfg-gated every `#[polars_expr]` plugin shim and its shim-only output-dtype helpers across 14 `src/expressions/*.rs` files so the `default-features = false` rlib build is warning-free, added a CI clippy gate plus a direct pyo3-absence dependency-tree guard, and corrected the Rust dependency instructions in README.md, docs/index.md and the `src/lib.rs` module doc to show a git-tag dependency (the crate is not on crates.io).

## What Was Built

**Task 1 (commit `f08e2b6`):** `Cargo.toml`'s `pyo3-polars` dependency is now `optional = true` and only pulled in by `python = ["dep:pyo3", "dep:numpy", "dep:pyo3-polars"]`. Every `use pyo3_polars::derive::polars_expr;` and every `#[polars_expr(...)]` attribute line in the 14 expression modules got a `#[cfg(feature = "python")]` line directly above it (222 insertions total — 14 `use` gates + 158 shim gates + 41 shim-only-helper gates + 9 split-import statements). The 41 clippy-reported shim-only warnings (26 never-used `*_output_dtype` fns in `regression.rs` alone, plus python-only imports in 9 other files) were resolved by gating the fn or splitting its `use` statement; `pub`/`pub(crate)` items that are public rlib API were left untouched. Two second-pass import fallouts (aft.rs and penalized_glm.rs both importing `predict_output_dtype_with_prefix`/`PredictKwargs` from `regression.rs`, which only became unused once `predict_output_dtype_with_prefix` itself was gated) were resolved the same way.

**Task 2 (commit `d4a5e8a`):** Added two new CI lint-job steps right after the existing `Clippy` step — `Clippy (rlib, no-default-features)` running `cargo clippy --no-default-features --lib -- -D warnings`, and `Assert pyo3 absent from rlib dependency tree` which counts `pyo3` matches in `cargo tree --no-default-features -e normal` and fails with a `::error::` annotation if non-zero (a clippy-only gate can't catch pyo3-polars being made non-optional again, since the gated shim code would just compile out — this is a direct guard on the dependency tree). Replaced the stale crates.io version-spec dependency line in README.md, docs/index.md and the `src/lib.rs` module doc with `polars-statistics = { git = "https://github.com/DataZooDE/polars-statistics", tag = "v0.9.0", default-features = false }` (dotted-table form in `src/lib.rs`'s doc comment), each accompanied by the verbatim phrase "not published on crates.io". Added a `### Fixed` section under `CHANGELOG.md [Unreleased]` with two bullets referencing `#51`.

## Verification Results

All of the plan's `<automated>` verify chains and the overall `<verification>` section passed:

- `cargo tree --no-default-features -e normal | grep -c pyo3` → **0** (was 13 at baseline `e026384`)
- `cargo clippy --no-default-features --lib -- -D warnings` → clean, 0 warnings
- `cargo clippy --all-targets -- -D warnings` → clean (default `python`-feature build unaffected)
- `cargo fmt --all -- --check` → clean
- Ungated `#[polars_expr]` lines: 0 (158 total shims, all gated); ungated `use pyo3_polars` lines: 0 (14 total, all gated)
- `grep -rn 'allow(dead_code)\|allow(unused' src | wc -l` → 15 (baseline unchanged, no lint-allow added)
- `cargo test --no-default-features --test rust_api` → 22 passed, 0 failed
- `cargo run --example rust_wls --no-default-features` → prints coefficients (intercept=1.0/2.0, coefs=[2.0]/[3.0], r2=1.0000 for both sites)
- `uv run --no-project --with pyyaml python -c "import yaml; yaml.safe_load(...)"` → `.github/workflows/ci.yml` parses as valid YAML
- `uvx --with mkdocs-material mkdocs build --strict` → exit 0, no errors
- All Task 2 grep/line-count assertions (CI step counts, exact dependency-line match in README/docs/lib.rs, 0 remaining stale-spec lines, `#51` under `[Unreleased]`) → pass
- Python no-regression: `uv sync --extra dev` (maturin/pytest were missing from the venv) → `uv run maturin develop --release` (7m06s fat-LTO build, exit 0) → `uv run pytest tests/ -q` → **1811 passed, 6 skipped, 0 failed** (9 pre-existing `FutureWarning` deprecation notices, unrelated to this change)
- `git diff --stat` touches exactly the 20 files in `files_modified` — no `Cargo.lock`, `pyproject.toml`, or version-string changes

## Deviations from Plan

None - plan executed exactly as written. The plan's own documentation of the expected 41-warning first pass and the two-file second pass (aft.rs/penalized_glm.rs `PredictKwargs`/`predict_output_dtype_with_prefix`) matched exactly what `cargo clippy` reported; no additional fixes were needed beyond what the plan specified.

## Known Stubs

None.

## Threat Flags

None — this is a build-configuration and documentation change with no new runtime input handling, matching the plan's own threat-model assessment (all dispositions `mitigate`/`accept`, no new network/auth/file-access surface).

## Self-Check: PASSED

All 19 `files_modified` entries verified present on disk (`FOUND`), and both task commits (`f08e2b6`, `d4a5e8a`) verified as ancestors of `HEAD`.
