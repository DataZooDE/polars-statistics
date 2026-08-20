---
plan: 03-04
phase: 03-statistics-api-parity
status: complete
completed: 2026-08-12
requirements: [STAT-01, STAT-02, STAT-03, STAT-04, STAT-05]
key_files:
  modified:
    - python/polars_statistics/__init__.py
    - python/polars_statistics/exprs/__init__.py
    - tests/rust_api.rs
completed_by: orchestrator (executor stalled on maturin/pytest step; finished inline)
---

# Plan 03-04 Summary — Registration + Full Gate (Statistics API Parity)

## What was done

Final integration plan for Phase 3. The spawned executor stalled on the long-running
`maturin develop`/`pytest` step (watchdog killed it after 600s with the key work already
applied but uncommitted). The autonomous orchestrator completed the remaining verification
and committing inline.

1. **Registration** — `energy_distance_nd` added to the Python export surface in both
   `python/polars_statistics/__init__.py` and `python/polars_statistics/exprs/__init__.py`
   (imports + `__all__`, in the "Modern tests" group). Plan 03-02 deliberately deferred this
   here. The ANOVA names (`one_way_anova`, `two_way_anova`, `repeated_measures_anova`) and
   `icc` were already registered by 03-01/03-03.

2. **Rust test migration (regression fix)** — `tests/rust_api.rs::correlation_fits` still
   called `icc_fit` with the OLD contract `(values, icc_type, conf_level)` and expected the
   all-NaN placeholder. Plan 03-03 changed the ICC contract to `(n_raters, icc_type,
   rater_cols…)` and updated the Python tests but missed this Rust test. Migrated it to the
   new matrix-input contract (3 raters × 6 subjects) and assert the returned `icc` is finite.

## Verification (phase gate)

- `cargo clippy --all-targets --features python -- -D warnings` — CLEAN
- `cargo fmt --check` — CLEAN
- `ruff` — the registration change adds ZERO errors vs. the committed baseline (pre-existing
  count unchanged; the project CI ruff config tolerates them)
- `cargo test --no-default-features --test rust_api` — **15 passed, 0 failed** (was 14/1 before
  the ICC test migration)
- Full Python suite `pytest tests/` — **477 passed** (457 baseline + 20 new statistics-parity)
- Statistics-parity + correlation smoke suites — 45 passed
- All 5 new expressions importable AND callable from top-level `polars_statistics`
  (`one_way_anova`, `two_way_anova`, `repeated_measures_anova`, `energy_distance_nd`, `icc`) —
  verified by the executing smoke tests (real struct output, non-NaN).

## Requirements

STAT-01/02/03 (ANOVA family), STAT-04 (energy nD), STAT-05 (real ICC) — all callable from the
Polars/Python expression API. Phase 3 goal met.

## Notes / deviations

- Executor stall on `maturin develop`/`pytest` recurred (same pattern as the first 03-02
  attempt). Root cause appears environmental (long-running maturin/pytest under the sandbox
  watchdog), not a logic error — the work itself was sound. Orchestrator finished the gate
  directly with backgrounded test runs.
- No `maturin develop` rebuild was needed at completion time: no Rust changed between 03-03's
  successful build (45/45) and this plan's Python-only registration; the ICC Rust-test fix was
  validated by a fresh `cargo test` compile.
