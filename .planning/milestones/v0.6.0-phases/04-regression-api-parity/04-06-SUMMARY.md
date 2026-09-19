---
plan: 04-06
phase: 04-regression-api-parity
status: complete
completed: 2026-08-12
requirements: [REGR-01, REGR-02, REGR-03, REGR-04, REGR-05, REGR-06]
key_files:
  modified:
    - tests/test_glm_diagnostics.py
completed_by: orchestrator (maturin/pytest gate stalls executor subagents; run inline)
---

# Plan 04-06 Summary — Full CI + maturin+pytest Phase Gate

## What was done

Final phase-gate plan for Phase 4. Per the plan (and the executor-stall constraint), the
`maturin develop` + `pytest` step stalls executor subagents in this sandbox, so the
autonomous orchestrator ran the definitive gate inline.

1. **Static gates:** `cargo clippy --all-targets --features python -- -D warnings` CLEAN;
   `cargo fmt --check` CLEAN; `cargo test --no-default-features --test rust_api` → 15 passed.
   ruff F-rules (real-bug category: undefined names, unused imports, redefinitions) — all
   checks passed on every new/changed Python file. (Full style-category ruff across the CI
   matrix is Phase 6 / TEST-05.)
2. **Extension rebuild:** `maturin develop` rebuilt the cdylib with all Phase 4 PyModels
   (43.7s). Note: `maturin develop` must be run in the FOREGROUND here — backgrounded maturin
   is killed by the sandbox before it can relink the large cdylib.
3. **Definitive test gate:** full `pytest tests/` → **584 passed, 0 failed** (up from 479 at
   end of Phase 3 — +105 new regression-model tests).
4. **Callable check:** all 10 new PyModel classes (`Gamma`, `GLMM`, `PSpline`, `TheilSen`,
   `RANSAC`, `BayesianRidge`, `ARD`, `LARS`, `PassiveAggressive`, `MomentAccumulator`) and all
   5 new GLM-diagnostic expressions (`gamma_dispersion_deviance`, `gamma_dispersion_pearson`,
   `gamma_pearson_chi_squared`, `gamma_standardized_pearson_residuals`,
   `gamma_standardized_deviance_residuals`) import and are callable from top-level
   `polars_statistics`.

## Fix applied during the gate

- `tests/test_glm_diagnostics.py`: the `TestRidgeHcInference._xy(p)` and
  `TestWlsHcInference._xyw(p)` helpers hardcoded a length-2 coefficient vector, so the
  `test_shape_matches_features` tests (calling `_xy(p=3)` / `_xyw(p=4)`) raised a numpy matmul
  shape mismatch. Fixed to `np.linspace(..., p)` (reproduces the exact original values at the
  default `p=2`). This was a TEST bug, not an HC-implementation bug — all other HC tests passed.

## Requirements

REGR-01 (GLMM+PSpline), REGR-02 (PSpline), REGR-03 (Gamma), REGR-04 (Ridge/WLS HC),
REGR-05 (GLM dispersion + standardized residuals), REGR-06 (TheilSen/RANSAC/BayesianRidge/
ARD/LARS/PassiveAggressive/MomentAccumulator + fit_from_accumulator) — all callable. Phase 4
goal (full regression parity) met.

## Notes / deviations

- Executor stalls recurred on the FIRST executor of large plans (04-01, 04-03) via connection
  drops, and once via orphaned `cargo`/`rustc` processes thrashing the CPU. Recovery: sweep
  orphaned build processes between waves; re-dispatch a continuation executor from the RED/partial
  state. Root cause is environmental (long compiles + sandbox watchdog / process accumulation),
  not logic errors — the implementations themselves were sound.
