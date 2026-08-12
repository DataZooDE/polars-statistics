# Phase 4: Regression API Parity - Context

**Gathered:** 2026-08-12
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — 1 grey area, all recommendations accepted (full parity)

<domain>
## Phase Boundary

Make every unexposed `anofox-regression` 0.5.13 capability identified in the Phase 2
audit callable from Python — primarily as PyModel classes (matching the existing 40+
regression PyModel pattern), and as Polars expressions where a per-group expression is
natural. This is the milestone's largest phase and honors the locked "expose ALL
unexposed functions (full parity)" decision.

Scope = IMPLEMENTATION + exposure (PyModel classes, expressions where natural, output
schemas, Python bindings, Rust doc comments inline). Comprehensive R-validated testing
is Phase 6; user-facing docs pages are Phase 5. This phase must produce correct, callable
regressors/diagnostics with sensible output, plus at least smoke-level correctness checks.

The audit (`02-API-AUDIT.md`) is the authoritative scope list.

</domain>

<decisions>
## Implementation Decisions

### Regression Parity Scope, Surfaces & HC Framing (accepted from recommendations — full parity)

- **REGR-06 scope = ALL:** expose the full regression gap list. HIGH: `GammaRegressor` (REGR-03),
  `GlmmRegressor` + `FactorSummary` (REGR-01), `PSplineRegressor` (REGR-01/02). MEDIUM (REGR-06):
  the 6 new solvers `TheilSenRegressor`, `RANSACRegressor`, `BayesianRidge`, `ARDRegression`,
  `LARS`/`LarsRegressor`, `PassiveAggressiveRegressor`; `MomentAccumulator`; and the missing
  diagnostics (GLM dispersion estimates, standardized GLM/deviance/Pearson residuals). Use the
  exact crate names from `02-RESEARCH.md`/`02-API-AUDIT.md`.
- **Primary exposure surface = PyModel classes** (fit/predict/summary), following the existing
  `src/pymodels/*.rs` pattern (e.g. `PyOLS`, `PyElasticNet`). Add Polars expressions only where a
  per-group expression is natural (e.g. the diagnostics, which follow the existing diagnostic
  expression pattern). The audit's target-surface column guides each item.
- **REGR-04 HC framing = EXTEND:** HC output is already reachable for OLS (`ols_summary` `hc_type`
  param + `OLS.hc_inference()`). Extend HC output to the other regressors the crate's inference
  path supports (e.g. Ridge/WLS/GLM), via the existing summary/inference mechanism — not a
  from-scratch implementation. (Resolves audit Deferred Q4.)
- **PassiveAggressive & MomentAccumulator = PyModel-only** — stateful/streaming; no Polars
  expression wrapper. (Resolves audit Deferred Q2.)

### Claude's Discretion
- Per-model PyModel file layout and method set (fit/predict/summary/residuals as the crate supports),
  following the closest existing PyModel analog per model type (GLM-family → Poisson/Tweedie analogs;
  robust solvers → Huber/RLS analogs; PSpline → smoother; GLMM → its own).
- Output-struct/dict schemas — follow the crate result types and existing PyModel getter conventions;
  do not invent fields.
- Which diagnostics become expressions vs PyModel methods — follow the existing diagnostics pattern.
- Grouping of capabilities into plans/waves (the planner decides; likely grouped by model family with
  a tracer model wired end-to-end first).
- How broadly HC "extend" reaches — cover the regressors whose crate inference clearly supports HC;
  do not force HC onto models where the crate does not provide it.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `02-API-AUDIT.md` (Phase 2) — authoritative regression gap list with target surfaces + priorities.
- `02-RESEARCH.md` — crate-source enumeration with exact regressor/type/fn names and modules.
- Existing PyModel pattern: `src/pymodels/*.rs` (PyO3 classes with `#[pymethods]`, `#[getter]`,
  `fit`/`predict`; numpy↔faer via `ToFaer`/`IntoNumpy`), registered in `src/pymodels/mod.rs` and the
  `#[pymodule]` block in `src/lib.rs`, surfaced through `python/polars_statistics/__init__.py`.
- Existing diagnostics expressions (Cook's distance, VIF, leverage, DFFITS etc.) — the audit found these
  substantially complete; only GLM dispersion + standardized GLM residuals are missing.
- HC already wired for OLS: `ols_summary` `hc_type` param (`src/expressions/regression.rs`) and
  `OLS.hc_inference()` (`src/pymodels/py_ols.rs`) — the extension analog.

### Established Patterns
- PyModel: private fields, `Option<Fitted*>` state, getters, fit/predict methods, `PyRuntimeError`
  on "not fitted".
- Zero-copy numpy↔faer read-only borrows for array inputs.

### Integration Points
- New PyModel classes register in `src/pymodels/mod.rs` + the `#[pymodule]` in `src/lib.rs` +
  `python/polars_statistics/__init__.py` to be importable as `from polars_statistics import X`.

</code_context>

<specifics>
## Specific Ideas

- Use the exact crate names/result types from `02-RESEARCH.md` for every regressor and diagnostic.
- Mirror the closest existing PyModel analog per model family to stay consistent with the 40+ existing
  classes (GLM family, robust solvers, smoothers).
- HC extension reuses the existing inference/summary path rather than re-implementing HC math.

</specifics>

<deferred>
## Deferred Ideas

- Comprehensive R-validated tests for all new regressors → Phase 6 (Testing & Validation).
- mkdocs API pages + full user docs → Phase 5 (Documentation).
- Statistics-side parity → done in Phase 3.

</deferred>
