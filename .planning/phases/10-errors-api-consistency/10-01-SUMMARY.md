# Phase 10 — Errors & API Consistency — Summary 10-01

**Status:** Complete
**Requirements:** ERR-01 ✅, ERR-02 ✅, API-01 ✅, API-02 ✅

## What shipped

### New shared module — `src/pymodels/errors.rs`

Centralised, contextual error constructors + validation:

- `not_fitted_err(model)` / `not_fitted_test_err(model)` → `RuntimeError`
  `"<Model> is not fitted — call \`.fit(X, y)\` first."`
- `x_y_row_mismatch_err`, `too_few_samples_err`, `empty_input_err` → `ValueError`
- `validate_xy(model, &x, &y)` — pre-solver shape/degenerate guard
- `resolve_intercept(py, add_intercept, with_intercept, default)` — constructor
  intercept resolution + `FutureWarning`

### ERR-01 — contextual not-fitted errors

- **298** generic `"Model not fitted"` / `"Test not fitted"` sites replaced with
  the named helpers → **323** helper call sites now, each naming its class.
- Exception **type unchanged** (`RuntimeError`), so `pytest.raises(RuntimeError)`
  assertions keep holding.

### ERR-02 — actionable shape / degenerate-input errors

- `validate_xy` called at the top of all **27** regressor `fit` paths
  (incl. WLS + GLMM with their extra args); Isotonic guarded inline in `score`.
- Catches: empty input, `X`/`y` row mismatch (message names both counts and the
  model), and fewer rows than columns (rank-deficient by construction) — all as
  `ValueError` before the solver can panic or emit NaN-only output.

### API-01 — `add_intercept` deprecation, consistently

- **Expressions** (`exprs/regression.py`): already centralised on
  `_resolve_intercept`; fixed the stale *"removed in v0.6.0"* wording →
  *"a future release"* (removal is NOT scheduled this milestone).
- **Model classes** (23 files): every constructor exposing `with_intercept` now
  accepts `add_intercept` (preferred) **and** `with_intercept` (deprecated, kept
  fully working), routed through `resolve_intercept`. Covers `#[new]`
  constructors and the ALM (6) + GLMM (3) static factories.
- Both kwargs given → `ValueError`; only `with_intercept` → `FutureWarning`.
- **Decision:** `fit_intercept`-style classes (BayesianRidge, ARD, LARS) were
  left unchanged — API-01 scopes only `with_intercept`; a `fit_intercept`→
  `add_intercept` rename is a separate, larger change and is deferred to keep
  this milestone additive/non-breaking.

### API-02 — uniform fit/predict/score

- `score()` added to **23** classes:
  - **Continuous regressors** (OLS, Ridge, ElasticNet, WLS, BLS, RLS, Huber,
    Quantile, PLS, TheilSen, RANSAC, LARS, PassiveAggressive, BayesianRidge,
    ARD, ALM, LmDynamic, PSpline, Isotonic) → **R²** (`1 - SS_res/SS_tot`).
  - **Binary classifiers** (Logistic, Probit, Cloglog) → **mean accuracy**
    with a `threshold=0.5` kwarg (sklearn `ClassifierMixin.score` convention).
    LogisticRegression already had accuracy `score`.
- `fit`/`predict` signatures were already uniform (X 2-D, y 1-D); Isotonic keeps
  its natural single-feature 1-D `x` in both `predict` and `score`.

### Stubs

- `_polars_statistics.pyi` regenerated via `scripts/gen_stubs.py`; drift guard +
  mypy (`--python-version 3.12`) green.

## Tests

- New file `tests/test_phase10_errors_api.py` — **29** tests:
  ERR-01 (named unfitted across 8 models + predict + score), ERR-02
  (row-mismatch / empty / too-few-samples `ValueError`), API-01 (`FutureWarning`
  + both-kwargs error on **classes AND exprs** + static factory), API-02 (R²
  correctness, perfect-fit R²=1, classifier accuracy for LogisticRegression /
  Logistic / Probit, score present across regressors).
- **Full suite: 671 passed / 5 skipped** (was 642/5; +29 new). No regressions.

## Existing tests modified

**None.** The not-fitted exception type was preserved, so no existing assertion
needed weakening or updating. Pre-existing tests that pass `with_intercept=` now
emit the (expected) `FutureWarning` but still pass.

## Quality gates (local)

- `cargo fmt --all -- --check` ✅  ·  `cargo clippy --all-targets --all-features -- -D warnings` ✅
- `maturin develop` ✅  ·  `gen_stubs.py` + stub drift guard ✅  ·  mypy 3.12 ✅
- `pytest` 671 ✅  ·  `ruff` — no new violations on touched files (regression.py
  carries the same pre-existing UP007 count as `main`).

## Caveats

- `regression.py` has pre-existing `UP007` (`Union[...]`) ruff findings unrelated
  to this phase; not touched to keep the change scoped and non-breaking.
- Perfect-separation logistic still diverges in the solver (25-iter default);
  ERR-02 flags shape/degenerate *inputs*, and the separation-diagnostic
  expressions (`check_binary_separation`) remain the recommended pre-fit guard.
