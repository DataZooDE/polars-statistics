# Phase 9 — Result Ergonomics (Summary 09-01)

**Status:** Complete
**Branch:** `phase-9-result-ergonomics`
**Requirements delivered:** ERGO-01, ERGO-02, ERGO-03

## What shipped

### New public API

Model / test classes (all ~42 PyO3 classes):

- `__repr__(self) -> str` — e.g. `OLS(fitted=True, r_squared=0.9820, ...)` or
  `OLS(fitted=False)`. Never raises on unfitted state.
- `to_dict(self) -> dict` — every result getter materialised without manual
  extraction. Unfitted models return `{"fitted": False}`.
- `summary(self) -> str` — bespoke summaries retained where they existed
  (OLS, t-tests, Mann-Whitney, Kruskal-Wallis, Shapiro-Wilk, D'Agostino,
  Brown-Forsythe, Brunner-Munzel, Isotonic, Quantile, Yuen, Wilcoxon);
  a readable generic summary added everywhere else.

Expression (Polars Struct-column) surface:

- `polars_statistics.unnest(df, column=None, *, prefix=None, keep_others=True)
  -> pl.DataFrame` — flattens a result Struct column into flat columns in one
  call; auto-resolves a single Struct column; optional field prefixing.
- `polars_statistics.struct_to_dict(df, column=None) -> dict | list[dict]` —
  converts a result Struct column to a dict (1 row) or list of dicts (many).

### Implementation

- `src/pymodels/ergonomics.rs` — generic `repr` / `to_dict` / `generic_summary`
  built on runtime property introspection (MRO walk, `getset_descriptor`
  detection). One implementation reused by all classes; getters that raise the
  unfitted guard are treated as unavailable and skipped.
- Delegating methods added to every `src/pymodels/py_*.rs` class.
- `python/polars_statistics/results.py` — typed Python helpers.
- `scripts/gen_stubs.py` — maps `Bound<'py, pyo3::types::PyDict>` → `dict`.
- `.github/workflows/ci.yml` — `results.py` added to the mypy check.

## Unfitted-state decision

`to_dict()` returns `{"fitted": False}` (non-raising) so callers branch without
try/except; `__repr__` reports `Class(fitted=False)`; the generic `summary()`
returns a `"… (not fitted) — call .fit(...)"` message. Bespoke `summary()`
methods keep their existing "not fitted" errors (unchanged). Contextual
unfitted *errors* on result getters are Phase 10 (ERR-01) scope.

## Quality gates (local)

- `cargo fmt --all -- --check`: clean.
- `cargo clippy --all-targets --all-features -- -D warnings`: clean.
- `maturin develop`: builds; `scripts/gen_stubs.py` reproduces the committed
  `.pyi` (drift guard `tests/test_type_stubs.py`: 6 passed).
- `pytest tests/ -q`: 642 passed, 5 skipped (608 prior + 34 new).
- mypy (`--python-version 3.12 --ignore-missing-imports`, CI mirror): success.
- ruff: clean on all touched files.

## Caveats

- Stateless/config classes with no result getters (`Aid`, `StationaryBootstrap`,
  `CircularBlockBootstrap`) return `to_dict() == {}` — correct, they carry no
  results until their methods produce a result object (e.g. `Aid.classify()`
  returns an `AidResult`, which has full ergonomics).
- Pre-existing ruff findings (UP007 `Union`, etc.) in untouched files are out of
  scope and not run in CI (CI has no ruff step).
