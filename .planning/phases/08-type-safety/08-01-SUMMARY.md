# Phase 8 — Type Safety — Summary 08-01

**Status:** Complete
**Requirements:** TYPE-01 ✅, TYPE-02 ✅, TYPE-03 ✅

## What changed

- **`python/polars_statistics/py.typed`** (new): PEP 561 marker so type
  checkers treat the package as typed.
- **`python/polars_statistics/_polars_statistics.pyi`** (new, generated):
  full type stub for the compiled Rust extension — **43 classes** (OLS, Ridge,
  ElasticNet, WLS, RLS, BLS, Quantile, Isotonic, Huber, TheilSen, RANSAC,
  BayesianRidge, ARD, LARS, PassiveAggressive, MomentAccumulator, PSpline, PLS,
  Gamma, GLMM, Logistic, LogisticRegression, Poisson, NegativeBinomial,
  Tweedie, Probit, Cloglog, ALM, LmDynamic, Aid, AidResult, bootstrap classes,
  and all parametric/non-parametric/distributional test classes). Each class
  has an accurate `__init__` (from PyO3 `__text_signature__`), `@property`
  getters (numpy arrays → `numpy.ndarray`, scalars → `float`/`int`/`bool`,
  strings → `str`, dicts → `dict`), and methods (`fit` → the class itself,
  `predict` → `numpy.ndarray`, `summary` → `str`, `is_fitted` → `bool`).
- **`scripts/gen_stubs.py`** (new): regenerates the stub by walking the runtime
  module and mapping return types parsed from `src/pymodels/*.rs`. Keeps the
  stub reproducible and reviewable.
- **`tests/test_type_stubs.py`** (new): drift guard (6 tests).
- **`.github/workflows/ci.yml`**: new `type-check` job.

## How verified

- `.venv/bin/maturin develop` — clean build.
- `.venv/bin/python -m pytest tests/ -q` → **608 passed, 5 skipped**
  (602 pre-existing + 6 new stub tests).
- `.venv/bin/python -m mypy` on a public-API smoke snippet
  (`OLS(...).coefficients`, `.r_squared`, `.summary()`, `ps.ols(...)`,
  `ps.ttest_ind(...)`) → `Success: no issues found`; mypy on `exprs/`,
  `formula/`, and the `.pyi` → clean. No missing-stub/untyped errors on the
  public surface.
- `.venv/bin/ruff check` clean on new files (`.pyi` carries `# ruff: noqa`).
- `.venv/bin/maturin build` wheel contents include
  `polars_statistics/_polars_statistics.pyi` and `polars_statistics/py.typed`.
- Regenerating stubs (`python scripts/gen_stubs.py`) produces **no diff**.

## Requirement coverage

- **TYPE-01** — `py.typed` + `_polars_statistics.pyi` resolve every Rust-bound
  model/test class with accurate constructor, method, and getter signatures.
- **TYPE-02** — The 167 re-exported expression builders in `exprs/*.py` and the
  formula helpers in `formula/*.py` are fully annotated (params typed, return
  `-> pl.Expr`); the drift test asserts this and that they build real
  `pl.Expr` objects.
- **TYPE-03** — `tests/test_type_stubs.py` (run by pytest in the `test-python`
  matrix) fails if any public class/member is missing from the stub or if a
  builder loses its annotations. The new CI `type-check` job additionally
  re-runs the drift test, verifies `gen_stubs.py` reproduces the committed stub
  byte-for-byte, and mypy-checks the public API.

## Notes / caveats

- The CI `Lint` job never ran `ruff`; the repo has pre-existing ruff findings
  outside this phase's scope, so they were left untouched. New files are clean.
- The pre-existing `[tool.mypy] python_version = 3.9` emits a benign
  "3.9 not supported" note under modern mypy; CI pins `mypy==1.11.2` and passes
  `--python-version 3.10` for the smoke check. Version numbers were not touched
  (Phase 12 owns the bump).
- A handful of getters returning enum-strings or inference dicts are typed
  `Any` (e.g. `distribution`, `hc_inference`); this is correct and type-checks
  clean without over-constraining.
