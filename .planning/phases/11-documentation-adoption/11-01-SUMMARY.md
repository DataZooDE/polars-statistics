# Phase 11 — Documentation & Adoption — Summary 11-01

**Status:** Complete
**Requirements:** DOCS-05 ✅, DOCS-06 ✅, DOCS-07 ✅, DOCS-08 ✅, DOCS-09 ✅, DOCS-10 ✅, DOCS-11 ✅

## What shipped

Docs + runnable examples only. **No Rust changes; no `.py` changes under
`python/polars_statistics/`** — so the CI stub-drift guard stays satisfied.

### DOCS-05 — Robust & sparse regression

- `docs/examples/robust-regression.md` — TheilSen, RANSAC, BayesianRidge, ARD, LARS
  (Lasso path), PassiveAggressive. Scenarios: response outliers, sparse features,
  online streaming. Each section explains how to read the output.
- `examples/06_robust_regression.py` — runnable, verified exit 0.

### DOCS-06 — GLMs, smoothers & streaming

- `docs/examples/glm-smoothers-streaming.md` — Gamma GLM (log link),
  GLMM (`GLMM.gaussian()` family factory, random intercepts), PSpline (GCV smoothing),
  MomentAccumulator (streaming/mergeable sufficient statistics + `Ridge.fit_from_accumulator`).
- `examples/07_glm_smoothers_streaming.py` — runnable, verified exit 0.

### DOCS-07 — ANOVA

- `docs/examples/anova.md` — one-way (Fisher + Welch), two-way (main effects +
  interaction), repeated-measures (with Mauchly sphericity + Greenhouse-Geisser /
  Huynh-Feldt corrections). Uses `ps.struct_to_dict` for field access.
- `examples/08_anova.py` — runnable, verified exit 0.

### DOCS-08 — Model-selection matrix

- `docs/model-selection.md` — comparison tables across use case / robustness /
  interpretability / speed / formula support, grouped by family (linear+regularized,
  robust+sparse, GLM, smoothers+mixed+streaming). Includes the "two API surfaces"
  (expression vs class) guidance and dimension definitions.

### DOCS-09 — Migration guide

- `docs/migration.md`:
  - **icc**: single-column/long → matrix/wide (one column per rater) side-by-side,
    with a pivot recipe and verified output fields.
  - **with_intercept → add_intercept**: deprecated in 0.7.0, still works with
    `FutureWarning`, removal not scheduled; timeline table + both-keywords-conflict note.
- Cross-linked from the icc note in `docs/api/tests/correlation.md`.

### DOCS-10 — sklearn migration

- `docs/sklearn-migration.md` — shared `fit`/`predict`/`score` contract, a
  class-to-class map (LinearRegression→OLS, Ridge, Lasso/ElasticNet, LogisticRegression,
  TheilSen, RANSAC, BayesianRidge, ARD, LARS, PassiveAggressive, Huber, Quantile, GLMs),
  worked side-by-side comparisons, naming-convention table (dropped trailing `_`,
  `alpha`→`lambda_`), and the three Polars-native advantages (per-group expressions,
  R-validated inference, typed API).

### DOCS-11 — README refresh

- Added a "Why polars-statistics?" narrative (Polars-native, sklearn-compatible,
  ergonomic results, typed, R-validated).
- Rewrote Quick Start around the two headline flows: per-group OLS as a lazy
  expression with `ps.unnest`, and the sklearn-style class API with `.to_dict()`/
  `.summary()`.
- Added the three new example scripts and three new cookbook pages to the tables;
  added a "Guides" list linking model-selection, sklearn-migration, migration.

### Navigation

- `mkdocs.yml`: new top-level **Guides** section (Model Selection, Coming from
  scikit-learn, Migration Guide); ANOVA + Robust & Sparse + GLMs/Smoothers/Streaming
  added under Examples.

## Verification

| Gate | Result |
|------|--------|
| `examples/06_robust_regression.py` | exit 0 |
| `examples/07_glm_smoothers_streaming.py` | exit 0 |
| `examples/08_anova.py` | exit 0 |
| `ruff check` (new scripts) | clean |
| `mkdocs build --strict` | pass (exit 0, no link warnings) |
| `pytest tests/ -q` | 671 passed, 5 skipped |
| Stub-drift guard | satisfied (no runtime `.py` touched) |

All documented signatures/getters/kwargs were verified against the built extension
(`GLMM.gaussian()` factory + `fixed_effects` layout, `MomentAccumulator.merge`,
`icc` wide contract + real field names, `with_intercept` FutureWarning and
both-keyword conflict).

## Notes / caveats

- The GLMM example's fitted intercept (~2.0 vs nominal 1.0) reflects the non-zero
  mean of the random intercepts under the example seed; the *slope* (3.001 vs 3.0)
  is the point being demonstrated and is exact.
- The icc reference note historically said "0.6.0" for the contract change; the
  migration page describes it version-agnostically ("when real ICC computation
  landed") to avoid overstating.
- `LARS`/`ARD`/`BayesianRidge` use a `fit_intercept` kwarg (not `add_intercept`);
  that was an intentional Phase 10 scoping decision and is documented as-is.

## CHANGELOG

Deferred to Phase 12 (which finalizes the 0.7.0 release date and version bumps). No
`CHANGELOG` entry added here.

## Deferred / follow-ups

- REL-05 (version bump) and REL-06 (PyPI publish) remain for Phase 12.
