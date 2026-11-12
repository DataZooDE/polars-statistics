# Requirements: polars-statistics

**Defined:** 2026-11-11
**Milestone:** v0.10.0 Upstream Wave 3
**Core Value:** Every public statistical capability in the backing `anofox-*` crates is exposed through the Polars/Python API — correctly (post-bugfix), documented, and tested — and shipped to users via PyPI.

**Research:** `.planning/research/UPSTREAM-DELTA.md` (version deltas, file:line evidence, stepping-stone versions).

**Behaviour-change policy:** when upstream changes an output to match R, adopt it. Document every changed output under "Behaviour changes" in the CHANGELOG and docs, and lock it with an R-reference test. Same policy as v0.8.0, with no deprecation warnings.

## v1 Requirements

### Upstream upgrade (UPST)

- [ ] **UPST-11**: Build against `anofox-statistics =0.10.0` with every existing test green. The posthoc pairwise-t CI test, the fixture `posthoc_reference.csv` and the docstring are updated for the 0.4.10 real-CI behaviour.
- [ ] **UPST-12**: The anofox-statistics 0.8.x behaviour changes are R-validated and documented. Users get correct SPA/MSPE-SPA p-values (the `p_value_consistent`/`p_value_upper` swap is fixed), the new MCS p-values, the Shrout–Fleiss ICC(2) CI, one-sided t-test CIs in the PyModels, joint-control `semi_partial_cor`, and the Lecoutre Huynh–Feldt ε. Each has an R reference test (MCSprocedure, psych::ICC, t.test, ppcor, afex).
- [ ] **UPST-13**: Build against `anofox-regression` 0.8.x. Required fixes: `Augment` is constructed without a struct literal, penalized-GLM `model_type` is `"glm"`, the double φ scaling of PenalizedGLM `vcov` is removed, and the LmDynamic `max_models` default is raised so fits with 7+ predictors still work. Every R-parity fixture that moved under the 0.8 numerics is re-validated against R, not just loosened. Affected: elastic-net CD, Poisson SE, IRLS tolerance, VIF/GVIF/condition labels, ALM, BLS, P-spline, lmDynamic, BayesianRidge/ARD, LARS, GLMM.
- [ ] **UPST-14**: Build against `anofox-regression =0.12.2` with clippy `-D warnings` clean. This covers the `*Regressor` renames, `with_intercept`, `IterativeFit`, `ColRef` returns, `non_exhaustive` enums/structs and the `RegressionError` type. The 0.11/0.12 semantics (∞/NaN VIF when undefined, non-zero P-spline AIC/BIC) are tested and documented.

### Polars / PyO3 stack (STACK)

- [ ] **STACK-01**: The crate builds on pyo3-polars 0.28 / polars 0.55 / pyo3 0.29 / numpy 0.29. All plugin expressions and PyModels behave identically, and the full test suite passes.
- [ ] **STACK-02**: Unused dependencies (`rand`, `rand_chacha`, `polars-arrow`, `thiserror`) are removed. `rust-version = "1.87"` is declared. faer stays at 0.23 and statrs at 0.18 to match the anofox crates. The no-default-features pyo3-absence CI guard stays green.
- [ ] **STACK-03**: `pyproject.toml` declares a tested Python `polars` range, and a CI job installs the declared minimum polars and runs the test suite against it.

### New statistics surface (STAT)

- [ ] **STAT-01**: Small statistics additions:
  - `ps.pairwise_t_test(..., conf_level=)`.
  - `ps.p_adjust(..., n=)` (`p_adjust_n`).
  - `ps.ttest_1samp` and a matching PyModel or expression, with Cohen's d.
  - SPA output gains `p_value_lower`; MCS output gains `mcs_p_values` / `elimination_order`.
  - `ps.mmd_test` supports the Gaussian median-heuristic bandwidth.
  - All R-validated where references exist.
- [ ] **STAT-02**: Bootstrap CI expressions accept `method` (iid / stationary / circular / moving block) and `block_length`. A `MovingBlockBootstrap` class matches the existing bootstrap classes.
- [ ] **STAT-03**: Two-sample / k-sample distribution tests `ps.ks_test_2sample`, `ps.ad_test_ksamp`, `ps.cvm_test_2sample` and `ps.epps_singleton_test`, each working in `group_by`/`over` and validated against R (`ks.test`, `kSamples::ad.test`, `twosamples`) or scipy.
- [ ] **STAT-04**: Time-series tests `ps.acf`, `ps.pacf`, `ps.ljung_box`, `ps.box_pierce` and `ps.durbin_watson_test` (exact p-value), validated against R (`acf`, `pacf`, `Box.test`, `lmtest::dwtest`).

### New regression surface (REG)

- [ ] **REG-01**: Inference:
  - Prediction standard errors (`predict_se`) for OLS/WLS/GLM, as both an expression option and a PyModel method.
  - Cluster-robust and HC covariance (`cluster_vcov`, `hc_vcov`).
  - Nested-model F test and ANOVA table (`nested_f_test`, `anova_table`).
  - R-validated against `predict(se.fit=TRUE)`, `sandwich::vcovCL`/`vcovHC` and `anova()`.
- [ ] **REG-02**: Diagnostics:
  - `breusch_pagan` and `white_test` (vs `lmtest::bptest`).
  - `partial_residuals` and `added_variable` (vs `car`).
  - `detect_separation`.
  - LOO/PRESS fields in `augment`.
- [ ] **REG-03**: Model kwargs and fields:
  - `weights` for Poisson/Tweedie/Gamma, and Poisson `estimate_dispersion` (quasi-Poisson).
  - Quantile `alpha` (L1), `HuberScale::Joint`, ridge `conditioning_aware`, Theil–Sen CIs.
  - GLMM `satterthwaite` / random-effects SE / `vcov`.
  - Bayesian `credible_intervals` / `predict_std`, and P-spline `lambda`.
  - All exposed on the PyModels and, where an expression exists, as expression kwargs. Stubs (`.pyi`) updated.

### Documentation & skills (DOC)

- [ ] **DOC-01**: Docs are updated for the whole milestone surface: mkdocs pages, docstrings, README and the CHANGELOG `[0.10.0]` with a "Behaviour changes" section listing every changed output from UPST-12..14. `mkdocs build --strict` is green, and `.pyi` stubs are regenerated with the drift guard green.
- [ ] **SKILL-07**: The five Claude Code skills document the new surface and behaviour changes, with every example verified against the live package (`scripts/verify_skill_examples.py` 0 FAIL). The bundled `_plugin/` copy has no drift.

### Release (REL)

- [ ] **REL-09**: The version is bumped 0.9.0 → 0.10.0 across all 18 version sources (`check_version_alignment.py --expect 0.10.0`), and the GitHub release `v0.10.0` publishes wheels for all 5 platforms plus the sdist to production PyPI via OIDC.
- [ ] **REL-10**: A post-release smoke check in a clean venv on Python 3.9 and 3.12 with the latest polars confirms that `ps.__version__ == "0.10.0"`, that an upgraded statistics output (SPA) and a regression output (OLS + `predict_se`) work, and that `group_by` still works.

## Future Requirements

Deferred to later milestones and tracked here, not in this roadmap.

### Large upstream families

- **FAM-01**: Survival: `coxph`, `survfit`/`kaplan_meier`, `survdiff`/`log_rank`, `cox_zph`, `calibrate_cph`, AFT `survival_curve`/`quantiles`.
- **FAM-02**: Ordinal / continuation-ratio / multinomial logistic / GLS models.
- **FAM-03**: rms toolkit: splines (`Rcs`, `Lsp`, `BSpline`), effects/contrasts/nomogram, `validate_*`/`calibrate_*`/`bootcov`, `areg_impute` + `pool_rubin`, `fastbw`, `pentrace`, `transcan`/`varclus`/`redun`.
- **FAM-04**: Drift module: PSI, KL, JS, Hellinger, TVD, Wasserstein, column/dataset/embedding drift.
- **FAM-05**: Classification and ranking metrics: ROC/PR/calibration curves, AUC, Brier, log-loss, classification/regression/ranking metric suites.
- **FAM-06**: PCA and monitoring: `pca`, CBPE, reconstruction drift, `threshold_band`, Isotonic/Platt calibrators.
- **FAM-07**: Regularisation paths and CV (`fit_path`, `cv_path_splits`, `nested_cv_path`, group/stratified/time-series folds) and `mediate`.
- **FAM-08**: Model persistence (pickle support via the upstream `serde` feature / `from_parts`).

### Other

- **FCST-01**: Evaluate exposing `anofox-forecast` capabilities.
- **DEP-01**: faer 0.24 / statrs 0.19, once both anofox crates move to them.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Large upstream families (FAM-01..08) | Each is a module-sized effort. v0.10.0 is kept to the upgrade plus small surface so it stays shippable. |
| pyo3 0.30 / numpy 0.30 | pyo3-polars 0.28 pins pyo3 0.29, and the build can only contain one pyo3. |
| faer 0.24 / statrs 0.19 | Both anofox crates require faer 0.23 / statrs 0.18; bumping ours would duplicate or break type passing. |
| Dropping Python 3.9 | pyo3 0.29 still supports abi3-py39, and dropping it is a separate compatibility decision. |
| FutureWarnings on behaviour-changed outputs | The user chose "adopt + document", the same policy as v0.8.0. |
| Novel algorithms not in the anofox crates | The milestone is about exposure and parity, not new statistics. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| UPST-11 | Phase 18 | Pending |
| UPST-12 | Phase 18 | Pending |
| UPST-13 | Phase 19 | Pending |
| UPST-14 | Phase 20 | Pending |
| STACK-01 | Phase 23 | Pending |
| STACK-02 | Phase 23 | Pending |
| STACK-03 | Phase 23 | Pending |
| STAT-01 | Phase 21 | Pending |
| STAT-02 | Phase 21 | Pending |
| STAT-03 | Phase 21 | Pending |
| STAT-04 | Phase 21 | Pending |
| REG-01 | Phase 22 | Pending |
| REG-02 | Phase 22 | Pending |
| REG-03 | Phase 22 | Pending |
| DOC-01 | Phase 24 | Pending |
| SKILL-07 | Phase 24 | Pending |
| REL-09 | Phase 25 | Pending |
| REL-10 | Phase 25 | Pending |

**Coverage:**
- v1 requirements: 18 total
- Mapped to phases: 18 ✓
- Unmapped: 0 ✓

---
*Requirements defined: 2026-11-11*
*Last updated: 2026-11-12 after roadmap creation (Phases 18–25)*
