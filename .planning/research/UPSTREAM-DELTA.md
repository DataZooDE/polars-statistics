# Upstream Delta Research: v0.10.0 Upstream Wave 3

**Researched:** 2026-11-11
**Method:** Diffed the crate sources in `~/.cargo/registry` (anofox-statistics 0.4.7 vs 0.10.0, anofox-regression 0.5.22 vs 0.12.2) and read the upstream CHANGELOGs. Downloaded the pyo3-polars 0.26/0.27/0.28 `.crate` files from crates.io to read their dependency pins. Smoke-tested the polars-statistics 0.9.0 wheel against Python polars 1.34.0, 1.39.3 and 2.1.0. Nothing was compiled against the new versions, so the API-break findings come from reading source.

## Versions

| Crate | Pinned | Latest | Notes |
|---|---|---|---|
| anofox-statistics | =0.4.7 | 0.10.0 | Requires statrs 0.18, rand 0.8, rand_chacha 0.3, thiserror 2. No faer, polars or anofox-regression. |
| anofox-regression | =0.5.22 | 0.12.2 | Requires faer 0.23, statrs 0.18, thiserror 2, and basin 3.1 (replaces argmin). MSRV 1.87. No rand, polars or anofox-statistics. |
| pyo3-polars | 0.25 | 0.28 | 0.26 → polars 0.53 / pyo3 0.27. 0.27 → polars 0.54.4 / pyo3 0.28. 0.28 → polars 0.55.1 / pyo3 0.29. |
| pyo3 / numpy | 0.26 | 0.30 | **Capped at 0.29** by pyo3-polars 0.28. The numpy crate tracks pyo3 version-for-version. pyo3 0.29/0.30 still supports abi3-py39. |
| faer | 0.23.2 | 0.24.4 | **Keep 0.23.** anofox-regression requires 0.23 and our `Mat`/`Col` values are passed straight into it. |
| statrs | 0.18 | 0.19.1 | **Keep 0.18.** Both anofox crates require 0.18. Only one direct use (`aft.rs:24`). |
| rand, rand_chacha, polars-arrow, thiserror | — | — | **No direct use in `src/`.** Remove from Cargo.toml. |
| Python polars | `>=1.0.0` (no cap) | 2.1.0 | The 0.9.0 wheel works on 1.34 / 1.39 / 2.1 (grouped ols + spearman smoke test). CI never pins polars, so it only tests the newest. |

The two anofox crates do not depend on each other or on polars/pyo3, so each upgrade is independent.

## anofox-statistics 0.4.7 → 0.10.0

**Compile breaks for polars-statistics:** none. All 68 imported functions, and every enum and result struct we use, keep their signatures. A few result structs only gained fields: `SPAResult.p_value_lower`, `MCSResult.{elimination_order, mcs_p_values}` and `MCSEliminationStep.mcs_p_value`.

**Behaviour changes** (upstream CHANGELOG line numbers at tag v0.10.0):

| Version | Item | Change | Impact |
|---|---|---|---|
| 0.4.10 (CL:366) | `pairwise_t_test` conf_low/conf_high | `None` → a real CI | **Breaks** `tests/test_posthoc.py:295-296` (expects NaN). Regenerate `tests/fixtures/posthoc_reference.csv`. Fix the docstring at `exprs/posthoc.py:332-333`. |
| 0.8.0 (CL:173) | `spa_test` / `mspe_adjusted_spa` | Statistic is √n-rescaled. **`p_value_consistent` and `p_value_upper` were swapped.** Hansen p-values. `n_bootstrap==0`, n<2 and block length ≤0 now return an error. | `src/expressions/forecast.rs:153-156, 291-294`. The new errors become NaN. |
| 0.8.0 (CL:185) | `model_confidence_set` | Shared bootstrap indices, running-max p-values, no `(c+1)/(B+1)` correction, deterministic order | `mcs_p_value` and `included_models` change (`forecast.rs:233-237`). |
| 0.8.0 (CL:192) | ICC(2,1) / ICC(2,k) CI | Shrout–Fleiss / McGraw–Wong with Satterthwaite df | `ci_*` for `icc_type="icc2"` (the default) change. Check `test_statistics_parity.py:564-568` and `test_correlation.py:396-470`. |
| 0.8.0 (CL:217) | one-sided `t_test` CI | One-sided interval with a ±inf bound | PyModel `conf_int()` in `py_ttest_ind.rs:193` and `py_ttest_paired.rs:180`. |
| 0.8.1 (CL:145) | `semi_partial_cor` with ≥2 controls | Joint residualization (ppcor) | Values change. |
| 0.8.1 (CL:159,164) | RM-ANOVA HF ε, Mauchly p | Lecoutre HF (capped at 1, no GG floor). Mauchly gains a second-order term (k≥4 only). | `hf_epsilon` and `hf_p_value` change. |

Seeded bootstraps are unchanged on 64-bit targets.

**New small surface (in scope):** `p_adjust_n`; `pairwise_t_test_with_conf_level`; `t_test_one_sample` (→ `ttest_1samp`); `bootstrap_*_ci_with_options` + `BootstrapMethod` (iid/stationary/circular/moving block); `MovingBlockBootstrap`; the SPA/MCS new fields; `Kernel::GaussianMedianHeuristic` for MMD; two-sample `ks_test_2sample` / `ad_test_ksamp` / `cvm_test_2sample` / `epps_singleton_test`; `acf` / `pacf` / `ljung_box` / `box_pierce` / `durbin_watson_test` (exact p).

**Deferred (large):** the drift module (psi, KL, JS, Hellinger, Wasserstein, dataset/embedding drift), Kaplan–Meier / log-rank, ROC/PR curves and metric suites, PCA and monitoring (CBPE), calibrators, and the unified `TestResult` refactor.

**Suggested stepping stones:** 0.4.11 (posthoc CI only) → 0.7.0 (no change to what we use) → 0.8.3 (the semantic batch) → 0.10.0 (additive).

## anofox-regression 0.5.22 → 0.12.2

**API breaks** (most land in 0.9):

| Item (our location) | Change | Migration |
|---|---|---|
| `LogisticRegression`, `BayesianRidge`, `ArdRegression` (`expressions/regression.rs:25,1011`; `py_logistic_regression.rs:12`; `py_bayesian_ridge.rs:6`; `py_ard.rs:6`) | Renamed to `*Regressor`. The old names are deprecated, which fails clippy `-D warnings`. | Rename. |
| `.fit_intercept()` on the Lars, BayesianRidge and ARD builders | Becomes `with_intercept` | Rename. |
| `.converged()` / `.iterations()` / `.n_iter()` (penalized_glm, glmm, aft, gamma, passive_aggressive, logistic_regression) | Now on the `IterativeFit` trait. `n_iter` removed. Counts are usize. | Import the trait; `n_iter` → `iterations`. |
| `FittedGlmm` accessors, `sigma_diag`, `lambdas`, `alphas`, `FittedPenalizedGlm::fitted_values` | Return `ColRef`/`Col` instead of slices/Vec | `.iter().copied().collect()` |
| `compute_hc_inference` (`py_ridge.rs:246-256`) | `Result<_, RegressionError>` | `map_err(\|e\| PyValueError::new_err(e.to_string()))` |
| `Augment {..}` literal (`regression.rs:3015`) | New fields (0.5.35/0.5.37), then `#[non_exhaustive]` (0.9) | Build the empty output without the literal. |
| `EbShrinkOptions {..}` (`shrinkage.rs:145`) | `#[non_exhaustive]` | Builder or setters on `default()`. |
| Matches on `ConditionSeverity`, `SeparationType`, `DemandType`, `DemandDistribution`, `AnomalyType`, `PenalizedGlmFamily` | `#[non_exhaustive]` | Add a `_ =>` arm. |
| PenalizedGLM `vcov` × `penalized_glm_covariance_scale` (`py_penalized_glm.rs:626`, `expressions/penalized_glm.rs:345`) | Upstream `vcov` includes φ since 0.8 | **Remove our scaling, or φ is applied twice.** |
| Penalized-GLM `model_info().model_type` | Becomes `"glm"` (0.5.23) | `tests/test_model_info.py:347,352` |
| LmDynamic `max_models=64` default (`py_lm_dynamic.rs:63`, `regression.rs:6317`) | All 2^p subsets are enumerated; errors past max_models (0.8) | Raise the default (upstream uses 4096). |
| Errors | `NumericalError` is split. Infinite input errors. NaN rows are dropped by default (`NaAction`). | Only error-message tests are affected. |

**Numerical changes, mostly 0.8** (expect R-fixture failures):
- **Elastic net:** coordinate descent.
- **GLMs:**
  - Poisson standard errors use dispersion 1.
  - The IRLS tolerance is 1e-8.
  - Collinear columns are aliased.
  - The binomial log-likelihood with trial weights changes.
  - L2-logistic standard errors include the penalty.
- **VIF / condition diagnostics:** BKW-scaled condition labels. Fox–Monette GVIF. VIF/GVIF give ∞/NaN instead of 1 when undefined (0.11.1).
- **Smoothers and constrained fits:**
  - P-spline matches mgcv exactly, and its AIC/BIC are no longer 0 (0.12.0).
  - BLS uses BVLS.
  - The PLS explained-variance fix.
  - Quantile pseudo-R².
  - The WLS zero-weight df.
- **Robust estimators:** RANSAC tie-breaking, Theil–Sen changes.
- **Precision:** p-values no longer underflow to 0; (X'X)⁻¹ via QR.
- **ALM:** Box–Cox normal coefficients are now on the transformed scale.
- **Other models:** lmDynamic, BayesianRidge/ARD (sklearn-exact), the LARS IC and LassoLars drop rule, and GLMM θ / fixed-NB-θ AIC.

**New small surface (in scope):** `breusch_pagan`, `white_test`, `cluster_vcov`, `hc_vcov`, `predict_se` (PredictWithSe), `nested_f_test`, `anova_table`, the press/LOO fields in Augment, `FittedRidge::loo`, `partial_residuals`, `added_variable`, `detect_separation`, `generalized_vif_terms`, Poisson/Tweedie/Gamma `weights`, Poisson `estimate_dispersion`, quantile `alpha`, `HuberScale::Joint`, ridge `conditioning_aware`, Theil–Sen CIs, GLMM `satterthwaite` / `random_effects_se` / `vcov`, Bayesian `credible_intervals` / `predict_std`, and `FittedPSpline::lambda`.

**Deferred (large):** coxph, survfit, survdiff and cox_zph; Ordinal, ContinuationRatio, GLS and Multinomial; mediate; the rms toolkit (splines, effects/nomogram, validate/calibrate/bootcov, areg_impute, transcan/varclus); fit_path / CV; serde persistence.

**Suggested stepping stones:** 0.5.34 → 0.5.38 (Augment) → 0.6/0.7 (basin, ALM) → **0.8.0 (numerics)** → **0.9.0 (API)** → 0.12.2 (additive, apart from the VIF/P-spline semantics).

## Polars / PyO3 stack

- **pyo3 0.26→0.29:**
  - `downcast` → `cast`; one site, `py_penalized_glm.rs:31`.
  - 0.30's by-value pyclass extraction is not hit, since no pyclass derives Clone. 0.29 is the target anyway.
  - Estimate: fewer than 10 edits.
- **numpy 0.26→0.29:** no API breaks on the paths we use.
- **polars 0.52→0.55:**
  - Not verified offline.
  - Exposure: 323 `#[polars_expr]` fns taking `&[Series]`, 155 `output_type_func`, 424 `Series::new`, 315 `Field::new`, 60 `StructChunked::from_series`.
  - Main risk: a Series→Column change in the plugin signatures, which would be mechanical but large.
- **Order:** anofox first, on the trusted stack. Then the stack as one PR, falling back to 0.26→0.27→0.28 if the polars changes turn out large.
- **Python:** set a tested polars range (polars 2.x is current) and add a CI job that tests the declared minimum.
