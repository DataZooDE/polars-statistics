# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Upstream crate refresh: `anofox-statistics` 0.4.2 → 0.4.7 and `anofox-regression`
0.5.13 → 0.5.22 — new API exposed and R-matching fixes adopted, plus two new
model families (penalized GLM, AFT survival regression). This release is an
exposure/correctness pass on the existing `anofox-*` surface — the
confidence-level/effect-size/bootstrap additions those crates now provide — and
also exposes the family-generic penalized GLM and accelerated-failure-time
survival engines `anofox-regression` added in 0.5.16.

### Added

- **Effect sizes** — `ps.cohens_d("x", "y", kind="student"|"welch"|"paired", mu)` and
  `ps.cohens_d_one_sample("x", mu)`, matching R `effectsize::cohens_d` exactly.
- **Rank-biserial correlation** — `ps.rank_biserial("x", "y", mu)`,
  `ps.matched_pairs_rank_biserial("x", "y", mu)`, and
  `ps.rank_biserial_from_u(u1, n1, n2)` (scalar literal args), cross-checked
  against `MannWhitneyU.statistic`.
- **Normality** — `ps.jarque_bera("x")` returns
  `struct{statistic, p_value, skewness, kurtosis, n}`, matching R
  `tseries::jarque.bera.test`.
- **Bootstrap confidence intervals** — `ps.bootstrap_mean_ci("x", n_bootstrap,
  conf_level, block_length, seed)` with IID, stationary, and circular-block
  resampling; `ps.bootstrap_ci("x", statistic="mean"|"median"|"std", n_bootstrap,
  conf_level, seed)` — a fixed-statistic dispatch to native Rust closures (no
  arbitrary Python callables; the upstream generic closure cannot cross the
  Python FFI boundary without a GIL-reentrancy hazard). `bootstrap_ci("mean", ...)`
  is bit-identical to `bootstrap_mean_ci(block_length=0, ...)` for the same seed.
- **`conf_level` kwarg** (default `0.95`, unchanged default behaviour) on
  `ps.binom_test`, `ps.prop_test_one`, `ps.prop_test_two`, and `ps.fisher_exact`.
- **`ps.fisher_exact_conditional(a, b, c, d, alternative, conf_level)`** — R
  `fisher.test`-semantics conditional-MLE odds ratio and exact conditional
  interval, distinct from `fisher_exact`'s sample odds ratio + Woolf interval.
- **Robust regression inference** — `Huber(compute_inference=False,
  confidence_level=0.95)` and `RANSAC(compute_inference=False,
  confidence_level=0.95)` PyModel kwargs, each with 9 Option-safe inference
  getters (`std_errors`, `t_statistics`, `p_values`, `conf_interval_lower`,
  `conf_interval_upper`, `intercept_std_error`, `intercept_t_statistic`,
  `intercept_p_value`, `intercept_conf_interval`); validated against
  `MASS::rlm(method="XtX")`.
- **`RANSAC` inlier-fit statistics** — 9 inlier-OLS fit-stat getters
  (`adj_r_squared`, `mse`, `rmse`, `f_statistic`, `f_pvalue`, `aic`, `bic`,
  `log_likelihood`) and `RANSAC.inlier_fit()`, which returns a real fitted `OLS`
  instance over the inlier rows.
- **`TheilSen(univariate_pairwise=True)`** — selects the classical
  median-of-pairwise-slopes estimator (Sen 1968); `False` restores sklearn's
  spatial-median-over-subsamples estimator.
- **`NegativeBinomial(estimate_dispersion=False)`** PyModel kwarg + `.dispersion`
  getter, plus the `estimate_dispersion` kwarg on `ps.negative_binomial` /
  `negative_binomial_summary` / `negative_binomial_predict` /
  `negative_binomial_formula`; validated against `MASS::glm.nb()` Pearson-dispersion
  standard errors.
- **`RLS(initial_p_diagonal=1e6)`** PyModel kwarg (sets `P0 = delta * I`) +
  `initial_p_diagonal_value()` getter, plus the `initial_p_diagonal` kwarg on
  `ps.rls` / `rls_summary` / `rls_predict` / `rls_formula`.
- **`GLMM.z_values`, `GLMM.p_values`, `GLMM.conf_int(level=0.95)`** — Wald
  inference getters/method, validated against `lme4` 2.0.1.
- **`variance_factor()` / `leverage_new(x_new)`** on `OLS`, `WLS`, and `Ridge` —
  shared variance-factor and leverage API (column-count mismatches raise
  `ValueError`, never a panic).
- **`Ridge.std_errors` / `t_statistics` / `p_values` / `intercept_std_error`** —
  sandwich standard errors for `lambda_ > 0` (t-statistics/p-values are `None`,
  since a t-test is not valid on shrunken coefficients), delegating fully to OLS
  inference at `lambda_ = 0`; R-reference validated.
- **Penalized GLM** — `PenalizedGLM` class (`family="poisson"|"binomial"|
  "logistic"|"negative_binomial"|"gamma"|"tweedie"` with links, per-coefficient
  Normal/Laplace priors as plain tuples or dicts, ridge `lambda_`,
  `vcov_type="laplace"|"sandwich"|"naive"`, offsets, `estimate_dispersion` for
  quasipoisson standard errors, inference including the intercept, scaled
  `vcov`, `information`, separation) and the expressions `ps.penalized_glm`,
  `ps.penalized_glm_summary` (tidy table with confidence intervals),
  `ps.penalized_glm_predict` (response/link scale), `ps.penalized_glm_formula`,
  `_formula_summary`, `_formula_predict`; validated against R `glm`,
  `MASS::glm.nb()`, `negative.binomial(theta)`, and `quasipoisson`.
- **AFT survival regression** — `AFT` class (`distribution="weibull"|
  "exponential"|"lognormal"|"loglogistic"`, right censoring via an event
  indicator, priors, `vcov_type`, inference, `predict_quantile`/`_median`/
  `_survival`/`_cdf`) and `ps.aft`, `ps.aft_summary`, `ps.aft_predict_quantile`/
  `_median`/`_survival`/`_cdf`, `ps.aft_formula`, `ps.aft_formula_summary`,
  `ps.aft_formula_predict_quantile`/`_median`/`_survival`/`_cdf`; validated
  against R `survival::survreg`.

### Changed

- **`ps.fisher_exact`** output struct gains trailing `ci_lower` / `ci_upper`
  fields (Woolf/logit interval). `statistic` and `p_value` are unchanged in
  name, order, and value.
- **`ps.mann_whitney_u` / `ps.wilcoxon_signed_rank`** output structs gain
  trailing `estimate`, `ci_lower`, `ci_upper` fields (Hodges-Lehmann estimate
  and interval, `NaN` unless `conf_level` is given, matching R
  `wilcox.test(conf.int = TRUE)`). `statistic` and `p_value` are unchanged in
  name, order, and value — code that unnests the struct will see three more
  trailing columns.

### Behaviour changes (results now match R)

- **Statistics** (`anofox-statistics` 0.4.3–0.4.6): Shapiro-Wilk now uses R's
  exact `swilk.c` coefficients for n ≤ 11; Kendall's tau uses tie-corrected
  variance (tau-b); Mann-Whitney / Wilcoxon no longer apply a continuity
  correction at the null centre, and report p = 1 when all observations are
  tied; `binom_test` uses the exact Clopper-Pearson interval and R's relative
  tolerance for the exact two-sided p-value; proportion and Fisher intervals
  use a qnorm-based calculation with R's continuity correction, and one-sided
  alternatives now produce one-sided intervals; the exact two-sided
  Mann-Whitney p-value is computed correctly (previously capped at 1 above the
  mean); constant input data now produces a NaN p-value (t-test, ANOVA,
  Brown-Forsythe, Yuen, Pearson) instead of crashing; NaN input to the test
  classes raises `ValueError` ("non-finite value") instead of panicking,
  ±Inf is ranked the way R ranks it, and Inf − Inf pairs are dropped (the
  expression builders already drop non-finite values first, so this only
  affects direct test-class use).
- **Regression** (`anofox-regression` 0.5.14–0.5.16): linear-solver
  log-likelihood / AIC / AICc / BIC now equal R's `logLik.lm` for OLS, WLS,
  Ridge, Huber, ElasticNet, BLS/NNLS, PLS, and RLS; no-intercept R² / adjusted
  R² / F-statistic use the uncentred TSS; Ridge standard errors are sandwich
  SEs, and t-statistics/p-values are unavailable (`NaN` in `ridge_summary`)
  for `lambda > 0`; Ridge prediction intervals use the sandwich variance
  factor; a constant column in a design without an intercept is now kept as
  the intercept rather than producing a singular fit; RANSAC's reported
  statistics are those of the inlier-only OLS fit — `n_observations` is the
  inlier count, inliers are selected by `|r| <= threshold`, and a constant
  response now fits; Poisson / Negative Binomial log-likelihood is the full
  family log-likelihood (NegBin's AIC counts `theta` as a parameter only when
  it is estimated); GLMM Poisson/binomial `log_likelihood` includes the
  saturated-model term; Tweedie / Gamma standard errors use the Pearson
  dispersion, and log-likelihood / AIC / BIC are `NaN` unless `var_power` is
  0, 1, 2, or 3 — the default `var_power=1.5` now reports `NaN` `aic`/`bic`;
  Negative Binomial standard errors use dispersion 1 by default (matching
  `MASS::glm.nb()`; opt into Pearson dispersion with
  `estimate_dispersion=True`); NegBin's `theta` is now the exact ML estimate,
  and zero counts no longer produce negative deviances; WLS with constant
  non-unit weights now scales the way R's `lm(weights=)` does; one-feature
  Theil-Sen is the classical pairwise-median estimator, and rank-deficient
  designs are now aliased instead of failing; quantile regression now reaches
  the exact LP optimum; GLMs with a constant column plus an intercept now
  alias the constant column.
- **Statistics** (`anofox-statistics` 0.4.5 / 0.4.6): the `MannWhitneyU` /
  `WilcoxonSignedRank` Hodges-Lehmann estimate and confidence interval now
  follow R `wilcox.test(conf.int = TRUE)` — a root-found asymptotic interval
  with the continuity correction applied, one-sided intervals for one-sided
  alternatives, and the signed-rank interval computed on `x - y`; `exact=True`
  is now honoured only for `n1 * n2 <= 10,000` (rank sum) or `<= 300` non-zero
  differences (signed rank) — larger samples fall back to the normal
  approximation instead of overflowing; `mcnemar_exact` p-values now come from
  the binomial CDF (matching R `binom.test(b, b + c, 0.5)`), fixing `NaN`
  results for large discordant-pair counts.
- **Regression** (`anofox-regression` 0.5.19–0.5.21): `OLS` / `WLS` alias
  rank-deficient designs the way R's `lm` does (`dqrdc2`, relative tolerance
  1e-7) for every `solve_method` — the **later** of two exactly or
  near-exactly collinear columns gets a `NaN` coefficient (previously `qr`
  aliased the earlier column and `svd` returned a minimum-norm split across
  both collinear columns), and the aliasing decision is scale-independent;
  `WLS` without an intercept now reports standard errors from the reduced
  (non-aliased) fit. Every GLM `*_summary` expression (Binomial, Poisson,
  Negative Binomial, Tweedie, Gamma, Probit, Cloglog) now reports a finite
  intercept `std_error` / `statistic` / `p_value` (previously `NaN`), and
  Tweedie / Gamma also report per-coefficient `statistic` / `p_value`; probit
  standard errors use `statrs` `erfc` / `erfc_inv` and match R to full
  precision (previously off R in the 5th digit); GLM Wald p-values use the
  survival function and no longer underflow to `0`. `LARS` / `LassoLars` now
  alias the **later** of exactly collinear columns (previously
  platform-dependent).

### Fixed

- No panics on NaN/±Inf input anywhere in the statistics layer.
- Exact rank-sum and signed-rank null distributions no longer overflow for
  larger samples (`n1 + n2 > ~62`, `n >= 64`).
- `mcnemar_exact` no longer returns `NaN` for more than ~1,074 discordant
  pairs.
- Removed O(n²) memory usage from the Mann-Whitney / Wilcoxon confidence
  intervals, Kendall's tau, distance correlation, the Wilcoxon TOST
  equivalence tests (which also no longer panic for `alpha > 0.5`), 1-D MMD,
  and 1-D energy distance.
- Removed O(n²) memory usage from the QR/SVD regression solvers, fixing
  out-of-memory failures on large-n designs.

### Added (Rust API)

For `rlib` consumers (all additive; no existing `*_fit` signature or input
index changed):

- New `*_fit` functions: `cohens_d_fit`, `cohens_d_one_sample_fit`,
  `rank_biserial_fit`, `matched_pairs_rank_biserial_fit`,
  `rank_biserial_from_u_fit`, `jarque_bera_fit`, `bootstrap_mean_ci_fit`,
  `bootstrap_ci_fit`, `fisher_exact_conditional_fit`.
- `negative_binomial_fit_with` / `negative_binomial_summary_fit_with`
  (`NegativeBinomialKwargs`), `negative_binomial_predict_fit_with`
  (`NegativeBinomialPredictKwargs`); `rls_fit_with` / `rls_summary_fit_with`
  (`RlsKwargs`), `rls_predict_fit_with` (`RlsPredictKwargs`).
- `binom_test_fit` / `prop_test_one_fit` / `prop_test_two_fit` /
  `fisher_exact_fit` now accept an optional trailing `conf_level` input — the
  old arity still works unchanged.
- New `*_fit` functions: `penalized_glm_fit`, `penalized_glm_summary_fit`,
  `penalized_glm_predict_fit` (`PenalizedGlmKwargs`, `PriorKwarg`), `aft_fit`,
  `aft_summary_fit`, `aft_predict_quantile_fit`, `aft_predict_median_fit`,
  `aft_predict_survival_fit`, `aft_predict_cdf_fit` (`AftKwargs`), plus the
  shared helpers `parse_penalized_glm_family`, `prior_spec_from_parts`,
  `parse_vcov_type`, `parse_aft_distribution`, and the output-dtype builders
  `inference_summary_output_dtype` / `rank_location_output_dtype`.
- `mann_whitney_u_fit` / `wilcoxon_signed_rank_fit` keep their existing input
  contract unchanged and now return the widened (5-field) struct.

### Dependencies

- `anofox-statistics` 0.4.2 → 0.4.7.
- `anofox-regression` 0.5.13 → 0.5.22 — the 0.5.16 penalized GLM and AFT
  survival solvers are now exposed (see Added). The other 0.4.7 / 0.5.17–0.5.22
  additions (post-hoc tests, `p_adjust`, `augment`, `ModelInfo`, GLMM families,
  empirical-Bayes shrinkage) are planned for a later release.

## [0.7.0] - 2026-09-20

Ergonomics, adoption & documentation release. Additive and backward-compatible —
no breaking changes. Focuses on making the full v0.6.0 API surface best-in-class
to adopt and use: type-safe, ergonomic to consume, clear on errors, and thoroughly
documented.

### Added

- **Type safety** — the package now ships a `py.typed` marker and `.pyi` type
  stubs for every Rust-bound model and test class, so IDEs and type checkers
  (mypy/pyright) resolve accurate constructor, method, and getter signatures. The
  Polars expression builders and formula helpers are fully annotated. A CI drift
  guard keeps the stubs in sync with the runtime API.
- **Result ergonomics** — every model/test class gains `.to_dict()`, `.summary()`,
  and an informative `__repr__`. New package-level helpers
  `polars_statistics.unnest(df, column=...)` and
  `polars_statistics.struct_to_dict(df, column=...)` flatten expression Struct
  results without manual `.struct.field()` extraction.
- **sklearn-style `score()`** — added uniformly across regressors (returns R²) and
  binary classifiers (returns mean accuracy), alongside consistent `fit`/`predict`.
- **`add_intercept`** — a canonical intercept keyword on the regression
  expressions and model constructors (see Deprecated).
- **Documentation** — new cookbook pages for robust/sparse regression
  (TheilSen, RANSAC, BayesianRidge, ARD, LARS, PassiveAggressive), GLM/smoother/
  streaming models (Gamma, GLMM, PSpline, MomentAccumulator), and the ANOVA
  family; a model-selection decision matrix; a migration guide; an
  sklearn-migration guide; and an adoption-focused README refresh.

### Changed

- **Contextual errors** — calling a method on an unfitted model now raises an
  error naming the model and the required `.fit(...)` call (type unchanged:
  `RuntimeError`). Regressor `fit` paths validate input shapes up front and raise
  actionable `ValueError`s for empty input, row mismatches, and rank-deficient
  designs, instead of opaque failures or panics.

### Deprecated

- **`with_intercept`** is deprecated in favor of `add_intercept` across the
  regression expressions and model constructors. `with_intercept` still works and
  emits a `FutureWarning`; removal is not scheduled for this release. Passing both
  keywords raises. (Classes that use `fit_intercept` — BayesianRidge, ARD, LARS —
  are unchanged.)

## [0.6.0] - 2026-08-17

### Added

- **New statistics API** (`anofox-statistics` 0.4.1 → 0.4.2):
  - `one_way_anova` — one-way ANOVA with Fisher and Welch variants; returns
    F-statistic, degrees of freedom, p-value, sums of squares, and η² effect size.
  - `two_way_anova` — two-way ANOVA with A×B interaction term; partitions variance
    into main effects (A, B) and interaction (A×B).
  - `repeated_measures_anova` — repeated-measures ANOVA with Mauchly's sphericity
    test and Greenhouse-Geisser / Huynh-Feldt ε corrections.
  - `energy_distance_nd` — multivariate energy distance for two multivariate samples
    (generalization of `energy_distance` to arbitrary dimension).
  - `icc` — intraclass correlation with real matrix-input contract; replaced the
    previous all-NaN stub (see Breaking Changes).
- **New regression model classes** (`anofox-regression` 0.5.4 → 0.5.13):
  - **GLM family:** `Gamma` (log link, positive data), `GLMM` (generalized linear
    mixed model with random intercepts), `PSpline` (penalized B-spline smoother).
  - **Robust / sparse linear:** `TheilSen` (median-of-slopes, ~29% breakdown point),
    `RANSAC` (random sample consensus), `BayesianRidge` (evidence-maximisation
    hyperparameter tuning), `ARD` (automatic relevance determination with per-feature
    precision), `LARS` (least angle regression with full path), `PassiveAggressive`
    (online PA/PA-I/PA-II regression).
  - **Online learning:** `MomentAccumulator` — accumulates sufficient statistics
    (XᵀX, Xᵀy) from streaming batches; pass to `OLS.fit_from_accumulator` or
    `Ridge.fit_from_accumulator`.
  - **HC-robust inference:** `Ridge.hc_inference(X, y, hc_type)` — HC0–HC3
    sandwich standard errors for Ridge fits.
  - **GLM diagnostic expressions** for Gamma GLM fits:
    `gamma_dispersion_deviance`, `gamma_dispersion_pearson`,
    `gamma_pearson_chi_squared`,
    `gamma_standardized_pearson_residuals`,
    `gamma_standardized_deviance_residuals`.

### Changed

- Updated `anofox-statistics` dependency 0.4.1 → **0.4.2**.
- Updated `anofox-regression` dependency 0.5.4 → **0.5.13**.
- **Column-pivot correctness fix** (inherited from `anofox-regression` 0.5.13):
  OLS, WLS, and NNLS models trained on differently-scaled designs (e.g. one predictor
  in units of thousands while another is in units of ones) now return correct
  coefficients. The previous release could pivot to the wrong column when building the
  design matrix, causing coefficients to be assigned to the wrong predictor.

### Breaking Changes

- **`icc` new contract:** the expression now accepts one column per rater
  (`ps.icc("rater1", "rater2", "rater3", icc_type="icc3")`) and computes real
  ICC values validated against R's `irr::icc()`. The old single-column stub always
  returned NaN and is removed. Update call sites to pass all rater columns explicitly.

### Known Limitations

- `WLS.hc_inference` raises `NotImplementedError` — heteroscedasticity-consistent
  inference is not yet implemented for WLS; use `OLS.hc_inference` or
  `Ridge.hc_inference` as alternatives.
- `icc` matrix-input contract change: code passing only a single column to `ps.icc()`
  must be updated to pass one column per rater.

---

## [0.5.0] - 2026-05-28

### Added

- **Hybrid crate (Rust + Python)** — `polars-statistics` now builds as both a
  `cdylib` (Python plugin) and an `rlib` (Rust dependency). Every Polars
  expression has a public `<name>_fit` Rust entry point in
  `polars_statistics::expressions`. New `python` Cargo feature gates the
  pyo3 / numpy / pymodels surface; downstream Rust crates use
  `default-features = false`. (#13)
- **New regression model wrappers**:
  - `Huber` — M-estimator robust to outliers; class + `huber()` expression. (#14)
  - `LogisticRegression` — sklearn-style API with `predict_proba`,
    `decision_function`, `score`, `penalty="l2"`, `C` kwarg. Distinct from
    the existing `Logistic` wrapper. (#14)
  - `PLS` — Partial Least Squares with `transform()` for the latent space;
    class + `pls()` expression. (#19)
- **Penalized IRLS `lambda_` kwarg** on the GLM PyClass models — `PyLogistic`,
  `PyPoisson`, `PyNegativeBinomial`, `PyTweedie`, `PyProbit`, `PyCloglog`
  (the expression layer already supported it). (#15)
- **ALM expression parity with `PyALM`** — all 25 distributions reachable
  from `ps.alm(...)`, plus `loss`, `link`, `role_trim`, and
  `extra_parameter` kwargs. (#16)
- **Summary / predict completeness** — added the matching expressions for
  families that previously only had a base fit (#18):
  - `quantile_summary`, `quantile_predict`
  - `isotonic_predict`
  - `lm_dynamic_predict`
- **Diagnostics toolkit** (#17 + #27):
  - Multicollinearity: `vif`, `generalized_vif`, `high_vif_predictors`
  - OLS residual battery: `standardized_residuals`,
    `studentized_residuals`, `externally_studentized_residuals`,
    `residual_outliers`
  - GLM residuals (logistic + Poisson): `*_pearson_residuals`,
    `*_deviance_residuals`, `*_working_residuals` for each family
  - Influence / leverage: `leverage`, `cooks_distance`, `dffits`,
    `influential_cooks`, `influential_dffits`, `high_leverage_points`
  - Goodness of fit: `pearson_chi_squared_logistic`,
    `pearson_chi_squared_poisson`
- **Documentation**:
  - Sweep of the README and mkdocs site to cover every v0.5.0 addition.
  - New "Use from Rust" section + `examples/rust_wls.rs` walking through
    the rlib path.

### Changed

- Updated `anofox-regression` dependency to v0.5.4 (introduces
  `HuberRegressor` and the sklearn-style `LogisticRegression`)
- Updated `anofox-statistics` dependency to v0.4.1
- Python `__version__` caught up from 0.3.0 → 0.5.0 (was lagging two
  minor versions behind the wheel metadata)

### Backwards compatibility

Additive. `PyLogistic` is unchanged — the new sklearn-style
`LogisticRegression` is a separate class. The `alm()` expression's input
contract grew but all existing keyword-only callers continue to work
because the new kwargs default to `None` / `"likelihood"`.

## [0.4.0] - 2026-01-17

### Added

- **Robust Regression**:
  - `quantile` - Quantile regression (median and arbitrary quantiles)
  - `isotonic` - Isotonic (monotonic) regression using PAVA algorithm
  - `Quantile` and `Isotonic` model classes for scikit-learn-style API

- **Regression Diagnostics**:
  - `condition_number` - Detect multicollinearity via condition number analysis
  - `check_binary_separation` - Detect complete/quasi-complete separation in logistic regression
  - `check_count_sparsity` - Detect sparsity issues in Poisson/count regression

- **Conda-forge Support**:
  - Added conda-forge recipe for `conda install polars-statistics`
  - CI workflow for conda package building and testing

- **Documentation**:
  - Reorganized API documentation into modular structure
  - Added detailed descriptions for all statistical tests
  - New docs for TOST equivalence tests, correlation, categorical tests, and forecast comparison

### Changed

- Updated `anofox-regression` dependency to v0.5.1
- Updated `pyo3` to v0.27
- Updated `faer` to v0.23.2
- Improved cross-platform test compatibility for GLM functions

## [0.3.0] - 2024-12-22

### Added

- **TOST Equivalence Tests** (10 functions):
  - `tost_t_test_one_sample` - One-sample equivalence test
  - `tost_t_test_two_sample` - Two-sample equivalence test
  - `tost_t_test_paired` - Paired samples equivalence test
  - `tost_correlation` - Correlation equivalence test (Pearson/Spearman)
  - `tost_prop_one` - One-proportion equivalence test
  - `tost_prop_two` - Two-proportion equivalence test
  - `tost_wilcoxon_paired` - Non-parametric paired equivalence test
  - `tost_wilcoxon_two_sample` - Non-parametric two-sample equivalence test
  - `tost_bootstrap` - Bootstrap-based equivalence test
  - `tost_yuen` - Trimmed means equivalence test

- **Correlation Tests** (7 functions):
  - `pearson` - Pearson correlation with confidence intervals
  - `spearman` - Spearman rank correlation
  - `kendall` - Kendall's tau (variants a, b, c)
  - `distance_cor` - Distance correlation (detects nonlinear relationships)
  - `partial_cor` - Partial correlation controlling for covariates
  - `semi_partial_cor` - Semi-partial (part) correlation
  - `icc` - Intraclass correlation coefficient

- **Categorical Tests** (13 functions):
  - `binom_test` - Exact binomial test
  - `prop_test_one` - One-sample proportion test
  - `prop_test_two` - Two-sample proportion test
  - `chisq_test` - Chi-square test for independence
  - `chisq_goodness_of_fit` - Chi-square goodness of fit test
  - `g_test` - G-test (likelihood ratio test)
  - `fisher_exact` - Fisher's exact test for 2x2 tables
  - `mcnemar_test` - McNemar's test for paired proportions
  - `mcnemar_exact` - McNemar's exact test
  - `cohen_kappa` - Cohen's Kappa for inter-rater agreement
  - `cramers_v` - Cramer's V for association strength
  - `phi_coefficient` - Phi coefficient for 2x2 tables
  - `contingency_coef` - Contingency coefficient (Pearson's C)

- Comprehensive test suite for all new statistical tests

### Changed

- Updated `anofox-statistics` dependency to v0.4.0
- Updated `anofox-regression` dependency to v0.4.0

## [0.2.0] - 2024-12-15

### Added

- **R-style Formula Syntax**:
  - `ols_formula`, `ridge_formula`, `elastic_net_formula`, etc.
  - Support for polynomial terms: `poly(x, 2)`
  - Support for interactions: `x1 * x2` expands to `x1 + x2 + x1:x2`
  - Support for explicit transforms: `I(x^2)`

- **Summary Functions** for tidy coefficient output:
  - `ols_summary`, `ridge_summary`, `elastic_net_summary`
  - `wls_summary`, `rls_summary`, `bls_summary`
  - `logistic_summary`, `poisson_summary`, `negative_binomial_summary`
  - `tweedie_summary`, `probit_summary`, `cloglog_summary`, `alm_summary`
  - Returns: term, estimate, std_error, statistic, p_value

- **Prediction Functions** with intervals:
  - `ols_predict`, `ridge_predict`, `elastic_net_predict`, etc.
  - Support for confidence and prediction intervals
  - Configurable confidence level

- **Additional Regression Models**:
  - `WLS` - Weighted Least Squares
  - `RLS` - Recursive Least Squares
  - `BLS` - Bounded Least Squares
  - `NNLS` - Non-negative Least Squares

- **GLM Models**:
  - `Logistic` - Logistic regression
  - `Poisson` - Poisson regression
  - `NegativeBinomial` - Negative binomial regression
  - `Tweedie` - Tweedie regression
  - `Probit` - Probit regression
  - `Cloglog` - Complementary log-log regression

- **ALM (Augmented Linear Model)**:
  - Support for 24+ distributions (normal, laplace, cauchy, student-t, etc.)
  - Robust regression alternatives

- **Dynamic Models**:
  - `LmDynamic` - Dynamic linear model with forgetting factor
  - `lm_dynamic` expression for rolling regression

- **Demand Classification**:
  - `Aid` - Automatic Item-level Demand classification
  - `aid` and `aid_anomalies` expressions
  - Based on Kolassa (2025) methodology

- **Model Classes for Statistical Tests**:
  - `TTestInd`, `TTestPaired` - t-test classes
  - `BrownForsythe`, `YuenTest` - variance/robust tests
  - `MannWhitneyU`, `WilcoxonSignedRank`, `KruskalWallis`, `BrunnerMunzel`
  - `ShapiroWilk`, `DAgostino` - normality tests

- **Forecast Comparison Tests**:
  - `diebold_mariano` - Diebold-Mariano test
  - `clark_west` - Clark-West test
  - `spa_test` - Superior Predictive Ability test
  - `model_confidence_set` - Model Confidence Set
  - `mspe_adjusted` - MSPE-adjusted test
  - `permutation_t_test` - Permutation t-test

- **Modern Distribution Tests**:
  - `energy_distance` - Energy distance test
  - `mmd_test` - Maximum Mean Discrepancy test

- **Bootstrap Methods**:
  - `StationaryBootstrap` - Stationary bootstrap for time series
  - `CircularBlockBootstrap` - Circular block bootstrap

### Changed

- Renamed `regress-rs` dependency to `anofox-regression`
- Improved documentation with API reference

## [0.1.0] - 2024-11-01

### Added

- Initial release of polars-statistics

- **Core Statistical Tests**:
  - `ttest_ind` - Independent samples t-test
  - `ttest_paired` - Paired samples t-test
  - `brown_forsythe` - Brown-Forsythe test for variance equality
  - `yuen_test` - Yuen's test for trimmed means

- **Non-parametric Tests**:
  - `mann_whitney_u` - Mann-Whitney U test
  - `wilcoxon_signed_rank` - Wilcoxon signed-rank test
  - `kruskal_wallis` - Kruskal-Wallis H test
  - `brunner_munzel` - Brunner-Munzel test

- **Distributional Tests**:
  - `shapiro_wilk` - Shapiro-Wilk normality test
  - `dagostino` - D'Agostino-Pearson normality test

- **Regression Models**:
  - `OLS` - Ordinary Least Squares
  - `Ridge` - Ridge regression
  - `ElasticNet` - Elastic Net regression
  - Expression API: `ols`, `ridge`, `elastic_net`

- **Polars Integration**:
  - Full support for `group_by` aggregations
  - Full support for `over` window functions
  - Lazy evaluation support
  - Struct output for all statistical results

- **Performance**:
  - Rust-powered with zero-copy data transfer
  - SIMD-optimized linear algebra via faer
  - Automatic parallelization for group operations

[0.7.0]: https://github.com/DataZooDE/polars-statistics/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/DataZooDE/polars-statistics/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/DataZooDE/polars-statistics/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/DataZooDE/polars-statistics/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/DataZooDE/polars-statistics/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/DataZooDE/polars-statistics/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/DataZooDE/polars-statistics/releases/tag/v0.1.0
