# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.5.0]: https://github.com/DataZooDE/polars-statistics/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/DataZooDE/polars-statistics/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/DataZooDE/polars-statistics/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/DataZooDE/polars-statistics/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/DataZooDE/polars-statistics/releases/tag/v0.1.0
