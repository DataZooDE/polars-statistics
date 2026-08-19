# Phase 2: Public API Audit — 02-API-AUDIT.md

**Phase:** 02-public-api-audit
**Plan:** 02-01
**Status:** Complete
**Date:** 2026-08-11
**Consumed by:** Phase 3 (statistics parity planner), Phase 4 (regression parity planner)

---

## Overview

### Audited Crate Versions

| Crate | Audited Version | Source |
|-------|----------------|--------|
| `anofox-statistics` | **0.4.2** | `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-statistics-0.4.2/src/lib.rs` |
| `anofox-regression` | **0.5.13** | `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/lib.rs` + `src/solvers/mod.rs` + `src/diagnostics/mod.rs` + `src/inference/mod.rs` + `src/core/mod.rs` |

### Enumeration Method

Each crate's authoritative public surface was read directly from the Cargo registry source cache
(`~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/`) by scanning `lib.rs` top-level
`pub use` re-exports and referenced module-level `pub use` blocks. The "exposed?" determination
used `python/polars_statistics/__init__.py` as the ground-truth Python surface, cross-referenced
against `src/expressions/*.rs` and `src/pymodels/mod.rs`. A capability is counted as
**already-exposed** if it is reachable from Python via EITHER a Polars expression OR a PyModel
class — either surface suffices.

### Summary Counts

#### anofox-statistics 0.4.2

| Category | Total pub items | Already exposed | Gaps (user-facing) | Gaps (internal/low-priority) |
|----------|-----------------|-----------------|--------------------|------------------------------|
| Parametric | 16 | 8 | 8 (3 ANOVA fns + 5 ANOVA result types) | 0 |
| Nonparametric | 9 | 8 | 0 | 1 (`rank`) |
| Distributional | 4 | 4 | 0 | 0 |
| Correlation | 12 | 9 | 2 (`ICCResult`, `ICCType`) + 1 stub (`icc`) | 1 (`distance_cor` standalone) |
| Modern | 6 | 2 | 3 (`energy_distance_test` nD, `mmd_test` nD, `Kernel`) | 1 |
| Categorical | 21 | 21 | 0 | 0 |
| Equivalence (TOST) | 14 | 14 | 0 | 0 |
| Forecast | 13 | 13 | 0 | 0 |
| Resampling | 5 | 3 | 0 | 2 (`PermutationEngine`, `PermutationResult`) |
| Error types | 2 | 0 | 0 | 2 (internal) |
| **Total** | **102** | **82** | **13** | **7** |

**Actionable gaps for Phase 3:** `one_way_anova`, `two_way_anova`, `repeated_measures_anova`
(+ their result types `OneWayAnovaResult`, `TwoWayAnovaResult`, `RmAnovaResult`,
`SphericityResult`, `CorrectedResult`), `energy_distance_test` nD overload, plus the ICC stub fix.

#### anofox-regression 0.5.13

| Category | Already exposed | Gaps (HIGH priority) | Gaps (MEDIUM priority) | Internal/low |
|----------|-----------------|---------------------|------------------------|--------------|
| Solvers | 17 regressors | 3 (`GlmmRegressor`, `PSplineRegressor`, `GammaRegressor`) | 6 (`TheilSen`, `RANSAC`, `BayesianRidge`, `ARD`, `LARS`, `PassiveAggressive`) | 0 |
| HC Inference | `HcType` (partial) | 4 (`HcInference`, `HcResult`, `HcInterceptInference`, `compute_hc_inference`) | 1 (`compute_hc_standard_errors`) | 0 |
| Diagnostics | 18 items | 0 | 5 (`estimate_dispersion_*`, `pearson_chi_squared`, `standardized_deviance_residuals`, `standardized_pearson_residuals`) | 1 (`response_residuals`) |
| Streaming | 0 | 0 | 1 (`MomentAccumulator`) | 0 |
| Core/Utility | 10+ | 0 | 0 | 20+ internal |
| **Total gaps** | — | **7** | **13** | **20+** |

**Actionable gaps for Phase 4:** `GammaRegressor`, `GlmmRegressor`, `PSplineRegressor`,
`HcInference`/HC expressions (REGR-01 through REGR-05); then MEDIUM priority new solvers (REGR-06).

---

## anofox-statistics 0.4.2 — Gap List

> Satisfies **AUDIT-01**: authoritative documented gap list for every public `anofox-statistics` function.

### Parametric Tests

| Item | Kind | Exposed? | Target Surface | Priority | Notes |
|------|------|----------|----------------|----------|-------|
| `t_test` | fn | YES | expression (`ttest_ind`, `ttest_paired`) | — | Both independent and paired via `TTestKind` |
| `yuen_test` | fn | YES | expression (`yuen_test`) | — | |
| `brown_forsythe` | fn | YES | expression (`brown_forsythe`) | — | Called directly via `anofox_statistics::brown_forsythe` |
| `one_way_anova` | fn | **NO** | expression | HIGH | Required by STAT-01 |
| `two_way_anova` | fn | **NO** | expression | HIGH | Required by STAT-02 |
| `repeated_measures_anova` | fn | **NO** | expression | HIGH | Required by STAT-03; exercises sphericity/Mauchly |
| `Alternative` | enum | YES | expression (parameter) | — | Used in multiple expressions |
| `TTestKind` | enum | YES | expression (internal) | — | |
| `TTestResult` | struct | YES | expression (output) | — | |
| `AnovaKind` | enum | NO | expression (parameter) | MEDIUM | Needed to parameterize one_way_anova type |
| `AnovaTableRow` | struct | NO | expression (output) | MEDIUM | Output row for ANOVA table |
| `OneWayAnovaResult` | struct | NO | expression (output) | HIGH | Output for STAT-01 |
| `TwoWayAnovaResult` | struct | NO | expression (output) | HIGH | Output for STAT-02 |
| `RmAnovaResult` | struct | NO | expression (output) | HIGH | Output for STAT-03 |
| `SphericityResult` | struct | NO | expression (output) | HIGH | Embedded in `RmAnovaResult` |
| `CorrectedResult` | struct | NO | expression (output) | MEDIUM | Greenhouse-Geisser/Huynh-Feldt corrections in `RmAnovaResult` |
| `LeveneResult` | struct | YES | expression (internal, used by brown_forsythe) | — | |
| `YuenResult` | struct | YES | expression (output) | — | |
| `YuenConfInt` | struct | YES | expression (internal) | — | |

### Nonparametric Tests

| Item | Kind | Exposed? | Target Surface | Priority | Notes |
|------|------|----------|----------------|----------|-------|
| `mann_whitney_u` | fn | YES | expression (`mann_whitney_u`) | — | |
| `wilcoxon_signed_rank` | fn | YES | expression (`wilcoxon_signed_rank`) | — | |
| `kruskal_wallis` | fn | YES | expression (`kruskal_wallis`) | — | |
| `brunner_munzel` | fn | YES | expression (`brunner_munzel`) | — | |
| `rank` | fn | **NO** | util | LOW | Internal helper — computes rank array; not a user-facing test; see Internal-only section |
| `MannWhitneyResult` | struct | YES | expression (output) | — | |
| `WilcoxonResult` | struct | YES | expression (output) | — | |
| `KruskalResult` | struct | YES | expression (output) | — | |
| `BrunnerMunzelResult` | struct | YES | expression (output) | — | |

### Distributional Tests

| Item | Kind | Exposed? | Target Surface | Priority | Notes |
|------|------|----------|----------------|----------|-------|
| `shapiro_wilk` | fn | YES | expression (`shapiro_wilk`) | — | |
| `dagostino_k_squared` | fn | YES | expression (`dagostino`) | — | |
| `ShapiroWilkResult` | struct | YES | expression (output) | — | |
| `DAgostinoResult` | struct | YES | expression (output) | — | |

### Correlation

| Item | Kind | Exposed? | Target Surface | Priority | Notes |
|------|------|----------|----------------|----------|-------|
| `pearson` | fn | YES | expression (`pearson`) | — | |
| `spearman` | fn | YES | expression (`spearman`) | — | |
| `kendall` | fn | YES | expression (`kendall`) | — | |
| `distance_cor_test` | fn | YES | expression (`distance_cor`) | — | Permutation-based distance correlation test |
| `distance_cor` | fn | **NO** | util | LOW | Computes dcor coefficient only, no test; wrapper uses `distance_cor_test`; not needed separately |
| `partial_cor` | fn | YES | expression (`partial_cor`) | — | |
| `semi_partial_cor` | fn | YES | expression (`semi_partial_cor`) | — | |
| `icc` | fn | YES (stub) | expression (`icc`) — **STUB ONLY** | MEDIUM | Binding exists but returns all-NaN; see Special-Case Flags |
| `CorrelationResult` | struct | YES | expression (output) | — | |
| `DistanceCorResult` | struct | YES | expression (output) | — | |
| `ICCResult` | struct | NO | expression (output) | MEDIUM | Needed once `icc` is properly implemented |
| `ICCType` | enum | NO | expression (parameter) | MEDIUM | Needed to select ICC variant (ICC1, ICC2, ICC3, etc.) |
| `KendallVariant` | enum | YES | expression (parameter) | — | |
| `PartialCorResult` | struct | YES | expression (output) | — | |
| `CorrelationConfInt` | struct | YES | expression (internal) | — | |
| `CorrelationMethod` | enum | YES | expression (internal) | — | |

### Modern Tests

| Item | Kind | Exposed? | Target Surface | Priority | Notes |
|------|------|----------|----------------|----------|-------|
| `energy_distance_test_1d` | fn | YES | expression (`energy_distance`) | — | 1D (scalar series) overload — this is what the expression uses |
| `energy_distance_test` | fn | **NO** | expression | HIGH | Multi-dimensional overload — required by STAT-04 (higher-dimensional case); see Deferred Scope Questions |
| `mmd_test_1d` | fn | YES | expression (`mmd_test`) | — | 1D overload |
| `mmd_test` | fn | **NO** | expression | MEDIUM | Multi-dimensional MMD test overload — not yet exposed |
| `EnergyDistanceResult` | struct | YES | expression (output) | — | |
| `MMDResult` | struct | YES | expression (output) | — | |
| `Kernel` | enum | **NO** | expression (parameter) | MEDIUM | Kernel selection for multi-dim MMD; not needed for 1D (median heuristic used) |

### Categorical Tests

| Item | Kind | Exposed? | Target Surface | Priority | Notes |
|------|------|----------|----------------|----------|-------|
| `binom_test` | fn | YES | expression (`binom_test`) | — | |
| `prop_test_one` | fn | YES | expression (`prop_test_one`) | — | |
| `prop_test_two` | fn | YES | expression (`prop_test_two`) | — | |
| `chisq_test` | fn | YES | expression (`chisq_test`) | — | |
| `chisq_goodness_of_fit` | fn | YES | expression (`chisq_goodness_of_fit`) | — | |
| `g_test` | fn | YES | expression (`g_test`) | — | |
| `fisher_exact` | fn | YES | expression (`fisher_exact`) | — | |
| `mcnemar_test` | fn | YES | expression (`mcnemar_test`) | — | |
| `mcnemar_exact` | fn | YES | expression (`mcnemar_exact`) | — | |
| `cohen_kappa` | fn | YES | expression (`cohen_kappa`) | — | |
| `cramers_v` | fn | YES | expression (`cramers_v`) | — | |
| `phi_coefficient` | fn | YES | expression (`phi_coefficient`) | — | |
| `contingency_coef` | fn | YES | expression (`contingency_coef`) | — | |
| `BinomTestResult` | struct | YES | expression (output) | — | |
| `PropTestResult` | struct | YES | expression (output) | — | |
| `ChiSquareResult` | struct | YES | expression (output) | — | |
| `FisherResult` | struct | YES | expression (output) | — | |
| `McNemarkResult` | struct | YES | expression (output) | — | |
| `McNemarkExactResult` | struct | YES | expression (output) | — | |
| `KappaResult` | struct | YES | expression (output) | — | |
| `AssociationResult` | struct | YES | expression (output) | — | |

### Equivalence (TOST)

| Item | Kind | Exposed? | Target Surface | Priority | Notes |
|------|------|----------|----------------|----------|-------|
| `tost_t_test_one_sample` | fn | YES | expression | — | |
| `tost_t_test_two_sample` | fn | YES | expression | — | |
| `tost_t_test_paired` | fn | YES | expression | — | |
| `tost_wilcoxon_paired` | fn | YES | expression | — | |
| `tost_wilcoxon_two_sample` | fn | YES | expression | — | |
| `tost_correlation` | fn | YES | expression | — | |
| `tost_prop_one` | fn | YES | expression | — | |
| `tost_prop_two` | fn | YES | expression | — | |
| `tost_yuen` | fn | YES | expression | — | |
| `tost_bootstrap` | fn | YES | expression | — | |
| `TostResult` | struct | YES | expression (output) | — | |
| `EquivalenceBounds` | enum | YES | expression (parameter) | — | |
| `CorrelationTostMethod` | enum | YES | expression (parameter) | — | |
| `OneSidedTestResult` | struct | YES | expression (internal) | — | |

### Forecast

| Item | Kind | Exposed? | Target Surface | Priority | Notes |
|------|------|----------|----------------|----------|-------|
| `diebold_mariano` | fn | YES | expression (`diebold_mariano`) | — | |
| `clark_west` | fn | YES | expression (`clark_west`) | — | |
| `spa_test` | fn | YES | expression (`spa_test`) | — | |
| `mspe_adjusted_spa` | fn | YES | expression (`mspe_adjusted`) | — | |
| `model_confidence_set` | fn | YES | expression (`model_confidence_set`) | — | |
| `DMResult` | struct | YES | expression (output) | — | |
| `CWResult` | struct | YES | expression (output) | — | |
| `SPAResult` | struct | YES | expression (output) | — | |
| `MSPEAdjustedResult` | struct | YES | expression (output) | — | |
| `MCSResult` | struct | YES | expression (output) | — | |
| `MCSEliminationStep` | struct | YES | expression (internal) | — | |
| `LossFunction` | enum | YES | expression (parameter) | — | |
| `VarEstimator` | enum | YES | expression (parameter) | — | |
| `MCSStatistic` | enum | YES | expression (parameter) | — | |

### Resampling

| Item | Kind | Exposed? | Target Surface | Priority | Notes |
|------|------|----------|----------------|----------|-------|
| `permutation_t_test` | fn | YES | expression (`permutation_t_test`) | — | |
| `CircularBlockBootstrap` | struct | YES | PyModel (`CircularBlockBootstrap`) | — | Exposed as Python class only |
| `StationaryBootstrap` | struct | YES | PyModel (`StationaryBootstrap`) | — | Exposed as Python class only |
| `PermutationEngine` | struct | **NO** | util | LOW | Low-level engine used internally by `permutation_t_test`; not user-facing |
| `PermutationResult` | struct | **NO** | util | LOW | Return type of internal `PermutationEngine` calls; not needed directly |

### Internal-only Items (not Phase 3 action items)

These items are in the `anofox-statistics` public API surface (exported from `lib.rs`) but are
internal helpers with no meaningful user-facing purpose. They are not gaps that Phase 3 should expose.

| Item | Kind | Reason Not a User Gap |
|------|------|----------------------|
| `rank` | fn | Internal ranking helper used inside nonparametric test computations; not a user test |
| `distance_cor` | fn | Coefficient-only variant; wrapper correctly uses `distance_cor_test` (includes p-value) |
| `PermutationEngine` | struct | Low-level permutation engine; users access via `permutation_t_test` expression |
| `PermutationResult` | struct | Internal return type of `PermutationEngine`; not surfaced to users |
| `Result` | type alias | `anofox_statistics::Result<T>` = `std::result::Result<T, StatError>`; Rust-internal only |
| `StatError` | enum | Rust error type; converted to Python exceptions by expression wrappers; not user-facing |

---

## anofox-regression 0.5.13 — Gap List

> Satisfies **AUDIT-02**: authoritative documented gap list for every public `anofox-regression` capability.

### Core Types

**Source:** `anofox-regression-0.5.13/src/core/mod.rs`

| Item | Kind | Exposed? | Target Surface | Priority | Notes |
|------|------|----------|----------------|----------|-------|
| `SolverType` | enum | YES | expression (parameter, via `parse_solver_type`) | — | QR/SVD/Cholesky — already imported in `regression.rs` |
| `IntervalType` | enum | YES | expression (parameter) | — | Used in predict expressions |
| `RegressionOptions` | struct | YES | expression (internal builder) | — | |
| `RegressionOptionsBuilder` | struct | YES | expression (internal builder) | — | |
| `RegressionResult` | struct | YES | expression (output fields accessed) | — | |
| `PredictionResult` | struct | YES | expression (output) | — | |
| `PredictionType` | enum | YES | expression (parameter) | — | |
| `LambdaScaling` | enum | NO | util | LOW | Internal scaling option for regularization; not user-facing |
| `OptionsError` | enum | NO | util | LOW | Error from options builder; not user-facing |
| `GlmFamily` | trait | NO | util | LOW | Internal trait implemented by GLM families; not user-callable |
| `TweedieFamily` | struct | NO | util | LOW | Internal GLM family config; configured via TweedieRegressor parameter |
| `BinomialFamily` | struct | NO | util | LOW | Internal GLM family config |
| `BinomialLink` | enum | NO | util | LOW | Internal link function config |
| `NegativeBinomialFamily` | struct | NO | util | LOW | Internal GLM family config |
| `PoissonFamily` | struct | NO | util | LOW | Internal GLM family config |
| `PoissonLink` | enum | NO | util | LOW | Internal link function config |
| `NaAction` | enum | NO | util | LOW | NA handling policy; internal |
| `NaError` | enum | NO | util | LOW | Internal error type |
| `NaHandler` | struct | NO | util | LOW | Internal |
| `NaInfo` | struct | NO | util | LOW | Internal |
| `NaResult` | type alias | NO | util | LOW | Internal |
| `estimate_theta_ml` | fn | NO | util | LOW | Internal NegBin dispersion estimator |
| `estimate_theta_moments` | fn | NO | util | LOW | Internal NegBin dispersion estimator |

### Inference / HC Robust Errors

**Source:** `anofox-regression-0.5.13/src/inference/mod.rs`

| Item | Kind | Exposed? | Target Surface | Priority | Notes |
|------|------|----------|----------------|----------|-------|
| `HcType` | enum | YES | expression (parameter) + PyModel | MEDIUM | Exposed today: `ols_summary` accepts an `hc_type` parameter, and `OLS.hc_inference(x, hc_type)` accepts it. `parse_hc_type` maps the string. Fully reachable for OLS. |
| `HcInference` | struct | **PARTIAL (OLS only)** | expression (output) + PyModel | HIGH | HC output IS surfaced for OLS: `ols_summary` with `hc_type` routes through `HcInference` and populates `std_error`/`statistic`/`p_value` (`src/expressions/regression.rs:2640-2690`); `OLS.hc_inference()` returns a dict of HC SEs (`src/pymodels/py_ols.rs:371-405`). **Gap (REGR-04) = extend HC output to non-OLS regressors (Ridge/WLS/GLM…) and/or broaden the expression surface — NOT implement from scratch.** |
| `HcResult` | struct | PARTIAL (OLS path) | expression (internal) | LOW | Helper struct within the already-wired OLS HC inference path |
| `HcInterceptInference` | struct | PARTIAL (OLS path) | expression (output) | MEDIUM | Intercept-specific HC inference; reachable via the OLS HC path, gap = non-OLS coverage |
| `compute_hc_inference` | fn | PARTIAL (OLS path) | expression (backing fn) | MEDIUM | Backing fn already invoked by the OLS `hc_type` path; gap = wire it for other regressors |
| `compute_hc_standard_errors` | fn | PARTIAL (OLS path) | expression (backing fn) | LOW | Lower-level SEs-only path; used within the OLS HC route |
| `CoefficientInference` | struct | NO | util | LOW | Internal inference struct used by OLS/Ridge summary; not user-facing directly |
| `compute_prediction_intervals` | fn | NO | util | LOW | Internal prediction interval computation; already surfaced via `*_predict` expressions |
| `compute_xtx_inverse*` (8 variants) | fn | NO | util | LOW | Matrix inversion helpers; internal |

### Solvers — Already Exposed

**Source:** `src/expressions/regression.rs`, `src/pymodels/mod.rs`, `python/polars_statistics/__init__.py`

| Solver | Kind | Exposed? | Surface | Notes |
|--------|------|----------|---------|-------|
| `OlsRegressor` / `FittedOls` | regressor | YES | expression (`ols`, `ols_summary`, `ols_predict`, `ols_formula*`) + PyModel (`OLS`) | |
| `RidgeRegressor` / `FittedRidge` | regressor | YES | expression + PyModel (`Ridge`) | |
| `ElasticNetRegressor` / `FittedElasticNet` | regressor | YES | expression + PyModel (`ElasticNet`) | |
| `WlsRegressor` / `FittedWls` | regressor | YES | expression + PyModel (`WLS`) | |
| `RlsRegressor` / `FittedRls` | regressor | YES | expression + PyModel (`RLS`) | |
| `BlsRegressor` / `FittedBls` | regressor | YES | expression (`bls`, `nnls`) + PyModel (`BLS`) | |
| `HuberRegressor` / `FittedHuber` | regressor | YES | expression + PyModel (`Huber`) | |
| `PlsRegressor` / `FittedPls` | regressor | YES | expression + PyModel (`PLS`) | |
| `QuantileRegressor` / `FittedQuantile` | regressor | YES | expression + PyModel (`Quantile`) | |
| `IsotonicRegressor` / `FittedIsotonic` | regressor | YES | expression + PyModel (`Isotonic`) | |
| `LogisticRegression` / `FittedLogistic` | regressor | YES | expression + PyModel (`Logistic`, `LogisticRegression`) | |
| `BinomialRegressor` / `FittedBinomial` | regressor | YES | expression (`logistic`) + PyModel (`Logistic`) | `BinomialRegressor` is the backing type |
| `PoissonRegressor` / `FittedPoisson` | regressor | YES | expression + PyModel (`Poisson`) | |
| `NegativeBinomialRegressor` / `FittedNegativeBinomial` | regressor | YES | expression + PyModel (`NegativeBinomial`) | |
| `TweedieRegressor` / `FittedTweedie` | regressor | YES | expression + PyModel (`Tweedie`) | |
| `AlmRegressor` / `FittedAlm` | regressor | YES | expression (`alm`, `alm_summary`, `alm_predict`) + PyModel (`ALM`) | |
| `AidClassifier` | classifier | YES | expression (`aid`, `aid_anomalies`) + PyModel (`Aid`) | |
| `LmDynamicRegressor` / `FittedLmDynamic` | regressor | YES | expression (`lm_dynamic`, `lm_dynamic_predict`) + PyModel (`LmDynamic`) | |

### Solvers — NOT YET Exposed (Gaps)

| Solver | Kind | Exposed? | Target Surface | Priority | Notes |
|--------|------|----------|----------------|----------|-------|
| `GammaRegressor` / `FittedGamma` | regressor | **NO** | both | HIGH | Required by REGR-03; Gamma GLM (thin wrapper over Tweedie with var_power=2); `predict_mu`, `predict_eta`, `predict_with_offset`, `converged`, `inner` methods on `FittedGamma` |
| `GlmmRegressor` / `FittedGlmm` / `GlmmRegressorBuilder` | regressor | **NO** | PyModel | HIGH | Required by REGR-01; Gaussian/Poisson/Binomial random-intercept and random-slope GLMM; `fit` (single grouping factor) + `fit_crossed` (multiple factors); requires group column input |
| `PSplineRegressor` / `FittedPSpline` | regressor | **NO** | both | HIGH | Required by REGR-02; P-spline smoother (GAM-style); GCV-selected lambda; single predictor column as input |
| `FactorSummary` | struct | **NO** | PyModel (output from GlmmRegressor) | HIGH | Per-factor random-effects summary returned by `FittedGlmm::factors()`; HIGH priority — see Special-Case Flags |
| `TheilSenRegressor` / `FittedTheilSen` | regressor | **NO** | both | MEDIUM | Robust regression; breakdown point 29.3%; matches sklearn; spatial median of pairwise OLS vectors |
| `RansacRegressor` / `FittedRansac` | regressor | **NO** | both | MEDIUM | RANSAC robust regression |
| `BayesianRidge` / `FittedBayesianRidge` / `BayesianRidgeBuilder` | regressor | **NO** | both | MEDIUM | Empirical-Bayes Ridge; matches sklearn `BayesianRidge` |
| `ArdRegression` / `FittedArd` / `ArdRegressionBuilder` | regressor | **NO** | both | MEDIUM | Automatic Relevance Determination; per-feature precision |
| `LarsRegressor` / `FittedLars` / `LarsRegressorBuilder` / `LarsMethod` | regressor | **NO** | both | MEDIUM | LARS / Lasso-LARS; `LarsMethod` selects LARS vs LASSO path |
| `PassiveAggressiveRegressor` / `FittedPassiveAggressive` / `PaLoss` / `PaState` | regressor | **NO** | PyModel | MEDIUM | Online learning regressor; streaming fit; stateful across calls — does not map cleanly to per-group Polars expressions; see Deferred Scope Questions |
| `MomentAccumulator` | struct | **NO** | PyModel utility | MEDIUM | Streaming suffix-statistics accumulator for OLS/Ridge `fit_from_moments`/`fit_from_accumulator`; large-panel use case; stateful — PyModel only; see Deferred Scope Questions |

### Diagnostics

**Source:** `anofox-regression-0.5.13/src/diagnostics/mod.rs`

| Item | Kind | Exposed? | Target Surface | Priority | Notes |
|------|------|----------|----------------|----------|-------|
| `condition_number` | fn | YES | expression (`condition_number`) | — | |
| `condition_diagnostic` | fn | YES | expression (backing `condition_number_fit`) | — | |
| `classify_condition_number` | fn | YES | expression (backing, internal use) | — | |
| `variance_decomposition_proportions` | fn | YES | expression (in `condition_number_fit` output) | — | |
| `ConditionDiagnostic` | struct | YES | expression (output) | — | |
| `ConditionSeverity` | enum | YES | expression (output) | — | |
| `compute_leverage` | fn | YES | expression (`leverage`) | — | |
| `compute_leverage_with_aliased` | fn | YES | expression (backing, aliased columns) | — | |
| `high_leverage_points` | fn | YES | expression (`high_leverage_points`) | — | |
| `cooks_distance` | fn | YES | expression (`cooks_distance`) | — | |
| `dffits` | fn | YES | expression (`dffits`) | — | |
| `influential_cooks` | fn | YES | expression (`influential_cooks`) | — | |
| `influential_dffits` | fn | YES | expression (`influential_dffits`) | — | |
| `variance_inflation_factor` | fn | YES | expression (`vif`) | — | |
| `generalized_vif` | fn | YES | expression (`generalized_vif`) | — | |
| `high_vif_predictors` | fn | YES | expression (`high_vif_predictors`) | — | |
| `standardized_residuals` | fn | YES | expression (`standardized_residuals`) | — | |
| `studentized_residuals` | fn | YES | expression (`studentized_residuals`) | — | |
| `externally_studentized_residuals` | fn | YES | expression (`externally_studentized_residuals`) | — | |
| `residual_outliers` | fn | YES | expression (`residual_outliers`) | — | |
| `check_binary_separation` | fn | YES | expression (`check_binary_separation`) | — | |
| `check_count_sparsity` | fn | YES | expression (`check_count_sparsity`) | — | |
| `SeparationCheck` | struct | YES | expression (output) | — | |
| `SeparationType` | enum | YES | expression (output) | — | |
| `deviance_residuals` | fn | YES | expression (`logistic_deviance_residuals`, `poisson_deviance_residuals`) | — | Logistic + Poisson |
| `pearson_residuals` | fn | YES | expression (`logistic_pearson_residuals`, `poisson_pearson_residuals`) | — | Logistic + Poisson |
| `working_residuals` | fn | YES | expression (`logistic_working_residuals`, `poisson_working_residuals`) | — | Logistic + Poisson |
| `estimate_dispersion_deviance` | fn | **NO** | expression | MEDIUM | Deviance-based dispersion estimate; not yet wired; needed for GLM diagnostic completeness |
| `estimate_dispersion_pearson` | fn | **NO** | expression | MEDIUM | Pearson-based dispersion estimate; not yet wired |
| `pearson_chi_squared` | fn | **NO** | expression | MEDIUM | GLM goodness-of-fit; wrapper has `pearson_chi_squared_logistic` and `pearson_chi_squared_poisson` but not a general form for Tweedie/NegBin/Gamma |
| `standardized_deviance_residuals` | fn | **NO** | expression | MEDIUM | Standardized deviance residuals for GLMs; not wired |
| `standardized_pearson_residuals` | fn | **NO** | expression | MEDIUM | Standardized Pearson residuals for GLMs; not wired |
| `response_residuals` | fn | **NO** | expression | LOW | Raw `y - mu_hat` residuals; low priority (simple to compute manually) |

### Traits and Utility Exports

| Item | Kind | Exposed? | Target Surface | Priority | Notes |
|------|------|----------|----------------|----------|-------|
| `Regressor` | trait | YES | expression (internal, used in all fit functions) | — | Foundational but not directly user-callable |
| `FittedRegressor` | trait | YES | expression (internal, for `predict`) | — | |
| `OutOfBounds` | enum | YES | PyModel (`IsotonicRegressor` parameter) | — | Exposed as parameter of `PyIsotonic` |
| `Penalty` | enum | YES | PyModel (`LogisticRegression` parameter) | — | |
| `LinkFunction` | enum | YES | expression + PyModel (ALM link) | — | |
| `AlmDistribution` | enum | YES | expression + PyModel (ALM) | — | |
| `AlmLoss` | enum | YES | expression + PyModel (ALM) | — | |
| `AnomalyType` | enum | YES | expression + PyModel (AID) | — | |
| `DemandDistribution` | enum | YES | expression + PyModel (AID) | — | |
| `DemandType` | enum | YES | expression + PyModel (AID) | — | |
| `DistributionParameters` | struct | YES | expression (AID output) | — | |
| `DemandClassification` | struct | YES | expression (AID output) | — | |
| `InformationCriterion` | enum | YES | expression (`LmDynamic` parameter) | — | |
| `ModelSpec` | struct | YES | expression (`LmDynamic` output) | — | |
| `RegressionError` | enum | NO | util | LOW | Internal Rust error; converted to Python exceptions by wrapper |
| `AidClassifierBuilder` | struct | NO | util | LOW | Builder pattern internal; users use `AidClassifier` directly |

### Internal-only Items (not Phase 4 action items)

| Item | Kind | Reason Not a User Gap |
|------|------|----------------------|
| `LambdaScaling` | enum | Internal scaling option for regularization; configured via RegressionOptions |
| `OptionsError` | enum | Error from options builder; converted by wrapper |
| `GlmFamily` | trait | Internal trait; users configure family via regressor type |
| `TweedieFamily`, `BinomialFamily`, `PoissonFamily`, `NegativeBinomialFamily` | struct | Internal GLM family configs; set via regressor parameters |
| `BinomialLink`, `PoissonLink` | enum | Internal link function configs |
| `NaAction`, `NaError`, `NaHandler`, `NaInfo`, `NaResult` | mixed | NA handling internals |
| `estimate_theta_ml`, `estimate_theta_moments` | fn | Internal NegBin dispersion estimators |
| `CoefficientInference` | struct | Internal inference used by OLS/Ridge summary; not user-facing |
| `compute_prediction_intervals` | fn | Internal; already surfaced via `*_predict` expressions |
| `compute_xtx_inverse*` (8 variants) | fn | Matrix inversion helpers; internal |
| `RegressionError` | enum | Internal Rust error type |
| `AidClassifierBuilder` | struct | Builder pattern; users use `AidClassifier` directly |

---

## Known-Candidate Checklist

All 21 known candidates from the CONTEXT.md enumeration, with explicit verdicts.

| Candidate | Present in Crate? | Exposed? | Resolved |
|-----------|------------------|----------|---------|
| `one_way_anova` | YES (statistics 0.4.2) | NO | Gap confirmed — Phase 3 action item |
| `two_way_anova` | YES (statistics 0.4.2) | NO | Gap confirmed — Phase 3 action item |
| `repeated_measures_anova` | YES (statistics 0.4.2) | NO | Gap confirmed — Phase 3 action item |
| `energy_distance_test` | YES (both 1D and nD overloads) | 1D: YES, nD: NO | Partial — nD overload is a gap; see Deferred Scope Questions |
| `GlmmRegressor` | YES (regression 0.5.12+) | NO | Gap confirmed — Phase 4 action item |
| `PSplineRegressor` | YES (regression 0.5.10+) | NO | Gap confirmed — Phase 4 action item |
| Gamma GLM (`GammaRegressor`) | YES (regression 0.5.5+) | NO | Gap confirmed — Phase 4 action item |
| `HcInference` / `HcType` | YES (both) | PARTIAL — HC output IS wired for OLS (`ols_summary` `hc_type` param + `OLS.hc_inference()`) | Partial — reachable for OLS; REGR-04 gap = extend HC to non-OLS regressors, not implement from scratch |
| Cook's distance | YES | YES | No gap — `cooks_distance`, `influential_cooks`, `influential_dffits` all exposed |
| VIF | YES | YES | No gap — `vif`, `high_vif_predictors`, `generalized_vif` all exposed |
| Leverage | YES | YES | No gap — `leverage`, `high_leverage_points` exposed |
| Residual variants | YES | YES | No gap — `standardized_residuals`, `studentized_residuals`, `externally_studentized_residuals`, `residual_outliers` all exposed; GLM residuals (pearson/deviance/working) for Logistic and Poisson also exposed |
| Condition diagnostics | YES | YES | No gap — `condition_number` expression exposed |
| Streaming moments (`MomentAccumulator`) | YES (regression 0.5.9+) | NO | Gap confirmed — Phase 4 MEDIUM priority |
| Theil-Sen | YES (regression 0.5.5+) | NO | Gap confirmed — Phase 4 MEDIUM priority |
| RANSAC | YES (regression 0.5.5+) | NO | Gap confirmed — Phase 4 MEDIUM priority |
| LOWESS | INTERNAL ONLY (not in `solvers/mod.rs` pub use list) | N/A | **Not a user-facing gap** — `lowess_smooth_weights` is an internal helper in `solvers/lowess.rs` used by `LmDynamicRegressor`; no `LowessRegressor` type exists in the pub API |
| Bayesian Ridge | YES (regression 0.5.5+) | NO | Gap confirmed — Phase 4 MEDIUM priority |
| Passive-Aggressive | YES (regression 0.5.5+) | NO | Gap confirmed — Phase 4 MEDIUM priority; PyModel only (stateful) |
| LARS | YES (regression 0.5.5+) | NO | Gap confirmed — Phase 4 MEDIUM priority |
| ARD | YES (regression 0.5.5+) | NO | Gap confirmed — Phase 4 MEDIUM priority |

> Note: ARD (`ArdRegression`) is the 21st candidate, implied by CONTEXT.md's "Bayesian/passive-aggressive/LARS" group — confirmed present and absent.

All 21 known candidates fully resolved.

---

## Special-Case Flags

### (a) ICC Stub — "Exposed-but-Stubbed"

`icc` appears in `python/polars_statistics/__init__.py` and `icc_fit` is registered as a
`#[polars_expr]` in `src/expressions/correlation.rs`. However, the Rust implementation is a
non-functional placeholder:

```rust
// src/expressions/correlation.rs:icc_fit
// ICC requires a 2D matrix structure - this is a simplified placeholder
// The actual implementation would need proper matrix handling
// TODO: Implement proper ICC with matrix input
let estimate = Series::new("estimate".into(), &[f64::NAN]);
// ... all fields return f64::NAN
```

**Classification:** "exposed-but-stubbed" — the binding exists and the Python name is importable,
but calling `ps.icc(...)` returns all-NaN regardless of input.

**This is distinct from a missing-binding gap.** Phase 3 must implement the actual matrix-input
ICC logic — not just register a new expression. The `ICCResult` and `ICCType` types from the
crate are also not yet wired into the output schema.

**Phase 3 action:** Implement proper matrix-input ICC in `icc_fit`, wire `ICCType` as a parameter,
and produce a correctly-typed `ICCResult` output struct. Count as STAT-05.

### (b) FactorSummary — HIGH-priority user-facing output, NOT internal

`FactorSummary` is `pub` in `anofox-regression` and is the return type of
`FittedGlmm::factors() -> &[FactorSummary]`. It contains the per-factor random-effects summary
(factor name, level count, variance, BLUPs) that users need to interpret multi-factor GLMM fits.

**This is NOT an internal struct** despite the "Summary" naming pattern. It is the primary output
type for multi-factor `GlmmRegressor` fits and must be exposed as part of the Phase 4 GLMM wrapper
(REGR-01). Mark it HIGH priority alongside `GlmmRegressor` / `FittedGlmm`.

---

## Deferred Scope Questions (for Phase 3/4 planning)

These four questions were identified during enumeration/verification and carry a recommendation but
have NOT been resolved here — they are deferred as explicit decisions for the Phase 3 and Phase 4 planners.

### Q1: Does STAT-04 require the nD `energy_distance_test` overload, or is the existing 1D wrapper sufficient?

- **What we know:** The existing `energy_distance` expression wraps `energy_distance_test_1d` (scalar
  series). The crate also exports `energy_distance_test` accepting `&[Vec<f64>]` (multiple samples,
  each a Vec). STAT-04 states "user can compute the energy distance test via the Polars expression API."
- **What is unclear:** Whether the intent is to also support the multi-dimensional case.
- **Recommendation (from RESEARCH.md):** Phase 3 planner should clarify. If only 1D is required,
  STAT-04 may already be partially satisfied — the expression exists and needs validation +
  documentation only. If the nD overload is in scope, it adds a moderate new wrapper task.
- **Status:** Deferred to Phase 3 planning.

### Q2: Should `PassiveAggressiveRegressor` and `MomentAccumulator` be PyModel-only or also Polars expressions?

- **What we know:** Both are stateful across calls. `PassiveAggressiveRegressor` has `PaState` for
  `partial_fit` incremental updates. `MomentAccumulator` is designed for streaming batch processing.
  The Polars expression model processes per-group chunks statelessly — stateful streaming does not
  map cleanly.
- **Recommendation (from RESEARCH.md):** `PassiveAggressiveRegressor` → PyModel only.
  `MomentAccumulator` → PyModel utility class (no Polars expression equivalent). Document in Phase 4
  audit table notes.
- **Status:** Deferred to Phase 4 planning. This audit records both as "target surface: PyModel" with
  a note about the stateful streaming rationale.

### Q3: Is the ICC stub fix in scope for Phase 3 as an "unexposed capability"?

- **What we know:** `pl_icc` is registered and `icc` appears in `__init__.py`. The Rust function
  returns all NaN (confirmed by reading `src/expressions/correlation.rs:icc_fit` this session).
- **Recommendation (from RESEARCH.md):** Treat it as an "unexposed capability" — the binding exists
  but produces no output, so the capability is effectively absent. Phase 3 should implement the actual
  matrix-input ICC (real `ICCResult` output with `ICCType` parameter).
- **Status:** Deferred to Phase 3 planning. Recorded in this audit as MEDIUM priority under STAT-05,
  with the "exposed-but-stubbed" special-case flag above.

### Q4: How should REGR-04 (HC inference) be framed, given HC is already reachable for OLS?

- **What we know (corrected during Phase 2 verification):** HC inference is NOT unexposed. It is
  already reachable for OLS via two surfaces: the `ols_summary` expression accepts an `hc_type`
  parameter that routes through `HcInference` and populates `std_error`/`statistic`/`p_value`
  (`src/expressions/regression.rs:2640-2690`), and the `OLS` PyModel exposes
  `.hc_inference(x, hc_type)` returning a dict of HC standard errors (`src/pymodels/py_ols.rs:371-405`).
  `HcType`/`parse_hc_type` are wired. The earlier "NO / not wired" rows in this audit were inaccurate
  and have been corrected to PARTIAL (OLS only).
- **What is unclear:** REGR-04's target scope. Two framings:
  - **Option A (conservative):** REGR-04 = validate + document the existing OLS HC surface only.
  - **Option B (extend):** REGR-04 = extend HC output to non-OLS regressors (Ridge/WLS/GLM…) and/or
    broaden the expression surface beyond `ols_summary`.
- **Recommendation:** Phase 4 planner decides A vs B based on how broadly HC standard errors are
  expected across regressor types. Do NOT plan REGR-04 as "implement HC from scratch" — that work
  already exists for OLS.
- **Status:** Deferred to Phase 4 planning.

---

## Cross-Reference: Requirement to Gap

| Requirement | Gap(s) | Phase | Priority |
|-------------|--------|-------|----------|
| STAT-01 | `one_way_anova` fn + `OneWayAnovaResult` struct | Phase 3 | HIGH |
| STAT-02 | `two_way_anova` fn + `TwoWayAnovaResult` struct | Phase 3 | HIGH |
| STAT-03 | `repeated_measures_anova` fn + `RmAnovaResult` + `SphericityResult` + `CorrectedResult` structs | Phase 3 | HIGH |
| STAT-04 | `energy_distance_test` nD overload (gap confirmed) + Q1: clarify whether existing 1D `energy_distance` expression satisfies requirement | Phase 3 | HIGH (nD gap) / pending Q1 |
| STAT-05 | ICC stub fix (`icc_fit` → real matrix-input implementation; wire `ICCType` + `ICCResult`); `AnovaKind`, `AnovaTableRow`, `CorrectedResult` (shared with STAT-03) | Phase 3 | MEDIUM |
| REGR-01 | `GlmmRegressor` / `FittedGlmm` / `GlmmRegressorBuilder` / `FactorSummary` | Phase 4 | HIGH |
| REGR-02 | `PSplineRegressor` / `FittedPSpline` | Phase 4 | HIGH |
| REGR-03 | `GammaRegressor` / `FittedGamma` | Phase 4 | HIGH |
| REGR-04 | HC inference is ALREADY reachable for OLS (`ols_summary` `hc_type` param + `OLS.hc_inference()`). REGR-04 scope = **extend** HC output to non-OLS regressors (Ridge/WLS/GLM…) and/or broaden the expression surface — NOT implement from scratch. See Deferred Scope Question #4. | Phase 4 | HIGH |
| REGR-05 | GLM dispersion/residual variants: `estimate_dispersion_deviance`, `estimate_dispersion_pearson`, `pearson_chi_squared` (general form), `standardized_deviance_residuals`, `standardized_pearson_residuals` — most diagnostics already exposed | Phase 4 | MEDIUM |
| REGR-06 | `TheilSenRegressor`, `RansacRegressor`, `BayesianRidge` / `ArdRegression`, `LarsRegressor`, `PassiveAggressiveRegressor`, `MomentAccumulator` | Phase 4 | MEDIUM |

---

## Verification

The following automated check was run to confirm this document's completeness:

```
test -f .planning/phases/02-public-api-audit/02-API-AUDIT.md \
  && grep -q "anofox-statistics 0.4.2" .planning/phases/02-public-api-audit/02-API-AUDIT.md \
  && grep -q "anofox-regression 0.5.13" .planning/phases/02-public-api-audit/02-API-AUDIT.md \
  && grep -q "Known-Candidate Checklist" .planning/phases/02-public-api-audit/02-API-AUDIT.md \
  && grep -q "Deferred Scope Questions" .planning/phases/02-public-api-audit/02-API-AUDIT.md \
  && grep -qi "one_way_anova" .planning/phases/02-public-api-audit/02-API-AUDIT.md \
  && grep -qi "GlmmRegressor" .planning/phases/02-public-api-audit/02-API-AUDIT.md \
  && grep -qi "HcInference" .planning/phases/02-public-api-audit/02-API-AUDIT.md
```

Result: PASS (all seven conditions met).
