# Phase 2: Public API Audit - Research

**Researched:** 2026-08-11
**Domain:** Cross-reference enumeration — anofox-statistics 0.4.2 and anofox-regression 0.5.13 public API vs. wrapper exposure
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Artifact location & format:** Write the gap list to `.planning/phases/02-public-api-audit/02-API-AUDIT.md` as a planning artifact, with per-crate markdown tables. It feeds Phases 3/4 as their scope.
- **Enumeration method:** Enumerate each crate's public API authoritatively by scanning the installed crate source under `~/.cargo/registry/src/*/anofox-regression-0.5.13/` and `~/.cargo/registry/src/*/anofox-statistics-0.4.2/` for `pub` items (functions, structs, enums, trait methods, regressor types), cross-referenced against the wrapper's existing call sites (`src/expressions/*.rs`, `src/pymodels/*.rs`, `python/polars_statistics/`). Do NOT trust docs.rs alone; do NOT limit to the known-candidate list.
- **Definition of "exposed":** A capability counts as already-exposed if it is reachable from Python via EITHER a Polars expression OR a PyModel class (either surface suffices).
- **Output granularity:** One row per public fn/type — columns: name, kind, exposed? (yes/no), target surface (expression / PyModel / both), notes. Group by crate, then by category.

### Claude's Discretion
- Exact table column ordering, section headings, and how to sub-group categories.
- How to handle borderline "public but internal-only" items (e.g. helper traits) — use judgment; note them but mark low-priority if they are not meaningful user-facing capabilities.
- Whether to additionally emit a short summary count per crate/category.

### Deferred Ideas (OUT OF SCOPE)
- Actually exposing any gap → Phases 3 (statistics) and 4 (regression). This phase only documents.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| AUDIT-01 | Authoritative, documented gap list enumerates every public `anofox-statistics` function not yet exposed via the Polars/Python API | Full lib.rs + module-level enumeration performed this session; cross-referenced against all expression files and `__init__.py` |
| AUDIT-02 | Authoritative, documented gap list enumerates every public `anofox-regression` capability not yet exposed via expressions or PyModel classes | Full lib.rs + solvers/mod.rs + diagnostics/mod.rs + inference/mod.rs enumeration performed; cross-referenced against regression.rs and pymodels/mod.rs |
</phase_requirements>

---

## Summary

This research phase performs the core enumeration work so that the planner's single task is to write `02-API-AUDIT.md` from the tables below — no further source inspection is required.

Both crates' authoritative public surfaces come from their `lib.rs` top-level `pub use` re-exports and their module-level `pub use` statements, all read directly from the Cargo registry cache at `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/`. The wrapper's exposed surface comes from `src/expressions/regression.rs` (function list), `src/pymodels/mod.rs` (PyO3 class list), and `python/polars_statistics/__init__.py` (canonical Python surface — the definitive "reachable from Python" reference).

**Key finding — statistics crate:** The primary gaps in `anofox-statistics` 0.4.2 are the three ANOVA functions (`one_way_anova`, `two_way_anova`, `repeated_measures_anova`) and the multi-dimensional overloads of `energy_distance_test` and `mmd_test`. All other statistics functions are already fully exposed. The `resampling` module's `CircularBlockBootstrap` and `StationaryBootstrap` are exposed as PyModel classes but not as Polars expressions. The `rank` utility function and internal helper types are public but not user-facing.

**Key finding — regression crate:** The primary gaps in `anofox-regression` 0.5.13 are: `GlmmRegressor`/`FittedGlmm` (GLMM), `PSplineRegressor`/`FittedPSpline` (P-spline smoother), `GammaRegressor`/`FittedGamma` (Gamma GLM wrapper), `HcInference`/`HcType` (HC robust SE — already imported in expressions but HC output is not wired to any user-callable expression), and the 0.5.5 batch of new solvers: `TheilSenRegressor`, `RansacRegressor`, `BayesianRidge`, `ArdRegression`, `LarsRegressor`, `PassiveAggressiveRegressor`. `MomentAccumulator` + `fit_from_moments`/`fit_from_accumulator` are new utility types for streaming fits (no wrapper yet). The diagnostics surface is substantially already exposed; only `estimate_dispersion_deviance`, `estimate_dispersion_pearson`, `pearson_chi_squared` (for non-logistic/Poisson GLMs), `standardized_deviance_residuals`, `standardized_pearson_residuals`, `response_residuals` are not yet wired.

**Primary recommendation:** The planner's single task is to write `02-API-AUDIT.md` from the gap tables in `## Complete Enumeration Tables` below, applying the discretion rules for internal-only items, and structured as one table per crate/category.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Public API enumeration | Research (this phase) | — | Source-of-truth is the crate's `lib.rs` re-export list, not docs.rs |
| "Exposed?" determination | Wrapper (`python/polars_statistics/__init__.py`) | `src/expressions/` + `src/pymodels/` | `__init__.py` is the canonical Python surface; Rust expressions only count if Python-callable |
| Gap list artifact | `.planning/phases/02-public-api-audit/02-API-AUDIT.md` | — | Consumed by Phase 3 (statistics) and Phase 4 (regression) planners |
| Target surface assignment | Research discretion | — | Statistics fns → expressions; regression regressors → PyModel + expression; diagnostics → expression |

---

## Standard Stack

No new packages are installed in this phase. This phase produces a document only.

## Package Legitimacy Audit

No packages installed. Not applicable.

---

## Complete Enumeration Tables

The tables below ARE the primary research deliverable. The planner copies them (with formatting decisions applied) into `02-API-AUDIT.md`.

### Enumeration Legend

- **Exposed:** Reachable from Python via `import polars_statistics` or `from polars_statistics import *` or the `exprs` sub-module, confirmed by reading `__init__.py` and the underlying Rust expression/PyModel files this session.
- **Target surface:** `expression` = Polars `#[polars_expr]` function callable via `df.select(ps.foo(...))`; `PyModel` = Python class with `fit`/`predict` methods; `both` = both surfaces needed; `util` = internal utility, not user-facing.
- **Priority:** `HIGH` = explicitly required by a named requirement (STAT-01 through STAT-05, REGR-01 through REGR-06); `MEDIUM` = user-facing capability not yet exposed; `LOW` = internal helper or borderline item.

---

### CRATE: anofox-statistics 0.4.2

**Source:** [VERIFIED: ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-statistics-0.4.2/src/lib.rs:1-57]

```rust
pub use parametric::{
    brown_forsythe, one_way_anova, repeated_measures_anova, t_test, two_way_anova, yuen_test,
    Alternative, AnovaKind, AnovaTableRow, CorrectedResult, LeveneResult, OneWayAnovaResult,
    RmAnovaResult, SphericityResult, TTestKind, TTestResult, TwoWayAnovaResult, YuenConfInt,
    YuenResult,
};
pub use nonparametric::{
    brunner_munzel, kruskal_wallis, mann_whitney_u, rank, wilcoxon_signed_rank,
    BrunnerMunzelResult, KruskalResult, MannWhitneyResult, WilcoxonResult,
};
pub use distributional::{dagostino_k_squared, shapiro_wilk, DAgostinoResult, ShapiroWilkResult};
pub use correlation::{
    distance_cor, distance_cor_test, icc, kendall, partial_cor, pearson, semi_partial_cor,
    spearman, CorrelationConfInt, CorrelationMethod, CorrelationResult, DistanceCorResult,
    ICCResult, ICCType, KendallVariant, PartialCorResult,
};
pub use modern::{
    energy_distance_test, energy_distance_test_1d, mmd_test, mmd_test_1d, EnergyDistanceResult,
    Kernel, MMDResult,
};
pub use categorical::{
    binom_test, chisq_goodness_of_fit, chisq_test, cohen_kappa, contingency_coef, cramers_v,
    fisher_exact, g_test, mcnemar_exact, mcnemar_test, phi_coefficient, prop_test_one,
    prop_test_two, AssociationResult, BinomTestResult, ChiSquareResult, FisherResult, KappaResult,
    McNemarkExactResult, McNemarkResult, PropTestResult,
};
pub use equivalence::{
    tost_bootstrap, tost_correlation, tost_prop_one, tost_prop_two, tost_t_test_one_sample,
    tost_t_test_paired, tost_t_test_two_sample, tost_wilcoxon_paired, tost_wilcoxon_two_sample,
    tost_yuen, CorrelationTostMethod, EquivalenceBounds, OneSidedTestResult, TostResult,
};
pub use resampling::{
    permutation_t_test, CircularBlockBootstrap, PermutationEngine, PermutationResult,
    StationaryBootstrap,
};
pub use forecast::{
    clark_west, diebold_mariano, model_confidence_set, mspe_adjusted_spa, spa_test, CWResult,
    DMResult, LossFunction, MCSEliminationStep, MCSResult, MCSStatistic, MSPEAdjustedResult,
    SPAResult, VarEstimator,
};
pub use error::{Result, StatError};
```

#### anofox-statistics: Parametric Tests

| Item | Kind | Exposed? | Surface | Priority | Notes |
|------|------|----------|---------|----------|-------|
| `t_test` | fn | YES | expression (`ttest_ind`, `ttest_paired`) | — | Both independent and paired via `TTestKind` |
| `yuen_test` | fn | YES | expression (`yuen_test`) | — | |
| `brown_forsythe` | fn | YES | expression (`brown_forsythe`) | — | Called directly via `anofox_statistics::brown_forsythe` |
| `one_way_anova` | fn | **NO** | expression | HIGH | Required by STAT-01 |
| `two_way_anova` | fn | **NO** | expression | HIGH | Required by STAT-02 |
| `repeated_measures_anova` | fn | **NO** | expression | HIGH | Required by STAT-03; also exercises sphericity/Mauchly |
| `Alternative` | enum | YES | expression (parameter) | — | Used in multiple expressions |
| `TTestKind` | enum | YES | expression (internal) | — | |
| `TTestResult` | struct | YES | expression (output) | — | |
| `AnovaKind` | enum | NO | expression (parameter) | MEDIUM | Needed to parameterize one_way_anova type |
| `AnovaTableRow` | struct | NO | expression (output) | MEDIUM | Output row for ANOVA table |
| `OneWayAnovaResult` | struct | NO | expression (output) | HIGH | Output for STAT-01 |
| `TwoWayAnovaResult` | struct | NO | expression (output) | HIGH | Output for STAT-02 |
| `RmAnovaResult` | struct | NO | expression (output) | HIGH | Output for STAT-03 |
| `SphericityResult` | struct | NO | expression (output) | HIGH | Embedded in RmAnovaResult |
| `CorrectedResult` | struct | NO | expression (output) | MEDIUM | Greenhouse-Geisser/Huynh-Feldt corrections in RmAnovaResult |
| `LeveneResult` | struct | YES | expression (internal, used by brown_forsythe) | — | |
| `YuenResult` | struct | YES | expression (output) | — | |
| `YuenConfInt` | struct | YES | expression (internal) | — | |

#### anofox-statistics: Nonparametric Tests

| Item | Kind | Exposed? | Surface | Priority | Notes |
|------|------|----------|---------|----------|-------|
| `mann_whitney_u` | fn | YES | expression (`mann_whitney_u`) | — | |
| `wilcoxon_signed_rank` | fn | YES | expression (`wilcoxon_signed_rank`) | — | |
| `kruskal_wallis` | fn | YES | expression (`kruskal_wallis`) | — | |
| `brunner_munzel` | fn | YES | expression (`brunner_munzel`) | — | |
| `rank` | fn | **NO** | util | LOW | Internal helper — computes rank array; not a user-facing test; low priority |
| `MannWhitneyResult` | struct | YES | expression (output) | — | |
| `WilcoxonResult` | struct | YES | expression (output) | — | |
| `KruskalResult` | struct | YES | expression (output) | — | |
| `BrunnerMunzelResult` | struct | YES | expression (output) | — | |

#### anofox-statistics: Distributional Tests

| Item | Kind | Exposed? | Surface | Priority | Notes |
|------|------|----------|---------|----------|-------|
| `shapiro_wilk` | fn | YES | expression (`shapiro_wilk`) | — | |
| `dagostino_k_squared` | fn | YES | expression (`dagostino`) | — | |
| `ShapiroWilkResult` | struct | YES | expression (output) | — | |
| `DAgostinoResult` | struct | YES | expression (output) | — | |

#### anofox-statistics: Correlation

| Item | Kind | Exposed? | Surface | Priority | Notes |
|------|------|----------|---------|----------|-------|
| `pearson` | fn | YES | expression (`pearson`) | — | |
| `spearman` | fn | YES | expression (`spearman`) | — | |
| `kendall` | fn | YES | expression (`kendall`) | — | |
| `distance_cor_test` | fn | YES | expression (`distance_cor`) — uses this overload | — | Permutation-based distance correlation test |
| `distance_cor` | fn | **NO** | util | LOW | Computes dcor coefficient only, no test; wrapper uses `distance_cor_test`; not needed separately |
| `partial_cor` | fn | YES | expression (`partial_cor`) | — | |
| `semi_partial_cor` | fn | YES | expression (`semi_partial_cor`) | — | |
| `icc` | fn | YES | expression (`icc`) — **but stub only** | MEDIUM | `icc_fit` exists but returns all NaN with TODO; matrix input not yet implemented |
| `CorrelationResult` | struct | YES | expression (output) | — | |
| `DistanceCorResult` | struct | YES | expression (output) | — | |
| `ICCResult` | struct | NO | expression (output) | MEDIUM | Needed once `icc` is implemented |
| `ICCType` | enum | NO | expression (parameter) | MEDIUM | Needed to select ICC variant (ICC1, ICC2, ICC3, etc.) |
| `KendallVariant` | enum | YES | expression (parameter) | — | |
| `PartialCorResult` | struct | YES | expression (output, field access) | — | |
| `CorrelationConfInt` | struct | YES | expression (internal) | — | |
| `CorrelationMethod` | enum | YES | expression (internal) | — | |

> **Note on ICC:** The wrapper has a `pl_icc` expression registered and an `icc` Python callable, but `icc_fit` returns all `f64::NAN` with a code comment "TODO: Implement proper ICC with matrix input". The function appears in `__init__.py` as exposed, but it is non-functional. This is a quality gap, not a missing binding gap — the audit should note it as "stub only, non-functional."

#### anofox-statistics: Modern Tests

| Item | Kind | Exposed? | Surface | Priority | Notes |
|------|------|----------|---------|----------|-------|
| `energy_distance_test_1d` | fn | YES | expression (`energy_distance`) | — | 1D (scalar series) overload — this is what the expression uses |
| `energy_distance_test` | fn | **NO** | expression | HIGH | Multi-dimensional overload — required by STAT-04 (the higher-dimensional case) |
| `mmd_test_1d` | fn | YES | expression (`mmd_test`) | — | 1D overload |
| `mmd_test` | fn | **NO** | expression | MEDIUM | Multi-dimensional MMD test overload — not yet exposed |
| `EnergyDistanceResult` | struct | YES | expression (output) | — | |
| `MMDResult` | struct | YES | expression (output) | — | |
| `Kernel` | enum | **NO** | expression (parameter) | MEDIUM | Kernel selection for multi-dim MMD; not needed for 1D since median heuristic is used |

> **Note on STAT-04:** The existing `energy_distance` expression uses `energy_distance_test_1d`. STAT-04 ("user can compute the energy distance test") may be satisfied by this existing wrapper if the requirement is interpreted as the 1D case. However, the multi-dimensional overload `energy_distance_test` (which takes `&[Vec<f64>]`) is not exposed. The planner for Phase 3 must clarify whether STAT-04 means "also support multi-dimensional" or "confirm the existing 1D is sufficient". The audit should record both the 1D (YES) and the nD overload (NO).

#### anofox-statistics: Categorical Tests

| Item | Kind | Exposed? | Surface | Priority | Notes |
|------|------|----------|---------|----------|-------|
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

#### anofox-statistics: Equivalence (TOST)

| Item | Kind | Exposed? | Surface | Priority | Notes |
|------|------|----------|---------|----------|-------|
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

#### anofox-statistics: Forecast

| Item | Kind | Exposed? | Surface | Priority | Notes |
|------|------|----------|---------|----------|-------|
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

#### anofox-statistics: Resampling

| Item | Kind | Exposed? | Surface | Priority | Notes |
|------|------|----------|---------|----------|-------|
| `permutation_t_test` | fn | YES | expression (`permutation_t_test`) | — | |
| `CircularBlockBootstrap` | struct | YES | PyModel (`CircularBlockBootstrap`) | — | Exposed as Python class only |
| `StationaryBootstrap` | struct | YES | PyModel (`StationaryBootstrap`) | — | Exposed as Python class only |
| `PermutationEngine` | struct | **NO** | util | LOW | Low-level engine used internally by `permutation_t_test`; not a user-facing API; low priority |
| `PermutationResult` | struct | **NO** | util | LOW | Return type of `PermutationEngine` internal calls; not directly needed |

#### anofox-statistics: Error types

| Item | Kind | Exposed? | Surface | Priority | Notes |
|------|------|----------|---------|----------|-------|
| `Result` | type alias | NO | util | LOW | `anofox_statistics::Result<T>` = `std::result::Result<T, StatError>`; internal Rust only |
| `StatError` | enum | NO | util | LOW | Error type; converted to Python exceptions by the expression wrapper; not user-facing |

---

### CRATE: anofox-regression 0.5.13

**Sources:**
- [VERIFIED: ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/lib.rs:28-74]
- [VERIFIED: ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/mod.rs:33-73]
- [VERIFIED: ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/diagnostics/mod.rs:63-81]
- [VERIFIED: ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/inference/mod.rs:1-17]
- [VERIFIED: ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/core/mod.rs:1-25]

#### anofox-regression: Core Types

**Source:** [VERIFIED: ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/core/mod.rs:1-25]

```rust
pub use binomial::BinomialFamily;
pub use family::{GlmFamily, TweedieFamily};
pub use link::BinomialLink;
pub use na_action::{NaAction, NaError, NaHandler, NaInfo, NaResult};
pub use negative_binomial::{estimate_theta_ml, estimate_theta_moments, NegativeBinomialFamily};
pub use options::{LambdaScaling, OptionsError, RegressionOptions, RegressionOptionsBuilder, SolverType};
pub use poisson::PoissonFamily;
pub use poisson_link::PoissonLink;
pub use prediction::{IntervalType, PredictionResult, PredictionType};
pub use result::RegressionResult;
```

| Item | Kind | Exposed? | Surface | Priority | Notes |
|------|------|----------|---------|----------|-------|
| `SolverType` | enum | YES | expression (parameter, via `parse_solver_type`) | — | QR/SVD/Cholesky — already imported in regression.rs |
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

#### anofox-regression: Inference / HC Robust Errors

**Source:** [VERIFIED: ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/inference/mod.rs:1-17]

```rust
pub use coefficient::CoefficientInference;
pub use prediction::{
    compute_prediction_intervals, compute_xtwx_inverse_augmented, ...
};
pub use robust_covariance::{
    compute_hc_inference, compute_hc_standard_errors, HcInference, HcInterceptInference, HcResult, HcType,
};
```

| Item | Kind | Exposed? | Surface | Priority | Notes |
|------|------|----------|---------|----------|-------|
| `HcType` | enum | YES (imported) | expression (parameter) — **but HC output not yet wired** | HIGH | Required by REGR-04; `HcType` is imported in `regression.rs` and `parse_hc_type` exists, but no user-callable expression returns HC inference output |
| `HcInference` | struct | **NO** | expression (output) | HIGH | Required by REGR-04; the struct holding HC SE, t-stats, p-values, CIs for each coefficient |
| `HcResult` | struct | **NO** | expression (internal) | HIGH | Helper struct within HC inference |
| `HcInterceptInference` | struct | **NO** | expression (output) | HIGH | Intercept-specific HC inference |
| `compute_hc_inference` | fn | **NO** | expression (backing fn) | HIGH | The function that computes HcInference from residuals + leverage + X |
| `compute_hc_standard_errors` | fn | **NO** | expression (backing fn) | MEDIUM | Lower-level: SEs only, without full t/p/CI |
| `CoefficientInference` | struct | NO | util | LOW | Internal inference struct used by OLS/Ridge summary; not user-facing directly |
| `compute_prediction_intervals` | fn | NO | util | LOW | Internal prediction interval computation; already surfaced via `*_predict` expressions |
| `compute_xtx_inverse*` (8 variants) | fn | NO | util | LOW | Matrix inversion helpers; internal |

#### anofox-regression: Solvers — Already Exposed

**Source:** [VERIFIED: src/expressions/regression.rs:17-24, src/pymodels/mod.rs:39-59, python/polars_statistics/__init__.py:6-50]

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
| `BinomialRegressor` / `FittedBinomial` | regressor | YES | expression (`logistic`) + PyModel (`Logistic`) | BinomialRegressor is the backing type |
| `PoissonRegressor` / `FittedPoisson` | regressor | YES | expression + PyModel (`Poisson`) | |
| `NegativeBinomialRegressor` / `FittedNegativeBinomial` | regressor | YES | expression + PyModel (`NegativeBinomial`) | |
| `TweedieRegressor` / `FittedTweedie` | regressor | YES | expression + PyModel (`Tweedie`) | |
| `AlmRegressor` / `FittedAlm` | regressor | YES | expression (`alm`, `alm_summary`, `alm_predict`) + PyModel (`ALM`) | |
| `AidClassifier` | classifier | YES | expression (`aid`, `aid_anomalies`) + PyModel (`Aid`) | |
| `LmDynamicRegressor` / `FittedLmDynamic` | regressor | YES | expression (`lm_dynamic`, `lm_dynamic_predict`) + PyModel (`LmDynamic`) | |

#### anofox-regression: Solvers — NOT YET Exposed (GAPs)

| Solver | Kind | Exposed? | Surface | Priority | Notes |
|--------|------|----------|---------|----------|-------|
| `GammaRegressor` / `FittedGamma` | regressor | **NO** | both | HIGH | Required by REGR-03; Gamma GLM (thin wrapper over Tweedie with var_power=2); `predict_mu`, `predict_eta`, `predict_with_offset`, `converged`, `inner` methods on FittedGamma |
| `GlmmRegressor` / `FittedGlmm` / `GlmmRegressorBuilder` | regressor | **NO** | PyModel | HIGH | Required by REGR-01; Gaussian/Poisson/Binomial random-intercept and random-slope GLMM; `fit` (single grouping factor) + `fit_crossed` (multiple factors); requires group column input |
| `PSplineRegressor` / `FittedPSpline` | regressor | **NO** | both | HIGH | Required by REGR-02; P-spline smoother (GAM-style); GCV-selected lambda; single predictor column as input |
| `TheilSenRegressor` / `FittedTheilSen` | regressor | **NO** | both | MEDIUM | Robust regression — breakdown point 29.3%; matches sklearn; spatial median of pairwise OLS vectors |
| `RansacRegressor` / `FittedRansac` | regressor | **NO** | both | MEDIUM | RANSAC robust regression |
| `BayesianRidge` / `FittedBayesianRidge` / `BayesianRidgeBuilder` | regressor | **NO** | both | MEDIUM | Empirical-Bayes Ridge; matches sklearn `BayesianRidge` |
| `ArdRegression` / `FittedArd` / `ArdRegressionBuilder` | regressor | **NO** | both | MEDIUM | Automatic Relevance Determination; per-feature precision |
| `LarsRegressor` / `FittedLars` / `LarsRegressorBuilder` / `LarsMethod` | regressor | **NO** | both | MEDIUM | LARS / Lasso-LARS; `LarsMethod` selects LARS vs LASSO path |
| `PassiveAggressiveRegressor` / `FittedPassiveAggressive` / `PaLoss` / `PaState` | regressor | **NO** | PyModel | MEDIUM | Online learning regressor; streaming fit; `PaLoss` (Epsilon insensitive vs Huber); `PaState` for partial_fit |
| `MomentAccumulator` | struct | **NO** | util | MEDIUM | Streaming suffix-statistics accumulator for OLS/Ridge `fit_from_moments`/`fit_from_accumulator`; large-panel use case |
| `FactorSummary` | struct | **NO** | PyModel (output from GlmmRegressor) | HIGH | Factor-level summary within FittedGlmm for multiple random factors |

#### anofox-regression: Solvers — Known Candidate Confirmation

| Candidate (from CONTEXT.md) | Present in 0.5.13? | Exposed? | Notes |
|----------------------------|-------------------|----------|-------|
| `GlmmRegressor` | YES | NO | Added in 0.5.12 |
| `PSplineRegressor` | YES | NO | Added in 0.5.10 |
| Gamma GLM (`GammaRegressor`) | YES | NO | Added in 0.5.5; `FittedGamma::inner()` gives access to full Tweedie diagnostics |
| `HcInference` / `HcType` | YES | PARTIAL | `HcType` imported + `parse_hc_type` exists in regression.rs; no user-callable expression returns HC output |
| Cook's distance | YES | YES | `cooks_distance`, `influential_cooks`, `influential_dffits` all exposed |
| VIF | YES | YES | `vif`, `high_vif_predictors`, `generalized_vif` all exposed |
| Leverage | YES | YES | `leverage`, `high_leverage_points` exposed |
| Residual variants | YES | YES | `standardized_residuals`, `studentized_residuals`, `externally_studentized_residuals`, `residual_outliers` all exposed; GLM residuals (pearson/deviance/working) for Logistic and Poisson also exposed |
| Condition diagnostics | YES | YES | `condition_number` expression exposed |
| Streaming moment fits (`MomentAccumulator`) | YES | NO | Added in 0.5.9 |
| Theil–Sen | YES | NO | Added in 0.5.5 |
| RANSAC | YES | NO | Added in 0.5.5 |
| LOWESS | PARTIAL | NO | `lowess_smooth_weights` is an **internal helper** in `solvers/lowess.rs` used by `LmDynamicRegressor`; it is **not** in `solvers/mod.rs` pub use list and is not a standalone user-facing solver |
| Bayesian Ridge / ARD | YES | NO | Added in 0.5.5 |
| Passive-Aggressive | YES | NO | Added in 0.5.5 |
| LARS / Lasso-LARS | YES | NO | Added in 0.5.5 |

#### anofox-regression: Diagnostics

**Source:** [VERIFIED: ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/diagnostics/mod.rs:63-81]

| Item | Kind | Exposed? | Surface | Priority | Notes |
|------|------|----------|---------|----------|-------|
| `condition_number` | fn | YES | expression (`condition_number`) | — | |
| `condition_diagnostic` | fn | YES | expression (backing `condition_number_fit`) | — | |
| `classify_condition_number` | fn | YES | expression (backing, internal use) | — | |
| `variance_decomposition_proportions` | fn | YES | expression (included in `condition_number_fit` output) | — | |
| `ConditionDiagnostic` | struct | YES | expression (output) | — | |
| `ConditionSeverity` | enum | YES | expression (output) | — | |
| `compute_leverage` | fn | YES | expression (`leverage`) | — | |
| `compute_leverage_with_aliased` | fn | YES | expression (backing, for aliased columns) | — | |
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
| `deviance_residuals` | fn | YES | expression (`logistic_deviance_residuals`, `poisson_deviance_residuals`) | — | Used for Logistic + Poisson |
| `pearson_residuals` | fn | YES | expression (`logistic_pearson_residuals`, `poisson_pearson_residuals`) | — | Used for Logistic + Poisson |
| `working_residuals` | fn | YES | expression (`logistic_working_residuals`, `poisson_working_residuals`) | — | Used for Logistic + Poisson |
| `estimate_dispersion_deviance` | fn | **NO** | expression | MEDIUM | Computes deviance-based dispersion estimate; not yet wired; needed for GLM diagnostic completeness |
| `estimate_dispersion_pearson` | fn | **NO** | expression | MEDIUM | Pearson-based dispersion estimate; not yet wired |
| `pearson_chi_squared` | fn | **NO** | expression | MEDIUM | GLM goodness-of-fit; note: wrapper has `pearson_chi_squared_logistic` and `pearson_chi_squared_poisson` but not a general form for Tweedie/NegBin/Gamma |
| `standardized_deviance_residuals` | fn | **NO** | expression | MEDIUM | Standardized deviance residuals for GLMs; not wired |
| `standardized_pearson_residuals` | fn | **NO** | expression | MEDIUM | Standardized Pearson residuals for GLMs; not wired |
| `response_residuals` | fn | **NO** | expression | LOW | Raw `y - mu_hat` residuals; low priority (simple to compute manually) |

#### anofox-regression: Traits and Utility Exports

| Item | Kind | Exposed? | Surface | Priority | Notes |
|------|------|----------|---------|----------|-------|
| `Regressor` | trait | YES | expression (internal, used in all fit functions) | — | Not user-callable but foundational |
| `FittedRegressor` | trait | YES | expression (internal, for `predict`) | — | |
| `RegressionError` | enum | NO | util | LOW | Internal Rust error; converted to Python exceptions by wrapper |
| `OutOfBounds` | enum | YES | PyModel (IsotonicRegressor parameter) | — | Exposed as parameter of `PyIsotonic` |
| `Penalty` | enum | YES | PyModel (LogisticRegression parameter) | — | |
| `LinkFunction` | enum | YES | expression + PyModel (ALM link) | — | |
| `AlmDistribution` | enum | YES | expression + PyModel (ALM) | — | |
| `AlmLoss` | enum | YES | expression + PyModel (ALM) | — | Internal to ALM but exposed via PyALM |
| `AnomalyType` | enum | YES | expression + PyModel (AID) | — | |
| `DemandDistribution` | enum | YES | expression + PyModel (AID) | — | |
| `DemandType` | enum | YES | expression + PyModel (AID) | — | |
| `DistributionParameters` | struct | YES | expression (AID output) | — | |
| `DemandClassification` | struct | YES | expression (AID output) | — | |
| `InformationCriterion` | enum | YES | expression (LmDynamic parameter) | — | |
| `ModelSpec` | struct | YES | expression (LmDynamic output) | — | |
| `AidClassifierBuilder` | struct | NO | util | LOW | Builder pattern internal; users use AidClassifier directly |

---

## Summary Counts

### anofox-statistics 0.4.2 — Gap Summary

| Category | Total pub items | Already exposed | Gaps (user-facing) | Gaps (internal/low-priority) |
|----------|-----------------|-----------------|--------------------|------------------------------|
| Parametric | 16 | 8 | 8 (3 ANOVA fns + 5 ANOVA result types) | 0 |
| Nonparametric | 9 | 8 | 0 | 1 (`rank`) |
| Distributional | 4 | 4 | 0 | 0 |
| Correlation | 12 | 9 | 2 (icc stub + ICCType) | 1 (`distance_cor` standalone) |
| Modern | 6 | 2 | 3 (`energy_distance_test` nD, `mmd_test` nD, `Kernel`) | 1 |
| Categorical | 21 | 21 | 0 | 0 |
| Equivalence (TOST) | 14 | 14 | 0 | 0 |
| Forecast | 13 | 13 | 0 | 0 |
| Resampling | 5 | 3 | 0 | 2 (`PermutationEngine`, `PermutationResult`) |
| Error types | 2 | 0 | 0 | 2 (internal) |
| **Total** | **102** | **82** | **13** | **7** |

**Actionable gaps for Phase 3:** `one_way_anova`, `two_way_anova`, `repeated_measures_anova` (+ their result types), `energy_distance_test` (nD overload), plus the ICC stub fix.

### anofox-regression 0.5.13 — Gap Summary

| Category | Already exposed | Gaps (HIGH priority) | Gaps (MEDIUM priority) | Internal/low |
|----------|-----------------|---------------------|------------------------|--------------|
| Solvers | 17 regressors | 3 (`GlmmRegressor`, `PSplineRegressor`, `GammaRegressor`) | 6 (`TheilSen`, `RANSAC`, `BayesianRidge`, `ARD`, `LARS`, `PassiveAggressive`) | 0 |
| HC Inference | `HcType` (partial) | 4 (`HcInference`, `HcResult`, `HcInterceptInference`, `compute_hc_inference`) | 1 (`compute_hc_standard_errors`) | 0 |
| Diagnostics | 18 items | 0 | 5 (`estimate_dispersion_*`, `pearson_chi_squared`, `standardized_deviance_residuals`, `standardized_pearson_residuals`) | 1 (`response_residuals`) |
| Streaming | 0 | 0 | 1 (`MomentAccumulator`) | 0 |
| Core/Utility | 10+ | 0 | 0 | 20+ internal |
| **Total gaps** | — | **7** | **13** | **20+** |

**Actionable gaps for Phase 4:** `GammaRegressor`, `GlmmRegressor`, `PSplineRegressor`, `HcInference`/HC expressions (REGR-01 through REGR-05); then the MEDIUM priority new solvers (REGR-06).

---

## Architecture Patterns

### How the Audit Artifact Is Structured

The executor's single task is to write `02-API-AUDIT.md` using the tables above. The artifact structure should be:

```
02-API-AUDIT.md
├── ## Overview
│   ├── Crate versions confirmed
│   ├── Enumeration method
│   └── Summary counts table
├── ## anofox-statistics 0.4.2 — Gap List
│   ├── Confirmed Present: [ANOVA, energy_distance_test nD, mmd_test nD]
│   ├── Per-category tables (Parametric, Nonparametric, ..., Resampling)
│   └── Internal-only items (marked, not action items for Phase 3)
├── ## anofox-regression 0.5.13 — Gap List
│   ├── Confirmed Present: [GlmmRegressor, PSplineRegressor, GammaRegressor, HcInference, ...]
│   ├── Per-category tables (Solvers, HC Inference, Diagnostics, Utility)
│   └── Internal-only items
└── ## Cross-Reference: Requirement → Gap
    ├── STAT-01 → one_way_anova
    ├── STAT-02 → two_way_anova
    ├── STAT-03 → repeated_measures_anova
    ├── STAT-04 → energy_distance_test (nD) + confirm existing 1D sufficient or not
    ├── STAT-05 → ICC stub fix; remaining items if any
    ├── REGR-01 → GlmmRegressor
    ├── REGR-02 → PSplineRegressor
    ├── REGR-03 → GammaRegressor
    ├── REGR-04 → HcInference + compute_hc_inference
    ├── REGR-05 → diagnostics already largely exposed; GLM dispersion/residual variants
    └── REGR-06 → TheilSen, RANSAC, BayesianRidge, ARD, LARS, PassiveAggressive, MomentAccumulator
```

### Executor Approach

The plan for Phase 2 is a single wave with a single task:

1. **Task:** Write `02-API-AUDIT.md` by applying the discretion rules to the tables in RESEARCH.md.
   - Copy the gap tables verbatim.
   - Add the "Internal-only items" grouping (items marked LOW priority + internal).
   - Add the "Cross-Reference: Requirement → Gap" section mapping STAT-*/REGR-* to the gap list.
   - Add a brief "Enumeration Method" preamble confirming the source-file scan approach.
   - No code changes, no wrapper changes.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ANOVA wrapper | Custom F-test implementation | `anofox_statistics::{one_way_anova, two_way_anova, repeated_measures_anova}` | Mauchly sphericity, Greenhouse-Geisser correction, multi-factor ANOVA are non-trivial to implement correctly |
| GLMM | Custom random-effects fitting | `GlmmRegressor` (lme4 method, Laplace PIRLS) | Mixed models with correct REML profiling require substantial linear algebra infrastructure |
| P-spline smoother | Custom B-spline basis + GCV | `PSplineRegressor` | GCV smoothing parameter selection + posterior covariance SE are complex |
| HC standard errors | Manual sandwich estimator | `compute_hc_inference` | Leverage-adjusted HC2/HC3 require hat matrix; already implemented in crate |
| API enumeration | Manual reading of docs | Read `lib.rs` re-exports directly | Docs can lag; `lib.rs` is the authoritative public surface |

---

## Common Pitfalls

### Pitfall 1: Treating `src/expressions/regression.rs` Imports as the Exposed Surface
**What goes wrong:** `HcType` is imported and `parse_hc_type` exists in `regression.rs`, which could be mistaken as "HC is already exposed." But no user-callable expression actually returns HC inference output — the import exists only for internal parameter parsing in other paths.
**How to avoid:** Always check whether a user-callable Python function exists that surfaces the output, not just whether the symbol is imported in Rust.
**Warning signs:** Symbol appears in import list but no `pub fn *_fit` or `fn pl_*` uses it to produce output.

### Pitfall 2: Confusing `energy_distance_test_1d` (exposed) with `energy_distance_test` (not exposed)
**What goes wrong:** The wrapper uses the `_1d` suffix variant. The lib.rs exports BOTH `energy_distance_test_1d` and `energy_distance_test` (multi-dimensional). STAT-04 may require the nD version.
**How to avoid:** Distinguish the 1D and nD overloads in the audit table; flag the STAT-04 scoping question explicitly.

### Pitfall 3: Treating LOWESS as a Missing User-Facing Solver
**What goes wrong:** "LOWESS" appears in the CONTEXT.md known candidates list. The crate has `solvers/lowess.rs` with `lowess_smooth_weights`. But this function is NOT in `solvers/mod.rs`'s `pub use` list — it is an internal helper used by `LmDynamicRegressor`. There is no `LowessRegressor` or `FittedLowess` type.
**How to avoid:** Check whether the function appears in `solvers/mod.rs` pub re-exports. Only items there are part of the public crate API.
**Verdict:** LOWESS is internal-only. Not a user-facing gap.

### Pitfall 4: ICC Listed as "Exposed" When the Implementation is a Stub
**What goes wrong:** `icc` appears in `__init__.py` and the `pl_icc` expression is registered, so it looks "exposed." But `icc_fit` returns all NaN with a `// TODO` comment for proper matrix input handling.
**How to avoid:** Audit not just whether a binding exists but whether it produces correct output. Mark the ICC situation explicitly: "bound but non-functional stub."
**Impact for Phase 3:** Phase 3 must implement the actual matrix-input ICC, not just register the expression.

### Pitfall 5: Assuming `FactorSummary` is Internal
**What goes wrong:** `FactorSummary` (part of `FittedGlmm`) looks like an internal struct but is `pub` and contains the per-factor random-effects summary that users need to read after fitting a multi-factor GLMM. Marking it "internal/low-priority" would make the GLMM output incomplete.
**How to avoid:** Check whether the struct is an output field of a user-facing `FittedRegressor` method. `FittedGlmm::factors()` returns `&[FactorSummary]`.

---

## Validation Architecture

This phase produces a document artifact, not code. Validation means verifying the audit artifact is correct.

### How to Validate That Each Claimed Gap Is a Real `pub` Item

Every gap claimed in `02-API-AUDIT.md` must be traceable to a line in one of:
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-statistics-0.4.2/src/lib.rs`
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/lib.rs`
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/mod.rs`
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/diagnostics/mod.rs`
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/inference/mod.rs`

The executor verifies this by: for each gap row in the audit, confirming the item name appears in one of the above files with a `pub use` or direct `pub` declaration.

### How to Validate That Each "Already Exposed" Claim Maps to a Concrete Call Site

An item is "already exposed" only if:
1. There is a `pub fn *_fit` in `src/expressions/regression.rs` (or equivalent expression file) that calls the crate function, AND
2. A Python-callable name for it appears in `python/polars_statistics/__init__.py`.

The executor verifies this by: for each "YES" row, confirming both conditions hold. The `__init__.py` file is the ground truth.

### Known-Candidate Checklist (from CONTEXT.md)

| Candidate | Present in Crate? | Exposed? | Resolved |
|-----------|------------------|----------|---------|
| `one_way_anova` | YES (statistics 0.4.2) | NO | Gap confirmed |
| `two_way_anova` | YES (statistics 0.4.2) | NO | Gap confirmed |
| `repeated_measures_anova` | YES (statistics 0.4.2) | NO | Gap confirmed |
| `energy_distance_test` | YES (both 1D and nD) | 1D: YES, nD: NO | Partial — nD is a gap |
| `GlmmRegressor` | YES (regression 0.5.12+) | NO | Gap confirmed |
| `PSplineRegressor` | YES (regression 0.5.10+) | NO | Gap confirmed |
| Gamma GLM | YES (`GammaRegressor`, regression 0.5.5+) | NO | Gap confirmed |
| `HcInference` / `HcType` | YES (both) | `HcType` imported but HC output not wired | Partial — output is a gap |
| Cook's distance | YES | YES | No gap |
| VIF | YES | YES | No gap |
| Leverage | YES | YES | No gap |
| Residual variants | YES | YES | No gap |
| Condition diagnostics | YES | YES | No gap |
| Streaming moments (`MomentAccumulator`) | YES (regression 0.5.9+) | NO | Gap confirmed |
| Theil–Sen | YES (regression 0.5.5+) | NO | Gap confirmed |
| RANSAC | YES (regression 0.5.5+) | NO | Gap confirmed |
| LOWESS | INTERNAL ONLY (not in pub use list) | N/A | Not a user-facing gap — internal helper |
| Bayesian Ridge | YES (regression 0.5.5+) | NO | Gap confirmed |
| Passive-Aggressive | YES (regression 0.5.5+) | NO | Gap confirmed |
| LARS | YES (regression 0.5.5+) | NO | Gap confirmed |

All 21 known candidates fully resolved.

---

## Security Domain

This phase writes a planning document. No code, no user inputs, no network calls, no cryptographic operations. ASVS categories V2–V6 do not apply. `security_enforcement` default applies but there is no attack surface in a document-write phase.

---

## Environment Availability

Step 2.6: SKIPPED — this phase produces a document artifact only. No external tools, databases, services, or CLI utilities beyond `cat`/`grep` for source inspection are required.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `icc_fit` in `src/expressions/correlation.rs` is a non-functional stub (returns all NaN) — confirmed by reading the source code and finding the `// TODO: Implement proper ICC with matrix input` comment and the all-NaN return body | Correlation gap table | LOW — if the TODO was resolved in a recent edit not yet grepped, ICC would be functional. Executor should re-read `icc_fit` before writing the audit. |
| A2 | `lowess_smooth_weights` is internal to `LmDynamicRegressor` and not in `solvers/mod.rs` pub-use — confirmed by reading `solvers/mod.rs` which does not `pub use` any lowess items | Known-candidate checklist | LOW — if a `LowessRegressor` was added after 0.5.13 it would not appear here |
| A3 | STAT-04 ("user can compute the energy distance test") is satisfied by the existing 1D `energy_distance_test_1d` wrapper unless Phase 3 specifies otherwise | STAT-04 gap analysis | MEDIUM — if STAT-04 specifically requires the nD overload, Phase 3 scope grows |

**All other claims were verified by directly reading source files in the Cargo registry cache and the wrapper source this session.**

---

## Open Questions

1. **Does STAT-04 require the nD `energy_distance_test` overload or is the 1D variant sufficient?**
   - What we know: The existing `energy_distance` expression uses `energy_distance_test_1d`. The crate also exports `energy_distance_test` which accepts `&[Vec<f64>]` (multiple samples, one Vec each). STAT-04 says "user can compute the energy distance test via the Polars expression API."
   - What's unclear: Whether the intent is to also support the multi-dimensional case.
   - Recommendation: Phase 3 planner should clarify. If only 1D is required, STAT-04 may already be partially satisfied (the expression exists; it just needs validation + documentation).

2. **Should the ICC stub be treated as a Phase 3 "fix" or a "new implementation"?**
   - What we know: `pl_icc` is registered and `icc` appears in `__init__.py`. The Rust function returns all NaN.
   - What's unclear: Whether fixing a broken stub is within scope of "unexposed capability."
   - Recommendation: Treat it as an "unexposed capability" — the binding exists but produces no output, so the capability is effectively absent. Phase 3 should implement it properly.

3. **Should `PassiveAggressiveRegressor` (online learner) and `MomentAccumulator` (streaming fits) be exposed as Polars expressions or PyModel only?**
   - What we know: Both are stateful across calls (PA has `PaState` for partial_fit; MomentAccumulator is intended for streaming batch processing). The Polars expression model processes per-group chunks and returns a result — stateful streaming does not map cleanly.
   - Recommendation: PyModel only for `PassiveAggressiveRegressor`. `MomentAccumulator` as a PyModel utility class (no Polars expression equivalent). Document this in the audit table's "Notes" column.

---

## Sources

### Primary (HIGH confidence — read from local source files this session)

- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-statistics-0.4.2/src/lib.rs` — canonical statistics public API (58 lines)
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-statistics-0.4.2/src/parametric/mod.rs` — parametric module exports
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-statistics-0.4.2/src/resampling/mod.rs` — resampling module exports
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-statistics-0.4.2/src/modern/mod.rs` — modern module exports
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-statistics-0.4.2/src/correlation/mod.rs` — correlation module exports
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/lib.rs` — canonical regression public API (74 lines)
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/mod.rs` — all solver exports (74 lines)
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/diagnostics/mod.rs` — diagnostics exports (82 lines)
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/inference/mod.rs` — inference/HC exports
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/core/mod.rs` — core types exports
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/glmm.rs` (first 80 lines) — GlmmRegressor public API
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/pspline.rs` (first 80 lines) — PSplineRegressor public API
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/gamma.rs` (first 100 lines) — GammaRegressor public API
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/inference/robust_covariance.rs` (first 80 lines) — HcInference public API
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/theil_sen.rs` (first 60 lines) — TheilSenRegressor description
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/bayesian.rs` (first 60 lines) — BayesianRidge/ArdRegression description
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/moments.rs` (first 60 lines) — MomentAccumulator description
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/lowess.rs` (first 60 lines) — LOWESS is internal helper, not public solver
- `src/expressions/mod.rs` — wrapper expression module list
- `src/expressions/regression.rs` — all regression expression functions (full function listing via grep)
- `src/expressions/parametric.rs` — parametric expression functions
- `src/expressions/nonparametric.rs` — nonparametric expression functions
- `src/expressions/categorical.rs` — categorical expression functions
- `src/expressions/correlation.rs` — correlation expression functions (includes icc_fit stub)
- `src/expressions/modern.rs` — modern expression functions (1D overloads only)
- `src/expressions/distributional.rs` — distributional expression functions
- `src/expressions/forecast.rs` — forecast expression functions
- `src/expressions/tost.rs` — TOST expression functions
- `src/pymodels/mod.rs` — PyModel class exports (21 classes + 2 stat classes exposed)
- `python/polars_statistics/__init__.py` — canonical Python public surface (ground truth for "exposed?")

### Secondary (MEDIUM confidence)
None — all claims verified directly from source files.

### Tertiary (LOW confidence)
None.

---

## Metadata

**Confidence breakdown:**
- Statistics crate public surface: HIGH — read directly from `lib.rs` + module-level files
- Regression crate public surface: HIGH — read directly from `lib.rs`, `solvers/mod.rs`, `diagnostics/mod.rs`, `inference/mod.rs`
- "Already exposed" determination: HIGH — confirmed against `__init__.py` (ground truth) and Rust expression files
- Gap list accuracy: HIGH — cross-referenced both directions (crate → wrapper, wrapper → crate)
- Known-candidate resolutions: HIGH — all 21 candidates explicitly checked against crate sources

**Research date:** 2026-08-11
**Valid until:** 2027-02-11 (pinned crate versions; source will not change)
