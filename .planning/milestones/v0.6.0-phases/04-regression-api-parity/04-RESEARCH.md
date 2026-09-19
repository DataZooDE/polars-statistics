# Phase 4: Regression API Parity - Research

**Researched:** 2026-08-12
**Domain:** Rust/PyO3 wrapper implementation — anofox-regression 0.5.13 gap closure
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **REGR-06 scope = ALL:** expose the full regression gap list. HIGH: `GammaRegressor` (REGR-03),
  `GlmmRegressor` + `FactorSummary` (REGR-01), `PSplineRegressor` (REGR-01/02). MEDIUM (REGR-06):
  the 6 new solvers `TheilSenRegressor`, `RANSACRegressor`, `BayesianRidge`, `ARDRegression`,
  `LARS`/`LarsRegressor`, `PassiveAggressiveRegressor`; `MomentAccumulator`; and the missing
  diagnostics (GLM dispersion estimates, standardized GLM/deviance/Pearson residuals). Use the
  exact crate names from `02-RESEARCH.md`/`02-API-AUDIT.md`.
- **Primary exposure surface = PyModel classes** (fit/predict/summary), following the existing
  `src/pymodels/*.rs` pattern (e.g. `PyOLS`, `PyElasticNet`). Add Polars expressions only where a
  per-group expression is natural (e.g. the diagnostics, which follow the existing diagnostic
  expression pattern). The audit's target-surface column guides each item.
- **REGR-04 HC framing = EXTEND:** HC output is already reachable for OLS (`ols_summary` `hc_type`
  param + `OLS.hc_inference()`). Extend HC output to the other regressors the crate's inference
  path supports (e.g. Ridge/WLS/GLM), via the existing summary/inference mechanism — not a
  from-scratch implementation. (Resolves audit Deferred Q4.)
- **PassiveAggressive & MomentAccumulator = PyModel-only** — stateful/streaming; no Polars
  expression wrapper. (Resolves audit Deferred Q2.)

### Claude's Discretion

- Per-model PyModel file layout and method set (fit/predict/summary/residuals as the crate supports),
  following the closest existing PyModel analog per model type (GLM-family → Poisson/Tweedie analogs;
  robust solvers → Huber/RLS analogs; PSpline → smoother; GLMM → its own).
- Output-struct/dict schemas — follow the crate result types and existing PyModel getter conventions;
  do not invent fields.
- Which diagnostics become expressions vs PyModel methods — follow the existing diagnostics pattern.
- Grouping of capabilities into plans/waves (the planner decides; likely grouped by model family with
  a tracer model wired end-to-end first).
- How broadly HC "extend" reaches — cover the regressors whose crate inference clearly supports HC;
  do not force HC onto models where the crate does not provide it.

### Deferred Ideas (OUT OF SCOPE)

- Comprehensive R-validated tests for all new regressors → Phase 6 (Testing & Validation).
- mkdocs API pages + full user docs → Phase 5 (Documentation).
- Statistics-side parity → done in Phase 3.

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REGR-01 | User can fit a GLMM (`GlmmRegressor`) via a PyModel class | `GlmmRegressor` / `FittedGlmm` / `GlmmRegressorBuilder` / `FactorSummary` fully read this session — exact API documented below |
| REGR-02 | User can fit a P-spline smoother via the API | `PSplineRegressor` / `FittedPSpline` fully read — exact API documented below |
| REGR-03 | User can fit a Gamma GLM via the API | `GammaRegressor` / `FittedGamma` / `GammaRegressorBuilder` fully read — exact API documented below |
| REGR-04 | User can obtain HC robust SEs for regression fits | HC already wired for OLS; `compute_hc_inference` / `compute_hc_standard_errors` read — extension path documented |
| REGR-05 | User can compute regression diagnostics | Five missing GLM diagnostic functions read from `glm_residuals.rs` — expression wiring pattern documented |
| REGR-06 | Every remaining unexposed anofox-regression capability is callable | Six solver files, `MomentAccumulator`, `PaState` all read this session — APIs documented in full |

</phase_requirements>

---

## Summary

Phase 4 closes all anofox-regression 0.5.13 API gaps identified in Phase 2. This is the largest
implementation phase: 12 new PyModel files, 5 new Polars expression functions, and HC extension for
3 additional regressor families.

The audit established two classes of gap: **HIGH priority** (GammaRegressor, GlmmRegressor,
PSplineRegressor — needed for REGR-01/02/03) and **MEDIUM priority** (six additional solvers,
MomentAccumulator, missing GLM diagnostics, HC extension). All have now been read from crate source
in this session; their exact public APIs are recorded verbatim below.

The implementation pattern is fully established by the 40+ existing PyModel classes. The tracer
model is `GammaRegressor` — it is a thin wrapper over `TweedieRegressor` (already exposed), shares
all IRLS numerics, and establishes the PyModel-add pattern before the more complex GLMM and
P-spline models. Shared registration files (`src/pymodels/mod.rs`, `src/lib.rs` `#[pymodule]` block,
`python/polars_statistics/__init__.py`) are touched once per wave to avoid conflicts.

**Primary recommendation:** Implement in 5 waves grouped by model family. Start with GammaRegressor
as the end-to-end tracer (Wave 1), then GlmmRegressor + FactorSummary (Wave 2), PSplineRegressor
(Wave 2), the 5 GLM diagnostic expressions + HC extension (Wave 3), the 4 sklearn-style solvers
(Wave 4), and PassiveAggressive + MomentAccumulator + LARS (Wave 5). Batch all shared-file edits
within each wave into a single registration task.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| New PyModel regressors | `src/pymodels/` (Rust) | `python/polars_statistics/__init__.py` | All state and fitting logic lives in the Rust PyO3 class; Python only imports and re-exports |
| PyModel registration | `src/pymodels/mod.rs` + `src/lib.rs #[pymodule]` | — | Rust module system gates visibility; pymodule block registers each class |
| New Polars expressions (diagnostics) | `src/expressions/regression.rs` | `python/polars_statistics/exprs/regression.py` | Expression functions live in Rust; Python builder wraps the plugin call |
| HC inference extension | `src/pymodels/py_ridge.rs`, `py_wls.rs`, `py_poisson.rs` (etc.) | `src/expressions/regression.rs` | Add `hc_inference()` method to existing PyModel structs that have residuals and leverage accessible |
| numpy ↔ faer bridge | `src/utils/` (ToFaer + IntoNumpy) | — | Existing bridge; no changes needed |
| Python public surface | `python/polars_statistics/__init__.py` | `exprs/regression.py` | New names must be added to `__init__.py` imports for `from polars_statistics import X` to work |

---

## Exact Crate APIs (read this session)

All items below were verified by reading the named source files during this research session.

### REGR-03: GammaRegressor

**Source:** [VERIFIED: ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/gamma.rs:37-231]

```rust
// GammaRegressor — entry point
pub struct GammaRegressor { inner: TweedieRegressor }
impl GammaRegressor {
    pub fn builder() -> GammaRegressorBuilder
}
impl Regressor for GammaRegressor {
    type Fitted = FittedGamma;
    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<FittedGamma, RegressionError>
}

// GammaRegressorBuilder — all public setters
pub struct GammaRegressorBuilder { /* defaults: with_intercept=true, compute_inference=true,
    confidence_level=0.95, max_iterations=25, tolerance=1e-8, lambda=0.0,
    link_power=None, offset=None, error_on_non_convergence=true */ }
impl GammaRegressorBuilder {
    pub fn with_intercept(self, include: bool) -> Self
    pub fn compute_inference(self, compute: bool) -> Self
    pub fn confidence_level(self, level: f64) -> Self
    pub fn max_iterations(self, max_iter: usize) -> Self
    pub fn tolerance(self, tol: f64) -> Self
    pub fn lambda(self, lambda: f64) -> Self
    pub fn link_power(self, q: f64) -> Self
    pub fn offset(self, offset: Col<f64>) -> Self
    pub fn error_on_non_convergence(self, error: bool) -> Self
    pub fn build(self) -> GammaRegressor
}

// FittedGamma — public methods
pub struct FittedGamma { inner: FittedTweedie }
impl FittedGamma {
    pub fn predict_mu(&self, x: &Mat<f64>) -> Col<f64>
    pub fn predict_eta(&self, x: &Mat<f64>) -> Col<f64>
    pub fn predict_with_offset(&self, x: &Mat<f64>, offset: &Col<f64>) -> Col<f64>
    pub fn converged(&self) -> bool
    pub fn inner(&self) -> &FittedTweedie
}
impl FittedRegressor for FittedGamma {
    fn predict(&self, x: &Mat<f64>) -> Col<f64>    // delegates to predict_mu
    fn result(&self) -> &RegressionResult
    fn predict_with_interval(&self, x, interval, level) -> PredictionResult
}
```

**Key facts:**
- `GammaRegressor` is a thin re-skin of `TweedieRegressor` with `var_power=2.0`, log link. All GLM
  numerics (IRLS, coefficient inference, AIC/BIC) are shared with `TweedieRegressor`.
- `predict()` (via `FittedRegressor`) delegates to `predict_mu()` — returns response-scale
  (`exp(η)`) predictions. `predict_eta()` returns the linear predictor.
- `result()` returns `&RegressionResult` which carries: `coefficients`, `intercept`, `residuals`,
  `fitted_values`, `std_errors`, `p_values`, `r_squared`, `aic`, `bic`, `log_likelihood`,
  `n_observations`, `n_parameters`.
- `converged()` is `false` only when `error_on_non_convergence(false)` and IRLS exhausted
  `max_iterations`.
- `inner()` gives full access to `FittedTweedie` for GLM diagnostics (pearson_residuals,
  deviance_residuals, dispersion estimate, etc.).

**Closest PyModel analog:** `src/pymodels/py_tweedie.rs` (PyTweedie). The Gamma wrapper follows an
identical pattern — private fields, `Option<FittedGamma>` state, `fit`/`predict`/`is_fitted`
methods, getters for `coefficients`/`intercept`/`std_errors`/`p_values`/`aic`/`bic`.
[VERIFIED: src/pymodels/py_tweedie.rs:36-236]

**Target surfaces:** PyModel (primary) + Polars expression (secondary; `gamma_fit` paralleling
`poisson_fit`/`tweedie_fit`). Expression can use the same output schema as the Tweedie expression.

---

### REGR-01: GlmmRegressor + FittedGlmm + FactorSummary

**Source:** [VERIFIED: ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/glmm.rs:84-960]

```rust
// Entry points — factory methods produce builders
impl GlmmRegressor {
    pub fn gaussian() -> GlmmRegressorBuilder
    pub fn poisson() -> GlmmRegressorBuilder
    pub fn binomial() -> GlmmRegressorBuilder
    // fit — single grouping factor
    pub fn fit(&self, x: &Mat<f64>, y: &Col<f64>, group: &[usize]) -> Result<FittedGlmm, RegressionError>
    // fit_crossed — multiple crossed/nested random-intercept factors
    pub fn fit_crossed(&self, x: &Mat<f64>, y: &Col<f64>, groups: &[&[usize]]) -> Result<FittedGlmm, RegressionError>
}

// Builder — all public setters (defaults: with_intercept=true, random_intercept=true,
//   random_slopes=[], reml=true, max_iterations=100, tolerance=1e-8, theta_max=1000.0)
pub struct GlmmRegressorBuilder { ... }
impl GlmmRegressorBuilder {
    pub fn with_intercept(self, include: bool) -> Self
    pub fn random_intercept(self, include: bool) -> Self
    pub fn random_slopes(self, cols: Vec<usize>) -> Self   // 0-based column indices into x
    pub fn reml(self, reml: bool) -> Self                  // Gaussian only; ignored for GLMMs
    pub fn max_iterations(self, max_iter: usize) -> Self
    pub fn tolerance(self, tol: f64) -> Self
    pub fn theta_max(self, theta_max: f64) -> Self
    pub fn build(self) -> GlmmRegressor
}

// FittedGlmm — all public methods
pub struct FittedGlmm { ... }
impl FittedGlmm {
    pub fn fixed_effects(&self) -> &[f64]           // intercept first when with_intercept
    pub fn std_errors(&self) -> &[f64]              // SEs of fixed effects
    pub fn intercept(&self) -> Option<f64>
    pub fn slopes(&self) -> &[f64]                  // non-intercept fixed effects
    pub fn random_effects(&self) -> &[f64]          // random intercept BLUP per group
    pub fn random_effects_matrix(&self) -> &[Vec<f64>]  // full n_groups × q BLUPs
    pub fn n_random_effects(&self) -> usize          // q (1 for plain random intercept)
    pub fn random_cov(&self) -> &[Vec<f64>]         // Σ (q × q)
    pub fn random_sd(&self) -> Vec<f64>             // sqrt(diag Σ)
    pub fn random_corr(&self) -> Vec<Vec<f64>>      // correlation matrix (q × q)
    pub fn theta(&self) -> f64                       // profiled ratio σ_b/σ
    pub fn sigma(&self) -> f64                       // residual SD (1.0 for Poisson/Binomial)
    pub fn sd_random(&self) -> f64                   // σ_b = sqrt(Σ[0][0])
    pub fn var_random(&self) -> f64                  // σ_b² = Σ[0][0]
    pub fn deviance(&self) -> f64                    // −2 log-likelihood / REML criterion
    pub fn log_likelihood(&self) -> f64              // −deviance / 2
    pub fn n_groups(&self) -> usize
    pub fn factors(&self) -> &[FactorSummary]        // non-empty only for fit_crossed
    pub fn n_factors(&self) -> usize
    pub fn is_reml(&self) -> bool
    pub fn converged(&self) -> bool
    pub fn iterations(&self) -> usize
    pub fn predict_fixed(&self, x: &Mat<f64>) -> Col<f64>  // marginal/new-group predictions
}

// FactorSummary — per-factor output for fit_crossed
pub struct FactorSummary {
    pub n_levels: usize,
    pub sd: f64,          // random-intercept SD σ_f
    pub blups: Vec<f64>,  // BLUP per level (0-based)
}
```

**Key facts:**
- `fit()` requires `group: &[usize]` — group label per observation (any `usize`, compacted internally).
- `fit_crossed()` requires `groups: &[&[usize]]` — one slice per factor. For crossed: `&[&sku_ids, &region_ids]`. For nested `(1|a/b)`: pass `a` and `interaction_id`.
- Random slopes: specify 0-based column indices of `x` via `random_slopes(vec![0, 1])`.
- `factors()` is empty for single-factor fits; use `random_effects()` / `random_cov()` for those.
- No `predict` with group BLUPs — `predict_fixed()` gives marginal (new-group) predictions only.
- PyModel must expose group IDs as numpy integer arrays or Python lists; convert to `Vec<usize>` in the wrapper.

**Closest PyModel analog:** No exact analog. Use `src/pymodels/py_poisson.rs` for the fit/predict
pattern and `src/pymodels/py_tweedie.rs` for the factory-method-based construction pattern. GLMM
is unique in requiring group ID arrays as additional fit inputs — the `fit` signature must accept
`(x, y, group)` not just `(x, y)`.

**Target surfaces:** PyModel-only (CONTEXT.md locked decision — stateful, non-group-aggregable).
GLMM does not map to per-group Polars expressions because it is itself a cross-group model.

---

### REGR-02: PSplineRegressor + FittedPSpline

**Source:** [VERIFIED: ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/pspline.rs:24-269]

```rust
// Construction — builder or direct
pub struct PSplineRegressor {
    n_basis: usize,       // 0 = automatic (~n/4, clamped 6..40)
    penalty_order: usize, // default 2
    lambda: Option<f64>,  // None = GCV selection
}
impl PSplineRegressor {
    pub fn new() -> Self                          // defaults: n_basis=0, penalty_order=2, lambda=None
    pub fn with_n_basis(self, k: usize) -> Self
    pub fn with_penalty_order(self, order: usize) -> Self
    pub fn with_lambda(self, lambda: f64) -> Self
    pub fn builder() -> Self                      // alias for new()
    pub fn build(self) -> Self                    // no-op finaliser
}
impl Regressor for PSplineRegressor {
    type Fitted = FittedPSpline;
    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<FittedPSpline, RegressionError>
    // x must be n×1 (single predictor column); the basis is built internally
}

// FittedPSpline — all public methods
pub struct FittedPSpline { knots, beta, chol_l, sigma2, edf, n, p, xmin, xmax, result }
impl FittedPSpline {
    pub fn edf(&self) -> f64       // effective degrees of freedom (trace of smoother matrix)
    pub fn sigma2(&self) -> f64    // residual variance estimate σ²
}
impl FittedRegressor for FittedPSpline {
    fn predict(&self, x: &Mat<f64>) -> Col<f64>
    fn result(&self) -> &RegressionResult         // r_squared, rmse, mse, coefficients
    fn predict_with_interval(&self, x, interval, level) -> PredictionResult
    // IntervalType::Confidence = Bayesian posterior SE; IntervalType::Prediction adds σ²
}
```

**Key facts:**
- `x` must be `n × 1` — a single predictor column. The B-spline basis is built internally.
- `n_basis = 0` → automatic: `(n/4 + 4).clamp(6, 40).min(n - 1)`, further clamped to `max(degree + penalty_order + 1, ...)`.
- `lambda = None` → GCV search over `{1e-8, 1e-7, ..., 1e8} * scale`. Fixed `lambda` → exact solve.
- `edf` = trace of the smoother / hat matrix. For a linear signal, `edf ≈ 2`.
- Confidence intervals use the Bayesian posterior covariance (same as `mgcv` default).
- `result()` → `RegressionResult` carries: `coefficients` (B-spline coefficients, not user-interpretable), `fitted_values`, `residuals`, `r_squared`, `rmse`, `mse`. AIC/BIC not set.

**Closest PyModel analog:** No smoother analog exists. Use `src/pymodels/py_ridge.rs` for the
builder pattern, but the fit input contract is simpler (single-predictor x) and the output is
smoother-specific (expose `edf`, `sigma2` as getters rather than inference tables).

**Target surfaces:** PyModel (primary) + Polars expression `pspline_fit` (CONTEXT.md locked as
"both" for REGR-02). The Polars expression version accepts a single column name and fits a
per-group smoother, outputting `edf`, `sigma2`, `r_squared` as struct fields.

---

### REGR-04: HC Inference Extension

**Source:** [VERIFIED: ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/inference/robust_covariance.rs:1-120]

```rust
// HcType enum — four variants
pub enum HcType { HC0, HC1, HC2, HC3 }

// HcResult — SEs only
pub struct HcResult {
    pub std_errors: Col<f64>,
    pub intercept_std_error: Option<f64>,
}

// HcInference — full inference output
pub struct HcInference {
    pub hc_type: HcType,
    pub std_errors: Col<f64>,
    pub t_statistics: Col<f64>,
    pub p_values: Col<f64>,
    pub conf_interval_lower: Col<f64>,
    pub conf_interval_upper: Col<f64>,
    pub confidence_level: f64,
    pub intercept: Option<HcInterceptInference>,
}

// HcInterceptInference
pub struct HcInterceptInference {
    pub std_error: f64,
    pub t_statistic: f64,
    pub p_value: f64,
    pub conf_interval: (f64, f64),
}

pub fn compute_hc_standard_errors(
    x: &Mat<f64>,
    residuals: &Col<f64>,
    aliased: &[bool],
    with_intercept: bool,
    hc_type: HcType,
) -> Result<HcResult, &'static str>

pub fn compute_hc_inference(
    x: &Mat<f64>,
    residuals: &Col<f64>,
    aliased: &[bool],
    with_intercept: bool,
    hc_type: HcType,
    coef: &Col<f64>,
    intercept: Option<f64>,
    confidence_level: f64,
    df: usize,
) -> Result<HcInference, &'static str>
```

**OLS existing implementation (the template):** [VERIFIED: src/pymodels/py_ols.rs:355-413]
The `PyOLS.hc_inference(x, hc_type)` method calls `fitted.hc_inference(&x_mat, hc)` on
`FittedOls`, which returns an `HcInference` struct. The method returns a Python dict with keys:
`std_errors`, `t_statistics`, `p_values`, `conf_interval_lower`, `conf_interval_upper`, and
optionally `intercept_std_error`, `intercept_t_statistic`, `intercept_p_value`.

**HC extension strategy (REGR-04):** The `FittedRegressor` trait gives `.result()` → `&RegressionResult`
which carries `residuals`, `fitted_values`, `n_observations`, `n_parameters`. The HC sandwich
computation needs: `x` (design matrix, WITHOUT intercept column), `residuals`, `aliased` mask,
`with_intercept`, `hc_type`, coefficients, intercept, confidence level, df.

**Regressors where HC extension is natural** (those whose `FittedRegressor` impl carries sufficient
residuals/leverage and for which the sandwich estimator is statistically meaningful):
- `FittedRidge` — residuals available; HC meaningful for penalized regression only if λ is small
- `FittedWls` — residuals available; HC extension adds heteroskedasticity-robust SEs on top of WLS
- `FittedOls` — already done (template)
- NOT meaningful for GLMs (Poisson/Binomial have fixed dispersion; HC SE for non-Gaussian families
  requires different sandwich formulation), NOT for `FittedPassiveAggressive` (online learner),
  NOT for `FittedTheilSen`/`FittedRansac` (robust estimators already resistant to HC issues).

**Implementation:** Add `hc_inference(x, hc_type="hc1")` method to `PyRidge` and `PyWLS`, mirroring
`PyOLS.hc_inference()` exactly. Call `compute_hc_inference` directly (not through `FittedRidge.hc_inference()`
unless that method exists). The `aliased` mask should be `vec![false; x.ncols()]` for standard
non-aliased cases.

---

### REGR-05: Missing GLM Diagnostic Expressions

**Source:** [VERIFIED: ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/diagnostics/glm_residuals.rs:67-130]

```rust
// Five missing functions (exact signatures)

pub fn standardized_pearson_residuals<F: GlmFamily>(
    y: &Col<f64>, mu: &Col<f64>, family: &F, leverage: &Col<f64>, dispersion: f64,
) -> Col<f64>
// r_P / sqrt(φ * (1 - h_ii))

pub fn standardized_deviance_residuals<F: GlmFamily>(
    y: &Col<f64>, mu: &Col<f64>, family: &F, leverage: &Col<f64>, dispersion: f64,
) -> Col<f64>
// r_D / sqrt(φ * (1 - h_ii))

pub fn estimate_dispersion_deviance(
    y: &Col<f64>, mu: &Col<f64>, n: usize, p: usize,
    unit_deviance_fn: &dyn Fn(f64, f64) -> f64,
) -> f64
// deviance / (n - p)

pub fn estimate_dispersion_pearson(
    y: &Col<f64>, mu: &Col<f64>, n: usize, p: usize,
    variance_fn: &dyn Fn(f64) -> f64,
) -> f64
// Σ(y-μ)²/V(μ) / (n - p)

pub fn pearson_chi_squared(
    y: &Col<f64>, mu: &Col<f64>, variance_fn: &dyn Fn(f64) -> f64,
) -> f64
// Σ(y-μ)²/V(μ)  — general form (not family-specific)
```

**Implementation approach:** Wire as Polars expressions in `src/expressions/regression.rs`, following
the existing `logistic_pearson_residuals` / `poisson_pearson_residuals` pattern. The existing pattern
takes `[residuals_series, mu_series, ...]` from the `Series` slice. Because the new functions
require `GlmFamily` trait objects, the expressions should accept a `family` string parameter
("poisson", "binomial", "gamma", "tweedie", "negbinom") and dispatch to the concrete family's
`variance(mu)` and `unit_deviance(y, mu)` methods internally. The `leverage` column needed for
standardized residuals comes from the existing `compute_leverage` expression output.

**Output type:** Single `Float64` series (one value per row), no struct wrapper needed.

---

### REGR-06a: TheilSenRegressor

**Source:** [VERIFIED: ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/theil_sen.rs:34-495]

```rust
pub struct TheilSenRegressor {
    with_intercept: bool,           // default true
    max_subpopulation: usize,       // default 10_000
    n_subsamples: Option<usize>,    // default None → n_features + 1
    max_iter: usize,                // default 300 (Weiszfeld iterations)
    tol: f64,                       // default 1e-3
    random_state: u64,              // default 0
}
impl TheilSenRegressor {
    pub fn builder() -> TheilSenRegressorBuilder
}
impl Regressor for TheilSenRegressor {
    type Fitted = FittedTheilSen;
    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<FittedTheilSen, RegressionError>
}

pub struct TheilSenRegressorBuilder { ... }
impl TheilSenRegressorBuilder {
    pub fn with_intercept(self, include: bool) -> Self
    pub fn max_subpopulation(self, value: usize) -> Self
    pub fn n_subsamples(self, value: usize) -> Self
    pub fn max_iter(self, value: usize) -> Self
    pub fn tolerance(self, value: f64) -> Self
    pub fn random_state(self, seed: u64) -> Self
    pub fn build(self) -> TheilSenRegressor
}

pub struct FittedTheilSen { result: RegressionResult, with_intercept: bool }
impl FittedTheilSen {
    pub fn with_intercept(&self) -> bool
}
impl FittedRegressor for FittedTheilSen {
    fn predict(&self, x: &Mat<f64>) -> Col<f64>
    fn result(&self) -> &RegressionResult
    fn predict_with_interval(&self, x, None/Some(iv), level) -> PredictionResult
    // NOTE: predict_with_interval always returns point predictions only (no CI)
}
```

**Key facts:**
- `result()` carries: `coefficients`, `intercept`, `residuals`, `fitted_values`, `r_squared`,
  `adj_r_squared`, `rmse`, `mse`. `f_statistic`, `f_pvalue`, `aic`, `aicc`, `bic`, `log_likelihood`
  are all `NAN` (Theil-Sen has no tractable analytic sampling distribution).
- No confidence intervals available from `predict_with_interval` (returns point predictions only).

**Closest PyModel analog:** `src/pymodels/py_huber.rs` (PyHuber) — robust regressor with
`fit`/`predict`/`is_fitted`/`coefficients`/`intercept`/`residuals` but no inference table.
[VERIFIED: src/pymodels/py_huber.rs:1-60]

**Target surfaces:** PyModel (primary) + Polars expression `theilsen_fit` (both per CONTEXT.md).

---

### REGR-06b: RansacRegressor

**Source:** [VERIFIED: ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/ransac.rs:48-481]

```rust
pub struct RansacRegressor {
    with_intercept: bool,              // default true
    min_samples: Option<usize>,        // default None → n_features + 1
    residual_threshold: Option<f64>,   // default None → MAD(y)
    max_trials: usize,                 // default 100
    stop_probability: f64,             // default 0.99
    stop_n_inliers: Option<usize>,     // default None
    random_state: u64,                 // default 0
}
impl RansacRegressor {
    pub fn builder() -> RansacRegressorBuilder
}

pub struct RansacRegressorBuilder { ... }
impl RansacRegressorBuilder {
    pub fn with_intercept(self, include: bool) -> Self
    pub fn min_samples(self, value: usize) -> Self
    pub fn residual_threshold(self, value: f64) -> Self
    pub fn max_trials(self, value: usize) -> Self
    pub fn stop_probability(self, value: f64) -> Self
    pub fn stop_n_inliers(self, value: usize) -> Self
    pub fn random_state(self, seed: u64) -> Self
    pub fn build(self) -> RansacRegressor
}

pub struct FittedRansac { result: RegressionResult, inlier_mask: Vec<bool>,
                          n_trials: usize, residual_threshold: f64 }
impl FittedRansac {
    pub fn inlier_mask(&self) -> &[bool]          // bool per observation
    pub fn n_inliers(&self) -> usize
    pub fn n_trials(&self) -> usize
    pub fn residual_threshold(&self) -> f64
}
impl FittedRegressor for FittedRansac {
    fn predict(&self, x: &Mat<f64>) -> Col<f64>
    fn result(&self) -> &RegressionResult
    fn predict_with_interval(&self, ...) -> PredictionResult  // point only
}
```

**Key facts:**
- Unique output: `inlier_mask` (boolean array). The PyModel must expose this as a numpy bool array
  getter.
- If RANSAC fails to find a consensus set of `min_samples` inliers, returns `RegressionError::ConvergenceFailed`.
- `result()`: `coefficients`, `intercept`, `residuals`, `fitted_values`, `r_squared`, `rank`,
  `n_parameters`, `n_observations`. `f_statistic` etc. are not set (NAN or 0).

**Closest PyModel analog:** `src/pymodels/py_huber.rs` for robust fit pattern. Extra getter:
`inlier_mask` → `PyArray1<bool>`.

**Target surfaces:** PyModel (primary) + Polars expression `ransac_fit` (both per CONTEXT.md).

---

### REGR-06c: BayesianRidge + ArdRegression

**Source:** [VERIFIED: ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/bayesian.rs:47-719]

```rust
// BayesianRidge
pub struct BayesianRidge {
    fit_intercept: bool,     // default true
    max_iter: usize,         // default 300
    tol: f64,                // default 1e-3
    alpha_1: f64,            // default 1e-6 (noise precision prior shape)
    alpha_2: f64,            // default 1e-6 (noise precision prior rate)
    lambda_1: f64,           // default 1e-6 (weight precision prior shape)
    lambda_2: f64,           // default 1e-6 (weight precision prior rate)
    alpha_init: Option<f64>, // default None
    lambda_init: Option<f64>,// default None
}
impl BayesianRidge {
    pub fn builder() -> BayesianRidgeBuilder
}
pub struct BayesianRidgeBuilder { ... }
impl BayesianRidgeBuilder {
    pub fn fit_intercept(self, v: bool) -> Self
    pub fn max_iter(self, v: usize) -> Self
    pub fn tolerance(self, v: f64) -> Self
    pub fn alpha_1(self, v: f64) -> Self
    pub fn alpha_2(self, v: f64) -> Self
    pub fn lambda_1(self, v: f64) -> Self
    pub fn lambda_2(self, v: f64) -> Self
    pub fn alpha_init(self, v: f64) -> Self
    pub fn lambda_init(self, v: f64) -> Self
    pub fn build(self) -> BayesianRidge
}
pub struct FittedBayesianRidge { result: RegressionResult, alpha: f64, lambda: f64,
                                  sigma_diag: Vec<f64> }
impl FittedBayesianRidge {
    pub fn alpha(&self) -> f64           // noise precision α
    pub fn lambda(&self) -> f64          // weight precision λ
    pub fn sigma_diag(&self) -> &[f64>   // posterior covariance diagonal (for CIs)
}
impl FittedRegressor for FittedBayesianRidge { ... }  // predict, result, predict_with_interval (point only)

// ArdRegression
pub struct ArdRegression {
    fit_intercept: bool,
    max_iter: usize,
    tol: f64,
    alpha_1: f64, alpha_2: f64, lambda_1: f64, lambda_2: f64,
    threshold_lambda: f64,  // default 10_000.0; features with λ_j > this are pruned
}
impl ArdRegression { pub fn builder() -> ArdRegressionBuilder }
pub struct ArdRegressionBuilder { ... }
impl ArdRegressionBuilder {  // same as BayesianRidgeBuilder + threshold_lambda()
    pub fn threshold_lambda(self, v: f64) -> Self
    pub fn build(self) -> ArdRegression
}
pub struct FittedArd { result: RegressionResult, alpha: f64, lambdas: Vec<f64> }
impl FittedArd {
    pub fn alpha(&self) -> f64           // noise precision
    pub fn lambdas(&self) -> &[f64]      // per-feature precision λ_j (pruned features → high)
}
impl FittedRegressor for FittedArd { ... }  // predict, result, predict_with_interval (point only)
```

**Key facts:**
- Both use SVD-based updates (BayesianRidge) or LLT factorization (ARD) — may error with
  `RegressionError::SingularMatrix` on degenerate designs.
- `fit_intercept` (not `with_intercept`) — parameter name differs from other solvers.
- BayesianRidge `sigma_diag` enables posterior prediction intervals (not yet wired in `predict_with_interval`).
- ARD `lambdas` vector has length `p` (original feature count); pruned features have `λ_j >> threshold_lambda`.

**Closest PyModel analog:** `src/pymodels/py_ridge.rs` (PyRidge) for the parameter-name style
(`fit_intercept`, `max_iter`, etc.). Both are regularized regression models with hyperparameter outputs.

**Target surfaces:** PyModel (primary) + Polars expression (both per CONTEXT.md).

---

### REGR-06d: LarsRegressor + LarsMethod

**Source:** [VERIFIED: ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/lars.rs:38-470]

```rust
pub enum LarsMethod { Lar, Lasso }   // default: Lar

pub struct LarsRegressor {
    method: LarsMethod,              // default Lar
    fit_intercept: bool,             // default true
    n_nonzero_coefs: Option<usize>,  // default None → min(n-1, p)
    alpha: f64,                      // default 0.0; LassoLars stop alpha
    standardize: bool,               // default false
    eps: f64,                        // default f64::EPSILON
}
impl LarsRegressor { pub fn builder() -> LarsRegressorBuilder }
pub struct LarsRegressorBuilder { ... }  // all fields settable, pub fn build(self) -> LarsRegressor

pub struct FittedLars {
    result: RegressionResult,
    alphas: Vec<f64>,               // path alphas (max abs correlation at each step)
    coefs_path: Vec<Vec<f64>>,      // coefficient path (one Vec<f64> per step)
    // x_mean, x_norm, y_mean, method, fit_intercept stored but not all public
}
impl FittedLars {
    pub fn alphas(&self) -> &[f64]
    // coefs_path is NOT public — only final coefficients are accessible via result()
}
impl FittedRegressor for FittedLars { fn predict, fn result, fn predict_with_interval (point only) }
```

**Key facts:**
- `LarsMethod` enum must be exposed as a Python string parameter ("lar" / "lasso") in the PyModel.
- `alphas()` returns the regularization path α values — expose as numpy array getter.
- `coefs_path` is private (the field is defined but not pub-exposed). Only the final coefficients
  are accessible via `result().coefficients`.
- For LassoLars with `alpha > 0`: path is interpolated to produce exactly `alpha`-regularized coefficients.

**Closest PyModel analog:** `src/pymodels/py_elastic_net.rs` (PyElasticNet) — path-based
regularized regression with `alpha` / `method` string parameters.

**Target surfaces:** PyModel (primary) + Polars expression `lars_fit` (both per CONTEXT.md).

---

### REGR-06e: PassiveAggressiveRegressor + PaState

**Source:** [VERIFIED: ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/passive_aggressive.rs:30-413]

```rust
pub enum PaLoss { EpsilonInsensitive, SquaredEpsilonInsensitive }  // default: EpsilonInsensitive

pub struct PassiveAggressiveRegressor {
    c: f64,               // default 1.0 (regularisation / aggressiveness)
    epsilon: f64,         // default 0.1 (insensitivity band)
    with_intercept: bool, // default true
    max_iter: usize,      // default 1000
    tol: f64,             // default 1e-3
    shuffle: bool,        // default true
    loss: PaLoss,         // default EpsilonInsensitive
    random_state: u64,    // default 0
}
impl PassiveAggressiveRegressor {
    pub fn builder() -> PassiveAggressiveRegressorBuilder
    pub fn partial_fit(&self, state: &mut PaState, x_row: &[f64], y_value: f64)
        -> Result<(), RegressionError>  // single-sample online update
}
impl Regressor for PassiveAggressiveRegressor {
    type Fitted = FittedPassiveAggressive;
    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<FittedPassiveAggressive, RegressionError>
}

pub struct PassiveAggressiveRegressorBuilder { ... }
impl PassiveAggressiveRegressorBuilder {
    pub fn c(self, value: f64) -> Self
    pub fn epsilon(self, value: f64) -> Self
    pub fn with_intercept(self, include: bool) -> Self
    pub fn max_iter(self, value: usize) -> Self
    pub fn tolerance(self, value: f64) -> Self
    pub fn shuffle(self, value: bool) -> Self
    pub fn loss(self, value: PaLoss) -> Self
    pub fn random_state(self, seed: u64) -> Self
    pub fn build(self) -> PassiveAggressiveRegressor
}

pub struct FittedPassiveAggressive { result: RegressionResult, weights: Vec<f64>,
                                      intercept: f64, with_intercept: bool, n_iter: usize }
impl FittedPassiveAggressive {
    pub fn n_iter(&self) -> usize
}
impl FittedRegressor for FittedPassiveAggressive { fn predict, fn result, fn predict_with_interval (point) }

pub struct PaState { pub weights: Vec<f64>, pub intercept: f64 }
impl PaState {
    pub fn new(n_features: usize) -> Self
}
```

**Key facts:**
- `PaLoss` must be exposed as a Python string: `"epsilon_insensitive"` / `"squared_epsilon_insensitive"`.
- `partial_fit` requires a mutable `PaState` — stateful across calls. The PyModel must hold
  a `PaState` as an optional field alongside the fitted model, and expose `partial_fit(x_row, y)`.
- **PyModel-only** per CONTEXT.md locked decision (no Polars expression).
- `n_iter()` on `FittedPassiveAggressive` returns early-stop iteration count (not max_iter).

**Closest PyModel analog:** No direct analog. `src/pymodels/py_rls.rs` (PyRLS — Recursive Least
Squares) is the closest in spirit (stateful regressor with update semantics), but RLS does not have
a `partial_fit` pattern.

---

### REGR-06f: MomentAccumulator

**Source:** [VERIFIED: ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/moments.rs:51-162]

```rust
pub struct MomentAccumulator {
    n_features: usize,
    n: usize,
    sum_x: Col<f64>,    // Σxᵢ
    sum_y: f64,         // Σyᵢ
    xtx: Mat<f64>,      // Σxᵢxᵢᵀ (p × p)
    xty: Col<f64>,      // Σxᵢyᵢ
}
impl MomentAccumulator {
    pub fn new(n_features: usize) -> Self
    pub fn n_features(&self) -> usize
    pub fn n(&self) -> usize
    pub fn sum_x(&self) -> &Col<f64>
    pub fn sum_y(&self) -> f64
    pub fn xtx(&self) -> &Mat<f64>
    pub fn xty(&self) -> &Col<f64>
    pub fn push_row(&mut self, x_row: &[f64], y: f64) -> Result<(), RegressionError>
    pub fn merge(&mut self, other: &Self) -> Result<(), RegressionError>
    pub fn clear(&mut self)
}
// Used by:
//   OlsRegressor::fit_from_accumulator(&acc) -> Result<FittedOls, RegressionError>
//   RidgeRegressor::fit_from_accumulator(&acc) -> Result<FittedRidge, RegressionError>
//   (also fit_from_moments(n, sum_x, sum_y, xtx, xty) variants on OlsRegressor/RidgeRegressor)
```

**Key facts:**
- `push_row` takes `x_row: &[f64]` — a plain Rust slice, not a faer type. The Python wrapper must
  accept a 1-D numpy array and convert to `Vec<f64>` (no `ToFaer` needed — just `.as_slice()`).
- `merge` enables parallel accumulation — expose in Python so users can accumulate on worker
  threads and merge.
- `xtx()` returns `&Mat<f64>` — expose as 2-D numpy array via `IntoNumpy`.
- `sum_x()` returns `&Col<f64>` — expose as 1-D numpy array.
- After accumulation, user calls `OLS().fit_from_accumulator(&acc)` or
  `Ridge().fit_from_accumulator(&acc)` on the existing PyModel objects — but these methods are
  NOT yet exposed on `PyOLS`/`PyRidge`. The PyModel for `MomentAccumulator` itself should be a
  utility class, and `PyOLS`/`PyRidge` must get a `fit_from_accumulator(acc)` method.
- **PyModel-only** per CONTEXT.md locked decision.

**Closest PyModel analog:** None — unique utility class. Implement as `PyMomentAccumulator` in a new
file `src/pymodels/py_moment_accumulator.rs`. Additionally add `fit_from_accumulator` methods to
`PyOLS` and `PyRidge`.

---

## Standard Stack

### Core (no new packages required)

All implementation uses the packages already in `Cargo.toml`. No new Rust crate dependencies.
No new Python package dependencies.

| Library | Version | Purpose | Notes |
|---------|---------|---------|-------|
| `anofox-regression` | 0.5.13 | All gap regressors and diagnostics | Already in Cargo.toml |
| `pyo3` | 0.26 | PyO3 bindings | Already in Cargo.toml (python feature) |
| `numpy` (pyo3) | pyo3-0.26 compat | `PyReadonlyArray*`, `PyArray1` | Already in Cargo.toml |
| `faer` | 0.23.2 | `Mat<f64>`, `Col<f64>` used by crate types | Already in Cargo.toml |

### Supporting

| Library | Purpose | Notes |
|---------|---------|-------|
| `crate::utils::{ToFaer, IntoNumpy}` | numpy↔faer bridge | Already in `src/utils/` — use as-is |
| `pyo3_polars::derive::polars_expr` | Expression macro | For new diagnostic expressions only |

## Package Legitimacy Audit

No new packages are installed in this phase. All implementation uses existing project dependencies.
Not applicable.

---

## Architecture Patterns

### System Architecture Diagram

```
Python user code
    │
    ├── from polars_statistics import Gamma, GLMM, PSpline, ...
    │        └── python/polars_statistics/__init__.py (imports from _polars_statistics)
    │
    ├── model.fit(X, y)  ←→  PyModel#[pymethods]
    │        └── src/pymodels/py_gamma.rs, py_glmm.rs, py_pspline.rs, ...
    │                 └── ToFaer(numpy array) → faer Mat/Col
    │                 └── anofox_regression::solvers::GammaRegressor::builder()...fit()
    │                 └── FittedGamma stored in Option<FittedGamma>
    │
    ├── df.select(ps.gamma_fit(...))  ←→  Polars expression
    │        └── python/polars_statistics/exprs/regression.py (builder fn)
    │        └── src/expressions/regression.rs #[polars_expr] gamma_fit
    │                 └── anofox_regression::solvers::GammaRegressor::fit()
    │                 └── Output: Struct series with named fields
    │
    └── Registration path (shared files, edited once per wave):
             src/pymodels/mod.rs  ─── mod py_gamma; pub use py_gamma::PyGamma;
             src/lib.rs           ─── m.add_class::<pymodels::PyGamma>()?;
             python/__init__.py   ─── from ..._polars_statistics import Gamma
```

### Recommended Project Structure (new files only)

```
src/pymodels/
├── py_gamma.rs              # REGR-03: GammaRegressor (tracer)
├── py_glmm.rs               # REGR-01: GlmmRegressor + FactorSummary
├── py_pspline.rs            # REGR-02: PSplineRegressor
├── py_theil_sen.rs          # REGR-06a: TheilSenRegressor
├── py_ransac.rs             # REGR-06b: RansacRegressor
├── py_bayesian_ridge.rs     # REGR-06c: BayesianRidge
├── py_ard.rs                # REGR-06c: ArdRegression
├── py_lars.rs               # REGR-06d: LarsRegressor + LarsMethod
├── py_passive_aggressive.rs # REGR-06e: PassiveAggressiveRegressor + PaState
└── py_moment_accumulator.rs # REGR-06f: MomentAccumulator
```

### Pattern 1: Standard PyModel class (fit/predict)

This is the established pattern from `py_tweedie.rs` and `py_huber.rs`.

```rust
// Source: src/pymodels/py_tweedie.rs (verified this session)
use anofox_regression::solvers::{GammaRegressor, FittedGamma, Regressor, FittedRegressor};
use crate::utils::{IntoNumpy, ToFaer};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyclass(name = "Gamma")]
pub struct PyGamma {
    with_intercept: bool,
    max_iter: usize,
    tol: f64,
    lambda_: f64,
    fitted: Option<FittedGamma>,
}

#[pymethods]
impl PyGamma {
    #[new]
    #[pyo3(signature = (with_intercept=true, max_iter=25, tol=1e-8, lambda_=0.0))]
    fn new(with_intercept: bool, max_iter: usize, tol: f64, lambda_: f64) -> Self { ... }

    fn fit<'py>(mut slf: PyRefMut<'py, Self>,
                x: PyReadonlyArray2<'py, f64>,
                y: PyReadonlyArray1<'py, f64>) -> PyResult<PyRefMut<'py, Self>> {
        let x_mat = x.to_faer();
        let y_col = y.to_faer();
        let model = GammaRegressor::builder()
            .with_intercept(slf.with_intercept)
            .max_iterations(slf.max_iter)
            .tolerance(slf.tol)
            .lambda(slf.lambda_)
            .build();
        let fitted = model.fit(&x_mat, &y_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        slf.fitted = Some(fitted);
        Ok(slf)
    }

    fn predict<'py>(&self, py: Python<'py>,
                    x: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self.fitted.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(fitted.predict(&x.to_faer()).into_numpy(py))
    }

    #[getter] fn coefficients<'py>(...) -> PyResult<...> { ... }
    #[getter] fn intercept(&self) -> PyResult<Option<f64>> { ... }
    #[getter] fn converged(&self) -> PyResult<bool> { ... }
    fn is_fitted(&self) -> bool { self.fitted.is_some() }
}
```

### Pattern 2: GLM model fit method (GLMM — extra group input)

```rust
// GLMM has a non-standard fit signature: fit(x, y, group)
fn fit<'py>(mut slf: PyRefMut<'py, Self>,
            x: PyReadonlyArray2<'py, f64>,
            y: PyReadonlyArray1<'py, f64>,
            group: Vec<usize>,  // Python list[int] → Vec<usize>
           ) -> PyResult<PyRefMut<'py, Self>> {
    let x_mat = x.to_faer();
    let y_col = y.to_faer();
    let model = GlmmRegressor::gaussian()
        .with_intercept(slf.with_intercept)
        .random_slopes(slf.random_slopes.clone())
        .build();
    let fitted = model.fit(&x_mat, &y_col, &group)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    slf.fitted = Some(fitted);
    Ok(slf)
}
// For fit_crossed: groups: Vec<Vec<usize>>
fn fit_crossed<'py>(..., groups: Vec<Vec<usize>>) -> PyResult<...> {
    let group_refs: Vec<&[usize]> = groups.iter().map(|g| g.as_slice()).collect();
    model.fit_crossed(&x_mat, &y_col, &group_refs)...
}
```

### Pattern 3: Polars expression for new solver (gamma_fit)

```rust
// Source: parallel to existing ols_fit in src/expressions/regression.rs
#[polars_expr(output_type_func = gamma_output_dtype)]
fn gamma_fit(inputs: &[Series], kwargs: GammaKwargs) -> PolarsResult<Series> {
    // Extract y (inputs[0]) and x columns (inputs[1..])
    // Build Mat<f64>, fit GammaRegressor, extract RegressionResult fields
    // Return Struct series with fields matching gamma_output_dtype
}
fn gamma_output_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "gamma".into(),
        DataType::Struct(vec![
            Field::new("intercept".into(), DataType::Float64),
            Field::new("coefficients".into(), DataType::List(Box::new(DataType::Float64))),
            Field::new("r_squared".into(), DataType::Float64),
            Field::new("aic".into(), DataType::Float64),
            Field::new("converged".into(), DataType::Boolean),
        ])
    ))
}
```

### Pattern 4: Registration (shared files — edit once per wave)

```rust
// src/pymodels/mod.rs — add at bottom of each section
mod py_gamma;
pub use py_gamma::PyGamma;

// src/lib.rs #[pymodule] block — add under "GLM Models" comment
m.add_class::<pymodels::PyGamma>()?;

// python/polars_statistics/__init__.py — add to import list
from polars_statistics._polars_statistics import (
    ...
    Gamma,
    ...
)
```

### Anti-Patterns to Avoid

- **Inventing extra fields:** Do not add fields to PyModel output that are not in the crate's result struct. `FittedTheilSen.result()` has `NAN` for `f_statistic`/`aic`/`bic` — expose them as `NaN` not as missing Python keys.
- **Using `.to_faer()` on 1-D MomentAccumulator rows:** `push_row` takes `&[f64]` (a slice), not `Col<f64>`. Use `x_row.as_slice().unwrap()` on `PyReadonlyArray1`.
- **Calling `fit_crossed` with wrong group slice order:** `groups: &[&[usize]]` — each inner slice is a different factor. Python side must validate that all slices have the same length as `y`.
- **Using `with_intercept` vs `fit_intercept`:** `BayesianRidge`, `ArdRegression`, and `LarsRegressor` use `fit_intercept` (matching sklearn); `GammaRegressor`, `TheilSenRegressor`, `RansacRegressor`, `PassiveAggressiveRegressor` use `with_intercept`. Do not mix these up in the builder calls.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| GLMM fitting | Custom REML/PIRLS | `GlmmRegressor::gaussian().build().fit()` | Profiled REML with Schur complement elimination — correct lme4 semantics |
| P-spline basis | Custom B-spline basis + GCV | `PSplineRegressor::new().fit()` | GCV grid search + Bayesian posterior SE already implemented |
| HC standard errors | Manual sandwich estimator | `compute_hc_inference()` | Leverage-adjusted HC2/HC3 require hat matrix; already correct |
| Theil-Sen multivariate | Custom spatial median | `TheilSenRegressor::builder().build().fit()` | Vardi–Zhang Weiszfeld convergence details are non-trivial |
| RANSAC inlier selection | Custom random sampling | `RansacRegressor::builder().build().fit()` | Fischler–Bolles early termination wired in |
| Bayesian posterior updates | SVD-based EM | `BayesianRidge::builder().build().fit()` | Evidence approximation matches sklearn numerics exactly |
| LARS path | Active set algorithm | `LarsRegressor::builder().build().fit()` | Equiangular vector updates are numerically sensitive |
| Streaming moments | Custom accumulation loop | `MomentAccumulator::new(p)` + `push_row()` | Avoids materializing N×p matrix; parallelisable via `merge()` |

---

## Common Pitfalls

### Pitfall 1: GLMM group array type mismatch

**What goes wrong:** Python `list[int]` or numpy `int64` array passed as the group argument.
`fit` requires `group: &[usize]` in Rust. `usize` on 64-bit Linux/macOS = `u64` but on 32-bit
= `u32`. Using `numpy::PyReadonlyArray1<u64>` then `.as_slice().unwrap()` will type-check on
64-bit but fail to compile on 32-bit targets.

**How to avoid:** Accept `group: Vec<u64>` in the PyModel method signature (pyo3 converts Python
`list[int]` or numpy int array to this), then `.iter().map(|&g| g as usize).collect::<Vec<usize>>()`.
This is portable across all platforms and wheel targets.

**Warning signs:** Compile error mentioning `u64` vs `usize` mismatch on 32-bit CI runner.

### Pitfall 2: PSplineRegressor x must be n×1

**What goes wrong:** User passes a multi-column design matrix `X` of shape `(n, p)` with `p > 1`.
`FittedPSpline::eval()` reads only `x[(i, 0)]` — silently ignores all other columns.

**How to avoid:** In the PyModel `fit` method, validate `x.ncols() == 1` and raise a Python
`ValueError` with a clear message: `"PSpline requires a single-column predictor matrix (n, 1)."`.

**Warning signs:** No error at fit time; predict returns wrong shape if x has extra columns.

### Pitfall 3: GammaRegressor convergence flag not checked

**What goes wrong:** IRLS diverges with `error_on_non_convergence(false)`. The `fitted` field is
set to `Some(FittedGamma { inner: FittedTweedie { converged: false, ... } })`. All predictions
are made on a non-converged fit without warning.

**How to avoid:** Either default `error_on_non_convergence` to `true` (Python raises an exception
on divergence) OR expose `converged()` as a Python property and document that users should check it.
The PyModel should default to `error_on_non_convergence=true` matching the crate default.

### Pitfall 4: MomentAccumulator push_row arity error

**What goes wrong:** `push_row` validates `x_row.len() == n_features` and returns
`Err(RegressionError::DimensionMismatch)` if the lengths differ. If the Python wrapper converts
to `Vec<f64>` silently (without checking), the error propagates only after accumulation — wasted work.

**How to avoid:** Validate `x_row_array.len() == self.n_features` before calling `push_row`, and
raise Python `ValueError` immediately with a clear message.

### Pitfall 5: Shared registration file conflicts in same-wave tasks

**What goes wrong:** Two tasks in the same plan wave both edit `src/pymodels/mod.rs` to add their
respective `mod pyXXX; pub use pyXXX::PyXXX;` lines. The second edit overwrites or conflicts with
the first.

**How to avoid:** Always have a single dedicated "registration" task per wave that edits
`src/pymodels/mod.rs`, `src/lib.rs`, and `python/polars_statistics/__init__.py` after all PyModel
files are written. Never put registration edits in the same task as the PyModel file creation.

### Pitfall 6: HC extension calling wrong residuals for Ridge

**What goes wrong:** Ridge coefficients are shrinkage-biased; the residuals `y - Xβ̂_ridge` are
NOT the same as OLS residuals. HC2/HC3 use leverage values from `(XᵀX + λI)⁻¹Xᵀ` (penalized hat
matrix), not `X(XᵀX)⁻¹Xᵀ`. Using `compute_leverage` (which uses the unpenalized hat matrix) for
Ridge HC SEs is statistically incorrect.

**How to avoid:** For `PyRidge.hc_inference()`, pass the Ridge residuals from
`fitted.result().residuals` — these are already the correct penalized residuals. For the leverage,
either use `compute_leverage_with_aliased` (unpenalized — consistent with OLS HC implementation and
widely used in practice) or document the caveat. The safest approach: match the OLS implementation
exactly and document that HC for Ridge uses the OLS leverage formula.

### Pitfall 7: LarsMethod enum — string vs enum in Python

**What goes wrong:** Exposing `LarsMethod` as a Rust enum in PyO3 requires extra registration.
If forgotten, the Python user cannot construct a LARS model with the Lasso variant.

**How to avoid:** Accept `method: &str` in the Python constructor (`"lar"` / `"lasso"`) and convert
to `LarsMethod` in the builder call:
```rust
let method = match method_str { "lasso" => LarsMethod::Lasso, _ => LarsMethod::Lar };
```
Do not register `LarsMethod` as a `#[pyclass]` enum — string dispatch is simpler and matches
the existing convention for `HcType`, `SolverType`, `PaLoss`, etc.

---

## HC Extension — Recommended Scope

Based on statistical correctness and implementation effort:

| Regressor | HC Extension | Rationale |
|-----------|-------------|-----------|
| `OLS` | Already done | Template |
| `Ridge` | YES — add `hc_inference()` to `PyRidge` | Ridge residuals available; common use case |
| `WLS` | YES — add `hc_inference()` to `PyWLS` | WLS residuals available; HC corrects for model misspecification on top of WLS |
| `ElasticNet` | SKIP | Highly regularized; HC SEs on penalized fit are misleading |
| `BayesianRidge`/`ARD` | SKIP | Posterior SEs already available via `sigma_diag`; HC conflates Bayesian and frequentist inference |
| GLMs (Poisson/Binomial/etc.) | SKIP | Requires quasi-likelihood sandwich — different from OLS HC; not what `compute_hc_inference` implements |
| `TheilSen`/`RANSAC` | SKIP | Robust estimators don't benefit from HC; sampling distribution is not derived from OLS residuals |

This covers 2 additional regressors (Ridge, WLS) with minimal risk of statistical misuse.

---

## Validation Architecture

> `workflow.nyquist_validation` absent from `.planning/config.json` — treat as enabled.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (7.0+) |
| Config file | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| Quick run command | `maturin develop --features python && pytest tests/ -x -q` |
| Full suite command | `maturin develop --features python && pytest tests/ -v --tb=short` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Status |
|--------|----------|-----------|-------------------|-------------|
| REGR-01 | `GLMM.fit(X, y, group)` returns finite fixed_effects | smoke | `pytest tests/test_glmm.py -x` | Wave 0 gap |
| REGR-01 | `GLMM.fit_crossed(X, y, [g1, g2])` returns FactorSummary | smoke | `pytest tests/test_glmm.py::test_fit_crossed -x` | Wave 0 gap |
| REGR-02 | `PSpline.fit(X, y)` returns finite edf, r_squared > 0 | smoke | `pytest tests/test_pspline.py -x` | Wave 0 gap |
| REGR-03 | `Gamma.fit(X, y)` returns finite coefficients, converged=True | smoke | `pytest tests/test_gamma.py -x` | Wave 0 gap |
| REGR-04 | `OLS.hc_inference(X).std_errors` all finite | smoke | `pytest tests/test_ols.py::test_hc -x` | Partial — extend |
| REGR-04 | `Ridge.hc_inference(X).std_errors` all finite | smoke | `pytest tests/test_ridge.py::test_hc -x` | Wave 0 gap |
| REGR-05 | `ps.glm_dispersion_deviance(...)` returns finite float | smoke | `pytest tests/test_glm_diagnostics.py -x` | Wave 0 gap |
| REGR-06 | `TheilSen.fit(X, y).coefficients` finite, `predict()` shape == n | smoke | `pytest tests/test_theil_sen.py -x` | Wave 0 gap |
| REGR-06 | `RANSAC.fit(X, y).inlier_mask` length == n | smoke | `pytest tests/test_ransac.py -x` | Wave 0 gap |
| REGR-06 | `BayesianRidge.fit(X, y).alpha` finite | smoke | `pytest tests/test_bayesian.py -x` | Wave 0 gap |
| REGR-06 | `ARD.fit(X, y).lambdas` length == n_features | smoke | `pytest tests/test_bayesian.py::test_ard -x` | Wave 0 gap |
| REGR-06 | `LARS.fit(X, y).alphas` non-empty | smoke | `pytest tests/test_lars.py -x` | Wave 0 gap |
| REGR-06 | `PassiveAggressive.fit(X, y)` then `predict` shape == n | smoke | `pytest tests/test_pa.py -x` | Wave 0 gap |
| REGR-06 | `MomentAccumulator(p).push_row(x, y)` then `OLS.fit_from_accumulator` finite | smoke | `pytest tests/test_moments.py -x` | Wave 0 gap |

### Sampling Rate

- **Per task commit:** `maturin develop --features python && pytest tests/test_{new_model}.py -x -q`
- **Per wave merge:** `maturin develop --features python && pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps (test files to create before or alongside implementation)

- [ ] `tests/test_gamma.py` — REQ-REGR-03
- [ ] `tests/test_glmm.py` — REQ-REGR-01
- [ ] `tests/test_pspline.py` — REQ-REGR-02
- [ ] `tests/test_theil_sen.py` — REQ-REGR-06a
- [ ] `tests/test_ransac.py` — REQ-REGR-06b
- [ ] `tests/test_bayesian.py` — REQ-REGR-06c (both BayesianRidge and ARD)
- [ ] `tests/test_lars.py` — REQ-REGR-06d
- [ ] `tests/test_pa.py` — REQ-REGR-06e
- [ ] `tests/test_moments.py` — REQ-REGR-06f
- [ ] `tests/test_glm_diagnostics.py` — REQ-REGR-05 (GLM dispersion + standardized residuals)
- [ ] Extend `tests/test_ridge.py` — add HC inference test

Each test file needs: one smoke test (fit on a small design, assert finite coefficients, predict
shape matches n, check `is_fitted()` returns True). Comprehensive R-validation is Phase 6.

---

## Recommended Plan / Wave Decomposition

### Wave 0 — Infrastructure (create test stubs)

**Purpose:** Create all smoke test files before implementation begins. These are the gate at each
task commit.

Tasks:
- Create `tests/test_gamma.py`, `test_glmm.py`, `test_pspline.py` (one task)
- Create `tests/test_theil_sen.py`, `test_ransac.py`, `test_bayesian.py`, `test_lars.py`,
  `test_pa.py`, `test_moments.py`, `test_glm_diagnostics.py` (one task)

### Wave 1 — Tracer: GammaRegressor (REGR-03)

**Purpose:** Establish the end-to-end PyModel-add pattern for GLM-family solvers. The simplest
new regressor (thin Tweedie wrapper).

Tasks:
1. `src/pymodels/py_gamma.rs` — implement `PyGamma` (fit/predict/is_fitted/converged/predict_mu/predict_eta/coefficients/intercept/std_errors/p_values/aic/bic getters)
2. Registration: `src/pymodels/mod.rs` + `src/lib.rs` + `python/polars_statistics/__init__.py`
3. Smoke test: `pytest tests/test_gamma.py -x`

**Shared-file conflict mitigation:** All three registration files are touched in a single task (task 2) AFTER task 1 completes.

### Wave 2 — HIGH Priority: GlmmRegressor + PSplineRegressor (REGR-01/02)

**Purpose:** Largest and most novel models. Parallelisable (independent PyModel files).

Tasks (can run in parallel):
- 2a: `src/pymodels/py_glmm.rs` — implement `PyGLMM` with `gaussian()`/`poisson()`/`binomial()` factory methods, `fit(x, y, group)`, `fit_crossed(x, y, groups)`, all `FittedGlmm` getters, `FactorSummary` exposed as Python list of dicts
- 2b: `src/pymodels/py_pspline.rs` — implement `PyPSpline` with `fit(x, y)`, `predict`, `predict_with_interval`, `edf`/`sigma2` getters
- 2c (after 2a+2b): Registration for both + smoke tests

**FactorSummary exposure strategy:** Return `factors()` output as Python `list[dict]` where each dict has keys `n_levels: int`, `sd: float`, `blups: np.ndarray`. Do not create a separate `PyFactorSummary` class — a dict is simpler and avoids extra class registration.

### Wave 3 — GLM Diagnostics + HC Extension (REGR-04/05)

**Purpose:** Diagnostic expressions (in `regression.rs`) and HC method additions to existing PyModels.

Tasks (can run in parallel):
- 3a: `src/expressions/regression.rs` — add 5 new `#[polars_expr]` functions:
  `glm_dispersion_deviance_fit`, `glm_dispersion_pearson_fit`, `pearson_chi_squared_glm_fit`,
  `standardized_deviance_residuals_fit`, `standardized_pearson_residuals_fit`
  + Python expression builders in `exprs/regression.py`
- 3b: `src/pymodels/py_ridge.rs` — add `hc_inference(x, hc_type)` method to `PyRidge`
- 3c: `src/pymodels/py_wls.rs` — add `hc_inference(x, hc_type)` method to `PyWLS`
- 3d (after 3a): Registration for new expression names in `__init__.py` / `exprs/__init__.py`

### Wave 4 — MEDIUM Solvers: TheilSen, RANSAC, BayesianRidge, ARD (REGR-06a/b/c)

**Purpose:** Standard sklearn-style solvers. Similar PyModel pattern; parallelisable.

Tasks (can run in parallel):
- 4a: `src/pymodels/py_theil_sen.rs` — `PyTheilSen`
- 4b: `src/pymodels/py_ransac.rs` — `PyRansac` (with `inlier_mask` getter)
- 4c: `src/pymodels/py_bayesian_ridge.rs` + `src/pymodels/py_ard.rs` — `PyBayesianRidge` and `PyARD`
- 4d (after 4a-4c): Registration for all four + smoke tests

### Wave 5 — Streaming: LARS, PassiveAggressive, MomentAccumulator (REGR-06d/e/f)

**Purpose:** Path-based and online models. Most novel Python API design.

Tasks (can run in parallel):
- 5a: `src/pymodels/py_lars.rs` — `PyLARS` with `alphas` getter and `method="lar"/"lasso"` string param
- 5b: `src/pymodels/py_passive_aggressive.rs` — `PyPassiveAggressive` with `partial_fit(x_row, y_value)` and `PaState`-based streaming
- 5c: `src/pymodels/py_moment_accumulator.rs` — `PyMomentAccumulator`; ALSO extend `py_ols.rs` and `py_ridge.rs` to add `fit_from_accumulator(acc)` method
- 5d (after 5a-5c): Registration for all three + smoke tests

### Wave 6 — Final validation and CI check

**Purpose:** Full pytest suite + clippy + ruff green.

Tasks:
- Run `cargo clippy -- -D warnings` and fix any new warnings
- Run `ruff check python/` and fix
- Run full pytest suite

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual IRLS + Tweedie wrapper | `GammaRegressor` as thin re-skin | 0.5.5 | Simpler for users, identical numerics |
| Single-factor GLMM only | `fit_crossed` for crossed/nested factors | 0.5.12 | Multi-factor random effects now supported |
| Offline-only regression | `MomentAccumulator` + `fit_from_moments` | 0.5.9 | Large-panel streaming without materializing N×p matrix |

---

## Security Domain

> `security_enforcement` absent from config — treat as enabled.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | No auth in library API |
| V3 Session Management | No | Library, no sessions |
| V4 Access Control | No | No access control required |
| V5 Input Validation | Yes | Array shape/size checks before `to_faer()` conversion; group array length must match n |
| V6 Cryptography | No | No cryptographic operations |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds on group array | Tampering | Validate `group.len() == x.nrows()` before calling `GlmmRegressor::fit` |
| Integer overflow on `n_features: usize` in `MomentAccumulator::new()` | Tampering | `faer::Mat::zeros(p, p)` allocation for large p — panic on OOM rather than UB |
| Malformed LARS path causing empty `coefs` vector | Tampering | `last_coefs()` panics on empty path — validate path is non-empty after `lars_path()` |

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `maturin` | Building wheels | ✓ | 1.7.4+ (per CLAUDE.md) | — |
| Rust stable toolchain | Compilation | ✓ | (project already building) | — |
| Python 3.9+ with venv | pytest | ✓ | (project already testing) | — |
| `anofox-regression` 0.5.13 | All new solvers | ✓ | 0.5.13 in Cargo.lock | — |

Step 2.6: No new external dependencies. All gap capability lives in the already-locked
`anofox-regression = "0.5.13"` dependency.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `FittedRidge` and `FittedWls` have residuals accessible via `result().residuals` which are correct inputs to `compute_hc_inference` | HC Extension section | LOW — `FittedRegressor::result()` is the standard accessor; would only fail if Ridge residuals use a different path |
| A2 | `compute_hc_inference` is importable from `anofox_regression::inference` without the `python` feature gate | HC Extension section | LOW — `src/inference/mod.rs` is in the main library path, not gated; confirmed by phase 2 research |
| A3 | `FittedLars.coefs_path` is private (not pub-exposed); only `alphas()` is public | LarsRegressor section | LOW — confirmed by reading the struct definition and `impl FittedLars`; `coefs_path` field is not annotated `pub` |
| A4 | OLS `fit_from_accumulator` and Ridge `fit_from_accumulator` methods exist on the respective regressor types | MomentAccumulator section | MEDIUM — the moments module docs describe these methods but they were not read from `ols.rs`/`ridge.rs` directly this session. Must verify at implementation time. |
| A5 | `PaLoss` can be matched as `"epsilon_insensitive"` / `"squared_epsilon_insensitive"` strings in Python (no `#[pyclass]` needed) | PassiveAggressive section | LOW — established pattern: `HcType`, `LarsMethod`, `SolverType` all use string dispatch |

---

## Open Questions

1. **Does `OlsRegressor` expose `fit_from_accumulator(&acc)`?**
   - What we know: `MomentAccumulator` module docs reference `OlsRegressor::fit_from_moments` and `fit_from_accumulator` methods.
   - What's unclear: Exact signature (is it `fit_from_accumulator` or `fit_from_moments`?); whether `PyOLS` already imports these.
   - Recommendation: Read `src/solvers/ols.rs` (first 100 lines) at implementation time to confirm the exact method name before writing `PyOLS.fit_from_accumulator`.

2. **GLM diagnostic expressions — family dispatch pattern**
   - What we know: `standardized_pearson_residuals` and `standardized_deviance_residuals` take `&dyn GlmFamily` — a trait object. The existing expressions for Logistic/Poisson hardcode the family inside the expression function.
   - What's unclear: Whether to implement one generic expression with a `family: &str` parameter or separate expressions per family (matching the existing `logistic_pearson_residuals`/`poisson_pearson_residuals` pattern).
   - Recommendation: Match the existing pattern (separate per-family expressions: `gamma_standardized_pearson_residuals`, etc.) to avoid string dispatch overhead in hot paths. Wave 3 task 3a decides the naming.

---

## Sources

### Primary (HIGH confidence — read from local crate source files this session)

- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/gamma.rs` — complete GammaRegressor API
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/glmm.rs:1-960` — complete GlmmRegressor + FittedGlmm + FactorSummary API
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/pspline.rs` — complete PSplineRegressor API
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/theil_sen.rs` — complete TheilSenRegressor API
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/ransac.rs` — complete RansacRegressor API
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/bayesian.rs` — complete BayesianRidge + ArdRegression API
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/lars.rs:1-470` — complete LarsRegressor + FittedLars API
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/passive_aggressive.rs` — complete PassiveAggressiveRegressor + PaState API
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/solvers/moments.rs` — complete MomentAccumulator API
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/inference/robust_covariance.rs:1-120` — HcInference/HcResult/compute_hc_inference signatures
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/diagnostics/glm_residuals.rs:1-130` — five missing diagnostic function signatures
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-regression-0.5.13/src/diagnostics/mod.rs:63-81` — diagnostics pub re-exports
- `src/pymodels/py_tweedie.rs` — full PyTweedie (primary template for GLM-family PyModels)
- `src/pymodels/py_huber.rs:1-60` — PyHuber (template for robust solver PyModels)
- `src/pymodels/py_ols.rs:350-413` — OLS hc_inference method (HC extension template)
- `src/pymodels/py_ols.rs:1-110` — OLS fit method pattern (ToFaer usage)
- `src/pymodels/mod.rs` — current PyModel registration list
- `src/lib.rs:1-85` — #[pymodule] block
- `python/polars_statistics/__init__.py:1-70` — Python public surface

### Secondary (MEDIUM confidence)

- Phase 2 `02-RESEARCH.md` and `02-API-AUDIT.md` — gap list and crate source enumeration
  (enumeration done in Phase 2 session, not re-verified beyond spot-check this session)

---

## Metadata

**Confidence breakdown:**
- Crate API signatures: HIGH — all read verbatim from source files this session
- PyModel implementation pattern: HIGH — read from existing py_tweedie.rs, py_huber.rs, py_ols.rs
- Wave decomposition: HIGH — based on file dependency analysis and established pattern
- HC extension scope: MEDIUM — Ridge/WLS HC applicability is a judgment call; statisticians may disagree

**Research date:** 2026-08-12
**Valid until:** 2027-02-12 (pinned crate version `anofox-regression = "0.5.13"`; source is immutable)
