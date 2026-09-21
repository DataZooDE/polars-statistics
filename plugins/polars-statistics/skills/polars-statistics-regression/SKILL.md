---
name: polars-statistics-regression
description: Use polars-statistics regression — fit OLS, Ridge, Lasso, Elastic Net, WLS, RLS, BLS, NNLS, Quantile, Huber, PLS, ALM, or any GLM (logistic, Poisson, Negative Binomial, Tweedie, Probit, Cloglog) as Polars expressions or as PyModel classes; use R-style formulas with interactions and polynomial terms.
version: "0.8.0"
user-invocable: false
---

## Overview

polars-statistics exposes a full regression library through two orthogonal API styles:

- **Expression API** — decorate `group_by`/`over` pipelines; per-group fitting; results are Polars Struct columns.
- **PyModel API** — `from polars_statistics import OLS; m = OLS().fit(X, y)` — sklearn-style imperative interface returning numpy arrays.

All expression families have fit, tidy-summary, per-row predict, and R-formula variants. Diagnostics (VIF, Cook's distance, leverage, residual outliers) are standalone expressions.

## When to use this skill

Route here when the task involves:

- Fitting or predicting with any regression model (linear, regularized, robust, GLM, mixed, demand)
- Calling any PyModel class: OLS, Ridge, ElasticNet, WLS, RLS, BLS, Logistic, Poisson, NegativeBinomial, Tweedie, Probit, Cloglog, Gamma, ALM, GLMM, Quantile, Huber, PLS, Isotonic, TheilSen, RANSAC, BayesianRidge, ARD, LARS, PassiveAggressive, LmDynamic, Aid, MomentAccumulator, StationaryBootstrap, CircularBlockBootstrap, PSpline
- Using any regression expression: ols, ridge, elastic_net, lasso, wls, rls, expanding_ols, bls, nnls, quantile, pls, huber, isotonic, logistic, logistic_regression, poisson, negative_binomial, tweedie, probit, cloglog, alm, lm_dynamic, aid, aid_anomalies
- R-style formula strings: `"y ~ x1 + x2"`, `"y ~ x1 * x2"`, `"y ~ poly(x, 2)"`
- Diagnostics: VIF, condition_number, leverage, Cook's distance, dffits, residual outliers, GLM residuals
- Summary (tidy coefficient tables): `*_summary` expressions
- Predictions with confidence/prediction intervals: `*_predict` expressions
- sklearn-to-polars migration for any regression class

## API Reference

### Universal PyModel Interface

Every PyModel class shares this interface (exceptions noted per class):

```python
Model(hyperparams...)
  .fit(x, y)          -> Model  (chainable; x: np.ndarray shape (n,p), y: shape (n,))
  .predict(x)         -> np.ndarray
  .score(x, y)        -> float  (R²)
  .is_fitted()        -> bool
  .summary()          -> str    (human-readable coefficient table)
  .to_dict()          -> dict   (machine-readable)
```

Guard against unfitted access: accessing `.coefficients`, `.r_squared`, etc. on an unfitted model raises `PyRuntimeError("Model not fitted")`.

---

### Group 1 — Linear Models

```python
OLS(
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,   # deprecated — use add_intercept=
    compute_inference: bool = True,
    confidence_level: float = 0.95,
    solve_method: str | None = None,      # "qr" | "svd" | "cholesky"
)
  .fit(x, y) -> OLS
  .fit_from_accumulator(acc: MomentAccumulator) -> OLS
  .hc_inference(x, hc_type: str = "hc1") -> dict   # hc_type: "hc0" | "hc1" | "hc2" | "hc3"
  # Properties:
  .coefficients, .intercept
  .r_squared, .adj_r_squared, .mse, .rmse
  .f_statistic, .f_pvalue, .aic, .bic, .log_likelihood
  .p_values, .std_errors, .t_statistics
  .n_observations, .n_parameters
  .fitted_values, .residuals

Ridge(
    lambda_: float = 1.0,                 # regularization strength
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,   # deprecated
    compute_inference: bool = True,
    confidence_level: float = 0.95,
)
  .fit(x, y) -> Ridge
  .fit_from_accumulator(acc: MomentAccumulator) -> Ridge
  .hc_inference(x, hc_type: str = "hc1") -> dict
  .coefficients, .intercept, .r_squared, .adj_r_squared, .lambda_value

ElasticNet(
    lambda_: float = 1.0,                 # total regularization strength
    alpha: float = 0.5,                   # L1 ratio: 0 = Ridge, 1 = Lasso
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,   # deprecated
    max_iter: int = 1000,
    tol: float = 1e-4,
)
  .fit(x, y) -> ElasticNet
  .coefficients, .intercept, .r_squared, .lambda_value, .alpha_value

WLS(
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,   # deprecated
    compute_inference: bool = True,
    confidence_level: float = 0.95,
)
  .fit(x, y, weights) -> WLS             # 3-ARG fit — weights is required, not optional
  .hc_inference(x, hc_type: str = "hc1") -> dict
  .coefficients, .intercept, .r_squared

RLS(
    forgetting_factor: float = 1.0,       # 0 < factor <= 1; 1.0 = expanding OLS
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,   # deprecated
)
  .fit(x, y) -> RLS
  .coefficients, .intercept, .r_squared, .forgetting_factor_value

BLS(
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,   # deprecated
    max_iter: int = 1000,
    tol: float = 1e-6,
)
  .fit(x, y) -> BLS
  .nnls(with_intercept: bool = True, max_iter: int = 1000, tol: float = 1e-6) -> BLS
  .coefficients, .intercept, .r_squared
```

---

### Group 2 — Robust & Sklearn-style Solvers

```python
Quantile(
    tau: float = 0.5,                     # quantile to fit (0 < tau < 1)
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,   # deprecated
    max_iterations: int = 100,            # NOTE: max_iterations, not max_iter
    tolerance: float = 1e-6,             # NOTE: tolerance, not tol
)
  .fit(x, y) -> Quantile
  .coefficients, .intercept, .tau
  .pseudo_r_squared, .check_loss, .fitted_values, .residuals

Isotonic(
    increasing: bool = True,
    out_of_bounds: str = "clip",          # "clip" | "nan"
)
  .fit(x, y) -> Isotonic                 # NOTE: single x (1-D), not *x
  .fit_2d(x, y) -> Isotonic
  .predict(x) -> np.ndarray
  .predict_2d(x) -> np.ndarray
  .fitted_values, .residuals, .r_squared, .is_increasing, .x_thresholds, .y_values

Huber(
    epsilon: float = 1.35,
    alpha: float = 0.0001,                # regularization
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,   # deprecated
    max_iter: int = 100,
    tol: float = 1e-5,
)
  .fit(x, y) -> Huber
  .coefficients, .intercept, .r_squared, .mse, .rmse, .scale
  .n_outliers, .outliers, .epsilon

PLS(
    n_components: int = 2,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,   # deprecated
    tol: float = 1e-6,
    scale: bool = True,
)
  .fit(x, y) -> PLS
  .transform(x) -> np.ndarray
  .coefficients, .intercept, .r_squared, .n_components, .explained_variance_ratio

TheilSen(
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,   # deprecated
    max_subpopulation: int = 10000,
    n_subsamples: int | None = None,
    max_iter: int = 300,
    tol: float = 1e-3,
    random_state: int = 0,
)
  .fit(x, y) -> TheilSen
  .coefficients, .intercept, .r_squared, .mse, .rmse, .residuals

RANSAC(
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,   # deprecated
    min_samples: int | None = None,
    residual_threshold: float | None = None,
    max_trials: int = 100,
    stop_probability: float = 0.99,
    stop_n_inliers: int | None = None,
    random_state: int = 0,
)
  .fit(x, y) -> RANSAC
  .coefficients, .intercept, .r_squared, .inlier_mask, .n_inliers, .n_trials, .residuals

BayesianRidge(
    fit_intercept: bool = True,
    max_iter: int = 300,
    tol: float = 1e-3,
    alpha_1: float = 1e-6,
    alpha_2: float = 1e-6,
    lambda_1: float = 1e-6,
    lambda_2: float = 1e-6,
    alpha_init: float | None = None,
    lambda_init: float | None = None,
)
  .fit(x, y) -> BayesianRidge
  .coefficients, .intercept, .r_squared, .residuals, .alpha_, .lambda_, .sigma_diag

ARD(
    fit_intercept: bool = True,
    max_iter: int = 300,
    tol: float = 1e-3,
    alpha_1: float = 1e-6,
    alpha_2: float = 1e-6,
    lambda_1: float = 1e-6,
    lambda_2: float = 1e-6,
    threshold_lambda: float = 10000.0,
)
  .fit(x, y) -> ARD
  .coefficients, .intercept, .r_squared, .residuals, .alpha_, .lambdas
```

---

### Group 3 — Streaming & Online

```python
LARS(
    method: str = "lar",                  # "lar" | "lasso"
    fit_intercept: bool = True,
    n_nonzero_coefs: int | None = None,
    alpha: float = 0.0,
    standardize: bool = False,
)
  .fit(x, y) -> LARS
  .coefficients, .intercept, .r_squared, .alphas

PassiveAggressive(
    c: float = 1.0,                       # regularization (inverse strength)
    epsilon: float = 0.1,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,   # deprecated
    max_iter: int = 1000,
    tol: float = 1e-3,
    shuffle: bool = True,
    loss: str = "epsilon_insensitive",    # "epsilon_insensitive" | "squared_epsilon_insensitive"
    random_state: int = 0,
)
  .fit(x, y) -> PassiveAggressive
  .partial_fit(x_row, y_value) -> None   # online incremental update (single row)
  .predict_from_state(x) -> np.ndarray
  .coefficients, .intercept, .r_squared, .n_iter

MomentAccumulator(n_features: int)
  .push_row(x_row, y: float) -> None     # add one observation
  .merge(other: MomentAccumulator) -> None
  .clear() -> None
  .n, .n_features, .sum_x, .sum_y, .xtx, .xty
  # Used with: OLS.fit_from_accumulator(acc), Ridge.fit_from_accumulator(acc)
```

---

### Group 4 — Smoothers

```python
PSpline(
    n_basis: int = 0,                     # 0 = auto-select
    penalty_order: int = 2,
    lambda_: float | None = None,         # None = cross-validated
)
  .fit(x, y) -> PSpline                  # x is 1-D numpy array
  .predict(x) -> np.ndarray
  .coefficients, .r_squared, .rmse, .edf, .sigma2
```

---

### Group 5 — GLM Models

```python
Logistic(
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,   # deprecated
    compute_inference: bool = True,
    confidence_level: float = 0.95,
    max_iter: int = 25,
    tol: float = 1e-8,
    lambda_: float = 0.0,                 # L2 regularization
)
  .fit(x, y) -> Logistic                 # y: binary 0/1
  .predict(x, threshold: float = 0.5) -> np.ndarray   # hard labels
  .predict_proba(x) -> np.ndarray        # probabilities in [0,1]
  .predict_linear(x) -> np.ndarray       # linear predictor (log-odds)
  .score(x, y, threshold: float = 0.5) -> float
  .coefficients, .intercept, .aic, .bic, .p_values, .std_errors
  .deviance_residuals, .pearson_residuals

LogisticRegression(
    penalty: str = ...,                   # "l1" | "l2" | "elasticnet" | "none"
    C: float = 1.0,                       # sklearn-style inverse regularization strength
    threshold: float = 0.5,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,   # deprecated
    max_iter: int = 100,
    tol: float = 1e-8,
    compute_inference: bool = True,
    confidence_level: float = 0.95,
)
  .fit(x, y) -> LogisticRegression
  .decision_function(x) -> np.ndarray
  .predict(x) -> np.ndarray
  .predict_proba(x) -> np.ndarray
  .score(x, y) -> float
  .n_iter

Poisson(
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,   # deprecated
    compute_inference: bool = True,
    confidence_level: float = 0.95,
    max_iter: int = 25,
    tol: float = 1e-8,
    lambda_: float = 0.0,
)
  .fit(x, y) -> Poisson                  # y: non-negative counts
  .predict(x) -> np.ndarray              # expected counts
  .predict_linear(x) -> np.ndarray       # log(mu) linear predictor
  .coefficients, .intercept, .aic, .bic, .p_values, .std_errors
  .deviance_residuals, .pearson_residuals

NegativeBinomial(
    theta: float | None = None,           # dispersion; None = estimate from data
    estimate_theta: bool = True,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,   # deprecated
    max_iter: int = 100,
    tol: float = 1e-6,
    lambda_: float = 0.0,
)
  .fit(x, y) -> NegativeBinomial
  .predict(x) -> np.ndarray
  .coefficients, .intercept, .aic, .bic, .p_values, .std_errors, .theta_estimated

Tweedie(
    var_power: float = 1.5,               # 0=Gaussian, 1=Poisson, 2=Gamma, 3=InvGaussian
    link_power: float | None = None,      # None = canonical link
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,   # deprecated
    max_iter: int = 100,
    tol: float = 1e-6,
    lambda_: float = 0.0,
)
  .fit(x, y) -> Tweedie
  # Factory methods (alternative to __init__):
  .gaussian(with_intercept: bool = True, max_iter: int = 100, tol: float = 1e-6, lambda_: float = 0.0) -> Tweedie
  .gamma(with_intercept: bool = True, ...) -> Tweedie
  .inverse_gaussian(with_intercept: bool = True, ...) -> Tweedie
  .predict(x) -> np.ndarray
  .coefficients, .intercept, .aic, .bic, .p_values, .std_errors, .var_power_value

Probit(
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,   # deprecated
    max_iter: int = 100,
    tol: float = 1e-6,
    lambda_: float = 0.0,
)
  .fit(x, y) -> Probit                   # y: binary 0/1
  .predict(x) -> np.ndarray
  .predict_proba(x) -> np.ndarray
  .score(x, y, threshold: float = 0.5) -> float
  .coefficients, .intercept, .aic, .bic, .p_values, .std_errors

Cloglog(
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,   # deprecated
    max_iter: int = 100,
    tol: float = 1e-6,
    lambda_: float = 0.0,
)
  .fit(x, y) -> Cloglog                  # y: binary 0/1
  .predict(x) -> np.ndarray
  .predict_proba(x) -> np.ndarray
  .score(x, y, threshold: float = 0.5) -> float
  .coefficients, .intercept, .aic, .bic, .p_values, .std_errors

Gamma(
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,   # deprecated
    max_iter: int = 25,
    tol: float = 1e-8,
    lambda_: float = 0.0,
)
  .fit(x, y) -> Gamma                    # y: strictly positive
  .predict(x) -> np.ndarray
  .predict_eta(x) -> np.ndarray          # linear predictor
  .coefficients, .intercept, .aic, .bic, .p_values, .std_errors, .converged
```

---

### Group 6 — Mixed & Augmented Models

```python
GLMM()
  # No constructor params — configure via builder methods:
  .gaussian(
      add_intercept: bool | None = None,
      with_intercept: bool | None = None,  # deprecated
      reml: bool = True,
      max_iter: int = 100,
      tol: float = 1e-8,
  ) -> GLMM
  .binomial(add_intercept=None, with_intercept=None, reml=False, max_iter=100, tol=1e-8) -> GLMM
  .poisson(add_intercept=None, with_intercept=None, reml=False, max_iter=100, tol=1e-8) -> GLMM
  .fit(x, y, group) -> GLMM              # group: array of integer group IDs (Vec<u64>)
  .fit_crossed(x, y, groups) -> GLMM    # multiple crossed random effects
  .predict_fixed(x) -> np.ndarray
  .fixed_effects, .intercept, .random_effects, .random_sd, .slopes
  .sigma, .theta, .deviance, .log_likelihood
  .n_groups, .iterations, .converged, .std_errors
  .factors() -> list[dict]

ALM(
    distribution: str = "normal",
    link: str | None = None,              # None = canonical link for the distribution
    loss: str = "likelihood",             # "likelihood" | "ols" | ...
    role_trim: float | None = None,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,   # deprecated
    compute_inference: bool = True,
    confidence_level: float = 0.95,
    max_iter: int = 100,
    tol: float = 1e-8,
    extra_parameter: float | None = None,
)
  .fit(x, y) -> ALM
  .predict(x) -> np.ndarray
  .score(x, y) -> float
  .coefficients, .intercept, .aic, .bic, .scale
  .p_values, .std_errors, .fitted_values, .residuals, .n_observations
  .distribution, .link_function
  # Distribution builder methods (shortcuts — same kwargs):
  .normal(add_intercept=None, with_intercept=None, compute_inference=True) -> ALM
  .gamma(add_intercept=None, with_intercept=None, compute_inference=True) -> ALM
  .laplace(add_intercept=None, with_intercept=None, compute_inference=True) -> ALM
  .poisson(add_intercept=None, with_intercept=None, compute_inference=True) -> ALM
  .student_t(df: float = 5.0, add_intercept=None, with_intercept=None, compute_inference=True) -> ALM
  .beta(add_intercept=None, with_intercept=None, compute_inference=True) -> ALM
  # Supported distributions (25 families): "normal", "laplace", "student_t", "logistic",
  # "lognormal", "loglaplace", "gamma", "inverse_gaussian", "exponential",
  # "beta", "poisson", "negative_binomial", "binomial", "geometric", and others
```

---

### Group 7 — Dynamic & Demand Models

```python
LmDynamic(
    ic: str = "aicc",                     # "aic" | "aicc" | "bic"
    distribution: str = "normal",
    lowess_span: float = ...,             # LOWESS smoothing span
    max_models: int = ...,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,   # deprecated
    confidence_level: float = 0.95,
)
  .fit(x, y) -> LmDynamic
  .predict(x) -> np.ndarray
  .score(x, y) -> float
  .coefficients, .intercept, .r_squared, .adj_r_squared, .mse, .rmse
  .fitted_values, .residuals, .dynamic_coefficients, .model_weights, .pointwise_ic
  .smoothed_weights, .n_observations
  .coefficient_at(obs_index: int, coef_index: int) -> float | None
  .coefficients_at(obs_index: int) -> np.ndarray | None

Aid(
    intermittent_threshold: float = 0.3,
    detect_anomalies: bool = True,
    ic: str = "aicc",
)
  .classify(y) -> AidResult              # NOTE: .classify(), not .fit()
  .summary() -> str
  .to_dict() -> dict

AidResult
  .demand_type, .distribution, .mean, .variance, .scale, .shape
  .zero_prob, .zero_proportion, .is_fractional
  .n_observations, .ic_values, .anomalies
  .is_intermittent() -> bool
  .is_regular() -> bool
  .is_new_product() -> bool
  .is_obsolete_product() -> bool
  .anomaly_counts() -> dict
  .has_stockouts() -> bool
  .summary() -> str
  .to_dict() -> dict
```

---

### Group 8 — Bootstrap

```python
StationaryBootstrap(
    expected_block_length: float,         # expected block length (e.g. 10.0)
    seed: int | None = None,
)
  .sample(data: np.ndarray, length: int | None = None) -> np.ndarray    # one bootstrap sample
  .samples(data: np.ndarray, n_samples: int, length: int | None = None) -> np.ndarray  # n samples

CircularBlockBootstrap(
    block_length: int,                    # fixed block length
    seed: int | None = None,
)
  .sample(data: np.ndarray, length: int | None = None) -> np.ndarray
  .samples(data: np.ndarray, n_samples: int, length: int | None = None) -> np.ndarray
```

---

### Regression Expression — Core Fit

All fit expressions return a Polars Struct containing regression results. Use inside `group_by(...).agg(...)` for per-group fitting, or `.select(...)` for whole-frame.

```python
# Linear
ps.ols(y, *x, add_intercept=None, with_intercept=None, solve_method=None)
ps.ridge(y, *x, lambda_=1.0, add_intercept=None, with_intercept=None, solve_method=None)
ps.elastic_net(y, *x, lambda_=1.0, alpha=0.5, add_intercept=None, with_intercept=None)
ps.lasso(y, *x, lambda_=1.0, add_intercept=None, with_intercept=None)       # elastic_net(alpha=1.0)
ps.wls(y, weights, *x, add_intercept=None, with_intercept=None, solve_method=None)  # weights BEFORE *x
ps.rls(y, *x, forgetting_factor=0.99, add_intercept=None, with_intercept=None)
ps.expanding_ols(y, *x, add_intercept=None, with_intercept=None)             # rls(forgetting_factor=1.0)
ps.bls(y, *x, lower_bound=None, upper_bound=None, add_intercept=None, with_intercept=None)
ps.nnls(y, *x, add_intercept=None, with_intercept=None)                      # bls(lower_bound=0.0)
ps.quantile(y, *x, tau=0.5, add_intercept=None, with_intercept=None)
ps.pls(y, *x, n_components=2, add_intercept=None, with_intercept=None, tol=1e-6, scale=True)
ps.huber(y, *x, epsilon=1.35, alpha=0.0001, add_intercept=None, with_intercept=None, max_iter=100, tol=1e-5)
ps.isotonic(y, x, increasing=True)      # NOTE: single x column (not *x)

# GLMs
ps.logistic(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.logistic_regression(y, *x, penalty="l2", C=1.0, threshold=0.5, add_intercept=None, with_intercept=None, max_iter=100, tol=1e-8)
ps.poisson(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.negative_binomial(y, *x, theta=None, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.tweedie(y, *x, var_power=1.5, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.probit(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.cloglog(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.alm(y, *x, distribution="normal", link=None, loss="likelihood", role_trim=None, extra_parameter=None, add_intercept=None, with_intercept=None)

# Dynamic / Demand
ps.lm_dynamic(y, *x, ic="aicc", distribution="normal", lowess_span=0.3, max_models=64, add_intercept=None, with_intercept=None)
ps.aid(y, intermittent_threshold=0.3, detect_anomalies=True)
ps.aid_anomalies(y, intermittent_threshold=0.3)
```

**OLS output struct fields:** `intercept, coefficients, r_squared, adj_r_squared, mse, rmse, f_statistic, f_pvalue, aic, bic, n_observations`

---

### Regression Expressions — Summary (Tidy Coefficients)

All `*_summary` variants return `List[Struct]` with fields: `{term, estimate, std_error, statistic, p_value}`. Tidy: `.explode(col).unnest(col)`.

```python
ps.ols_summary(y, *x, add_intercept=None, with_intercept=None, solve_method=None, hc_type=None)
# hc_type: "hc0" | "hc1" | "hc2" | "hc3" | None (homoskedastic)

ps.ridge_summary(y, *x, lambda_=1.0, add_intercept=None, with_intercept=None, solve_method=None)
ps.elastic_net_summary(y, *x, lambda_=1.0, alpha=0.5, add_intercept=None, with_intercept=None)
ps.lasso_summary(y, *x, lambda_=1.0, add_intercept=None, with_intercept=None)
ps.wls_summary(y, weights, *x, add_intercept=None, with_intercept=None, solve_method=None)
ps.rls_summary(y, *x, forgetting_factor=0.99, add_intercept=None, with_intercept=None)
ps.bls_summary(y, *x, lower_bound=None, upper_bound=None, add_intercept=None, with_intercept=None)
ps.quantile_summary(y, *x, tau=0.5, add_intercept=None, with_intercept=None)
# Note: quantile_summary std_error/statistic/p_value are NaN (no analytic SEs)

ps.logistic_summary(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.poisson_summary(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.negative_binomial_summary(y, *x, theta=None, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.tweedie_summary(y, *x, var_power=1.5, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.probit_summary(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.cloglog_summary(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.alm_summary(y, *x, distribution="normal", add_intercept=None, with_intercept=None)
```

---

### Regression Expressions — Predict (Per-row)

All `*_predict` variants return per-row struct: `{<prefix>_prediction, <prefix>_lower, <prefix>_upper}`. Use `.over("group")` for per-group predictions back to the original DataFrame.

**Common kwargs for predict expressions:**

| kwarg | type | default | description |
|-------|------|---------|-------------|
| `add_intercept` | bool | True | include intercept |
| `interval` | str\|None | None | `None` \| `"confidence"` \| `"prediction"` |
| `level` | float | 0.95 | CI level |
| `null_policy` | str | `"drop"` | `"drop"` \| `"drop_y_zero_x"` |
| `name` | str\|None | None | output prefix (default = model name) |

```python
ps.ols_predict(y, *x, add_intercept=True, solve_method=None, interval=None, level=0.95, null_policy="drop", name=None)
ps.ridge_predict(y, *x, lambda_=1.0, add_intercept=True, solve_method=None, interval=None, level=0.95, null_policy="drop", name=None)
ps.elastic_net_predict(y, *x, lambda_=1.0, alpha=0.5, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.lasso_predict(y, *x, lambda_=1.0, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.wls_predict(y, weights, *x, add_intercept=True, solve_method=None, interval=None, level=0.95, null_policy="drop", name=None)
ps.rls_predict(y, *x, forgetting_factor=0.99, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.bls_predict(y, *x, lower_bound=None, upper_bound=None, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.nnls_predict(y, *x, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.quantile_predict(y, *x, tau=0.5, add_intercept=True, null_policy="drop", name=None)
# Note: quantile_predict has no interval= param (bootstrap intervals not yet available)
ps.isotonic_predict(y, x, increasing=True, null_policy="drop", name=None)   # single x; no intervals

ps.logistic_predict(y, *x, lambda_=0.0, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.poisson_predict(y, *x, lambda_=0.0, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.negative_binomial_predict(y, *x, lambda_=0.0, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.tweedie_predict(y, *x, var_power=1.5, lambda_=0.0, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.probit_predict(y, *x, lambda_=0.0, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.cloglog_predict(y, *x, lambda_=0.0, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.alm_predict(y, *x, distribution="normal", add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)

ps.lm_dynamic_predict(y, *x, ic="aicc", distribution="normal", lowess_span=0.3, max_models=64, add_intercept=True, null_policy="drop", name=None)
# Note: lm_dynamic_predict lower/upper are NaN (no analytic intervals for time-varying mixture)
```

`null_policy` values:
- `"drop"` — drop rows with any nulls for fitting; NaN predictions for null rows
- `"drop_y_zero_x"` — drop null-target rows for fitting, zero-fill null features, predict on all rows

---

### Regression Expressions — Formula-Based

All formula expressions parse R-style formula strings and construct the corresponding Polars expression. They are convenience wrappers around the base expressions.

**Formula fit variants:**

```python
ps.ols_formula(formula: str, add_intercept=None, with_intercept=None, solve_method=None)
ps.ridge_formula(formula: str, lambda_=1.0, add_intercept=None, with_intercept=None, solve_method=None)
ps.elastic_net_formula(formula: str, lambda_=1.0, alpha=0.5, add_intercept=None, with_intercept=None)
ps.lasso_formula(formula: str, lambda_=1.0, add_intercept=None, with_intercept=None)
ps.wls_formula(formula: str, weights, add_intercept=None, with_intercept=None, solve_method=None)
# Note: wls_formula takes a weights positional arg (column name or expr) after formula
ps.rls_formula(formula: str, forgetting_factor=0.99, add_intercept=None, with_intercept=None)
ps.bls_formula(formula: str, lower_bound=None, upper_bound=None, add_intercept=None, with_intercept=None)
ps.nnls_formula(formula: str, add_intercept=None, with_intercept=None)
ps.logistic_formula(formula: str, add_intercept=None, with_intercept=None)
ps.poisson_formula(formula: str, add_intercept=None, with_intercept=None)
ps.negative_binomial_formula(formula: str, theta=None, add_intercept=None, with_intercept=None)
ps.tweedie_formula(formula: str, var_power=1.5, add_intercept=None, with_intercept=None)
ps.probit_formula(formula: str, add_intercept=None, with_intercept=None)
ps.cloglog_formula(formula: str, add_intercept=None, with_intercept=None)
ps.alm_formula(formula: str, distribution="normal", add_intercept=None, with_intercept=None)
```

**Formula summary variants** (return `List[Struct{term,estimate,std_error,statistic,p_value}]`):

```python
ps.ols_formula_summary(formula: str, add_intercept=None, with_intercept=None, solve_method=None, hc_type=None)
ps.ridge_formula_summary(formula: str, lambda_=1.0, add_intercept=None, with_intercept=None, solve_method=None)
ps.elastic_net_formula_summary(formula: str, lambda_=1.0, alpha=0.5, add_intercept=None, with_intercept=None)
ps.lasso_formula_summary(formula: str, lambda_=1.0, add_intercept=None, with_intercept=None)
ps.logistic_formula_summary(formula: str, add_intercept=None, with_intercept=None)
ps.poisson_formula_summary(formula: str, add_intercept=None, with_intercept=None)
ps.alm_formula_summary(formula: str, distribution="normal", add_intercept=None, with_intercept=None)
```

**Formula predict variants** (same `interval`/`level`/`null_policy`/`name` kwargs as base predict):

```python
ps.ols_formula_predict(formula: str, add_intercept=True, solve_method=None, interval=None, level=0.95, null_policy="drop", name=None)
ps.ridge_formula_predict(formula: str, lambda_=1.0, add_intercept=True, solve_method=None, interval=None, level=0.95, null_policy="drop", name=None)
ps.elastic_net_formula_predict(formula: str, lambda_=1.0, alpha=0.5, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.lasso_formula_predict(formula: str, lambda_=1.0, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.logistic_formula_predict(formula: str, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.poisson_formula_predict(formula: str, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.alm_formula_predict(formula: str, distribution="normal", add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
```

---

### Regression Expressions — Diagnostics

```python
# Structural / multicollinearity:
ps.condition_number(*x, add_intercept=None, with_intercept=None)
ps.vif(*x)
ps.generalized_vif(*x, group_sizes: list[int])          # GVIF for grouped predictors
ps.high_vif_predictors(*x, threshold=10.0)

# Influence:
ps.leverage(*x, add_intercept=None, with_intercept=None)
ps.cooks_distance(y, *x, add_intercept=None, with_intercept=None)
ps.dffits(y, *x, add_intercept=None, with_intercept=None)
ps.influential_cooks(y, *x, threshold=None, add_intercept=None, with_intercept=None)
ps.influential_dffits(y, *x, threshold=None, add_intercept=None, with_intercept=None)
ps.high_leverage_points(*x, threshold=None, add_intercept=True)   # no y arg

# Residuals:
ps.standardized_residuals(y, *x, add_intercept=None, with_intercept=None)
ps.studentized_residuals(y, *x, add_intercept=None, with_intercept=None)
ps.externally_studentized_residuals(y, *x, add_intercept=None, with_intercept=None)
ps.residual_outliers(y, *x, threshold=2.0, add_intercept=None, with_intercept=None)

# GLM model checks:
ps.check_binary_separation(y, *x)
ps.check_count_sparsity(y, *x)

# GLM residual diagnostics — Logistic:
ps.logistic_pearson_residuals(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.logistic_deviance_residuals(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.logistic_working_residuals(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.pearson_chi_squared_logistic(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)

# GLM residual diagnostics — Poisson:
ps.poisson_pearson_residuals(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.poisson_deviance_residuals(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.poisson_working_residuals(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.pearson_chi_squared_poisson(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)

# GLM residual diagnostics — Gamma:
ps.gamma_dispersion_deviance(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.gamma_dispersion_pearson(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.gamma_pearson_chi_squared(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.gamma_standardized_pearson_residuals(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.gamma_standardized_deviance_residuals(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
```

---

### R-style Formula Parser

End users typically call `ps.ols_formula(...)` — use the parser directly only for advanced expansion.

```python
from polars_statistics.formula.parser import FormulaParser
from polars_statistics.formula.expander import expand_terms_to_expressions

parser = FormulaParser()
parsed = parser.parse("y ~ x1 + x2 + x1:x2")
# parsed.response -> "y"
# parsed.terms -> list of Term objects

x_exprs, term_names = expand_terms_to_expressions(parsed.terms)
# x_exprs: list of pl.Expr ready to pass to ps.ols(response, *x_exprs)
# term_names: list of str (e.g. ["x1", "x2", "x1:x2"])
```

**Formula syntax reference:**

| Syntax | Meaning |
|--------|---------|
| `"y ~ x1 + x2"` | Main effects only |
| `"y ~ x1 * x2"` | Full crossing: expands to `x1 + x2 + x1:x2` |
| `"y ~ x1:x2"` | Interaction term only (no main effects) |
| `"y ~ poly(x, 2)"` | Orthogonal polynomial (Legendre, degree 2, centered within group) |
| `"y ~ poly(x, 2, raw=True)"` | Raw (non-centered) polynomial powers |
| `"y ~ I(x^2)"` | Explicit power transform |

## Worked Examples

### Example 1: OLS via Expression API (group_by)

```python
import polars as pl
import polars_statistics as ps
import numpy as np

rng = np.random.default_rng(42)
n = 200
df = pl.DataFrame({
    "group": ["A"] * 100 + ["B"] * 100,
    "y": rng.normal(0, 1, n).tolist(),
    "x1": rng.normal(0, 1, n).tolist(),
    "x2": rng.normal(0, 1, n).tolist(),
})

# Per-group OLS — returns struct per group
result = df.group_by("group").agg(
    ps.ols("y", "x1", "x2").alias("model")
)
# Extract R²:
result.with_columns(
    pl.col("model").struct.field("r_squared").alias("r2")
)

# Tidy coefficient table per group
coef_df = (
    df.group_by("group")
    .agg(ps.ols_summary("y", "x1", "x2").alias("coef"))
    .explode("coef")
    .unnest("coef")
)
```

### Example 2: OLS PyModel — sklearn-style

```python
import numpy as np
from polars_statistics import OLS

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (100, 2))
y = X[:, 0] * 2.0 + X[:, 1] * -1.5 + rng.normal(0, 0.5, 100)

m = OLS(add_intercept=True).fit(X, y)
assert m.is_fitted()
print(m.r_squared)          # float
print(m.summary())          # human-readable table
y_hat = m.predict(X)        # numpy array

# HC-robust inference:
hc = m.hc_inference(X, hc_type="hc3")
print(hc["std_errors"])

# Convert to dict:
d = m.to_dict()             # {"coefficients": [...], "r_squared": ..., ...}
```

### Example 3: Ridge regression

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({"y": [1., 2., 3., 4., 5.], "x1": [1., 2., 3., 4., 5.], "x2": [5., 4., 3., 2., 1.]})

# Expression API
result = df.select(ps.ridge("y", "x1", "x2", lambda_=0.1).alias("m"))

# PyModel API
import numpy as np
from polars_statistics import Ridge
X = df.select("x1", "x2").to_numpy()
y = df["y"].to_numpy()
m = Ridge(lambda_=0.1, add_intercept=True).fit(X, y)
print(m.coefficients, m.lambda_value)
```

### Example 4: Logistic regression with predictions

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "y":  [0., 0., 1., 1., 0., 1., 1., 0.],
    "x1": [1., 2., 3., 4., 1., 3., 5., 2.],
    "x2": [2., 1., 4., 3., 1., 5., 4., 2.],
})

# Per-row predicted probabilities using .over()
result = df.with_columns(
    ps.logistic_predict("y", "x1", "x2", interval="confidence", level=0.95)
    .alias("pred")
).unnest("pred")
# Columns: y, x1, x2, logistic_prediction, logistic_lower, logistic_upper
```

### Example 5: R-style formula with interactions

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "group": ["A"] * 60 + ["B"] * 60,
    "y":     [float(i) + (i % 3) for i in range(120)],
    "x1":    [float(i % 10) for i in range(120)],
    "x2":    [float((i + 5) % 8) for i in range(120)],
})

# Full crossing with formula — expands to x1 + x2 + x1:x2
result = df.group_by("group").agg(
    ps.ols_formula("y ~ x1 * x2").alias("model")
)

# Formula predict with prediction intervals
result = df.with_columns(
    ps.ols_formula_predict(
        "y ~ x1 * x2",
        interval="prediction",
        level=0.95,
        name="ols"
    ).over("group").alias("pred")
).unnest("pred")
```

### Example 6: Diagnostics — VIF and Cook's distance

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "y":  [1., 2., 3., 4., 5., 10.],   # last row is influential
    "x1": [1., 2., 3., 4., 5., 6.],
    "x2": [2., 4., 6., 8., 10., 12.],  # x2 = 2*x1 → high VIF
})

# Variance inflation factors
vif = df.select(ps.vif("x1", "x2"))

# Cook's distance to detect influential observations
cooks = df.select(ps.cooks_distance("y", "x1", "x2").alias("cooks_d"))

# Find influential points using default 4/n threshold
influential = df.select(ps.influential_cooks("y", "x1", "x2").alias("influential"))
```

### Example 7: WLS with weights

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "y":  [1., 2., 3., 4., 5.],
    "w":  [1., 2., 1., 2., 1.],   # observation weights
    "x":  [1., 2., 3., 4., 5.],
})

# Expression API: weights BEFORE *x
result = df.select(ps.wls("y", "w", "x").alias("m"))

# PyModel API: 3-arg fit(x, y, weights)
import numpy as np
from polars_statistics import WLS
X = df.select("x").to_numpy()
y = df["y"].to_numpy()
w = df["w"].to_numpy()
m = WLS(add_intercept=True).fit(X, y, w)
print(m.coefficients)
```

### Example 8: ALM with non-Gaussian distribution

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "y":  [0.1, 0.4, 0.6, 0.8, 0.9, 0.2, 0.7, 0.5],
    "x1": [1., 2., 3., 4., 5., 1., 3., 2.],
})

# Beta-distributed outcome (y in (0,1))
result = df.select(ps.alm("y", "x1", distribution="beta").alias("m"))

# Per-row predictions
result_pred = df.with_columns(
    ps.alm_predict("y", "x1", distribution="beta").alias("pred")
).unnest("pred")
```

## Gotchas / Notes

**Intercept kwarg deprecation.** Use `add_intercept=True/False` in all new code. The old parameter name emits `FutureWarning` at runtime. Both names are accepted for backward compatibility, but `add_intercept=` is the current name.

**WLS has a 3-argument fit.** The PyModel `WLS.fit` method takes three positional arguments: `fit(x, y, weights)`. Weights are required. In the expression API, the weights column comes before the feature columns: `ps.wls(y, weights, *x, ...)`.

**Quantile PyModel uses `max_iterations`/`tolerance`, not `max_iter`/`tol`.** This is the only PyModel that uses these longer names. `Quantile(max_iterations=200, tolerance=1e-8)` is correct; `max_iter` will raise `TypeError`.

**`ps.isotonic` takes a single x column, not varargs.** `ps.isotonic(y, x, increasing=True)` — pass one feature column. The PyModel `Isotonic.fit(x, y)` also takes 1-D arrays. Use `Isotonic.fit_2d` / `predict_2d` for 2-D input.

**`Aid.classify(y)` not `Aid.fit(y)`.** The demand-classification model uses `.classify()` instead of the universal `.fit()`. The result is an `AidResult` object.

**`ps.aid` expression has no `ic=` kwarg.** The expression form `ps.aid(y, intermittent_threshold=0.3, detect_anomalies=True)` does not expose the information criterion. Use the PyModel `Aid(ic="bic")` if you need to control IC.

**`quantile_predict` and `isotonic_predict` have no interval= parameter.** Quantile regression intervals require bootstrap (not yet wired); isotonic is a step function with no parametric uncertainty. These expressions' `lower`/`upper` fields are always NaN.

**`lm_dynamic_predict` lower/upper are always NaN.** No analytic intervals for the time-varying coefficient mixture model.

**`ps.wls` (expression) — weights position.** The argument order is `ps.wls(y, weights, *x, ...)` — weights comes second (before feature columns). This is different from `ps.ols(y, *x, ...)` where features immediately follow the target.

**`ps.high_leverage_points` takes no y argument.** `ps.high_leverage_points(*x, threshold=None, add_intercept=True)` — only feature columns, no target.

**`GLMM` is PyModel-only — no expression form.** Use `GLMM().gaussian(...).fit(x, y, group)` directly.

**`negative_binomial_formula` has a `theta=` kwarg.** `ps.negative_binomial_formula(formula, theta=None, add_intercept=None, ...)` — the theta dispersion parameter is exposed, unlike simpler formula variants.

**`ps.logistic_regression` (expression) vs `ps.logistic` (expression).** `logistic` is the lower-level IRLS GLM; `logistic_regression` is the sklearn-style wrapper with `penalty=`, `C=`, `threshold=`. For most use cases, prefer `ps.logistic`.

**`MomentAccumulator` for streaming OLS.** Push rows one at a time with `.push_row(x_row, y)`, then call `OLS().fit_from_accumulator(acc)` or `Ridge().fit_from_accumulator(acc)` to get the fitted model. Merge accumulators from parallel streams with `.merge(other)`.

## See also

- polars-statistics-hypothesis-tests — t-tests, Mann-Whitney, Kruskal-Wallis, normality, TOST equivalence
- polars-statistics-anova — one-way, two-way, repeated-measures ANOVA, ICC
- polars-statistics-results — struct extraction, unnest, to_dict/summary, error handling
- polars-statistics-getting-started — install, model-selection matrix, sklearn migration table
