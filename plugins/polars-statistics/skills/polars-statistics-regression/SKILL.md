---
name: polars-statistics-regression
description: Use polars-statistics regression — fit OLS, Ridge, Lasso, Elastic Net, WLS, RLS, BLS, NNLS, Quantile, Huber, PLS, ALM, or any GLM (logistic, Poisson, Negative Binomial, Tweedie, Probit, Cloglog, penalized GLM with priors and offsets), AFT survival regression with censoring, as Polars expressions or as PyModel classes; use R-style formulas with interactions and polynomial terms; get standard errors and p-values for Huber/RANSAC robust fits and GLMM fixed effects, or leverage/variance diagnostics for OLS/WLS/Ridge.
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

- Fitting or predicting with any regression model (linear, regularized, robust, GLM, mixed, demand, survival)
- Calling any PyModel class: OLS, Ridge, ElasticNet, WLS, RLS, BLS, Logistic, Poisson, NegativeBinomial, Tweedie, Probit, Cloglog, Gamma, PenalizedGLM, AFT, ALM, GLMM, Quantile, Huber, PLS, Isotonic, TheilSen, RANSAC, BayesianRidge, ARD, LARS, PassiveAggressive, LmDynamic, Aid, MomentAccumulator, StationaryBootstrap, CircularBlockBootstrap, PSpline
- Using any regression expression: ols, ridge, elastic_net, lasso, wls, rls, expanding_ols, bls, nnls, quantile, pls, huber, isotonic, logistic, logistic_regression, poisson, negative_binomial, tweedie, probit, cloglog, penalized_glm, alm, aft, lm_dynamic, aid, aid_anomalies
- R-style formula strings: `"y ~ x1 + x2"`, `"y ~ x1 * x2"`, `"y ~ poly(x, 2)"`
- Diagnostics: VIF, condition_number, leverage, Cook's distance, dffits, residual outliers, GLM residuals
- Summary (tidy coefficient tables): `*_summary` expressions
- Predictions with confidence/prediction intervals: `*_predict` expressions
- sklearn-to-polars migration for any regression class
- Survival / time-to-event data with right censoring (AFT: Weibull, log-normal, log-logistic, exponential)
- Count data with an exposure offset, or a Bayesian-style (Gaussian/Laplace) prior on a GLM coefficient (penalized GLM)

## API Reference

### Universal PyModel Interface

Every PyModel class shares this interface (exceptions noted per class):

```python
# verify: skip -- PyModel reference (indented method chaining with unicode R² and type annotations is illustrative, not valid Python)
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
# verify: skip -- PyModel class reference (constructor type annotations and indented method chaining are illustrative, not valid Python)
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
  .variance_factor() -> np.ndarray | None   # (p+1)x(p+1) with intercept, p x p without; None for moments-fits
  .leverage_new(x_new) -> np.ndarray        # hat-matrix leverage for new rows; ValueError on column-count mismatch
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
  .variance_factor() -> np.ndarray | None   # same contract as OLS.variance_factor()
  .leverage_new(x_new) -> np.ndarray
  .coefficients, .intercept, .r_squared, .adj_r_squared, .lambda_value
  # Inference getters (sandwich SEs; see the lambda_>0 gotcha below):
  .std_errors, .intercept_std_error        # always populated when compute_inference=True
  .p_values, .t_statistics                 # None when lambda_ > 0 (no valid t-test on shrunken coefficients)

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
  .variance_factor() -> np.ndarray | None   # same contract as OLS.variance_factor()
  .leverage_new(x_new) -> np.ndarray
  .coefficients, .intercept, .r_squared

RLS(
    forgetting_factor: float = 1.0,       # 0 < factor <= 1; 1.0 = expanding OLS
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,   # deprecated
    initial_p_diagonal: float = 1e6,      # P0 = initial_p_diagonal * I (prior covariance diagonal)
)
  .fit(x, y) -> RLS
  .coefficients, .intercept, .r_squared, .forgetting_factor_value, .initial_p_diagonal_value

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
# verify: skip -- PyModel class reference (constructor type annotations and indented method chaining are illustrative, not valid Python)
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
    compute_inference: bool = False,      # opt-in: MASS::rlm(method="XtX")-validated
    confidence_level: float = 0.95,
)
  .fit(x, y) -> Huber
  .coefficients, .intercept, .r_squared, .mse, .rmse, .scale
  .n_outliers, .outliers, .epsilon
  # Inference getters (None unless compute_inference=True; see Example 9):
  .std_errors, .t_statistics, .p_values
  .conf_interval_lower, .conf_interval_upper
  .intercept_std_error, .intercept_t_statistic, .intercept_p_value, .intercept_conf_interval

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
    univariate_pairwise: bool = True,     # True = classical median-of-pairwise-slopes (Sen 1968);
                                           # False = sklearn's spatial-median-over-subsamples estimator
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
    compute_inference: bool = False,
    confidence_level: float = 0.95,
)
  .fit(x, y) -> RANSAC
  .coefficients, .intercept, .r_squared, .inlier_mask, .n_inliers, .n_trials, .residuals
  # Fit statistics below are those of the inlier-only OLS fit (anofox-regression 0.5.14);
  # r_squared/n_observations above are also inlier-only, not whole-sample:
  .adj_r_squared, .mse, .rmse, .f_statistic, .f_pvalue, .aic, .bic, .log_likelihood
  # Same inference getters as Huber (None unless compute_inference=True):
  .std_errors, .t_statistics, .p_values
  .conf_interval_lower, .conf_interval_upper
  .intercept_std_error, .intercept_t_statistic, .intercept_p_value, .intercept_conf_interval
  .inlier_fit() -> OLS                  # a real fitted OLS instance on the inlier rows only;
                                         # coefficients/intercept/std_errors match RANSAC's own

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
# verify: skip -- PyModel class reference (constructor type annotations and indented method chaining are illustrative, not valid Python)
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
# verify: skip -- PyModel class reference (constructor type annotations and indented method chaining are illustrative, not valid Python)
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
# verify: skip -- PyModel class reference (constructor type annotations and indented method chaining are illustrative, not valid Python)
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
    estimate_dispersion: bool = False,    # False (default) = SEs use dispersion 1;
                                           # True = Pearson-dispersion SEs (R summary.glm(negative.binomial(theta)))
)
  .fit(x, y) -> NegativeBinomial
  .predict(x) -> np.ndarray
  .coefficients, .intercept, .aic, .bic, .p_values, .std_errors, .theta_estimated
  .dispersion                            # 1.0 by default; the estimated Pearson dispersion when estimate_dispersion=True

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
  # NOTE: Tweedie factory methods (.gaussian, .gamma, .inverse_gaussian) accept only
  # with_intercept= (not add_intercept=) — they have not yet been migrated to the new
  # parameter name. Use Tweedie().gaussian(with_intercept=True) not add_intercept=.
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

PenalizedGLM(
    family: str = "poisson",              # "poisson"|"binomial"|"logistic"|"negative_binomial"|"gamma"|"tweedie"
    link: str | None = None,              # None = family's canonical link
    theta: float | None = None,           # negative_binomial only; None = ML-estimated
    power: float | None = None,           # tweedie only; variance power in [1, 2], default 1.5
    threshold: float = 0.5,               # logistic only; classification threshold
    lambda_: float = 0.0,                 # uniform L2 ridge penalty (composes with priors)
    estimate_dispersion: bool = False,    # Poisson-only quasipoisson SEs; ValueError on other families
    priors: list | None = None,           # per-coefficient priors — see Priors gotcha below
    vcov_type: str = "laplace",           # "laplace" | "sandwich" | "naive"
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-8,
    compute_inference: bool = True,
    confidence_level: float = 0.95,
)
  .fit(x, y, offset=None) -> PenalizedGLM   # offset: array on the link scale, or None (= 0.0)
  .predict(x, offset=None) -> np.ndarray
  .predict_linear(x) -> np.ndarray
  .coefficients, .intercept, .family, .link
  .deviance, .null_deviance, .pseudo_r_squared, .log_likelihood, .aic, .bic
  .dispersion, .theta, .accuracy, .converged, .iterations, .n_observations
  .std_errors, .p_values, .z_values                         # None unless compute_inference=True
  .conf_interval_lower, .conf_interval_upper
  .intercept_std_error, .intercept_z_value, .intercept_p_value, .intercept_conf_interval
  .vcov, .information                    # scaled so sqrt(diag(vcov)) == std_errors
  .matrix_parameters                     # list[int | None] | None — solver-internal shapes
  .has_separation, .separated_predictors, .separation_warning
  .fitted_values, .valid_rows            # row indices kept after null/non-finite drop
```

---

### Group 6 — Mixed & Augmented Models

```python
# verify: skip -- PyModel class reference (indented builder methods with type annotations are illustrative, not valid Python)
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
  .z_values, .p_values                   # Wald inference: z = fixed_effects / std_errors
  .conf_int(level: float = 0.95) -> tuple[np.ndarray, np.ndarray]   # (lower, upper); all-NaN for invalid level
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
# verify: skip -- PyModel class reference (constructor type annotations and indented method chaining are illustrative, not valid Python)
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
# verify: skip -- PyModel class reference (constructor type annotations and indented method chaining are illustrative, not valid Python)
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

### Group 9 — Survival (AFT)

```python
# verify: skip -- PyModel class reference (constructor type annotations and indented method chaining are illustrative, not valid Python)
AFT(
    distribution: str = "weibull",        # "weibull"|"exponential"|"lognormal"|"loglogistic" (+ aliases, see Gotchas)
    priors: list | None = None,           # same format as PenalizedGLM's priors
    vcov_type: str = "laplace",           # "laplace" | "sandwich" | "naive"
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-9,
    compute_inference: bool = True,
    confidence_level: float = 0.95,
)
  .fit(x, time, event) -> AFT            # event: right-censoring indicator (bool or 0/1)
  .predict_quantile(x, p=0.5) -> np.ndarray
  .predict_median(x) -> np.ndarray
  .predict_survival(x, t) -> np.ndarray  # S(t|x) = P(T > t | x); t required, keyword-only on the expression
  .predict_cdf(x, t) -> np.ndarray       # F(t|x) = 1 - S(t|x)
  .predict_linear(x) -> np.ndarray
  .coefficients, .intercept, .scale, .distribution
  .log_likelihood, .null_log_likelihood, .aic, .bic
  .converged, .iterations, .n_observations, .n_events, .n_censored
  .std_errors, .p_values, .z_values      # None unless compute_inference=True
  .conf_interval_lower, .conf_interval_upper
  .intercept_std_error, .intercept_z_value, .intercept_p_value, .intercept_conf_interval
  .log_scale_std_error                   # SE of log(scale); None for exponential (scale fixed at 1)
  .vcov, .information                    # scaled so sqrt(diag(vcov)) == std_errors
```

---

### Regression Expression — Core Fit

All fit expressions return a Polars Struct containing regression results. Use inside `group_by(...).agg(...)` for per-group fitting, or `.select(...)` for whole-frame.

```python
# verify: skip -- expression API reference (y, *x are column-name placeholders, not runnable Python; ps not imported)
# Linear
ps.ols(y, *x, add_intercept=None, with_intercept=None, solve_method=None)
ps.ridge(y, *x, lambda_=1.0, add_intercept=None, with_intercept=None, solve_method=None)
ps.elastic_net(y, *x, lambda_=1.0, alpha=0.5, add_intercept=None, with_intercept=None)
ps.lasso(y, *x, lambda_=1.0, add_intercept=None, with_intercept=None)       # elastic_net(alpha=1.0)
ps.wls(y, weights, *x, add_intercept=None, with_intercept=None, solve_method=None)  # weights BEFORE *x
ps.rls(y, *x, forgetting_factor=0.99, add_intercept=None, with_intercept=None, initial_p_diagonal=1e6)
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
ps.negative_binomial(y, *x, theta=None, lambda_=0.0, add_intercept=None, with_intercept=None, estimate_dispersion=False)
ps.tweedie(y, *x, var_power=1.5, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.probit(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.cloglog(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.penalized_glm(y, *x, family="poisson", link=None, theta=None, power=None, threshold=0.5, lambda_=0.0, estimate_dispersion=False, priors=None, offset=None, add_intercept=True, max_iter=100, tol=1e-8)
ps.alm(y, *x, distribution="normal", link=None, loss="likelihood", role_trim=None, extra_parameter=None, add_intercept=None, with_intercept=None)

# Survival
ps.aft(time, event, *x, distribution="weibull", priors=None, add_intercept=True, max_iter=100, tol=1e-9)

# Dynamic / Demand
ps.lm_dynamic(y, *x, ic="aicc", distribution="normal", lowess_span=0.3, max_models=64, add_intercept=None, with_intercept=None)
ps.aid(y, intermittent_threshold=0.3, detect_anomalies=True)
ps.aid_anomalies(y, intermittent_threshold=0.3)
```

**OLS output struct fields:** `intercept, coefficients, r_squared, adj_r_squared, mse, rmse, f_statistic, f_pvalue, aic, bic, n_observations`

**`ps.penalized_glm` output struct fields:** `intercept, coefficients, deviance, null_deviance, pseudo_r_squared, log_likelihood, aic, bic, dispersion, theta, accuracy, has_separation, iterations, converged, n_observations` (15 fields; confirmed via live `df.select(...).schema`).

**`ps.aft` output struct fields:** `intercept, coefficients, scale, log_likelihood, null_log_likelihood, aic, bic, iterations, converged, n_observations, n_events, n_censored` (12 fields; confirmed via live `df.select(...).schema`).

---

### Regression Expressions — Summary (Tidy Coefficients)

All `*_summary` variants return `List[Struct]` with fields: `{term, estimate, std_error, statistic, p_value}`. Tidy: `.explode(col).unnest(col)`.

```python
# verify: skip -- expression API reference (y, *x are column-name placeholders, not runnable Python; ps not imported)
ps.ols_summary(y, *x, add_intercept=None, with_intercept=None, solve_method=None, hc_type=None)
# hc_type: "hc0" | "hc1" | "hc2" | "hc3" | None (homoskedastic)

ps.ridge_summary(y, *x, lambda_=1.0, add_intercept=None, with_intercept=None, solve_method=None)
ps.elastic_net_summary(y, *x, lambda_=1.0, alpha=0.5, add_intercept=None, with_intercept=None)
ps.lasso_summary(y, *x, lambda_=1.0, add_intercept=None, with_intercept=None)
ps.wls_summary(y, weights, *x, add_intercept=None, with_intercept=None, solve_method=None)
ps.rls_summary(y, *x, forgetting_factor=0.99, add_intercept=None, with_intercept=None, initial_p_diagonal=1e6)
ps.bls_summary(y, *x, lower_bound=None, upper_bound=None, add_intercept=None, with_intercept=None)
ps.quantile_summary(y, *x, tau=0.5, add_intercept=None, with_intercept=None)
# Note: quantile_summary std_error/statistic/p_value are NaN (no analytic SEs)

ps.logistic_summary(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.poisson_summary(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.negative_binomial_summary(y, *x, theta=None, lambda_=0.0, add_intercept=None, with_intercept=None, estimate_dispersion=False)
ps.tweedie_summary(y, *x, var_power=1.5, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.probit_summary(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.cloglog_summary(y, *x, lambda_=0.0, add_intercept=None, with_intercept=None)
ps.alm_summary(y, *x, distribution="normal", add_intercept=None, with_intercept=None)

ps.penalized_glm_summary(y, *x, family="poisson", link=None, theta=None, power=None, threshold=0.5, lambda_=0.0, estimate_dispersion=False, priors=None, vcov_type="laplace", conf_level=0.95, offset=None, add_intercept=True, max_iter=100, tol=1e-8)
ps.aft_summary(time, event, *x, distribution="weibull", priors=None, vcov_type="laplace", conf_level=0.95, add_intercept=True, max_iter=100, tol=1e-9)
```

**`ps.penalized_glm_summary` / `ps.aft_summary` return a widened 7-field schema** —
`List[Struct{term, estimate, std_error, statistic, p_value, ci_lower, ci_upper}]` — unlike
every other `*_summary` expression above, which share the 5-field
`{term, estimate, std_error, statistic, p_value}` schema (the shared `summary_output_dtype` and
every other `*_summary` expression are unchanged). The `intercept` row is read directly from the
upstream engine's own intercept inference, never recomputed locally. For `aft_summary`, rows are
`intercept` (when `add_intercept=True`), `x1..xp`, then `log_scale` (omitted for
`distribution="exponential"`, whose scale is fixed at 1 and not estimated).

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
# verify: skip -- expression API reference (y, *x are column-name placeholders, not runnable Python; ps not imported)
ps.ols_predict(y, *x, add_intercept=True, solve_method=None, interval=None, level=0.95, null_policy="drop", name=None)
ps.ridge_predict(y, *x, lambda_=1.0, add_intercept=True, solve_method=None, interval=None, level=0.95, null_policy="drop", name=None)
ps.elastic_net_predict(y, *x, lambda_=1.0, alpha=0.5, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.lasso_predict(y, *x, lambda_=1.0, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.wls_predict(y, weights, *x, add_intercept=True, solve_method=None, interval=None, level=0.95, null_policy="drop", name=None)
ps.rls_predict(y, *x, forgetting_factor=0.99, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None, initial_p_diagonal=1e6)
ps.bls_predict(y, *x, lower_bound=None, upper_bound=None, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.nnls_predict(y, *x, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.quantile_predict(y, *x, tau=0.5, add_intercept=True, null_policy="drop", name=None)
# Note: quantile_predict has no interval= param (bootstrap intervals not yet available)
ps.isotonic_predict(y, x, increasing=True, null_policy="drop", name=None)   # single x; no intervals

ps.logistic_predict(y, *x, lambda_=0.0, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.poisson_predict(y, *x, lambda_=0.0, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.negative_binomial_predict(y, *x, lambda_=0.0, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None, estimate_dispersion=False)
# Note: negative_binomial_predict does not expose theta= (unlike negative_binomial fit which
# accepts theta=). The predict variant estimates theta internally from the data.
ps.tweedie_predict(y, *x, var_power=1.5, lambda_=0.0, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.probit_predict(y, *x, lambda_=0.0, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.cloglog_predict(y, *x, lambda_=0.0, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.alm_predict(y, *x, distribution="normal", add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)

ps.penalized_glm_predict(y, *x, family="poisson", link=None, theta=None, power=None, threshold=0.5, lambda_=0.0, estimate_dispersion=False, priors=None, offset=None, add_intercept=True, max_iter=100, tol=1e-8, predict_type="response", null_policy="drop", name=None)
# Note: predict_type="response" applies the family's inverse link (+ offset); predict_type="link"
# returns the linear predictor + offset directly. lower/upper are always NaN (no
# interval-prediction method upstream). A null offset always yields a NaN prediction, regardless
# of null_policy.

ps.aft_predict_quantile(time, event, *x, p=0.5, distribution="weibull", priors=None, add_intercept=True, max_iter=100, tol=1e-9, null_policy="drop", name=None)
ps.aft_predict_median(time, event, *x, distribution="weibull", priors=None, add_intercept=True, max_iter=100, tol=1e-9, null_policy="drop", name=None)
ps.aft_predict_survival(time, event, *x, t, distribution="weibull", priors=None, add_intercept=True, max_iter=100, tol=1e-9, null_policy="drop", name=None)
ps.aft_predict_cdf(time, event, *x, t, distribution="weibull", priors=None, add_intercept=True, max_iter=100, tol=1e-9, null_policy="drop", name=None)
# Note: t= is keyword-only and required for aft_predict_survival/_cdf (must be finite and > 0).
# lower/upper are always NaN (no interval-prediction method upstream). Default output prefix is
# the function name (e.g. "aft_predict_quantile"), not "aft".

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
# verify: skip -- formula expression reference (formula: str uses type annotation syntax, not runnable Python)
ps.ols_formula(formula: str, add_intercept=None, with_intercept=None, solve_method=None)
ps.ridge_formula(formula: str, lambda_=1.0, add_intercept=None, with_intercept=None, solve_method=None)
ps.elastic_net_formula(formula: str, lambda_=1.0, alpha=0.5, add_intercept=None, with_intercept=None)
ps.lasso_formula(formula: str, lambda_=1.0, add_intercept=None, with_intercept=None)
ps.wls_formula(formula: str, weights, add_intercept=None, with_intercept=None, solve_method=None)
# Note: wls_formula takes a weights positional arg (column name or expr) after formula
ps.rls_formula(formula: str, forgetting_factor=0.99, add_intercept=None, with_intercept=None, initial_p_diagonal=1e6)
ps.bls_formula(formula: str, lower_bound=None, upper_bound=None, add_intercept=None, with_intercept=None)
ps.nnls_formula(formula: str, add_intercept=None, with_intercept=None)
ps.logistic_formula(formula: str, add_intercept=None, with_intercept=None)
ps.poisson_formula(formula: str, add_intercept=None, with_intercept=None)
ps.negative_binomial_formula(formula: str, theta=None, add_intercept=None, with_intercept=None, estimate_dispersion=False)
ps.tweedie_formula(formula: str, var_power=1.5, add_intercept=None, with_intercept=None)
ps.probit_formula(formula: str, add_intercept=None, with_intercept=None)
ps.cloglog_formula(formula: str, add_intercept=None, with_intercept=None)
ps.alm_formula(formula: str, distribution="normal", add_intercept=None, with_intercept=None)
ps.penalized_glm_formula(formula: str, family="poisson", link=None, theta=None, power=None, threshold=0.5, lambda_=0.0, estimate_dispersion=False, priors=None, offset=None, add_intercept=True, max_iter=100, tol=1e-8)
# Note: R's formula offset() term is not parsed — pass offset= as a column/expression directly.
ps.aft_formula(formula: str, event, distribution="weibull", priors=None, add_intercept=True, max_iter=100, tol=1e-9)
# Note: formula's left-hand side is the time column; event is passed separately (not formula syntax).
```

**Formula summary variants** (return `List[Struct{term,estimate,std_error,statistic,p_value}]`):

```python
# verify: skip -- formula expression reference (formula: str uses type annotation syntax, not runnable Python)
ps.ols_formula_summary(formula: str, add_intercept=None, with_intercept=None, solve_method=None, hc_type=None)
ps.ridge_formula_summary(formula: str, lambda_=1.0, add_intercept=None, with_intercept=None, solve_method=None)
ps.elastic_net_formula_summary(formula: str, lambda_=1.0, alpha=0.5, add_intercept=None, with_intercept=None)
ps.lasso_formula_summary(formula: str, lambda_=1.0, add_intercept=None, with_intercept=None)
ps.logistic_formula_summary(formula: str, add_intercept=None, with_intercept=None)
ps.poisson_formula_summary(formula: str, add_intercept=None, with_intercept=None)
ps.alm_formula_summary(formula: str, distribution="normal", add_intercept=None, with_intercept=None)
ps.penalized_glm_formula_summary(formula: str, family="poisson", link=None, theta=None, power=None, threshold=0.5, lambda_=0.0, estimate_dispersion=False, priors=None, vcov_type="laplace", conf_level=0.95, offset=None, add_intercept=True, max_iter=100, tol=1e-8)
# Note: returns the widened 7-field {term, estimate, std_error, statistic, p_value, ci_lower, ci_upper}
ps.aft_formula_summary(formula: str, event, distribution="weibull", priors=None, vcov_type="laplace", conf_level=0.95, add_intercept=True, max_iter=100, tol=1e-9)
# Note: same widened 7-field schema.
```

**Formula predict variants** (same `interval`/`level`/`null_policy`/`name` kwargs as base predict):

```python
# verify: skip -- formula expression reference (formula: str uses type annotation syntax, not runnable Python)
ps.ols_formula_predict(formula: str, add_intercept=True, solve_method=None, interval=None, level=0.95, null_policy="drop", name=None)
ps.ridge_formula_predict(formula: str, lambda_=1.0, add_intercept=True, solve_method=None, interval=None, level=0.95, null_policy="drop", name=None)
ps.elastic_net_formula_predict(formula: str, lambda_=1.0, alpha=0.5, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.lasso_formula_predict(formula: str, lambda_=1.0, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.logistic_formula_predict(formula: str, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.poisson_formula_predict(formula: str, add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.alm_formula_predict(formula: str, distribution="normal", add_intercept=True, interval=None, level=0.95, null_policy="drop", name=None)
ps.penalized_glm_formula_predict(formula: str, family="poisson", link=None, theta=None, power=None, threshold=0.5, lambda_=0.0, estimate_dispersion=False, priors=None, offset=None, add_intercept=True, max_iter=100, tol=1e-8, predict_type="response", null_policy="drop", name=None)
ps.aft_formula_predict_quantile(formula: str, event, p=0.5, distribution="weibull", priors=None, add_intercept=True, max_iter=100, tol=1e-9, null_policy="drop", name=None)
ps.aft_formula_predict_median(formula: str, event, distribution="weibull", priors=None, add_intercept=True, max_iter=100, tol=1e-9, null_policy="drop", name=None)
ps.aft_formula_predict_survival(formula: str, event, *, t, distribution="weibull", priors=None, add_intercept=True, max_iter=100, tol=1e-9, null_policy="drop", name=None)
ps.aft_formula_predict_cdf(formula: str, event, *, t, distribution="weibull", priors=None, add_intercept=True, max_iter=100, tol=1e-9, null_policy="drop", name=None)
# Note: formula's left-hand side is the time column; event is passed separately (not formula
# syntax). t= is keyword-only and required for _survival/_cdf (must be finite and > 0).
```

---

### Regression Expressions — Diagnostics

```python
# verify: skip -- diagnostics expression reference (group_sizes: list[int] uses type annotation syntax; ps and x not in scope)
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

### Example 9: Robust-regression inference and RANSAC's inlier fit

```python
import numpy as np
from polars_statistics import Huber, RANSAC, OLS

rng = np.random.default_rng(0)
n = 60
x = rng.normal(0, 1, n)
y = 2.0 * x + 1.0 + rng.normal(0, 0.3, n)
y[:5] += 20.0   # plant outliers in the first 5 rows
X = x.reshape(-1, 1)

# Huber: opt into Wald inference with compute_inference=True
h = Huber(compute_inference=True).fit(X, y)
print(h.std_errors)   # array of per-coefficient SEs (None when compute_inference=False)
print(h.p_values)

# RANSAC: same opt-in inference, plus the inlier-only fit statistics
r = RANSAC(compute_inference=True, random_state=0).fit(X, y)
print(r.n_inliers, r.std_errors, r.p_values)

# RANSAC.inlier_fit() returns a real fitted OLS instance on the inlier rows only
inlier_model = r.inlier_fit()
print(inlier_model.summary())
assert np.allclose(inlier_model.coefficients, r.coefficients)

# OLS.leverage_new() scores leverage for new observations against the fitted design
ols = OLS(add_intercept=True).fit(X, y)
print(ols.leverage_new(X[:3]))   # one leverage value per new row
```

### Example 10: Penalized Poisson with an exposure offset and priors; AFT with censoring

```python
import numpy as np
import polars as pl
import polars_statistics as ps
from polars_statistics import AFT

rng = np.random.default_rng(3)
n = 100
group = ["A"] * 50 + ["B"] * 50
x1 = rng.normal(0, 1, n)
log_exposure = rng.normal(0, 0.3, n)
mu = np.exp(0.2 + 0.5 * x1 + log_exposure)
y = rng.poisson(mu).astype(float)

df = pl.DataFrame({"group": group, "y": y, "x1": x1, "log_exposure": log_exposure})

# Penalized Poisson with an exposure offset and a Gaussian prior on x1, per group
glm_result = df.group_by("group").agg(
    ps.penalized_glm(
        "y", "x1",
        family="poisson",
        offset="log_exposure",
        priors=[None, ("normal", 0.0, 1.0)],   # flat on the intercept, N(0, 1) on x1
    ).alias("model")
)
print(glm_result.unnest("model").select("group", "coefficients", "dispersion"))

# Tidy summary with confidence intervals (7-field widened schema)
glm_summary = (
    df.select(
        ps.penalized_glm_summary(
            "y", "x1", family="poisson", offset="log_exposure",
            priors=[None, ("normal", 0.0, 1.0)],
        ).alias("s")
    )
    .explode("s")
    .unnest("s")
)
print(glm_summary)
assert "ci_lower" in glm_summary.columns

# AFT: Weibull survival model with right censoring
time_scale = 0.7
eta = 1.0 + 0.5 * x1
log_t = eta + time_scale * rng.gumbel(size=n)   # extreme-value errors -> Weibull AFT
time = np.exp(log_t)
censor_time = rng.exponential(scale=np.median(time) * 1.5, size=n)
observed_time = np.minimum(time, censor_time)
event = (time <= censor_time).astype(float)

aft_df = pl.DataFrame({"time": observed_time.tolist(), "event": event.tolist(), "x1": x1.tolist()})

X = aft_df.select("x1").to_numpy()
m = AFT(distribution="weibull").fit(X, aft_df["time"].to_numpy(), aft_df["event"].to_numpy())
print(m.predict_median(X[:5]))   # predicted median survival time per row
assert m.coefficients.shape[0] == 1

# Expression form: predicted survival probability S(t=5.0 | x)
survival_df = aft_df.select(
    ps.aft_predict_survival("time", "event", "x1", t=5.0).alias("pred")
).unnest("pred")
assert "aft_predict_survival_prediction" in survival_df.columns
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

**Ridge `lambda_ > 0` gives sandwich SEs with no t-test.** `Ridge.std_errors`/`.intercept_std_error` are always populated (a sandwich-estimator SE), but `.p_values`/`.t_statistics` are `None` whenever `lambda_ > 0` — a t-test on shrunken coefficients is not statistically valid. At `lambda_ == 0`, `Ridge` delegates fully to `OLS` semantics (full inference).

**RANSAC statistics describe the inlier-only OLS fit, not the whole sample.** Since anofox-regression 0.5.14, `RANSAC.r_squared`, `.n_observations`, `.adj_r_squared`, `.mse`, `.rmse`, `.f_statistic`/`.f_pvalue`, `.aic`/`.bic`/`.log_likelihood` are all computed on the inlier rows only (`n_observations == n_inliers`), matching `R summary(lm(y[inliers] ~ x[inliers]))`. Use `RANSAC.inlier_fit()` to get the equivalent standalone `OLS` object.

**Tweedie `aic`/`bic` are `NaN` for non-closed-form `var_power`.** Only `var_power` in `{0.0, 1.0, 2.0, 3.0}` (Gaussian, Poisson, Gamma, inverse-Gaussian) has a closed-form density for computing the log-likelihood-based `aic`/`bic`. The **default** `var_power=1.5` has no closed-form density and reports `NaN` for both — this is expected, not a bug. `coefficients`, `std_errors`, and `p_values` are unaffected.

**NegativeBinomial SEs use dispersion 1 by default.** `estimate_dispersion=False` (the default) matches the historical behaviour — standard errors assume dispersion exactly 1. Pass `estimate_dispersion=True` to get Pearson-dispersion-corrected SEs (`NegativeBinomial.dispersion` exposes the estimated value), matching R `summary.glm(negative.binomial(theta))`.

**Linear-solver `log_likelihood`/`aic`/`bic` now equal R's `logLik.lm`.** `OLS`/`WLS`/`Ridge` (at `lambda_ == 0`) report the exact R-matching Gaussian log-likelihood and its AIC/BIC since anofox-regression 0.5.14 — a previous constant-offset discrepancy is gone, but code that pinned the old numeric values will need updating.

**No-intercept R² uses the uncentred TSS.** For `OLS`/`WLS` fit without an intercept (`add_intercept=False`), `r_squared`/`adj_r_squared`/`f_statistic` are computed against the uncentred total sum of squares (`sum(y^2)`, not `sum((y - mean(y))^2)`), matching R's `lm(y ~ x - 1)` convention. This also affects any constant-column input that aliases to the intercept term.

**GLMM Poisson/binomial `log_likelihood` includes the saturated term.** `GLMM.log_likelihood` for non-Gaussian families matches `lme4`'s deviance-based log-likelihood (includes the saturated-model term), not a raw sum of per-observation log-densities.

**Huber/RANSAC inference is PyModel-only — no expression form.** `compute_inference`/`confidence_level` and the inference getters exist only on the `Huber`/`RANSAC` PyModel classes. `ps.huber(...)` (the expression) does not expose them.

**Rust `rlib` callers: new NegBin/RLS options arrive via additive `*_fit_with` functions.** If you consume `polars-statistics` as a Rust `rlib` (not through Python), `estimate_dispersion` and `initial_p_diagonal` are reachable through new `negative_binomial_fit_with`/`rls_fit_with` (and their `_summary`/`_predict` counterparts), which take an additional named kwargs struct. Every pre-existing `pub fn *_fit` signature is unchanged — this is purely additive.

### Penalized GLM / AFT: new-surface gotchas

**Priors are plain data, not a Python class.** `priors=` on `penalized_glm`/`penalized_glm_summary`/`penalized_glm_formula*`, `AFT`, and `aft`/`aft_summary`/`aft_formula*` accepts a sequence of `None`/`"flat"`, a `(kind, loc, scale)` tuple/list (or `(kind,)`/`(kind, loc)`), or a dict `{"kind", "loc", "scale"}` — `kind` is `"flat"`, `"normal"`, or `"laplace"`. A **leading** entry is optional and applies to the intercept, so `priors` may have `p` entries (one per feature) or `p + 1` entries (intercept first). `"normal"` is a ridge-like L2 prior; `"laplace"` is an L1 prior that can proximal-soft-threshold a coefficient to exactly `0.0` with `NaN` in its `std_errors`/`ci_lower`/`ci_upper` (no curvature SE for an inactive L1 coefficient).

**`offset=` is a column on the link scale, not a formula term.** `penalized_glm`'s `offset` (and `penalized_glm_summary`/`penalized_glm_predict`/`penalized_glm_formula*`) is a plain column/expression added to the linear predictor (e.g. `log(exposure)` precomputed yourself) — R's formula `offset()` syntax is **not** parsed by `penalized_glm_formula`; pass the offset as a separate argument. A null offset always produces a `NaN` prediction from `penalized_glm_predict`, regardless of `null_policy`.

**Poisson dispersion is fixed at `1.0` unless `estimate_dispersion=True` (Poisson only).** Matches R's `glm(family = poisson)` default. Pass `estimate_dispersion=True` to scale the covariance by the Pearson chi2/df dispersion estimate (R's `quasipoisson`) — every other family rejects this option with `ValueError` at build/construction time (Gamma and Tweedie always use their own Pearson-dispersion rule).

**`penalized_glm_summary`'s `statistic` is a Wald z (normal reference), not R's t.** R's `summary.glm` uses a t-reference for estimated-dispersion families (Gamma, Tweedie, quasipoisson); this library always reports the normal-reference Wald statistic/p-value for `penalized_glm_summary`, even for those families.

**`vcov` is scaled so `sqrt(diag(vcov)) == std_errors`, on both `PenalizedGLM` and `AFT`.** Matches R's `vcov(glm(...))` convention (for `NegativeBinomial`'s `theta` is a dispersion-unrelated shape parameter, so its scale factor is `1.0`, not `dispersion()`).

**AFT `event` must be boolean or 0/1 — R's 1/2 coding is rejected.** `aft`/`aft_summary`/`aft_predict_*`/`AFT.fit` accept a Boolean column/array directly, or a numeric column/array containing **only** `0`/`1`. Any other value (including R `survival::Surv`'s `1`/`2` censoring code) fails the whole fit rather than being silently misread — on the expression layer this is an all-`NaN` struct; on the `AFT` PyModel it is a `ValueError`. `time` must be strictly positive; a null `time` or a null `event` drops that row from the **fit**, but under `null_policy="drop_y_zero_x"` the four `aft_predict_*` expressions still **predict** that row (unlike `penalized_glm`'s `offset`, `event` is fit-only data, not required predict-input).

**`penalized_glm_predict` / `aft_predict_*` `lower`/`upper` are always `NaN`.** Neither `FittedPenalizedGlm` nor `FittedAft` has an interval-prediction method upstream — there is no bootstrap or analytic interval to compute. Only the point `*_prediction` field is populated.

**Use `tol=1e-10` on `AFT`/`aft*` to reproduce R `survival::survreg` to 1e-6.** The library's convergence tolerance defaults (`tol=1e-9` on the expression, `tol=1e-9` on the `AFT` constructor) are already tight; the plan's own R-reference tests tighten further to `tol=1e-10` to match `survreg`'s coefficients/scale to 1e-6 and log-likelihoods to 1e-7.

**Full `vcov`/`information` are PyModel-only.** `PenalizedGLM.vcov`/`.information` and `AFT.vcov`/`.information` expose the full covariance/information matrices; no expression returns them (`penalized_glm_summary`/`aft_summary` return only the diagonal-derived `std_error`/`ci_lower`/`ci_upper` per term).

### 0.5.19-0.5.21 upstream behaviour changes (anofox-regression bump)

**OLS/WLS alias the LATER collinear column, for every `solve_method` (0.5.21 #64).** Since this bump, `ps.ols`/`ps.wls`/`OLS`/`WLS` give a `NaN` coefficient to the **later** of two (or more) exactly collinear columns — matching R's `lm`'s `dqrdc2` convention — for `solve_method` in `{"qr", "svd", "cholesky"}` alike, and independent of column scaling (a scaled duplicate aliases exactly like an exact duplicate). Previously, `"svd"` silently returned a minimum-norm split with no `NaN` flag, and `"qr"` aliased the *earlier* column of a collinear pair.

**Every GLM `*_summary` now reports intercept inference (0.5.20 #66).** `logistic_summary`, `poisson_summary`, `negative_binomial_summary`, `tweedie_summary`, `probit_summary`, `cloglog_summary` (and the new `penalized_glm_summary`/`aft_summary`, whose intercept row this plan documents above) all report the intercept row's `std_error`/`statistic`/`p_value` — previously `NaN`. GLM Wald p-values are computed via the survival function, so they never underflow to exactly `0.0`.

**Tweedie / Gamma statistics are t-based (summary.glm convention, 0.5.20 #66).** `tweedie_summary` reports t-based statistics/p-values (t on the residual df, as R's `summary.glm`) for every row, where they previously could be `NaN`; the `Gamma` PyModel's `std_errors`/`p_values` gained the same t-based values. `Probit`'s standard errors now use `statrs`'s `erfc`/`erfc_inv` (full precision, matching R `summary.glm(binomial(link = "probit"))` exactly — previously off in the 5th digit).

**LARS aliases exactly collinear LATER columns (0.5.19 #62).** `LARS`/`LassoLars` now alias the **later** of two exactly collinear columns with a `NaN` coefficient (the fit equals the reduced model without it) instead of a platform-dependent result.

## See also

- polars-statistics-hypothesis-tests — t-tests, Mann-Whitney, Kruskal-Wallis, normality, TOST equivalence
- polars-statistics-anova — one-way, two-way, repeated-measures ANOVA, ICC
- polars-statistics-results — struct extraction, unnest, to_dict/summary, error handling
- polars-statistics-getting-started — install, model-selection matrix, sklearn migration table
