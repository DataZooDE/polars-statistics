# API Coverage — anofox-regression 0.5.16 `solvers::penalized_glm` + `solvers::aft`, and the anofox-statistics 0.4.6 Mann-Whitney / Wilcoxon expression gap

> Full coverage by default. Opt-outs are explicit, reasoned decisions.
> Surface enumerated from every `pub` item in `~/.cargo/registry/src/*/anofox-regression-0.5.16/src/solvers/penalized_glm/`
> and `.../src/solvers/aft/` (regressor.rs, types.rs, model.rs, distribution.rs, mod.rs re-exports), read at planning time,
> plus the `MannWhitneyResult` / `WilcoxonResult` fields in anofox-statistics 0.4.6 `src/nonparametric/wilcoxon.rs`.
> These rows are the "Deferred to Phase 16.2" items of `.planning/phases/16.1-upstream-crate-refresh/COVERAGE.md`.
> The reason column of INTEGRATE rows names where the capability lands (plan number).

| capability | decision | reason |
|---|---|---|
| pglm: PenalizedGlmFamily::Poisson (log / identity / sqrt link) | INTEGRATE | PenalizedGLM(family="poisson", link=) + ps.penalized_glm* — Plans 01, 03 |
| pglm: PenalizedGlmFamily::Binomial (logit / probit / cloglog link) | INTEGRATE | family="binomial", link= — Plans 01, 03 |
| pglm: PenalizedGlmFamily::Logistic { threshold } | INTEGRATE | family="logistic", threshold= — Plans 01, 03 |
| pglm: PenalizedGlmFamily::NegativeBinomial { theta: None / Some } | INTEGRATE | family="negative_binomial", theta=None (MASS::glm.nb) or fixed — Plans 01, 03 |
| pglm: PenalizedGlmFamily::Gamma | INTEGRATE | family="gamma" — Plans 01, 03 |
| pglm: PenalizedGlmFamily::Tweedie { power } | INTEGRATE | family="tweedie", power= (1..2, default 1.5) — Plans 01, 03 |
| pglm: builder with_intercept | INTEGRATE | add_intercept= on PyModel and every expression — Plans 01, 03 |
| pglm: builder max_iterations / tolerance | INTEGRATE | max_iter= / tol= on PyModel and every expression (R parity needs tol 1e-10) — Plans 01, 03 |
| pglm: builder compute_inference / confidence_level | INTEGRATE | PenalizedGLM(compute_inference=, confidence_level=); penalized_glm_summary(conf_level=) — Plans 01, 03 |
| pglm: builder lambda (uniform ridge) | INTEGRATE | lambda_= — Plans 01, 03 |
| pglm: builder priors (PriorSpec flat / normal / laplace) | INTEGRATE | priors= as plain tuples / dicts with an optional leading intercept entry (CONTEXT "Priors"); kwargs channel in expressions — Plans 01, 03 |
| pglm: builder vcov_type (VcovType laplace / sandwich / naive) | INTEGRATE | vcov_type= on PyModel and penalized_glm_summary — Plans 01, 03 |
| pglm: builder offset | INTEGRATE | PenalizedGLM.fit(x, y, offset=), predict(x, offset=); expression offset= column (default pl.lit(0.0)) — Plans 01, 03 |
| pglm: FittedPenalizedGlm coefficients / intercept | INTEGRATE | getters + fit struct — Plan 01 |
| pglm: FittedPenalizedGlm family() | INTEGRATE | PenalizedGLM.family / .link getters — Plan 01 |
| pglm: FittedPenalizedGlm deviance / null_deviance / pseudo_r_squared | INTEGRATE | getters + fit struct — Plan 01 |
| pglm: FittedPenalizedGlm log_likelihood / aic / bic | INTEGRATE | getters + fit struct — Plan 01 |
| pglm: FittedPenalizedGlm dispersion / theta | INTEGRATE | getters + fit struct (theta NaN / None for non-NegBin) — Plan 01 |
| pglm: FittedPenalizedGlm iterations / converged / n_observations | INTEGRATE | getters + fit struct — Plan 01 |
| pglm: FittedPenalizedGlm accuracy (Logistic only) | INTEGRATE | getter (None) + fit struct (NaN) for other families — Plan 01 |
| pglm: GlmInferenceResult std_errors / z_values / p_values / ci_lower / ci_upper | INTEGRATE | std_errors / z_values / p_values / conf_interval_lower / conf_interval_upper getters; 7-field penalized_glm_summary rows — Plans 01, 03 |
| pglm: intercept inference (vcov row matrix_parameters == None) | INTEGRATE | intercept_std_error / intercept_z_value / intercept_p_value / intercept_conf_interval getters + summary intercept row (same Wald formula as upstream laplace::inference) — Plans 01, 03 |
| pglm: vcov() + GlmInferenceResult information / matrix_parameters | INTEGRATE | PenalizedGLM.vcov / .information / .matrix_parameters (PyModel only — CONTEXT resolution 3; expressions report per-term SE/CI) — Plan 01 |
| pglm: FittedPenalizedGlm separation() (SeparationCheck) | INTEGRATE | has_separation / separated_predictors / separation_warning getters; has_separation in the fit struct — Plan 01 |
| pglm: FittedPenalizedGlm fitted_values() + result().valid_rows | INTEGRATE | fitted_values / valid_rows getters — Plan 01 |
| pglm: FittedPenalizedGlm predict / predict_with_offset / predict_linear | INTEGRATE | PenalizedGLM.predict(x, offset=) / predict_linear(x); ps.penalized_glm_predict(type="response" or "link") — Plans 01, 03 |
| pglm: formula interface (existing Poisson / logistic precedent) | INTEGRATE | penalized_glm_formula / _formula_summary / _formula_predict — Plans 01, 03 |
| pglm: column-major fit_* family functions + option / result structs | OPT-OUT | not needed: CONTEXT locks the builder entry point; upstream asserts the column-major functions give identical coefficients (see note 3) |
| pglm: low-level engine building blocks | OPT-OUT | not needed: internal pieces the builder composes; none is configurable through the builder API (see note 3) |
| pglm: PriorSpec::precision / is_flat, GlmPriorOptions::is_empty / has_l1 | OPT-OUT | not needed: helper predicates over the prior list the user already supplies |
| pglm: prediction intervals | OPT-OUT | not needed: FittedPenalizedGlm has no predict_with_interval upstream; *_predict lower/upper are NaN by design (documented) |
| aft: AftDistribution Weibull / LogNormal / LogLogistic / Exponential | INTEGRATE | from_name parsing used for the string argument; AFT(distribution=) + ps.aft* (from_name aliases accepted) — Plan 04 |
| aft: builder with_intercept / max_iterations / tolerance | INTEGRATE | add_intercept= / max_iter= / tol= on PyModel and every expression — Plan 04 |
| aft: builder compute_inference / confidence_level | INTEGRATE | AFT(compute_inference=, confidence_level=); aft_summary(conf_level=) — Plan 04 |
| aft: builder priors / vcov_type | INTEGRATE | priors= (same plain-data format as PenalizedGLM), vcov_type= — Plan 04 |
| aft: AftRegressor::fit(x, time, event: &[bool]) right censoring | INTEGRATE | AFT.fit(x, time, event); expressions take explicit time + event columns (CONTEXT "AFT inputs") — Plan 04 |
| aft: FittedAft coefficients / intercept / scale / distribution | INTEGRATE | getters + fit struct — Plan 04 |
| aft: FittedAft log_likelihood / null_log_likelihood / aic / bic / converged | INTEGRATE | getters + fit struct — Plan 04 |
| aft: AftFitResult n_observations / n_events / n_censored / iterations | INTEGRATE | getters + fit struct (n_features = len(coefficients)) — Plan 04 |
| aft: AftInference std_errors / z_values / p_values / ci_lower / ci_upper | INTEGRATE | getters + aft_summary rows — Plan 04 |
| aft: AftInference intercept_std_error / log_scale_std_error | INTEGRATE | getters + aft_summary intercept and log_scale rows — Plan 04 |
| aft: AftInference vcov / information | INTEGRATE | AFT.vcov / .information (PyModel only — CONTEXT resolution 3) — Plan 04 |
| aft: FittedAft predict_quantile / _median / _survival / _cdf | INTEGRATE | PyModel methods + four expressions aft_predict_quantile / _median / _survival / _cdf (CONTEXT resolution 2) — Plan 04 |
| aft: FittedAft predict_linear | INTEGRATE | AFT.predict_linear (PyModel); the four time-scale expressions cover the expression layer — Plan 04 |
| aft: formula interface | INTEGRATE | aft_formula / aft_formula_summary (time ~ x, event= column) — Plan 04 |
| aft: column-major fit_aft / AftOptions / AftResult | OPT-OUT | not needed: CONTEXT locks the builder entry point; upstream asserts fit_aft equals the builder |
| aft: AftRegressor::options() | OPT-OUT | not needed: echoes the configuration the AFT PyModel already stores from its constructor |
| aft: AftDistribution raw distribution functions and derivatives | OPT-OUT | not needed: raw (eta, sigma) functions behind the predict_* methods; the derivatives are Newton-Raphson internals (see note 3) |
| stats: MannWhitneyResult estimate / conf_int at the expression layer | INTEGRATE | ps.mann_whitney_u struct gains trailing estimate / ci_lower / ci_upper (CONTEXT) — Plan 02 |
| stats: WilcoxonResult estimate / conf_int at the expression layer | INTEGRATE | ps.wilcoxon_signed_rank struct gains trailing estimate / ci_lower / ci_upper (CONTEXT) — Plan 02 |
| stats: null_value and ConfidenceInterval.conf_level at the expression layer | OPT-OUT | MannWhitneyResult / WilcoxonResult — not needed: both echo the mu / conf_level arguments; CONTEXT fixes the added fields to estimate / ci_lower / ci_upper; the PyModel classes expose null_value |

## Notes

1. Documentation of the whole surface (stubs, docstrings, mkdocs, CHANGELOG, skills) is Plans 05-07; stubs are
   regenerated in Plans 01 and 04 where the PyModel classes land.
2. The PenalizedGLM `vcov` getter is scaled by the family's covariance scale (Pearson dispersion for Poisson
   floored at 1, Gamma and Tweedie; 1 for Binomial, Logistic and Negative Binomial) so that its diagonal square
   roots equal the reported standard errors, as R's `vcov(glm)` does. Upstream returns the unscaled matrix and
   applies the scale only to the standard errors (laplace::inference). `information` is returned as upstream.
3. Opted-out item lists. Column-major entry points: `fit_poisson`, `fit_binomial`, `fit_logistic`, `fit_negbinomial`,
   `fit_gamma`, `fit_tweedie`, the `*Options` structs, `GlmResult`, `LogisticResult` (and `fit_aft` / `AftOptions` / `AftResult`
   for AFT). Engine building blocks: the `engine`, `irls`, `laplace`, `loglik`, `normal_eq`, `penalty` and `design` modules,
   `EngineOptions`, `DispersionRule`, `LogLikKind`, `ConstantColumnPolicy`, `GlmEngineError`. AFT distribution functions:
   `log_density`, `log_survival`, `cdf`, `quantile`, `cdf_time`, `survival_time`, `quantile_time` and the `d_*` / `dd_*`
   derivatives.
