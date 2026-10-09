# API Coverage — anofox-regression 0.5.22 `solvers::penalized_glm` + `solvers::aft`, and the anofox-statistics 0.4.7 Mann-Whitney / Wilcoxon expression gap

> Full coverage by default. Opt-outs are explicit, reasoned decisions.
> Surface enumerated from every `pub` item in `solvers/penalized_glm/` and `solvers/aft/` of anofox-regression 0.5.22
> (`git -C ~/projects/rust/anofox-regression show v0.5.22:...`; regressor.rs, types.rs, model.rs, distribution.rs, mod.rs re-exports),
> diffed against 0.5.16, plus the `MannWhitneyResult` / `WilcoxonResult` fields of anofox-statistics 0.4.7.
> Retarget (CONTEXT "Retarget to 0.4.7 / 0.5.22", user decision 2026-10-09): Phase 16.2 bumps to
> anofox-statistics 0.4.7 / anofox-regression 0.5.22 (Plan 01). Items
> 0.5.17-0.5.22 added to the two engines are in the table; all other new 0.4.7 / 0.5.17-0.5.22 API is deferred to Phase 16.3
> (section at the end — the gate accepts only INTEGRATE / OPT-OUT rows).
> These rows are the "Deferred to Phase 16.2" items of `.planning/phases/16.1-upstream-crate-refresh/COVERAGE.md`.
> The reason column of INTEGRATE rows names where the capability lands (plan number).

| capability | decision | reason |
|---|---|---|
| pglm: PenalizedGlmFamily::Poisson (log / identity / sqrt link) | INTEGRATE | PenalizedGLM(family="poisson", link=) + ps.penalized_glm* — Plans 02, 04 |
| pglm: PenalizedGlmFamily::Binomial (logit / probit / cloglog link) | INTEGRATE | family="binomial", link= — Plans 02, 04 |
| pglm: PenalizedGlmFamily::Logistic { threshold } | INTEGRATE | family="logistic", threshold= — Plans 02, 04 |
| pglm: PenalizedGlmFamily::NegativeBinomial { theta: None / Some } | INTEGRATE | family="negative_binomial", theta=None (MASS::glm.nb) or fixed — Plans 02, 04 |
| pglm: PenalizedGlmFamily::Gamma | INTEGRATE | family="gamma" (logLik at deviance / n since 0.5.17) — Plans 02, 04 |
| pglm: PenalizedGlmFamily::Tweedie { power } | INTEGRATE | family="tweedie", power= (1..2, default 1.5) — Plans 02, 04 |
| pglm: builder with_intercept | INTEGRATE | add_intercept= on PyModel and every expression — Plans 02, 04 |
| pglm: builder max_iterations / tolerance | INTEGRATE | max_iter= / tol= on PyModel and every expression (R parity needs tol 1e-10) — Plans 02, 04 |
| pglm: builder compute_inference / confidence_level | INTEGRATE | PenalizedGLM(compute_inference=, confidence_level=); penalized_glm_summary(conf_level=) — Plans 02, 04 |
| pglm: builder lambda (uniform ridge) | INTEGRATE | lambda_= — Plans 02, 04 |
| pglm: builder priors (PriorSpec flat / normal / laplace) | INTEGRATE | priors= as plain tuples / dicts with an optional leading intercept entry (CONTEXT "Priors"); kwargs channel in expressions — Plans 02, 04 |
| pglm: builder vcov_type (VcovType laplace / sandwich / naive) | INTEGRATE | vcov_type= on PyModel and penalized_glm_summary — Plans 02, 04 |
| pglm: builder offset | INTEGRATE | PenalizedGLM.fit(x, y, offset=), predict(x, offset=); expression offset= column (default pl.lit(0.0)) — Plans 02, 04 |
| pglm: builder estimate_dispersion (0.5.17, Poisson quasipoisson scale) | INTEGRATE | estimate_dispersion= on PyModel and every expression; rejected for non-Poisson families (CONTEXT retarget exception) — Plans 02, 04 |
| pglm: FittedPenalizedGlm coefficients / intercept | INTEGRATE | getters + fit struct — Plan 02 |
| pglm: FittedPenalizedGlm family() | INTEGRATE | PenalizedGLM.family / .link getters — Plan 02 |
| pglm: FittedPenalizedGlm deviance / null_deviance / pseudo_r_squared | INTEGRATE | getters + fit struct; R-style null deviance with offsets / no intercept since 0.5.17 — Plan 02 |
| pglm: FittedPenalizedGlm log_likelihood / aic / bic | INTEGRATE | getters + fit struct — Plan 02 |
| pglm: FittedPenalizedGlm dispersion / theta | INTEGRATE | getters + fit struct (Poisson 1 unless estimate_dispersion; theta NaN / None for non-NegBin) — Plan 02 |
| pglm: FittedPenalizedGlm iterations / converged / n_observations | INTEGRATE | getters + fit struct — Plan 02 |
| pglm: FittedPenalizedGlm accuracy (Logistic only) | INTEGRATE | getter (None) + fit struct (NaN) for other families — Plan 02 |
| pglm: GlmInferenceResult std_errors / z_values / p_values / ci_lower / ci_upper | INTEGRATE | std_errors / z_values / p_values / conf_interval_lower / conf_interval_upper getters; 7-field summary rows — Plans 02, 04 |
| pglm: GlmInferenceResult intercept_* inference fields (0.5.20) | INTEGRATE | intercept_std_error / _z_value / _p_value / _ci_lower / _ci_upper feed the intercept_* getters and the summary intercept row directly — Plans 02, 04 |
| pglm: vcov() + GlmInferenceResult information / matrix_parameters | INTEGRATE | PenalizedGLM.vcov (scaled, see note 2) / .information / .matrix_parameters (PyModel only — CONTEXT resolution 3) — Plan 02 |
| pglm: FittedPenalizedGlm separation() (SeparationCheck) | INTEGRATE | has_separation / separated_predictors / separation_warning getters; has_separation in the fit struct — Plan 02 |
| pglm: FittedPenalizedGlm fitted_values() + result().valid_rows | INTEGRATE | fitted_values / valid_rows getters — Plan 02 |
| pglm: FittedPenalizedGlm predict / predict_with_offset / predict_linear | INTEGRATE | PenalizedGLM.predict(x, offset=) / predict_linear(x); ps.penalized_glm_predict(predict_type="response" or "link") — Plans 02, 04 |
| pglm: formula interface (existing Poisson / logistic precedent) | INTEGRATE | penalized_glm_formula / _formula_summary / _formula_predict — Plans 02, 04 |
| pglm: column-major fit_* family functions + option / result structs | OPT-OUT | not needed: CONTEXT locks the builder entry point; upstream asserts the column-major functions give identical coefficients (see note 3) |
| pglm: low-level engine building blocks | OPT-OUT | not needed: internal pieces the builder composes, incl. 0.5.17 engine::null_deviance; none is configurable via the builder API (see note 3) |
| pglm: PriorSpec::precision / is_flat, GlmPriorOptions::is_empty / has_l1 | OPT-OUT | not needed: helper predicates over the prior list the user already supplies |
| pglm: prediction intervals | OPT-OUT | not needed: FittedPenalizedGlm has no predict_with_interval upstream; *_predict lower/upper are NaN by design (documented) |
| aft: AftDistribution Weibull / LogNormal / LogLogistic / Exponential | INTEGRATE | from_name parsing for the string argument, as_str (0.5.20) for the getter; AFT(distribution=) + ps.aft* — Plan 05 |
| aft: builder with_intercept / max_iterations / tolerance | INTEGRATE | add_intercept= / max_iter= / tol= on PyModel and every expression — Plan 05 |
| aft: builder compute_inference / confidence_level | INTEGRATE | AFT(compute_inference=, confidence_level=); aft_summary(conf_level=) — Plan 05 |
| aft: builder priors / vcov_type | INTEGRATE | priors= (same plain-data format as PenalizedGLM), vcov_type= — Plan 05 |
| aft: AftRegressor::fit(x, time, event: &[bool]) right censoring | INTEGRATE | AFT.fit(x, time, event); expressions take explicit time + event columns (CONTEXT "AFT inputs") — Plan 05 |
| aft: FittedAft coefficients / intercept / scale / distribution | INTEGRATE | getters + fit struct — Plan 05 |
| aft: FittedAft log_likelihood / null_log_likelihood / aic / bic / converged | INTEGRATE | getters + fit struct — Plan 05 |
| aft: AftFitResult n_observations / n_events / n_censored / iterations | INTEGRATE | getters + fit struct (n_features = len(coefficients)) — Plan 05 |
| aft: AftInference std_errors / z_values / p_values / ci_lower / ci_upper | INTEGRATE | getters + aft_summary rows — Plan 05 |
| aft: AftInference intercept_std_error / _z_value / _p_value / _ci_* (0.5.20) | INTEGRATE | intercept_* getters and the aft_summary intercept row read these fields directly — Plan 05 |
| aft: AftInference log_scale_std_error | INTEGRATE | getter + aft_summary log_scale row (z / p / CI via the upstream Wald formula, proven equal on the intercept) — Plan 05 |
| aft: AftInference vcov / information | INTEGRATE | AFT.vcov / .information (PyModel only — CONTEXT resolution 3) — Plan 05 |
| aft: FittedAft predict_quantile / _median / _survival / _cdf | INTEGRATE | PyModel methods + four expressions aft_predict_quantile / _median / _survival / _cdf (CONTEXT resolution 2) — Plan 05 |
| aft: FittedAft predict_linear | INTEGRATE | AFT.predict_linear (PyModel); the four time-scale expressions cover the expression layer — Plan 05 |
| aft: formula interface | INTEGRATE | aft_formula / aft_formula_summary (time ~ x, event= column) — Plan 05 |
| aft: column-major fit_aft / AftOptions / AftResult | OPT-OUT | not needed: CONTEXT locks the builder entry point; upstream asserts fit_aft equals the builder |
| aft: AftRegressor::options() | OPT-OUT | not needed: echoes the configuration the AFT PyModel already stores from its constructor |
| aft: AftDistribution raw distribution functions and derivatives | OPT-OUT | not needed: raw (eta, sigma) functions behind the predict_* methods; the derivatives are Newton-Raphson internals (see note 3) |
| stats: MannWhitneyResult estimate / conf_int at the expression layer | INTEGRATE | ps.mann_whitney_u struct gains trailing estimate / ci_lower / ci_upper (CONTEXT) — Plan 03 |
| stats: WilcoxonResult estimate / conf_int at the expression layer | INTEGRATE | ps.wilcoxon_signed_rank struct gains trailing estimate / ci_lower / ci_upper (CONTEXT) — Plan 03 |
| stats: null_value and ConfidenceInterval.conf_level at the expression layer | OPT-OUT | MannWhitneyResult / WilcoxonResult — not needed: both echo the mu / conf_level arguments; CONTEXT fixes the added fields to estimate / ci_lower / ci_upper; the PyModel classes expose null_value |

## Notes

1. Plan 01 bumps the crates and adapts existing tests / doc pages to the 0.5.19-0.5.21 behaviour changes (lm-style aliasing,
   GLM intercept inference, probit accuracy, LARS collinear aliasing); those are behaviour changes of already-exposed API, not new
   capabilities, so they have no rows here. Documentation of the whole surface (stubs, docstrings, mkdocs, CHANGELOG, skills) is
   Plans 06-08; stubs are regenerated in Plans 02 and 05 where the PyModel classes land.
2. The PenalizedGLM `vcov` getter is scaled by the family's covariance scale so that its diagonal square roots equal the reported
   standard errors, as R's `vcov(glm)` does. Re-checked against 0.5.22: upstream `vcov()` is still the unscaled matrix, and
   laplace::inference still applies the scale only to the standard errors. 0.5.17 changed the Poisson scale from max(1, Pearson)
   to 1, or Pearson when `estimate_dispersion(true)`. The scale is 1 for Binomial, Logistic and Negative Binomial (theta is
   reported as the dispersion but does not scale the covariance), Pearson for Gamma and Tweedie. `information` is returned as
   upstream. AFT uses scale 1, so no scaling applies there.
3. Opted-out item lists. Column-major entry points: `fit_poisson`, `fit_binomial`, `fit_logistic`, `fit_negbinomial`,
   `fit_gamma`, `fit_tweedie`, the `*Options` structs (incl. 0.5.17 `PoissonOptions::estimate_dispersion`), `GlmResult`,
   `LogisticResult` (and `fit_aft` / `AftOptions` / `AftResult` for AFT). Engine building blocks: the `engine` (incl.
   `engine::null_deviance`), `irls` (incl. 0.5.22 `fit_irls_from`), `laplace`, `loglik`, `normal_eq`, `penalty` and `design` modules, `EngineOptions`,
   `DispersionRule`, `LogLikKind`, `ConstantColumnPolicy`, `GlmEngineError`. AFT distribution functions: `log_density`,
   `log_survival`, `cdf`, `quantile`, `cdf_time`, `survival_time`, `quantile_time` and the `d_*` / `dd_*` derivatives.
4. Target is anofox-regression 0.5.22 (user decision 2026-10-09). Diffing tags v0.5.21..v0.5.22 at planning time showed:
   performance only (#72); Cargo.toml dependency set unchanged; the R fixture and the upstream validation test byte-identical
   (sha256 07fd63e9...2bcde); one new low-level pub fn `irls::fit_irls_from` (an engine building block, covered by note 3).
   Cargo.lock is gitignored, so CI and the release build resolve the newest compatible patch, which is 0.5.22 / 0.4.7 at
   planning time. The R-reference tests assert upstream tolerances, so a later patch that changed results would fail CI loudly.

## Deferred to Phase 16.3 (not part of this phase's surface)

New API in anofox-statistics 0.4.7 / anofox-regression 0.5.17-0.5.22. It is **deferred to Phase 16.3** by user decision
(CONTEXT "Retarget to 0.4.7 / 0.5.22"), not opted out. 16.2 only adopts the versions and keeps every gate green on them. These
items stay out of the decision table above because the coverage gate accepts only INTEGRATE / OPT-OUT.

- anofox-statistics 0.4.7: `TestResult` (+ `From` conversions, `to_test_results`, builders, `alternative_name`), `p_adjust` / `PAdjustMethod`,
  `ptukey` / `qtukey`, post-hoc `pairwise_t_test` / `tukey_hsd` / `dunn_test` (`PairwiseResult`, `PairwiseComparison`).
- anofox-regression 0.5.17: AID moment heuristic (`solvers::aid::heuristic` — `compute_aid`, `compute_aid_anomalies`, `AidOptions`, `OutlierMethod`),
  empirical-Bayes shrinkage (`solvers::eb_shrink` — `eb_shrink`, `EbShrinkOptions`, `TauMethod`, `EbShrinkResult`, `ShrunkenGroup`).
- anofox-regression 0.5.18: GLMM `negative_binomial()` / `gamma()` / `tweedie(power)` families, `nb_theta`, GLMM `offset` +
  `predict_fixed_with_offset`, `random_slopes_per_factor`, `factor_random_cov` / `factor_random_sd` / `factor_random_effects_matrix`.
- anofox-regression 0.5.20: `ModelInfo` / `HasModelInfo` (on every fitted model, incl. FittedPenalizedGlm / FittedAft), `diagnostics::augment` /
  `augment_with` (`Augment`, `Augmentable`, `AugmentSpec`, `ResidualType`), intercept-inference GETTERS on the existing GLM PyModels (the
  existing `*_summary` expressions already show the filled intercept rows after Plan 01's bump).
- anofox-regression 0.5.22: `OlsRegressor::with_alias_tolerance` / `alias_tolerance` (OLS / WLS), `utils::lm_alias_columns`,
  `utils::LM_ALIAS_TOLERANCE`, LARS `adj_r_squared` / `mse` / `rmse` getters (filled upstream; the LARS PyModel exposes only `r_squared`).
