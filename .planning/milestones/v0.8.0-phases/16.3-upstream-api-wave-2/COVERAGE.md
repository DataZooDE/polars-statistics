# API Coverage — anofox-statistics 0.4.7 and anofox-regression 0.5.17–0.5.22 (Phase 16.3)

> Full coverage by default. Opt-outs are explicit, reasoned decisions.
> Surface enumerated from every `pub` item added between the tags `v0.4.6..v0.4.7` of anofox-statistics
> (`git -C ~/projects/rust/anofox-statistics-rs diff v0.4.6 v0.4.7 -- src`) and `v0.5.16..v0.5.22` of
> anofox-regression (`git -C ~/projects/rust/anofox-regression diff v0.5.16 v0.5.22 -- src`), plus the
> CHANGELOG "Added" items and the result fields those releases newly populate.
> Phase 16.2 owns the crate bump, penalized GLM `estimate_dispersion` and the `GlmInferenceResult` / `AftInference`
> intercept fields (16.2 COVERAGE.md). Those rows are not repeated here.
> The reason column of INTEGRATE rows names where the capability lands (16.3 plan number).
> Rows marked "pre-existing gap" are older upstream items that a 0.5.17–0.5.21 feature depends on; they are
> closed alongside it.

| capability | decision | reason |
|---|---|---|
| stats: p_adjust (elementwise, NaN / null kept and excluded from n) | INTEGRATE | ps.p_adjust(p, method=) bare Float64, returns_scalar=False — Plan 01 |
| stats: PAdjustMethod (holm, hochberg, hommel, bonferroni, BH/fdr, BY, none) | INTEGRATE | method= strings (case-insensitive, "fdr" alias) on p_adjust, pairwise_t_test, dunn_test — Plan 01 |
| stats: ptukey (q, nmeans, df, nranges, lower_tail) | INTEGRATE | ps.ptukey elementwise, column or scalar arguments broadcast — Plan 01 |
| stats: qtukey (p, nmeans, df, nranges, lower_tail) | INTEGRATE | ps.qtukey elementwise — Plan 01 |
| stats: tukey_hsd (conf_level) | INTEGRATE | ps.tukey_hsd(values, groups, conf_level=) long format → List[Struct] of pairwise rows — Plan 01 |
| stats: pairwise_t_test (pooled_sd, p_adjust_method, alternative) | INTEGRATE | ps.pairwise_t_test(values, groups, pooled_sd=, p_adjust_method=, alternative=) — Plan 01 |
| stats: dunn_test (p_adjust_method) | INTEGRATE | ps.dunn_test(values, groups, p_adjust_method=) — Plan 01 |
| stats: PairwiseComparison group1/group2/estimate/statistic/df/p_value/p_adj | INTEGRATE | pairwise row fields of the same names — Plan 01 |
| stats: PairwiseComparison conf_low / conf_high | INTEGRATE | row fields ci_lower / ci_upper (project naming, as fisher_exact / bootstrap_ci); NaN where upstream is None — Plan 01 |
| stats: PairwiseResult groups / group_sizes | INTEGRATE | per-row n1 / n2 (sizes of group1 / group2); the level set is the union of group1 / group2 — Plan 01 |
| stats: PairwiseResult method / p_adjust_method / alternative / conf_level | OPT-OUT | not needed: each echoes an argument of the same call |
| stats: generic group label type G: Ord + Clone | OPT-OUT | explicitly out of scope: CONTEXT casts the group column to Utf8; numeric labels appear (and sort) as strings, documented |
| stats: TestResult (+ From impls, to_test_results, builders, alternative_name) | OPT-OUT | not needed: every test class has to_dict() / summary() via ergonomics.rs (CONTEXT resolution 1); reversal: per-class `.to_test_result()` over the upstream From impls |
| reg: GLM intercept_std_error / _t_statistic / _p_value / _conf_interval (0.5.20) | INTEGRATE | getters on Logistic, Probit, Cloglog, LogisticRegression, Poisson, NegativeBinomial, Tweedie, Gamma — Plan 02 |
| reg: GLM slope conf_interval_lower / conf_interval_upper (0.5.20 #66) | INTEGRATE | getters on the same 8 GLM classes — Plan 02 |
| reg: GLM builder compute_inference / confidence_level | INTEGRATE | trailing ctor kwargs on Probit, Cloglog, NegativeBinomial, Tweedie (incl. its gaussian / gamma / inverse_gaussian factories), Gamma (Poisson, Logistic, LogisticRegression already have them) — Plan 02 |
| reg: OLS / WLS intercept inference + slope CIs | INTEGRATE | pre-existing gap named by CONTEXT resolution 3: intercept_* and conf_interval_* getters on OLS and WLS; WLS also std_errors / t_statistics / p_values — Plan 02 |
| reg: GLM slope t_statistics getter | INTEGRATE | pre-existing gap, closed with the same getter template on every GLM class and WLS — Plan 02 |
| reg: intercept inference / CIs at the expression layer (*_summary) | OPT-OUT | explicitly out of scope: CONTEXT resolution 3 keeps the shared 5-field summary_output_dtype unwidened; the *_summary intercept row already carries std_error / statistic / p_value after the 16.2 bump |
| reg: GlmmRegressor::negative_binomial / gamma / tweedie(power) (0.5.18 #29) | INTEGRATE | GLMM.negative_binomial(nb_theta=) / GLMM.gamma() / GLMM.tweedie(power=) factories — Plan 03 |
| reg: GlmmRegressorBuilder::nb_theta + FittedGlmm::nb_theta | INTEGRATE | factory kwarg + nb_theta getter — Plan 03 |
| reg: GlmmRegressorBuilder::offset + FittedGlmm::predict_fixed_with_offset | INTEGRATE | fit(..., offset=) / fit_crossed(..., offset=) / predict_fixed(x, offset=) — Plan 03 |
| reg: GlmmRegressorBuilder::random_slopes_per_factor | INTEGRATE | factory kwarg random_slopes_per_factor= (fit_crossed) — Plan 03 |
| reg: FittedGlmm factor_random_cov / _sd / _effects_matrix (0.5.18) | INTEGRATE | GLMM methods of the same names (factor index argument) — Plan 03 |
| reg: GlmmRegressorBuilder random_intercept / random_slopes / theta_max | INTEGRATE | pre-existing gap (random_slopes_per_factor is defined relative to them): factory kwargs on all six GLMM factories — Plan 03 |
| reg: FittedGlmm random_cov, random_effects_matrix, random_corr, n_*, is_reml | INTEGRATE | pre-existing gap (needed to read random-slope fits): GLMM getters random_cov / random_effects_matrix / random_corr / n_random_effects / n_factors / is_reml — Plan 03 |
| reg: FittedGlmm sd_random / var_random | OPT-OUT | not needed: equal random_sd[0] and random_cov[0, 0], both exposed |
| reg: GLMM expression layer | OPT-OUT | not needed: GLMM has no expression presence (PyModel-only, as 16.1's GLMM inference); CONTEXT adds the options "to its expressions, if those exist" |
| reg: eb_shrink (DerSimonian-Laird / none, fixed tau_squared) (0.5.17 #58) | INTEGRATE | ps.eb_shrink(estimate, std_error, method=, tau_squared=) expression-only (CONTEXT) — Plan 04 |
| reg: EbShrinkResult mu / mu_se / tau_squared / i_squared / q / n_groups | INTEGRATE | per-row struct fields (scalars repeated on every row) — Plan 04 |
| reg: ShrunkenGroup shrunken / shrunken_se / weight | INTEGRATE | per-row struct fields aligned with the input rows — Plan 04 |
| reg: ShrunkenGroup estimate / se | OPT-OUT | not needed: echo the input columns row-for-row |
| reg: EbShrinkError (EmptyInput/DimensionMismatch/InsufficientData/InvalidValue) | INTEGRATE | data errors → NaN rows (project convention); invalid tau_squared / method → ValueError at build time — Plan 04 |
| reg: diagnostics::augment / augment_with + Augment fields (0.5.20 #66) | INTEGRATE | one combined expression ps.augment(y, *x, model=, residual_type=) (CONTEXT resolution 5) — Plan 05; PyModel .augment() on OLS / WLS — Plan 05, on the 8 GLM classes — Plan 06 |
| reg: ResidualType response / pearson / deviance | INTEGRATE | residual_type= on the expression and the PyModel method; default response (linear) / deviance (GLM) as upstream — Plans 05, 06 |
| reg: Augment.resid_type | INTEGRATE | "resid_type" key of the PyModel dict; the expression's Python builder resolves the default explicitly, so its type is the argument — Plans 05, 06 |
| reg: Augmentable / AugmentSpec (trait plumbing) | OPT-OUT | not needed: trait machinery behind augment; every implementor (FittedOls / Wls / Poisson / Binomial / Logistic / NegativeBinomial / Tweedie / Gamma) is exposed |
| reg: augment for Probit / Cloglog | INTEGRATE | both PyModels store FittedBinomial, which implements Augmentable (corrects RESEARCH Pitfall 4) — Plans 05, 06 |
| reg: OLS / WLS builder alias_tolerance + with_alias_tolerance (0.5.21 #64) | INTEGRATE | OLS / WLS ctor kwarg and the ols / wls fit, summary, predict and formula expressions — Plan 07 |
| reg: utils::lm_alias_columns / utils::LM_ALIAS_TOLERANCE | OPT-OUT | not needed: internal aliasing pass every OLS / WLS fit runs; its result is the NaN-coefficient convention; the default 1e-7 is the alias_tolerance default |
| reg: LARS adj_r_squared / mse / rmse filled (0.5.21 #65) | INTEGRATE | LARS.adj_r_squared / .mse / .rmse getters — Plan 07 |
| reg: ModelInfo / HasModelInfo (0.5.20 #66) | INTEGRATE | `model_info` getter ({model_type, family, link}) on all 30 PyModel classes whose fitted type implements HasModelInfo, incl. 16.2's PenalizedGLM / AFT (CONTEXT resolution 4) — Plan 08 |
| reg: ModelInfo::new / gaussian / linear constructors | OPT-OUT | not needed: constructors for implementors; Python only reads a fitted model's info |
| reg: TweedieFamily::family_name / link_name | OPT-OUT | not needed: helpers behind model_info, surfaced through it |
| reg: AID heuristic compute_aid / compute_aid_anomalies + option types (0.5.17) | OPT-OUT | not needed: redundant with the richer IC-based Aid PyModel and ps.aid / ps.aid_anomalies; upstream: complements, not replaces (CONTEXT resolution 2); reversal: extend py_aid.rs |
| reg: irls::fit_irls_from (0.5.22) / utils::solve_on_kept_columns (0.5.21) | OPT-OUT | not needed: engine building blocks behind the builders already exposed |

## Notes

1. anofox-statistics' R reference data (`R/data/*.csv`) is excluded from the crates.io package (`exclude = ["R/"]`). Plan 01
   copies `p_adjust_reference.csv`, `ptukey_reference.csv`, `qtukey_reference.csv`, `posthoc_data.csv` and
   `posthoc_reference.csv` byte-identically from tag `v0.4.7` (commit 7175dc4), as 16.1 Plan 09 did for
   `wilcox_ci_reference.csv`.
2. anofox-regression's fixtures (`tests/fixtures/m1a_validation.rs`, `glmm_families_validation.rs`) ship in the crate and are
   byte-identical at 0.5.21 and 0.5.22 (sha256 237b3f71… and 7db4c5dd…). Plans 02 and 03 transcribe them into Python fixture
   modules with the full hash recorded.
3. anofox-regression 0.5.19 and 0.5.22 add no capability rows: 0.5.19 is the LARS collinear-aliasing fix and 0.5.22 a
   penalized-GLM performance fix (both behaviour, locked by 16.2 Plan 01).
4. The WLS, Ridge, ElasticNet, RLS and BLS PyModels store `Box<dyn FittedRegressor + Send + Sync>`. `HasModelInfo` and
   `Augmentable` are not supertraits of `FittedRegressor`, so Plan 05 stores `FittedWls` concretely (needed by `augment`)
   and Plan 08 captures `ModelInfo` from the concrete fitted value at each fit site of the other four.
