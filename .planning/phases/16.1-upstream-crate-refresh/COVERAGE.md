# API Coverage — anofox-statistics 0.4.2→0.4.4 and anofox-regression 0.5.13→0.5.15

> Full coverage by default. Opt-outs are explicit, reasoned decisions.
> Surface enumerated by diffing every `pub fn` / `pub struct` between the pinned and target
> crate sources in the cargo registry (not from CHANGELOG prose), plus CHANGELOG "Added" items.
> The reason column of INTEGRATE rows names where the capability lands (plan number).

| capability | decision | reason |
|---|---|---|
| stats: binom_test_with_conf_level | INTEGRATE | ps.binom_test(conf_level=) — Plan 02 |
| stats: prop_test_one_with_conf_level | INTEGRATE | ps.prop_test_one(conf_level=) — Plan 02 |
| stats: prop_test_two_with_conf_level | INTEGRATE | ps.prop_test_two(conf_level=) — Plan 02 |
| stats: fisher_exact_with_conf_level | INTEGRATE | ps.fisher_exact(conf_level=); struct gains ci_lower/ci_upper — Plan 02 |
| stats: fisher_exact_conditional | INTEGRATE | new ps.fisher_exact_conditional — Plan 02 |
| stats: cohens_d (Student / Welch / Paired) | INTEGRATE | new ps.cohens_d(kind=) — Plan 02 |
| stats: cohens_d_one_sample | INTEGRATE | new ps.cohens_d_one_sample — Plan 02 |
| stats: rank_biserial | INTEGRATE | new ps.rank_biserial — Plan 02 |
| stats: matched_pairs_rank_biserial | INTEGRATE | new ps.matched_pairs_rank_biserial — Plan 02 |
| stats: rank_biserial_from_u | INTEGRATE | new ps.rank_biserial_from_u (scalar literal args) — Plan 02 |
| stats: jarque_bera / JarqueBeraResult | INTEGRATE | new ps.jarque_bera — Plan 04 |
| stats: bootstrap_mean_ci (IID / stationary / circular block) | INTEGRATE | new ps.bootstrap_mean_ci — Plan 04 |
| stats: bootstrap_ci with fixed statistic (mean/median/std) | INTEGRATE | new ps.bootstrap_ci(statistic=) via Rust closure dispatch — Plan 04 |
| stats: bootstrap_ci with an arbitrary user closure | OPT-OUT | explicitly out of scope: a Rust Fn closure cannot cross FFI; Python callbacks in Polars worker threads risk GIL deadlock (see note 1) |
| stats: non-finite handling (#17: NaN error, ±Inf ranked) | INTEGRATE | locked by tests/test_nonfinite.py (Plan 01); documented (Plans 07, 08) |
| stats: validate_finite_data | OPT-OUT | not needed: input-validation helper, not a statistical capability; its effect is tested via the non-finite paths |
| reg: Huber compute_inference / confidence_level | INTEGRATE | Huber kwargs + inference getters — Plan 05 |
| reg: RANSAC compute_inference / confidence_level | INTEGRATE | RANSAC kwargs + inference and fit-stat getters — Plan 05 |
| reg: FittedRansac::inlier_fit | INTEGRATE | RANSAC.inlier_fit() returns a fitted OLS — Plan 05 |
| reg: FittedRansac::predict_with_interval | OPT-OUT | not needed: no PyModel exposes interval prediction and RANSAC has no expression family to extend (see note 2) |
| reg: NegativeBinomial estimate_dispersion | INTEGRATE | PyModel kwarg + negative_binomial fit/summary/predict/formula kwarg — Plan 03 |
| reg: FittedNegativeBinomial::dispersion | INTEGRATE | NegativeBinomial.dispersion getter — Plan 03 |
| reg: RLS initial_p_diagonal (builder) | INTEGRATE | PyModel kwarg + rls fit/summary/predict/formula kwarg — Plan 03 |
| reg: RlsRegressor::initial_p_diagonal() accessor | INTEGRATE | RLS.initial_p_diagonal_value() — Plan 03 |
| reg: FittedGlmm z_values / p_values / conf_int | INTEGRATE | GLMM getters + conf_int(level) method — Plan 03 |
| reg: TheilSen univariate_pairwise | INTEGRATE | TheilSen(univariate_pairwise=) — Plan 05 |
| reg: OLS / WLS / Ridge variance_factor | INTEGRATE | variance_factor() methods — Plan 05 |
| reg: OLS / WLS / Ridge leverage_new | INTEGRATE | leverage_new(x_new) methods — Plan 05 |
| reg: Ridge sandwich inference (#37) | INTEGRATE | Ridge std_errors/t_statistics/p_values getters (Plan 05); ridge_summary locked (Plan 01) |
| reg: inference free fns (variance factor / leverage / intervals) | OPT-OUT | not needed: raw-matrix forms of the fitted-model methods and *_predict intervals already exposed (see note 3) |
| reg: utils detect_constant_columns_relative / detect_zero_columns | OPT-OUT | not needed: internal design-matrix utilities; their aliasing effect surfaces in every fit and is tested |
| reg: FittedRls::update non-finite skip | OPT-OUT | not needed yet: PyRLS never exposed online update(); a pre-existing gap outside this upstream delta |

## Notes

1. RESEARCH "Don't Hand-Roll" and orchestrator resolution 2: `bootstrap_ci<F: Fn(&[f64]) -> f64>` takes a
   native Rust closure; bridging a Python callable would re-enter the GIL from inside Polars' multithreaded
   group_by. The fixed `mean` / `median` / `std` dispatch covers the crate's own use cases; the scope limit is
   documented in the builder docstring and the skill.
2. Interval prediction in this library is expression-only (`*_predict(interval=...)`); orchestrator resolution 1
   forbids inventing a RANSAC expression family. `RANSAC.inlier_fit().variance_factor()` / `.leverage_new()`
   provide the interval building blocks.
3. `inference::compute_variance_factor`, `compute_ridge_variance_factor`, `leverage_new`,
   `intervals_from_variance_factor` are matrix utilities whose capability is exposed through
   `OLS/WLS/Ridge.variance_factor()` / `.leverage_new()` and the `*_predict` interval expressions.
