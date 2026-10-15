# Output Structures

All return types from polars-statistics functions.

## Linear Model Output

Returned by `ols`, `ridge`, `elastic_net`, `wls`, `rls`, `bls`, `nnls`.

```
Struct {
    intercept: Float64,
    coefficients: List[Float64],
    r_squared: Float64,
    adj_r_squared: Float64,
    mse: Float64,
    rmse: Float64,
    f_statistic: Float64,
    f_pvalue: Float64,
    aic: Float64,
    bic: Float64,
    n_observations: UInt32,
}
```

---

## Quantile Regression Output

Returned by `quantile`.

```
Struct {
    intercept: Float64,
    coefficients: List[Float64],
    tau: Float64,
    pseudo_r_squared: Float64,
    check_loss: Float64,
    n_observations: UInt32,
}
```

---

## Isotonic Regression Output

Returned by `isotonic`.

```
Struct {
    r_squared: Float64,
    increasing: Boolean,
    fitted_values: List[Float64],
    n_observations: UInt32,
}
```

---

## Condition Number Output

Returned by `condition_number`.

```
Struct {
    condition_number: Float64,
    condition_number_xtx: Float64,
    singular_values: List[Float64],
    condition_indices: List[Float64],
    severity: String,           # "WellConditioned", "Moderate", "High", "Severe"
    warning: String,
}
```

---

## Separation Check Output

Returned by `check_binary_separation` and `check_count_sparsity`.

```
Struct {
    has_separation: Boolean,
    separated_predictors: List[UInt32],
    separation_types: List[String],  # "Complete", "Quasi", "MonotonicResponse"
    warning: String,
}
```

---

## VIF Output

Returned by `vif`.

```
Struct {
    terms: List[String],          # "x1", "x2", ...
    vif: List[Float64],
    n_observations: UInt32,
}
```

---

## VIF Mask Output

Returned by `high_vif_predictors`.

```
Struct {
    is_high: List[Boolean],       # one per input feature column
    n_high: UInt32,
    n_features: UInt32,
}
```

---

## GVIF Output

Returned by `generalized_vif`.

```
Struct {
    gvif: List[Float64],          # one value per group
    n_groups: UInt32,
}
```

---

## Leverage Output

Returned by `leverage`.

```
Struct {
    leverage: List[Float64],      # one h_ii per input row
    n_observations: UInt32,
}
```

---

## Cook's Distance Output

Returned by `cooks_distance`.

```
Struct {
    cooks_d: List[Float64],       # one D_i per input row
    n_observations: UInt32,
}
```

---

## DFFITS Output

Returned by `dffits`.

```
Struct {
    dffits: List[Float64],        # one value per input row
    n_observations: UInt32,
}
```

---

## Influence Mask Output

Returned by `influential_cooks`, `influential_dffits`, `high_leverage_points`.

```
Struct {
    is_influential: List[Boolean],
    n_influential: UInt32,
    n_observations: UInt32,
}
```

---

## Residual Diagnostics Output

Returned by `standardized_residuals`, `studentized_residuals`, `externally_studentized_residuals`, and all GLM residual functions (`logistic_*_residuals`, `poisson_*_residuals`).

```
Struct {
    residuals: List[Float64],     # one residual per input row
    n_observations: UInt32,
}
```

---

## Outlier Mask Output

Returned by `residual_outliers`.

```
Struct {
    is_outlier: List[Boolean],
    n_outliers: UInt32,
    n_observations: UInt32,
}
```

---

## Chi-Squared Output

Returned by `pearson_chi_squared_logistic` and `pearson_chi_squared_poisson`.

```
Struct {
    chi_squared: Float64,
    df_resid: UInt32,
    n_observations: UInt32,
}
```

---

## GLM Output

Returned by `logistic`, `poisson`, `negative_binomial`, `tweedie`, `probit`, `cloglog`.

```
Struct {
    intercept: Float64,
    coefficients: List[Float64],
    deviance: Float64,
    null_deviance: Float64,
    aic: Float64,
    bic: Float64,
    n_observations: UInt32,
}
```

---

## Penalized GLM Output

Returned by `penalized_glm`. See [Penalized GLM](regression/penalized-glm.md).

```
Struct {
    intercept: Float64,
    coefficients: List[Float64],
    deviance: Float64,
    null_deviance: Float64,
    pseudo_r_squared: Float64,
    log_likelihood: Float64,
    aic: Float64,
    bic: Float64,
    dispersion: Float64,       # 1.0 (Poisson unless estimate_dispersion=True; Binomial; Logistic),
                                # Pearson estimate (Gamma, Tweedie, quasipoisson), or theta (Negative Binomial)
    theta: Float64,            # NaN unless family="negative_binomial"
    accuracy: Float64,         # NaN unless family="logistic"
    has_separation: Boolean,   # family="binomial"/"logistic" only; False otherwise
    iterations: UInt32,
    converged: Boolean,
    n_observations: UInt32,
}
```

---

## AFT Output

Returned by `aft`. See [Survival Regression (AFT)](regression/survival.md).

```
Struct {
    intercept: Float64,
    coefficients: List[Float64],
    scale: Float64,                 # sigma; 1.0 (not estimated) for distribution="exponential"
    log_likelihood: Float64,        # on the time scale, as survreg reports it
    null_log_likelihood: Float64,   # intercept(+scale)-only model
    aic: Float64,
    bic: Float64,
    iterations: UInt32,
    converged: Boolean,
    n_observations: UInt32,
    n_events: UInt32,
    n_censored: UInt32,
}
```

---

## ALM Output

Returned by `alm`.

```
Struct {
    intercept: Float64,
    coefficients: List[Float64],
    aic: Float64,
    bic: Float64,
    log_likelihood: Float64,
    n_observations: UInt32,
}
```

---

## LmDynamic Output

Returned by `lm_dynamic`.

```
Struct {
    intercept: Float64,
    coefficients: List[Float64],
    r_squared: Float64,
    adj_r_squared: Float64,
    mse: Float64,
    rmse: Float64,
    n_observations: UInt32,
}
```

---

## AID Output

Returned by `aid`.

```
Struct {
    demand_type: String,           # "regular" or "intermittent"
    is_intermittent: Boolean,
    is_fractional: Boolean,
    distribution: String,          # Best-fit distribution name
    mean: Float64,
    variance: Float64,
    zero_proportion: Float64,
    n_observations: UInt32,
    has_stockouts: Boolean,
    is_new_product: Boolean,
    is_obsolete_product: Boolean,
    stockout_count: UInt32,
    new_product_count: UInt32,
    obsolete_product_count: UInt32,
    high_outlier_count: UInt32,
    low_outlier_count: UInt32,
}
```

---

## AID Anomalies Output

Returned by `aid_anomalies`. Per-row struct (use with `.over()` and `.unnest()`).

```
Struct {
    stockout: Boolean,
    new_product: Boolean,
    obsolete_product: Boolean,
    high_outlier: Boolean,
    low_outlier: Boolean,
}
```

---

## Summary Output

Returned by `*_summary` functions.

```
List[Struct {
    term: String,
    estimate: Float64,
    std_error: Float64,
    statistic: Float64,
    p_value: Float64,
}]
```

---

## Inference Summary Output

Returned by `penalized_glm_summary` and `aft_summary`. See
[Penalized GLM](regression/penalized-glm.md#penalized_glm_summary) and
[Survival Regression (AFT)](regression/survival.md#aft_summary).

```
List[Struct {
    term: String,
    estimate: Float64,
    std_error: Float64,
    statistic: Float64,   # Wald z, normal reference
    p_value: Float64,
    ci_lower: Float64,
    ci_upper: Float64,
}]
```

A widened, 7-field variant of [Summary Output](#summary-output) (adds `ci_lower`/`ci_upper`) —
additive, not a replacement; the shared `summary_output_dtype` and every other `*_summary`
expression are unchanged. Rows are `intercept` (when an intercept is fitted), `x1..xp`, and for
`aft_summary` a trailing `log_scale` row (omitted for `distribution="exponential"`). The
`intercept` row is read directly from the upstream engine's own intercept inference fields,
never recomputed locally.

---

## Prediction Output

Returned by `*_predict` functions.

```
Struct {
    prediction: Float64,
    lower: Float64,
    upper: Float64,
}
```

---

## Statistical Test Output

Returned by most statistical tests.

```
Struct {
    statistic: Float64,
    p_value: Float64,
}
```

Some tests include additional fields like `df`, `n`, `estimate`, `ci_lower`, `ci_upper`.

---

## Pairwise Comparison Output

Returned by `tukey_hsd`, `pairwise_t_test`, `dunn_test`. See
[Post-hoc Tests & Multiple Comparisons](tests/posthoc.md).

```
List[Struct {
    group1: String,
    group2: String,
    n1: UInt32,
    n2: UInt32,
    estimate: Float64,     # mean/mean-rank difference, group2 - group1
    statistic: Float64,
    df: Float64,           # NaN for pairwise_t_test (Welch df varies per pair, not reported) and dunn_test
    p_value: Float64,
    p_adj: Float64,        # == p_value for tukey_hsd (already simultaneous)
    ci_lower: Float64,     # NaN for pairwise_t_test and dunn_test
    ci_upper: Float64,     # NaN for pairwise_t_test and dunn_test
}]
```

One row per call (`returns_scalar=True`); explode to one row per pairwise comparison with
`.explode("col").struct.unnest()`. For `k` groups there are `k(k-1)/2` rows. Rows with a
null/NaN value or a null group are dropped before computing; fewer than 2 usable groups give
an empty list, never an exception.

---

## Elementwise Float64 Output

Returned by `p_adjust`, `ptukey`, `qtukey`. See
[Post-hoc Tests & Multiple Comparisons](tests/posthoc.md).

A bare `Float64` expression — no `Struct` wrapper — the same length as the input column(s).
`p_adjust` is a whole-column operation (adjusts the full family together, not value-by-value);
`ptukey`/`qtukey` are genuinely elementwise and broadcast scalar arguments against column
arguments. All three return `NaN` (never an exception) for invalid input rather than raising.

---

## Mann-Whitney / Wilcoxon Output

Returned by `mann_whitney_u` and `wilcoxon_signed_rank`. See
[Non-Parametric Tests](tests/nonparametric.md).

```
Struct {
    statistic: Float64,
    p_value: Float64,
    estimate: Float64,    # Hodges-Lehmann shift (mann_whitney_u) / pseudo-median (wilcoxon_signed_rank)
    ci_lower: Float64,
    ci_upper: Float64,
}
```

`estimate`/`ci_lower`/`ci_upper` are trailing fields added in 0.8.0 — the first two fields
(`statistic`, `p_value`) are unchanged in order and value. They are `NaN` unless `conf_level`
is given (R reports them only with `conf.int = TRUE`); a one-sided `alternative` gives an
infinite bound. Values match R `wilcox.test(conf.int = TRUE)` on 126 upstream reference cases
and equal the [`MannWhitneyU`/`WilcoxonSignedRank`](classes/tests.md) classes' `estimate`/
`conf_int` exactly.

---

## Correlation Output

Returned by `pearson`, `spearman`, `kendall`, `distance_cor`, `partial_cor`, `semi_partial_cor`, `icc`.

```
Struct {
    estimate: Float64,
    statistic: Float64,
    p_value: Float64,
    ci_lower: Float64,
    ci_upper: Float64,
    n: UInt32,
}
```

---

## TOST Equivalence Output

Returned by all `tost_*` functions.

```
Struct {
    estimate: Float64,
    ci_lower: Float64,
    ci_upper: Float64,
    bound_lower: Float64,
    bound_upper: Float64,
    tost_p_value: Float64,
    equivalent: Boolean,
    alpha: Float64,
    n: UInt32,
}
```

---

## Effect Size Output

Returned by `cohens_d`, `cohens_d_one_sample`, `rank_biserial`,
`matched_pairs_rank_biserial`, `rank_biserial_from_u`. See
[Effect Sizes & Bootstrap CIs](tests/effect-sizes.md).

```
Struct {
    estimate: Float64,
}
```

---

## Jarque-Bera Output

Returned by `jarque_bera`. See [Distributional Tests](tests/distributional.md).

```
Struct {
    statistic: Float64,
    p_value: Float64,
    skewness: Float64,
    kurtosis: Float64,
    n: UInt32,
}
```

Fewer than 3 finite values, a non-finite input, or constant data yield NaN
`statistic`/`p_value`/`skewness`/`kurtosis` and `n == 0` (never an exception).

---

## Bootstrap CI Output

Returned by `bootstrap_mean_ci` and `bootstrap_ci`. See
[Effect Sizes & Bootstrap CIs](tests/effect-sizes.md).

```
Struct {
    estimate: Float64,
    se: Float64,
    ci_lower: Float64,
    ci_upper: Float64,
    conf_level: Float64,
    n_bootstrap: UInt32,
}
```

`se` is the bootstrap standard error (the sample SD of the resampled replicates).
Invalid parameters (`n_bootstrap < 2`, `conf_level` outside `(0, 1)`, fewer than 2
finite observations) yield an all-NaN struct, never an exception.

---

## Fisher Exact Output

Returned by `fisher_exact` and `fisher_exact_conditional`. See
[Categorical Tests](tests/categorical.md).

```
Struct {
    statistic: Float64,
    p_value: Float64,
    ci_lower: Float64,
    ci_upper: Float64,
}
```

`statistic` is the odds ratio: for `fisher_exact` this is the sample odds ratio
(`ad/bc`) with a two-sided Woolf (logit) interval; for `fisher_exact_conditional`
it is R `fisher.test`'s conditional maximum-likelihood odds ratio with the exact
conditional interval. `ci_lower`/`ci_upper` were added to `fisher_exact` in 0.8.0 —
previously the struct had only `statistic`/`p_value`. One-sided `alternative` values
give a one-sided interval (`"greater"` → `ci_upper = +inf`, `"less"` → `ci_lower = 0.0`).
