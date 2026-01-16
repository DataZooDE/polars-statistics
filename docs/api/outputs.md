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
