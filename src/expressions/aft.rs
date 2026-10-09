//! Accelerated failure time survival regression expression (`ps.aft`).
//!
//! Wraps `anofox_regression::solvers::aft` — right-censored survival
//! regression (`log T = x'beta + sigma * W`) for the Weibull, log-normal,
//! log-logistic and exponential distributions, validated against R's
//! `survival::survreg`.
//!
//! Every scalar configuration value (distribution, priors, vcov_type,
//! with_intercept, max_iter, tol, conf_level, prefix, p, t, null_policy)
//! travels in one named serde kwargs struct, [`AftKwargs`], so no option can
//! be silently dropped or index-mismatched (T-16.2-15, mirroring
//! `PenalizedGlmKwargs`). `args` carry only per-row data `[time, event, *x]`.
//!
//! The parsing helper ([`parse_aft_distribution`]) and [`build_aft`] are also
//! used by the `AFT` PyO3 class (`src/pymodels/py_aft.rs`, Plan 05 Task 2) so
//! both surfaces thread options identically. The coefficient summary and the
//! four predict expressions are added in Plan 05 Task 3.

use faer::{Col, Mat};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use statrs::distribution::{ContinuousCDF, Normal};

use anofox_regression::solvers::aft::{AftDistribution, AftRegressor};

use crate::expressions::penalized_glm::{
    inference_summary_nan_output, inference_summary_output, inference_summary_output_dtype,
    parse_vcov_type, prior_spec_from_parts, PriorKwarg,
};
use crate::expressions::regression::{
    build_xy_with_null_policy, predict_output_dtype_with_prefix, prediction_nan_output_with_prefix,
    prediction_output_with_prefix, PredictKwargs,
};

/// Named kwargs for every AFT expression (fit / summary / the four predict
/// variants, added across Plan 05's tasks). Unknown-at-fit-time fields are
/// accepted here so one kwargs struct covers the whole family of
/// expressions; [`aft_fit`] itself reads only the fields it needs.
#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct AftKwargs {
    /// Error distribution: `"weibull"` (default), `"lognormal"`,
    /// `"loglogistic"`, or `"exponential"` (see [`parse_aft_distribution`] for
    /// accepted aliases).
    pub distribution: String,
    /// Per-coefficient priors, optionally with a leading intercept entry.
    pub priors: Vec<PriorKwarg>,
    /// Covariance type at the mode: `"laplace"` (default), `"sandwich"`, or
    /// `"naive"`.
    pub vcov_type: String,
    /// Whether to fit an intercept.
    pub with_intercept: bool,
    /// Maximum Newton iterations.
    pub max_iter: u32,
    /// Convergence tolerance on the log-likelihood.
    pub tol: f64,
    /// Confidence level for intervals. Unused by [`aft_fit`] (no inference is
    /// computed there); read by the Plan 05 Task 3 summary expression.
    pub conf_level: f64,
    /// Output column prefix. Unused by [`aft_fit`]; read by the Plan 05 Task
    /// 3 predict expressions.
    pub prefix: String,
    /// Quantile probability for `aft_predict_quantile` (Plan 05 Task 3).
    /// Unused elsewhere.
    pub p: f64,
    /// Time at which to evaluate `aft_predict_survival` / `aft_predict_cdf`
    /// (Plan 05 Task 3). Unused elsewhere.
    pub t: f64,
    /// Null-row handling for the Plan 05 Task 3 predict expressions. Unused
    /// by [`aft_fit`].
    pub null_policy: String,
}

impl Default for AftKwargs {
    fn default() -> Self {
        Self {
            distribution: "weibull".to_string(),
            priors: Vec::new(),
            vcov_type: "laplace".to_string(),
            with_intercept: true,
            max_iter: 100,
            tol: 1e-9,
            conf_level: 0.95,
            prefix: "aft".to_string(),
            p: 0.5,
            t: f64::NAN,
            null_policy: "drop".to_string(),
        }
    }
}

/// Parse a distribution name (case-insensitive; accepts upstream's aliases:
/// `"weibull"`, `"lognormal"`/`"log_normal"`/`"log-normal"`,
/// `"loglogistic"`/`"log_logistic"`/`"log-logistic"`, `"exponential"`/`"exp"`).
pub fn parse_aft_distribution(s: &str) -> Result<AftDistribution, String> {
    AftDistribution::from_name(s).ok_or_else(|| {
        format!(
            "unknown AFT distribution \"{s}\" (expected one of \"weibull\", \"lognormal\" \
             (or \"log_normal\"/\"log-normal\"), \"loglogistic\" (or \"log_logistic\"/\
             \"log-logistic\"), \"exponential\" (or \"exp\"))"
        )
    })
}

/// Build an [`AftRegressor`] from an [`AftKwargs`] and an explicit
/// `compute_inference` flag.
pub fn build_aft(kwargs: &AftKwargs, compute_inference: bool) -> Result<AftRegressor, String> {
    let distribution = parse_aft_distribution(&kwargs.distribution)?;
    let vcov_type = parse_vcov_type(&kwargs.vcov_type)?;
    let mut priors = Vec::with_capacity(kwargs.priors.len());
    for p in &kwargs.priors {
        priors.push(prior_spec_from_parts(&p.kind, p.loc, p.scale)?);
    }

    let mut builder = AftRegressor::builder()
        .distribution(distribution)
        .with_intercept(kwargs.with_intercept)
        .max_iterations(kwargs.max_iter)
        .tolerance(kwargs.tol)
        .compute_inference(compute_inference)
        .confidence_level(kwargs.conf_level)
        .vcov_type(vcov_type);

    if !priors.is_empty() {
        builder = builder.priors(priors);
    }

    Ok(builder.build())
}

/// Build `(x, time, event)` from `inputs = [time, event, *x]`.
///
/// `time` (`inputs[0]`) is Float64 with null -> NaN, except that a row whose
/// event is null is masked to NaN regardless of its own value (so upstream
/// drops it like any other non-finite row). `event` (`inputs[1]`) is used
/// as-is when Boolean; a numeric column is cast to Float64 and must contain
/// only `0.0`/`1.0` (anything else, e.g. R's `1`/`2` coding, is a coding
/// error mapped by the caller to the NaN struct — never silently
/// misinterpreted). `x` (`inputs[2..]`) is Float64 with null -> NaN.
#[allow(clippy::type_complexity)]
pub(crate) fn aft_inputs(inputs: &[Series]) -> Result<(Mat<f64>, Col<f64>, Vec<bool>), String> {
    let time_series = &inputs[0];
    let event_series = &inputs[1];
    let n = time_series.len();

    let time_ca = time_series
        .f64()
        .map_err(|e| format!("aft: time column must be Float64: {e}"))?;

    let mut row_valid = vec![true; n];
    let mut event_vals = vec![false; n];

    match event_series.dtype() {
        DataType::Boolean => {
            let ca = event_series
                .bool()
                .map_err(|e| format!("aft: event column: {e}"))?;
            for i in 0..n {
                match ca.get(i) {
                    Some(b) => event_vals[i] = b,
                    None => row_valid[i] = false,
                }
            }
        }
        _ => {
            let casted = event_series
                .cast(&DataType::Float64)
                .map_err(|e| format!("aft: event column must be boolean or numeric: {e}"))?;
            let ca = casted
                .f64()
                .map_err(|e| format!("aft: event column: {e}"))?;
            for i in 0..n {
                match ca.get(i) {
                    Some(v) => {
                        if v == 0.0 {
                            event_vals[i] = false;
                        } else if v == 1.0 {
                            event_vals[i] = true;
                        } else {
                            return Err(format!(
                                "aft: event must be boolean or 0.0/1.0 (got {v}); R's 1/2 \
                                 coding is not accepted"
                            ));
                        }
                    }
                    None => row_valid[i] = false,
                }
            }
        }
    }

    let time_vec: Vec<f64> = (0..n)
        .map(|i| {
            if row_valid[i] {
                time_ca.get(i).unwrap_or(f64::NAN)
            } else {
                f64::NAN
            }
        })
        .collect();
    let time = Col::from_fn(n, |i| time_vec[i]);

    let p = inputs.len() - 2;
    let x_cols: Vec<Vec<f64>> = (2..inputs.len())
        .map(|i| {
            let ca = inputs[i]
                .f64()
                .map_err(|e| format!("aft: feature column must be Float64: {e}"))?;
            Ok((0..n).map(|r| ca.get(r).unwrap_or(f64::NAN)).collect())
        })
        .collect::<Result<_, String>>()?;
    let x = Mat::from_fn(n, p, |row, col| x_cols[col][row]);

    Ok((x, time, event_vals))
}

/// Extract a `faer::Col<f64>` as a plain `Vec<f64>`.
pub(crate) fn col_to_vec(col: &Col<f64>) -> Vec<f64> {
    (0..col.nrows()).map(|i| col[i]).collect()
}

/// Output schema for `pl_aft`: `intercept`, `coefficients`, `scale`,
/// `log_likelihood`, `null_log_likelihood`, `aic`, `bic`, `iterations`,
/// `converged`, `n_observations`, `n_events`, `n_censored`.
pub fn aft_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("intercept".into(), DataType::Float64),
        Field::new(
            "coefficients".into(),
            DataType::List(Box::new(DataType::Float64)),
        ),
        Field::new("scale".into(), DataType::Float64),
        Field::new("log_likelihood".into(), DataType::Float64),
        Field::new("null_log_likelihood".into(), DataType::Float64),
        Field::new("aic".into(), DataType::Float64),
        Field::new("bic".into(), DataType::Float64),
        Field::new("iterations".into(), DataType::UInt32),
        Field::new("converged".into(), DataType::Boolean),
        Field::new("n_observations".into(), DataType::UInt32),
        Field::new("n_events".into(), DataType::UInt32),
        Field::new("n_censored".into(), DataType::UInt32),
    ];
    Ok(Field::new("aft".into(), DataType::Struct(fields)))
}

/// Build the `aft` output struct series.
#[allow(clippy::too_many_arguments)]
pub(crate) fn aft_output(
    intercept: Option<f64>,
    coefficients: &[f64],
    scale: f64,
    log_likelihood: f64,
    null_log_likelihood: f64,
    aic: f64,
    bic: f64,
    iterations: u32,
    converged: bool,
    n_observations: usize,
    n_events: usize,
    n_censored: usize,
) -> PolarsResult<Series> {
    let intercept_s = Series::new("intercept".into(), &[intercept.unwrap_or(f64::NAN)]);
    let coef_inner = Series::new("item".into(), coefficients);
    let coef_s = Series::new("coefficients".into(), &[coef_inner]);
    let scale_s = Series::new("scale".into(), &[scale]);
    let ll_s = Series::new("log_likelihood".into(), &[log_likelihood]);
    let null_ll_s = Series::new("null_log_likelihood".into(), &[null_log_likelihood]);
    let aic_s = Series::new("aic".into(), &[aic]);
    let bic_s = Series::new("bic".into(), &[bic]);
    let iter_s = Series::new("iterations".into(), &[iterations]);
    let conv_s = Series::new("converged".into(), &[converged]);
    let n_obs_s = Series::new("n_observations".into(), &[n_observations as u32]);
    let n_ev_s = Series::new("n_events".into(), &[n_events as u32]);
    let n_cens_s = Series::new("n_censored".into(), &[n_censored as u32]);

    StructChunked::from_series(
        "aft".into(),
        1,
        [
            &intercept_s,
            &coef_s,
            &scale_s,
            &ll_s,
            &null_ll_s,
            &aic_s,
            &bic_s,
            &iter_s,
            &conv_s,
            &n_obs_s,
            &n_ev_s,
            &n_cens_s,
        ]
        .into_iter(),
    )
    .map(|ca| ca.into_series())
}

/// NaN output for `pl_aft` on an event-coding or engine fit error.
pub(crate) fn aft_nan_output() -> PolarsResult<Series> {
    aft_output(
        None,
        &[],
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        0,
        false,
        0,
        0,
        0,
    )
}

/// Public Rust-callable variant. `inputs[0]` = time, `inputs[1]` = event
/// (Boolean or numeric 0/1), `inputs[2..]` = x columns. Config errors
/// (unknown distribution / vcov_type / prior) are a `ComputeError`;
/// event-coding and engine fit errors return the NaN struct.
pub fn aft_fit(inputs: &[Series], kwargs: &AftKwargs) -> PolarsResult<Series> {
    let regressor = build_aft(kwargs, false).map_err(|e| polars_err!(ComputeError: "{}", e))?;

    let (x, time, event) = match aft_inputs(inputs) {
        Ok(v) => v,
        Err(_) => return aft_nan_output(),
    };

    match regressor.fit(&x, &time, &event) {
        Ok(fitted) => {
            let fr = fitted.fit_result();
            aft_output(
                fitted.intercept(),
                &col_to_vec(fitted.coefficients()),
                fitted.scale(),
                fitted.log_likelihood(),
                fitted.null_log_likelihood(),
                fitted.aic(),
                fitted.bic(),
                fr.iterations,
                fr.converged,
                fr.n_observations,
                fr.n_events,
                fr.n_censored,
            )
        }
        Err(_) => aft_nan_output(),
    }
}

/// AFT fit expression (`ps.aft`). `inputs[0]` = time, `inputs[1]` = event,
/// `inputs[2..]` = x columns. kwargs: see [`AftKwargs`].
#[polars_expr(output_type_func=aft_output_dtype)]
fn pl_aft(inputs: &[Series], kwargs: AftKwargs) -> PolarsResult<Series> {
    aft_fit(inputs, &kwargs)
}

// ============================================================================
// Coefficient Summary (intercept + features + log_scale, with CIs)
// ============================================================================

/// Upstream's Wald inference formula (`laplace::inference`), reproduced here
/// only for the `log_scale` summary row (the intercept and feature rows are
/// read directly from upstream's own `AftInference` fields — never
/// recomputed). Returns `(z, p, ci_lower, ci_upper)`; NaN throughout when
/// `se` is not finite or not `> 0`.
fn log_scale_wald(estimate: f64, se: f64, conf_level: f64) -> (f64, f64, f64, f64) {
    if !se.is_finite() || se <= 0.0 {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let z = estimate / se;
    let p = match Normal::new(0.0, 1.0) {
        Ok(d) => 2.0 * (1.0 - d.cdf(z.abs())),
        Err(_) => f64::NAN,
    };
    let z_crit = normal_quantile(0.5 + conf_level / 2.0);
    (z, p, estimate - z_crit * se, estimate + z_crit * se)
}

/// Standard normal quantile; `statrs`'s `inverse_cdf` panics on the open
/// interval endpoints, so the input is clamped first (upstream's exact
/// convention).
fn normal_quantile(p: f64) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    match Normal::new(0.0, 1.0) {
        Ok(d) => d.inverse_cdf(p),
        Err(_) => f64::NAN,
    }
}

/// Public Rust-callable variant. Same input contract as [`aft_fit`]; always
/// fits with `compute_inference = true` regardless of the caller's intent,
/// since this is the summary expression.
///
/// Rows are `intercept` (when `kwargs.with_intercept`), `x1..xp`, then
/// `log_scale` (when the scale is estimated, i.e. not the exponential
/// distribution). The intercept row's std_error/statistic/p_value/CI come
/// straight from upstream's `AftInference::intercept_*` fields — never
/// recomputed locally; the `log_scale` row's statistic/p_value/CI are
/// computed by [`log_scale_wald`] from upstream's `log_scale_std_error`
/// (upstream itself does not expose a log-scale z/p/CI, only the SE).
pub fn aft_summary_fit(inputs: &[Series], kwargs: &AftKwargs) -> PolarsResult<Series> {
    let regressor = build_aft(kwargs, true).map_err(|e| polars_err!(ComputeError: "{}", e))?;

    let (x, time, event) = match aft_inputs(inputs) {
        Ok(v) => v,
        Err(_) => return inference_summary_nan_output(),
    };

    match regressor.fit(&x, &time, &event) {
        Ok(fitted) => {
            let with_intercept = kwargs.with_intercept;
            let n_features = inputs.len() - 2;
            let inference = fitted.inference();

            let mut terms = Vec::new();
            let mut estimates = Vec::new();
            let mut std_errors = Vec::new();
            let mut statistics = Vec::new();
            let mut p_values = Vec::new();
            let mut ci_lower = Vec::new();
            let mut ci_upper = Vec::new();

            if with_intercept {
                terms.push("intercept".to_string());
                estimates.push(fitted.intercept().unwrap_or(f64::NAN));
                std_errors.push(
                    inference
                        .and_then(|i| i.intercept_std_error)
                        .unwrap_or(f64::NAN),
                );
                statistics.push(
                    inference
                        .and_then(|i| i.intercept_z_value)
                        .unwrap_or(f64::NAN),
                );
                p_values.push(
                    inference
                        .and_then(|i| i.intercept_p_value)
                        .unwrap_or(f64::NAN),
                );
                ci_lower.push(
                    inference
                        .and_then(|i| i.intercept_ci_lower)
                        .unwrap_or(f64::NAN),
                );
                ci_upper.push(
                    inference
                        .and_then(|i| i.intercept_ci_upper)
                        .unwrap_or(f64::NAN),
                );
            }

            for j in 1..=n_features {
                terms.push(format!("x{j}"));
            }
            estimates.extend(col_to_vec(fitted.coefficients()));

            if let Some(inf) = inference {
                std_errors.extend(inf.std_errors.iter().copied());
                statistics.extend(inf.z_values.iter().copied());
                p_values.extend(inf.p_values.iter().copied());
                ci_lower.extend(inf.ci_lower.iter().copied());
                ci_upper.extend(inf.ci_upper.iter().copied());
            } else {
                std_errors.extend(vec![f64::NAN; n_features]);
                statistics.extend(vec![f64::NAN; n_features]);
                p_values.extend(vec![f64::NAN; n_features]);
                ci_lower.extend(vec![f64::NAN; n_features]);
                ci_upper.extend(vec![f64::NAN; n_features]);
            }

            if let Some(log_scale_se) = inference.and_then(|i| i.log_scale_std_error) {
                let estimate = fitted.scale().ln();
                let (z, p, lo, hi) = log_scale_wald(estimate, log_scale_se, kwargs.conf_level);
                terms.push("log_scale".to_string());
                estimates.push(estimate);
                std_errors.push(log_scale_se);
                statistics.push(z);
                p_values.push(p);
                ci_lower.push(lo);
                ci_upper.push(hi);
            }

            inference_summary_output(
                terms, estimates, std_errors, statistics, p_values, ci_lower, ci_upper,
            )
        }
        Err(_) => inference_summary_nan_output(),
    }
}

/// AFT coefficient summary expression (`ps.aft_summary`). Same input
/// contract as `pl_aft`. kwargs: see [`AftKwargs`] (`conf_level` /
/// `vcov_type` are read here).
#[polars_expr(output_type_func=inference_summary_output_dtype)]
fn pl_aft_summary(inputs: &[Series], kwargs: AftKwargs) -> PolarsResult<Series> {
    aft_summary_fit(inputs, &kwargs)
}

// ============================================================================
// Prediction (quantile / median / survival / cdf)
// ============================================================================

/// Which AFT prediction to compute.
enum AftPredictKind {
    Quantile(f64),
    Median,
    Survival(f64),
    Cdf(f64),
}

/// Output dtype for the four `pl_aft_predict_*` expressions — delegates to
/// the shared prefix-based prediction schema (`{prefix}_prediction`/
/// `_lower`/`_upper`).
fn aft_predict_output_dtype(input_fields: &[Field], kwargs: AftKwargs) -> PolarsResult<Field> {
    predict_output_dtype_with_prefix(
        input_fields,
        PredictKwargs {
            prefix: kwargs.prefix,
        },
    )
}

/// Shared implementation for the four AFT predict expressions. Same input
/// contract as [`aft_fit`] (`[time, event, *x]`); refits per group/row-batch
/// with the same options as `aft_fit`, then predicts on every row using
/// [`build_xy_with_null_policy`] with `time` as the nominal "y" column
/// (null where the row's event is null, matching [`aft_inputs`]'s masking)
/// and `event` gathered for the valid rows. `lower`/`upper` are always NaN —
/// the upstream engine has no interval-prediction method for any of the four
/// quantities.
fn aft_predict_impl(
    inputs: &[Series],
    kwargs: &AftKwargs,
    kind: AftPredictKind,
) -> PolarsResult<Series> {
    let n_rows = inputs[0].len();
    let prefix = &kwargs.prefix;

    if let AftPredictKind::Quantile(p) = kind {
        if !(0.0..=1.0).contains(&p) {
            return Err(
                polars_err!(ComputeError: "aft_predict_quantile: p must be in [0, 1] (got {})", p),
            );
        }
    }
    if let AftPredictKind::Survival(t) | AftPredictKind::Cdf(t) = kind {
        if !t.is_finite() || t <= 0.0 {
            return Err(
                polars_err!(ComputeError: "aft_predict: t must be finite and > 0 (got {})", t),
            );
        }
    }

    // Mask time at rows whose event is null, exactly like `aft_inputs`, so
    // the null-policy split below excludes them from the fit under every
    // policy (the event column itself is converted to `Vec<bool>` only for
    // the rows the split keeps).
    let event_series = &inputs[1];
    let event_null_mask: Vec<bool> = match event_series.dtype() {
        DataType::Boolean => {
            let ca = event_series.bool()?;
            (0..n_rows).map(|i| ca.get(i).is_none()).collect()
        }
        _ => {
            let casted = event_series.cast(&DataType::Float64)?;
            let ca = casted.f64()?;
            (0..n_rows).map(|i| ca.get(i).is_none()).collect()
        }
    };
    let event_bool: Vec<Option<bool>> = match event_series.dtype() {
        DataType::Boolean => {
            let ca = event_series.bool()?;
            (0..n_rows).map(|i| ca.get(i)).collect()
        }
        _ => {
            let casted = event_series.cast(&DataType::Float64)?;
            let ca = casted.f64()?;
            (0..n_rows).map(|i| ca.get(i).map(|v| v == 1.0)).collect()
        }
    };

    let time_ca = inputs[0].f64()?;
    let masked_time: Vec<Option<f64>> = (0..n_rows)
        .map(|i| {
            if event_null_mask[i] {
                None
            } else {
                time_ca.get(i)
            }
        })
        .collect();
    let mut processed_inputs: Vec<Series> = inputs.to_vec();
    processed_inputs[0] = Series::new(inputs[0].name().clone(), masked_time);

    let (x_fit, time_fit, valid_mask, x_pred) =
        match build_xy_with_null_policy(&processed_inputs, 0, 2, &kwargs.null_policy) {
            Ok(data) => data,
            Err(_) => return prediction_nan_output_with_prefix(n_rows, prefix),
        };

    let valid_indices: Vec<usize> = (0..n_rows).filter(|&i| valid_mask[i]).collect();
    let event_fit: Vec<bool> = valid_indices
        .iter()
        .map(|&i| event_bool[i].unwrap_or(false))
        .collect();

    let regressor = build_aft(kwargs, false).map_err(|e| polars_err!(ComputeError: "{}", e))?;

    match regressor.fit(&x_fit, &time_fit, &event_fit) {
        Ok(fitted) => {
            let raw: Vec<f64> = match kind {
                AftPredictKind::Quantile(p) => col_to_vec(&fitted.predict_quantile(&x_pred, p)),
                AftPredictKind::Median => col_to_vec(&fitted.predict_median(&x_pred)),
                AftPredictKind::Survival(t) => col_to_vec(&fitted.predict_survival(&x_pred, t)),
                AftPredictKind::Cdf(t) => col_to_vec(&fitted.predict_cdf(&x_pred, t)),
            };

            // A row with a null event is excluded from the FIT via the masked
            // `time` column above (so `valid_mask` is already false for it
            // under every policy), but it is still PREDICTED when the policy
            // is "drop_y_zero_x" — exactly like a row with a null time, since
            // the predict methods only need `x`, not `event`.
            let predictions: Vec<f64> = (0..n_rows)
                .map(|i| {
                    let drop_masked = kwargs.null_policy == "drop" && !valid_mask[i];
                    if drop_masked {
                        f64::NAN
                    } else {
                        raw[i]
                    }
                })
                .collect();

            prediction_output_with_prefix(
                predictions,
                vec![f64::NAN; n_rows],
                vec![f64::NAN; n_rows],
                prefix,
            )
        }
        Err(_) => prediction_nan_output_with_prefix(n_rows, prefix),
    }
}

/// Public Rust-callable variant of `ps.aft_predict_quantile`.
pub fn aft_predict_quantile_fit(inputs: &[Series], kwargs: &AftKwargs) -> PolarsResult<Series> {
    aft_predict_impl(inputs, kwargs, AftPredictKind::Quantile(kwargs.p))
}

/// Public Rust-callable variant of `ps.aft_predict_median`.
pub fn aft_predict_median_fit(inputs: &[Series], kwargs: &AftKwargs) -> PolarsResult<Series> {
    aft_predict_impl(inputs, kwargs, AftPredictKind::Median)
}

/// Public Rust-callable variant of `ps.aft_predict_survival`.
pub fn aft_predict_survival_fit(inputs: &[Series], kwargs: &AftKwargs) -> PolarsResult<Series> {
    aft_predict_impl(inputs, kwargs, AftPredictKind::Survival(kwargs.t))
}

/// Public Rust-callable variant of `ps.aft_predict_cdf`.
pub fn aft_predict_cdf_fit(inputs: &[Series], kwargs: &AftKwargs) -> PolarsResult<Series> {
    aft_predict_impl(inputs, kwargs, AftPredictKind::Cdf(kwargs.t))
}

/// AFT `p`-quantile prediction expression (`ps.aft_predict_quantile`). Same
/// input contract as `pl_aft`. kwargs: see [`AftKwargs`] (`p` / `prefix` /
/// `null_policy` are read here).
#[polars_expr(output_type_func_with_kwargs=aft_predict_output_dtype)]
fn pl_aft_predict_quantile(inputs: &[Series], kwargs: AftKwargs) -> PolarsResult<Series> {
    aft_predict_quantile_fit(inputs, &kwargs)
}

/// AFT median prediction expression (`ps.aft_predict_median`). Same input
/// contract as `pl_aft`. kwargs: see [`AftKwargs`].
#[polars_expr(output_type_func_with_kwargs=aft_predict_output_dtype)]
fn pl_aft_predict_median(inputs: &[Series], kwargs: AftKwargs) -> PolarsResult<Series> {
    aft_predict_median_fit(inputs, &kwargs)
}

/// AFT survival-probability prediction expression
/// (`ps.aft_predict_survival`). Same input contract as `pl_aft`. kwargs: see
/// [`AftKwargs`] (`t` / `prefix` / `null_policy` are read here).
#[polars_expr(output_type_func_with_kwargs=aft_predict_output_dtype)]
fn pl_aft_predict_survival(inputs: &[Series], kwargs: AftKwargs) -> PolarsResult<Series> {
    aft_predict_survival_fit(inputs, &kwargs)
}

/// AFT CDF prediction expression (`ps.aft_predict_cdf`). Same input contract
/// as `pl_aft`. kwargs: see [`AftKwargs`] (`t` / `prefix` / `null_policy` are
/// read here).
#[polars_expr(output_type_func_with_kwargs=aft_predict_output_dtype)]
fn pl_aft_predict_cdf(inputs: &[Series], kwargs: AftKwargs) -> PolarsResult<Series> {
    aft_predict_cdf_fit(inputs, &kwargs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `solvers::aft` module-doc example data: two groups
    /// (`x = i % 2`), 10 observations, 2 censored. Proves [`log_scale_wald`]
    /// reproduces upstream's `AftInference::intercept_{z_value,p_value,
    /// ci_lower,ci_upper}` exactly when applied to the intercept's own
    /// estimate/SE — the same `laplace::inference` Wald formula the
    /// `log_scale` summary row's statistic/p_value/CI must match.
    #[test]
    fn log_scale_wald_reproduces_upstream_intercept_inference() {
        let x = Mat::from_fn(10, 1, |i, _| (i % 2) as f64);
        let t = Col::from_fn(10, |i| {
            [5.0, 9.0, 6.0, 14.0, 4.0, 11.0, 7.0, 16.0, 5.5, 12.0][i]
        });
        let event = vec![true, true, true, false, true, true, true, true, false, true];

        let fit = AftRegressor::builder()
            .distribution(AftDistribution::Weibull)
            .compute_inference(true)
            .build()
            .fit(&x, &t, &event)
            .expect("fit should succeed on the module-doc example");

        let inf = fit.inference().expect("inference should be computed");
        let intercept = fit.intercept().expect("fit has an intercept");
        let se = inf
            .intercept_std_error
            .expect("intercept SE should be present");

        let (z, p, lo, hi) = log_scale_wald(intercept, se, 0.95);

        let rel_close = |a: f64, b: f64, tol: f64| (a - b).abs() / b.abs().max(1.0) < tol;
        assert!(rel_close(
            z,
            inf.intercept_z_value.expect("intercept z_value"),
            1e-12
        ));
        assert!(rel_close(
            p,
            inf.intercept_p_value.expect("intercept p_value"),
            1e-12
        ));
        assert!(rel_close(
            lo,
            inf.intercept_ci_lower.expect("intercept ci_lower"),
            1e-12
        ));
        assert!(rel_close(
            hi,
            inf.intercept_ci_upper.expect("intercept ci_upper"),
            1e-12
        ));
    }
}
