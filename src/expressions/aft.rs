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

use anofox_regression::solvers::aft::{AftDistribution, AftRegressor};

use crate::expressions::penalized_glm::{parse_vcov_type, prior_spec_from_parts, PriorKwarg};

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
