//! Family-generic penalized GLM expression (`ps.penalized_glm`).
//!
//! Wraps `anofox_regression::solvers::penalized_glm` — one IRLS engine shared
//! by Poisson, Binomial, Logistic, Negative Binomial, Gamma and Tweedie, with
//! explicit per-coefficient priors (flat / Gaussian / Laplace), a selectable
//! covariance at the mode, an offset and Poisson quasi-dispersion.
//!
//! Every scalar configuration value (family, link, theta, power, threshold,
//! lambda, estimate_dispersion, priors, vcov_type, with_intercept, max_iter,
//! tol, conf_level, prefix, predict_type, null_policy) travels in one named
//! serde kwargs struct, [`PenalizedGlmKwargs`], so no option can be silently
//! dropped or index-mismatched (T-16.2-03). `args` carry only per-row data
//! `[y, offset, *x]`.
//!
//! The parsing helpers ([`parse_penalized_glm_family`], [`prior_spec_from_parts`],
//! [`parse_vcov_type`]) and [`build_penalized_glm`] are also used by the
//! `PenalizedGLM` PyO3 class (`src/pymodels/py_penalized_glm.rs`) so both
//! surfaces thread options identically.

use faer::{Col, Mat};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use anofox_regression::core::{BinomialLink, PoissonLink};
use anofox_regression::solvers::penalized_glm::{
    FittedPenalizedGlm, PenalizedGlmFamily, PenalizedGlmRegressor, PriorSpec, VcovType,
};

use crate::expressions::regression::build_term_names;

/// Default prior scale (`+inf`, i.e. flat) used when a [`PriorKwarg`] omits
/// `scale`.
fn default_prior_scale() -> f64 {
    f64::INFINITY
}

/// One explicit per-coefficient prior, as plain deserializable data (no new
/// Python class — see CONTEXT "Priors"). `loc` / `scale` are ignored for
/// `kind = "flat"`.
#[derive(Deserialize, Clone, Debug)]
pub struct PriorKwarg {
    /// `"flat"`, `"normal"`, or `"laplace"` (case-insensitive).
    pub kind: String,
    /// Prior location (mean for normal, location for laplace).
    #[serde(default)]
    pub loc: f64,
    /// Prior scale (std-dev for normal, scale `b` for laplace); `+inf` is a
    /// flat prior.
    #[serde(default = "default_prior_scale")]
    pub scale: f64,
}

/// Named kwargs for the penalized GLM fit expression (and, additively, the
/// Plan 04 summary/predict expressions, which deserialize the identical
/// shape). Unknown-at-fit-time fields (`conf_level`, `vcov_type`, `prefix`,
/// `predict_type`, `null_policy`) are accepted here so one kwargs struct
/// covers the whole family of expressions; [`penalized_glm_fit`] itself reads
/// only the fields it needs.
#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct PenalizedGlmKwargs {
    /// Response family: `"poisson"` (default), `"binomial"`, `"logistic"`,
    /// `"negative_binomial"`, `"gamma"`, or `"tweedie"`.
    pub family: String,
    /// Link function; `None` uses the family default.
    pub link: Option<String>,
    /// Fixed Negative Binomial `theta`; `None` estimates it by ML
    /// (`MASS::glm.nb`).
    pub theta: Option<f64>,
    /// Tweedie variance power; `None` defaults to 1.5.
    pub power: Option<f64>,
    /// Logistic classification threshold, in `(0, 1)`.
    pub threshold: f64,
    /// Uniform ridge `lambda` on non-intercept coefficients.
    pub lambda: f64,
    /// Poisson only: scale the covariance by the Pearson dispersion (R
    /// `quasipoisson`). Rejected (`Err`) for every other family.
    pub estimate_dispersion: bool,
    /// Per-coefficient priors, optionally with a leading intercept entry.
    pub priors: Vec<PriorKwarg>,
    /// Covariance type at the mode: `"laplace"` (default), `"sandwich"`, or
    /// `"naive"`.
    pub vcov_type: String,
    /// Whether to fit an intercept.
    pub with_intercept: bool,
    /// Maximum IRLS iterations.
    pub max_iter: u32,
    /// Convergence tolerance.
    pub tol: f64,
    /// Confidence level for intervals. Unused by [`penalized_glm_fit`] (no
    /// inference is computed there); read by the Plan 04 summary expression.
    pub conf_level: f64,
    /// Output column prefix. Unused by [`penalized_glm_fit`]; read by the
    /// Plan 04 predict expression.
    pub prefix: String,
    /// Predicted-scale selector (`"response"` / `"link"`). Unused by
    /// [`penalized_glm_fit`]; read by the Plan 04 predict expression.
    pub predict_type: String,
    /// Null-row handling. Unused by [`penalized_glm_fit`] (the engine always
    /// drops non-finite rows); read by the Plan 04 predict expression.
    pub null_policy: String,
}

impl Default for PenalizedGlmKwargs {
    fn default() -> Self {
        Self {
            family: "poisson".to_string(),
            link: None,
            theta: None,
            power: None,
            threshold: 0.5,
            lambda: 0.0,
            estimate_dispersion: false,
            priors: Vec::new(),
            vcov_type: "laplace".to_string(),
            with_intercept: true,
            max_iter: 100,
            tol: 1e-8,
            conf_level: 0.95,
            prefix: "penalized_glm".to_string(),
            predict_type: "response".to_string(),
            null_policy: "drop".to_string(),
        }
    }
}

/// Build a [`PriorSpec`] from its parts. `kind` is case-insensitive
/// (`"flat"` | `"normal"` | `"laplace"`); `normal`/`laplace` require a finite
/// `loc` and a `scale` that is either `> 0` or `+inf` (upstream treats
/// `+inf` as flat).
pub fn prior_spec_from_parts(kind: &str, loc: f64, scale: f64) -> Result<PriorSpec, String> {
    match kind.to_lowercase().as_str() {
        "flat" => Ok(PriorSpec::flat()),
        "normal" => {
            if !loc.is_finite() {
                return Err(format!(
                    "prior loc must be finite for kind=\"normal\" (got {loc})"
                ));
            }
            if scale.is_nan() || scale <= 0.0 {
                return Err(format!(
                    "prior scale must be > 0 (or +inf for a flat prior) for kind=\"normal\" \
                     (got {scale})"
                ));
            }
            Ok(PriorSpec::normal(loc, scale))
        }
        "laplace" => {
            if !loc.is_finite() {
                return Err(format!(
                    "prior loc must be finite for kind=\"laplace\" (got {loc})"
                ));
            }
            if scale.is_nan() || scale <= 0.0 {
                return Err(format!(
                    "prior scale must be > 0 (or +inf for a flat prior) for kind=\"laplace\" \
                     (got {scale})"
                ));
            }
            Ok(PriorSpec::laplace(loc, scale))
        }
        other => Err(format!(
            "unknown prior kind \"{other}\" (expected one of \"flat\", \"normal\", \"laplace\")"
        )),
    }
}

/// Parse a covariance-type string (case-insensitive).
pub fn parse_vcov_type(s: &str) -> Result<VcovType, String> {
    match s.to_lowercase().as_str() {
        "laplace" => Ok(VcovType::Laplace),
        "sandwich" => Ok(VcovType::Sandwich),
        "naive" => Ok(VcovType::Naive),
        other => Err(format!(
            "unknown vcov_type \"{other}\" (expected one of \"laplace\", \"sandwich\", \"naive\")"
        )),
    }
}

/// Parse the family/link/theta/power/threshold combination into a
/// [`PenalizedGlmFamily`], rejecting every combination that upstream would
/// otherwise ignore silently (`theta` for a non-NegBin family, `power` for a
/// non-Tweedie family, a link invalid for the family, or `estimate_dispersion`
/// for a non-Poisson family — CONTEXT "Claude's discretion").
pub fn parse_penalized_glm_family(
    family: &str,
    link: Option<&str>,
    theta: Option<f64>,
    power: Option<f64>,
    threshold: f64,
    estimate_dispersion: bool,
) -> Result<PenalizedGlmFamily, String> {
    let family_lc = family.to_lowercase();

    if estimate_dispersion && family_lc != "poisson" {
        return Err(format!(
            "estimate_dispersion=True is only supported for family=\"poisson\" (got \"{family}\")"
        ));
    }

    let reject_theta = |theta: Option<f64>| -> Result<(), String> {
        if theta.is_some() {
            Err("theta is only valid for family=\"negative_binomial\"".to_string())
        } else {
            Ok(())
        }
    };
    let reject_power = |power: Option<f64>| -> Result<(), String> {
        if power.is_some() {
            Err("power is only valid for family=\"tweedie\"".to_string())
        } else {
            Ok(())
        }
    };

    match family_lc.as_str() {
        "poisson" => {
            reject_theta(theta)?;
            reject_power(power)?;
            let link = match link.map(|l| l.to_lowercase()) {
                None => PoissonLink::Log,
                Some(ref l) if l == "log" => PoissonLink::Log,
                Some(ref l) if l == "identity" => PoissonLink::Identity,
                Some(ref l) if l == "sqrt" => PoissonLink::Sqrt,
                Some(l) => {
                    return Err(format!(
                        "invalid link \"{l}\" for family=\"poisson\" \
                         (expected one of \"log\", \"identity\", \"sqrt\")"
                    ))
                }
            };
            Ok(PenalizedGlmFamily::Poisson(link))
        }
        "binomial" => {
            reject_theta(theta)?;
            reject_power(power)?;
            let link = match link.map(|l| l.to_lowercase()) {
                None => BinomialLink::Logit,
                Some(ref l) if l == "logit" => BinomialLink::Logit,
                Some(ref l) if l == "probit" => BinomialLink::Probit,
                Some(ref l) if l == "cloglog" => BinomialLink::Cloglog,
                Some(l) => {
                    return Err(format!(
                        "invalid link \"{l}\" for family=\"binomial\" \
                         (expected one of \"logit\", \"probit\", \"cloglog\")"
                    ))
                }
            };
            Ok(PenalizedGlmFamily::Binomial(link))
        }
        "logistic" => {
            reject_theta(theta)?;
            reject_power(power)?;
            if let Some(l) = link {
                if l.to_lowercase() != "logit" {
                    return Err(format!(
                        "invalid link \"{l}\" for family=\"logistic\" (only \"logit\" is supported)"
                    ));
                }
            }
            if !(threshold > 0.0 && threshold < 1.0) {
                return Err(format!("threshold must be in (0, 1) (got {threshold})"));
            }
            Ok(PenalizedGlmFamily::Logistic { threshold })
        }
        "negative_binomial" | "negativebinomial" | "negbin" => {
            reject_power(power)?;
            if let Some(l) = link {
                if l.to_lowercase() != "log" {
                    return Err(format!(
                        "invalid link \"{l}\" for family=\"negative_binomial\" \
                         (only \"log\" is supported)"
                    ));
                }
            }
            if let Some(t) = theta {
                if !(t.is_finite() && t > 0.0) {
                    return Err(format!("theta must be finite and > 0 (got {t})"));
                }
            }
            Ok(PenalizedGlmFamily::NegativeBinomial { theta })
        }
        "gamma" => {
            reject_theta(theta)?;
            reject_power(power)?;
            if let Some(l) = link {
                if l.to_lowercase() != "log" {
                    return Err(format!(
                        "invalid link \"{l}\" for family=\"gamma\" (only \"log\" is supported)"
                    ));
                }
            }
            Ok(PenalizedGlmFamily::Gamma)
        }
        "tweedie" => {
            reject_theta(theta)?;
            if let Some(l) = link {
                if l.to_lowercase() != "log" {
                    return Err(format!(
                        "invalid link \"{l}\" for family=\"tweedie\" (only \"log\" is supported)"
                    ));
                }
            }
            let power = power.unwrap_or(1.5);
            if !(1.0..=2.0).contains(&power) {
                return Err(format!(
                    "power must be in [1, 2] for family=\"tweedie\" (got {power})"
                ));
            }
            Ok(PenalizedGlmFamily::Tweedie { power })
        }
        other => Err(format!(
            "unknown family \"{other}\" (expected one of \"poisson\", \"binomial\", \"logistic\", \
             \"negative_binomial\", \"gamma\", \"tweedie\")"
        )),
    }
}

/// Lower-case `(family, link)` name pair for a fitted [`PenalizedGlmFamily`],
/// matching [`parse_penalized_glm_family`]'s vocabulary (`Logistic` reports
/// family `"logistic"`, link `"logit"`).
pub fn penalized_glm_family_names(family: &PenalizedGlmFamily) -> (&'static str, &'static str) {
    match *family {
        PenalizedGlmFamily::Poisson(link) => ("poisson", link.as_str()),
        PenalizedGlmFamily::Binomial(link) => ("binomial", link.as_str()),
        PenalizedGlmFamily::Logistic { .. } => ("logistic", "logit"),
        PenalizedGlmFamily::NegativeBinomial { .. } => ("negative_binomial", "log"),
        PenalizedGlmFamily::Gamma => ("gamma", "log"),
        PenalizedGlmFamily::Tweedie { .. } => ("tweedie", "log"),
    }
}

/// Scale applied to upstream's unscaled `vcov()` so its diagonal square roots
/// equal the reported standard errors (GOTCHA, CONTEXT retarget): `1.0` for
/// Negative Binomial (whose `dispersion()` is `theta`, not a covariance
/// scale), else `fitted.dispersion()`. Used only by the `PenalizedGLM` PyO3
/// class's `vcov` getter.
pub fn penalized_glm_covariance_scale(fitted: &FittedPenalizedGlm) -> f64 {
    match fitted.family() {
        PenalizedGlmFamily::NegativeBinomial { .. } => 1.0,
        _ => fitted.dispersion(),
    }
}

/// Build a [`PenalizedGlmRegressor`] from a [`PenalizedGlmKwargs`], an
/// explicit `compute_inference` flag and an optional pre-built offset column.
pub fn build_penalized_glm(
    kwargs: &PenalizedGlmKwargs,
    compute_inference: bool,
    offset: Option<Col<f64>>,
) -> Result<PenalizedGlmRegressor, String> {
    let family = parse_penalized_glm_family(
        &kwargs.family,
        kwargs.link.as_deref(),
        kwargs.theta,
        kwargs.power,
        kwargs.threshold,
        kwargs.estimate_dispersion,
    )?;
    let vcov_type = parse_vcov_type(&kwargs.vcov_type)?;
    let mut priors = Vec::with_capacity(kwargs.priors.len());
    for p in &kwargs.priors {
        priors.push(prior_spec_from_parts(&p.kind, p.loc, p.scale)?);
    }

    let mut builder = PenalizedGlmRegressor::builder()
        .family(family)
        .with_intercept(kwargs.with_intercept)
        .max_iterations(kwargs.max_iter)
        .tolerance(kwargs.tol)
        .compute_inference(compute_inference)
        .confidence_level(kwargs.conf_level)
        .lambda(kwargs.lambda)
        .estimate_dispersion(kwargs.estimate_dispersion)
        .vcov_type(vcov_type);

    if !priors.is_empty() {
        builder = builder.priors(priors);
    }
    if let Some(off) = offset {
        builder = builder.offset(off);
    }

    Ok(builder.build())
}

/// Build an `n`-length `Vec<f64>` from a Float64 Series, mapping null to NaN
/// (never `into_no_null_iter`, which yields undefined values in null slots).
fn series_to_vec_nan(s: &Series) -> PolarsResult<Vec<f64>> {
    let ca = s.f64()?;
    Ok((0..ca.len())
        .map(|i| ca.get(i).unwrap_or(f64::NAN))
        .collect())
}

/// Build the `n x p` design matrix from `inputs[x_start..]`, null -> NaN.
/// Upstream drops any row with a non-finite value in `y`, `x` or the offset,
/// so `n_observations` on the fitted result counts the rows actually used.
fn build_penalized_glm_x(inputs: &[Series], x_start: usize, n: usize) -> PolarsResult<Mat<f64>> {
    let p = inputs.len() - x_start;
    let cols: Vec<Vec<f64>> = (x_start..inputs.len())
        .map(|i| series_to_vec_nan(&inputs[i]))
        .collect::<PolarsResult<_>>()?;
    Ok(Mat::from_fn(n, p, |row, col| cols[col][row]))
}

/// Resolve the offset column (`inputs[1]`): length 1 broadcasts its value to
/// every row (a null scalar becomes `0.0`); length `n` is used per row (null
/// -> NaN, the row is then dropped upstream like any other non-finite
/// value); any other length is a `ComputeError`.
fn resolve_offset(offset: &Series, n: usize) -> PolarsResult<Col<f64>> {
    let ca = offset.f64()?;
    match ca.len() {
        1 => {
            let v = ca.get(0).unwrap_or(0.0);
            Ok(Col::from_fn(n, |_| v))
        }
        len if len == n => {
            let vec = series_to_vec_nan(offset)?;
            Ok(Col::from_fn(n, |i| vec[i]))
        }
        other => Err(polars_err!(
            ComputeError: "penalized_glm: offset must have length 1 or {} (got {})", n, other
        )),
    }
}

/// Extract a `faer::Col<f64>` as a plain `Vec<f64>`.
fn col_to_vec(col: &Col<f64>) -> Vec<f64> {
    (0..col.nrows()).map(|i| col[i]).collect()
}

/// Output schema for `pl_penalized_glm`: `intercept`, `coefficients`,
/// `deviance`, `null_deviance`, `pseudo_r_squared`, `log_likelihood`, `aic`,
/// `bic`, `dispersion`, `theta`, `accuracy`, `has_separation`, `iterations`,
/// `converged`, `n_observations`.
pub fn penalized_glm_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("intercept".into(), DataType::Float64),
        Field::new(
            "coefficients".into(),
            DataType::List(Box::new(DataType::Float64)),
        ),
        Field::new("deviance".into(), DataType::Float64),
        Field::new("null_deviance".into(), DataType::Float64),
        Field::new("pseudo_r_squared".into(), DataType::Float64),
        Field::new("log_likelihood".into(), DataType::Float64),
        Field::new("aic".into(), DataType::Float64),
        Field::new("bic".into(), DataType::Float64),
        Field::new("dispersion".into(), DataType::Float64),
        Field::new("theta".into(), DataType::Float64),
        Field::new("accuracy".into(), DataType::Float64),
        Field::new("has_separation".into(), DataType::Boolean),
        Field::new("iterations".into(), DataType::UInt32),
        Field::new("converged".into(), DataType::Boolean),
        Field::new("n_observations".into(), DataType::UInt32),
    ];
    Ok(Field::new("penalized_glm".into(), DataType::Struct(fields)))
}

/// Build the `penalized_glm` output struct series.
#[allow(clippy::too_many_arguments)]
fn penalized_glm_output(
    intercept: Option<f64>,
    coefficients: &[f64],
    deviance: f64,
    null_deviance: f64,
    pseudo_r_squared: f64,
    log_likelihood: f64,
    aic: f64,
    bic: f64,
    dispersion: f64,
    theta: Option<f64>,
    accuracy: Option<f64>,
    has_separation: bool,
    iterations: u32,
    converged: bool,
    n_observations: usize,
) -> PolarsResult<Series> {
    let intercept_s = Series::new("intercept".into(), &[intercept.unwrap_or(f64::NAN)]);
    let coef_inner = Series::new("item".into(), coefficients);
    let coef_s = Series::new("coefficients".into(), &[coef_inner]);
    let dev_s = Series::new("deviance".into(), &[deviance]);
    let null_dev_s = Series::new("null_deviance".into(), &[null_deviance]);
    let pr2_s = Series::new("pseudo_r_squared".into(), &[pseudo_r_squared]);
    let ll_s = Series::new("log_likelihood".into(), &[log_likelihood]);
    let aic_s = Series::new("aic".into(), &[aic]);
    let bic_s = Series::new("bic".into(), &[bic]);
    let disp_s = Series::new("dispersion".into(), &[dispersion]);
    let theta_s = Series::new("theta".into(), &[theta.unwrap_or(f64::NAN)]);
    let acc_s = Series::new("accuracy".into(), &[accuracy.unwrap_or(f64::NAN)]);
    let sep_s = Series::new("has_separation".into(), &[has_separation]);
    let iter_s = Series::new("iterations".into(), &[iterations]);
    let conv_s = Series::new("converged".into(), &[converged]);
    let n_s = Series::new("n_observations".into(), &[n_observations as u32]);

    StructChunked::from_series(
        "penalized_glm".into(),
        1,
        [
            &intercept_s,
            &coef_s,
            &dev_s,
            &null_dev_s,
            &pr2_s,
            &ll_s,
            &aic_s,
            &bic_s,
            &disp_s,
            &theta_s,
            &acc_s,
            &sep_s,
            &iter_s,
            &conv_s,
            &n_s,
        ]
        .into_iter(),
    )
    .map(|ca| ca.into_series())
}

/// NaN output for `pl_penalized_glm` on a config or engine fit error.
fn penalized_glm_nan_output() -> PolarsResult<Series> {
    penalized_glm_output(
        None,
        &[],
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        None,
        None,
        false,
        0,
        false,
        0,
    )
}

/// Public Rust-callable variant. Same input contract as the `pl_penalized_glm`
/// expression shim: `inputs[0]` = y, `inputs[1]` = offset (length 1 or n),
/// `inputs[2..]` = x columns.
pub fn penalized_glm_fit(inputs: &[Series], kwargs: &PenalizedGlmKwargs) -> PolarsResult<Series> {
    let y_series = inputs[0].f64()?;
    let n = y_series.len();

    let offset_col = resolve_offset(&inputs[1], n)?;
    let x = build_penalized_glm_x(inputs, 2, n)?;
    let y_vec = series_to_vec_nan(&inputs[0])?;
    let y = Col::from_fn(n, |i| y_vec[i]);

    let regressor = build_penalized_glm(kwargs, false, Some(offset_col))
        .map_err(|e| polars_err!(ComputeError: "{}", e))?;

    match regressor.fit(&x, &y) {
        Ok(fitted) => penalized_glm_output(
            fitted.intercept(),
            &col_to_vec(fitted.coefficients()),
            fitted.deviance(),
            fitted.null_deviance(),
            fitted.pseudo_r_squared(),
            fitted.log_likelihood(),
            fitted.aic(),
            fitted.bic(),
            fitted.dispersion(),
            fitted.theta(),
            fitted.accuracy(),
            fitted.separation().has_separation,
            fitted.iterations(),
            fitted.converged(),
            fitted.n_observations(),
        ),
        Err(_) => penalized_glm_nan_output(),
    }
}

/// Penalized GLM fit expression (`ps.penalized_glm`).
/// inputs[0] = y, inputs[1] = offset (length 1 broadcasts, or length n),
/// inputs[2..] = x columns. kwargs: see [`PenalizedGlmKwargs`].
#[polars_expr(output_type_func=penalized_glm_output_dtype)]
fn pl_penalized_glm(inputs: &[Series], kwargs: PenalizedGlmKwargs) -> PolarsResult<Series> {
    penalized_glm_fit(inputs, &kwargs)
}

// ============================================================================
// Coefficient Summary (tidy format, with confidence intervals)
// ============================================================================

/// Output type for the CI-carrying tidy summary (`ps.penalized_glm_summary`,
/// and the Plan 05 AFT summary). Separate from the shared 5-field
/// `summary_output_dtype` in `regression.rs`, which stays untouched (CONTEXT
/// resolution 3).
pub fn inference_summary_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("term".into(), DataType::String),
        Field::new("estimate".into(), DataType::Float64),
        Field::new("std_error".into(), DataType::Float64),
        Field::new("statistic".into(), DataType::Float64),
        Field::new("p_value".into(), DataType::Float64),
        Field::new("ci_lower".into(), DataType::Float64),
        Field::new("ci_upper".into(), DataType::Float64),
    ];
    Ok(Field::new(
        "summary".into(),
        DataType::List(Box::new(DataType::Struct(fields))),
    ))
}

/// Build the 7-column tidy summary output as List[Struct] (one row per term).
#[allow(clippy::too_many_arguments)]
pub(crate) fn inference_summary_output(
    terms: Vec<String>,
    estimates: Vec<f64>,
    std_errors: Vec<f64>,
    statistics: Vec<f64>,
    p_values: Vec<f64>,
    ci_lower: Vec<f64>,
    ci_upper: Vec<f64>,
) -> PolarsResult<Series> {
    let n = terms.len();

    let term_s = Series::new("term".into(), terms);
    let estimate_s = Series::new("estimate".into(), estimates);
    let std_error_s = Series::new("std_error".into(), std_errors);
    let statistic_s = Series::new("statistic".into(), statistics);
    let p_value_s = Series::new("p_value".into(), p_values);
    let ci_lower_s = Series::new("ci_lower".into(), ci_lower);
    let ci_upper_s = Series::new("ci_upper".into(), ci_upper);

    let struct_ca = StructChunked::from_series(
        "item".into(),
        n,
        [
            &term_s,
            &estimate_s,
            &std_error_s,
            &statistic_s,
            &p_value_s,
            &ci_lower_s,
            &ci_upper_s,
        ]
        .into_iter(),
    )?;

    let struct_series = struct_ca.into_series();
    Ok(Series::new("summary".into(), &[struct_series]))
}

/// Empty summary output (config or engine error).
pub(crate) fn inference_summary_nan_output() -> PolarsResult<Series> {
    inference_summary_output(vec![], vec![], vec![], vec![], vec![], vec![], vec![])
}

/// Public Rust-callable variant. Same input contract as [`penalized_glm_fit`]
/// (`[y, offset, *x]`); always fits with `compute_inference = true` regardless
/// of the caller's intent, since this is the summary expression.
///
/// Rows are `intercept` (when `kwargs.with_intercept`) then `x1..xp`
/// ([`build_term_names`]). The intercept row's std_error/statistic/p_value/CI
/// come straight from upstream's `GlmInferenceResult::intercept_*` fields
/// (anofox-regression 0.5.20+) — never recomputed locally. Feature rows come
/// from the parallel `std_errors`/`z_values`/`p_values`/`ci_lower`/`ci_upper`
/// vectors, which upstream already pads to feature length (`NaN` for an
/// aliased or inactive-Laplace-prior column).
pub fn penalized_glm_summary_fit(
    inputs: &[Series],
    kwargs: &PenalizedGlmKwargs,
) -> PolarsResult<Series> {
    let y_series = inputs[0].f64()?;
    let n = y_series.len();
    let n_features = inputs.len() - 2;

    let offset_col = resolve_offset(&inputs[1], n)?;
    let x = build_penalized_glm_x(inputs, 2, n)?;
    let y_vec = series_to_vec_nan(&inputs[0])?;
    let y = Col::from_fn(n, |i| y_vec[i]);

    let regressor = build_penalized_glm(kwargs, true, Some(offset_col))
        .map_err(|e| polars_err!(ComputeError: "{}", e))?;

    match regressor.fit(&x, &y) {
        Ok(fitted) => {
            let with_intercept = kwargs.with_intercept;
            let terms = build_term_names(n_features, with_intercept);
            let inference = fitted.inference();

            let mut estimates = Vec::with_capacity(terms.len());
            let mut std_errors = Vec::with_capacity(terms.len());
            let mut statistics = Vec::with_capacity(terms.len());
            let mut p_values = Vec::with_capacity(terms.len());
            let mut ci_lower = Vec::with_capacity(terms.len());
            let mut ci_upper = Vec::with_capacity(terms.len());

            if with_intercept {
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

            inference_summary_output(
                terms, estimates, std_errors, statistics, p_values, ci_lower, ci_upper,
            )
        }
        Err(_) => inference_summary_nan_output(),
    }
}

/// Penalized GLM coefficient summary expression (`ps.penalized_glm_summary`).
/// Same input contract as `pl_penalized_glm`. kwargs: see [`PenalizedGlmKwargs`]
/// (`conf_level` / `vcov_type` / `estimate_dispersion` are read here).
#[polars_expr(output_type_func=inference_summary_output_dtype)]
fn pl_penalized_glm_summary(inputs: &[Series], kwargs: PenalizedGlmKwargs) -> PolarsResult<Series> {
    penalized_glm_summary_fit(inputs, &kwargs)
}
