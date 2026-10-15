//! Empirical-Bayes shrinkage of per-group estimates toward their common mean,
//! wrapping `anofox_regression::solvers::eb_shrink` (crate version 0.5.17, issue
//! #58).
//!
//! `ps.eb_shrink(estimate, std_error)` shrinks noisy per-group estimates — for
//! example per-store effects computed with `group_by` — toward their
//! precision-weighted mean by an amount the data itself determines
//! (DerSimonian-Laird random-effects pooling, or complete pooling, or a fixed
//! `tau_squared`). `mu`, `mu_se`, `tau_squared`, `q` and the shrunken estimates
//! match `metafor::rma(yi, sei = se, method = "DL")` and `blup()` (see
//! `tests/r_validation_eb_shrink.rs` in the upstream crate).

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use anofox_regression::solvers::eb_shrink::{eb_shrink, EbShrinkError, EbShrinkOptions, TauMethod};

/// Parse an `eb_shrink` tau-method name (case-insensitive).
///
/// `"dl"` / `"dersimonian_laird"` / `"dersimonian-laird"` select the
/// DerSimonian-Laird moment estimator (the default); `"none"` selects
/// complete pooling (`tau_squared = 0`). Anything else is a `ComputeError`.
fn parse_tau_method(s: &str) -> PolarsResult<TauMethod> {
    match s.to_lowercase().as_str() {
        "dl" | "dersimonian_laird" | "dersimonian-laird" => Ok(TauMethod::DerSimonianLaird),
        "none" => Ok(TauMethod::None),
        _ => Err(
            polars_err!(ComputeError: "unknown eb_shrink method '{}' (expected 'dl' or 'none')", s),
        ),
    }
}

/// Output dtype: one struct per input row with the per-row shrunken fields
/// plus the whole-sample scalars repeated on every row.
pub fn eb_shrink_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("shrunken".into(), DataType::Float64),
        Field::new("shrunken_se".into(), DataType::Float64),
        Field::new("weight".into(), DataType::Float64),
        Field::new("mu".into(), DataType::Float64),
        Field::new("mu_se".into(), DataType::Float64),
        Field::new("tau_squared".into(), DataType::Float64),
        Field::new("i_squared".into(), DataType::Float64),
        Field::new("q".into(), DataType::Float64),
        Field::new("n_groups".into(), DataType::UInt32),
    ];
    Ok(Field::new("eb_shrink".into(), DataType::Struct(fields)))
}

/// Build the per-row struct output from raw columns, mirroring
/// `regression.rs`'s `prediction_output_with_prefix` / `posthoc.rs`'s
/// `pairwise_output`.
#[allow(clippy::too_many_arguments)]
fn eb_shrink_output(
    shrunken: Vec<f64>,
    shrunken_se: Vec<f64>,
    weight: Vec<f64>,
    mu: Vec<f64>,
    mu_se: Vec<f64>,
    tau_squared: Vec<f64>,
    i_squared: Vec<f64>,
    q: Vec<f64>,
    n_groups: Vec<u32>,
) -> PolarsResult<Series> {
    let n = shrunken.len();

    let shrunken_s = Series::new("shrunken".into(), shrunken);
    let shrunken_se_s = Series::new("shrunken_se".into(), shrunken_se);
    let weight_s = Series::new("weight".into(), weight);
    let mu_s = Series::new("mu".into(), mu);
    let mu_se_s = Series::new("mu_se".into(), mu_se);
    let tau_squared_s = Series::new("tau_squared".into(), tau_squared);
    let i_squared_s = Series::new("i_squared".into(), i_squared);
    let q_s = Series::new("q".into(), q);
    let n_groups_s = Series::new("n_groups".into(), n_groups);

    let struct_ca = StructChunked::from_series(
        "eb_shrink".into(),
        n,
        [
            &shrunken_s,
            &shrunken_se_s,
            &weight_s,
            &mu_s,
            &mu_se_s,
            &tau_squared_s,
            &i_squared_s,
            &q_s,
            &n_groups_s,
        ]
        .into_iter(),
    )?;

    Ok(struct_ca.into_series())
}

/// `n` rows of NaN (n_groups 0) — degenerate input (empty, single usable row,
/// or all rows unusable) stays aligned with the input, never an exception.
fn eb_shrink_nan_output(n: usize) -> PolarsResult<Series> {
    eb_shrink_output(
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![0u32; n],
    )
}

/// Public Rust-callable variant. Same input contract as the `pl_eb_shrink`
/// expression shim.
///
/// # Input contract
/// `inputs[0]`: estimate (f64, nullable). `inputs[1]`: std_error (f64,
/// nullable; same length as `inputs[0]`, else `ComputeError`). `inputs[2]`:
/// method (String scalar, "dl" / "none", default "dl"). `inputs[3]`:
/// tau_squared (f64 nullable scalar; `None` estimates via `method`, `Some`
/// overrides it).
///
/// Null estimate/std_error entries become NaN (upstream's "unusable row"
/// convention). A bad fixed `tau_squared` is a `ComputeError`
/// (`EbShrinkError::InvalidValue`); empty input, a single usable row, or all
/// rows unusable give `n` rows of NaN with `n_groups` 0 — never an exception.
pub fn eb_shrink_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let est_ca = inputs[0].f64()?;
    let se_ca = inputs[1].f64()?;
    if est_ca.len() != se_ca.len() {
        return Err(
            polars_err!(ComputeError: "eb_shrink: estimate and std_error must have the same length (got {} and {})", est_ca.len(), se_ca.len()),
        );
    }
    let n = est_ca.len();

    let method_str = inputs[2].str()?.get(0).unwrap_or("dl");
    let method = parse_tau_method(method_str)?;
    let tau_squared = inputs[3].f64()?.get(0);

    let estimates: Vec<f64> = est_ca.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
    let std_errors: Vec<f64> = se_ca.iter().map(|v| v.unwrap_or(f64::NAN)).collect();

    let options = EbShrinkOptions {
        method,
        tau_squared,
    };

    match eb_shrink(&estimates, &std_errors, &options) {
        Ok(r) => {
            let shrunken: Vec<f64> = r.groups.iter().map(|g| g.shrunken).collect();
            let shrunken_se: Vec<f64> = r.groups.iter().map(|g| g.shrunken_se).collect();
            let weight: Vec<f64> = r.groups.iter().map(|g| g.weight).collect();
            eb_shrink_output(
                shrunken,
                shrunken_se,
                weight,
                vec![r.mu; n],
                vec![r.mu_se; n],
                vec![r.tau_squared; n],
                vec![r.i_squared; n],
                vec![r.q; n],
                vec![r.n_groups as u32; n],
            )
        }
        Err(EbShrinkError::InvalidValue { field, message }) => {
            Err(polars_err!(ComputeError: "eb_shrink: invalid value for {}: {}", field, message))
        }
        Err(EbShrinkError::DimensionMismatch {
            estimates,
            standard_errors,
        }) => Err(
            polars_err!(ComputeError: "eb_shrink: {} estimates but {} standard errors", estimates, standard_errors),
        ),
        // EmptyInput / InsufficientData: degrade to aligned NaN rows, never an
        // exception.
        Err(_) => eb_shrink_nan_output(n),
    }
}

/// Empirical-Bayes (DerSimonian-Laird) shrinkage of per-group estimates
/// toward their precision-weighted mean, matching
/// `metafor::rma(yi, sei = se, method = "DL")` / `blup()`.
#[polars_expr(output_type_func=eb_shrink_output_dtype)]
fn pl_eb_shrink(inputs: &[Series]) -> PolarsResult<Series> {
    eb_shrink_fit(inputs)
}
