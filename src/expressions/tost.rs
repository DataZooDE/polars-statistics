//! TOST (Two One-Sided Tests) equivalence testing expressions.

use polars::prelude::*;
#[cfg(feature = "python")]
use pyo3_polars::derive::polars_expr;

use anofox_statistics::{
    tost_bootstrap, tost_correlation, tost_prop_one, tost_prop_two, tost_t_test_one_sample,
    tost_t_test_paired, tost_t_test_two_sample, tost_wilcoxon_paired, tost_wilcoxon_two_sample,
    tost_yuen, CorrelationTostMethod, EquivalenceBounds,
};

#[cfg(feature = "python")]
use crate::expressions::output_types::tost_output_dtype;

/// Helper to create TostResult output
fn tost_output(result: &anofox_statistics::TostResult, name: &str) -> PolarsResult<Series> {
    let estimate = Series::new("estimate".into(), &[result.estimate]);
    let ci_lower = Series::new("ci_lower".into(), &[result.ci.0]);
    let ci_upper = Series::new("ci_upper".into(), &[result.ci.1]);
    let bound_lower = Series::new("bound_lower".into(), &[result.bounds.0]);
    let bound_upper = Series::new("bound_upper".into(), &[result.bounds.1]);
    let tost_p_value = Series::new("tost_p_value".into(), &[result.tost_p_value]);
    let equivalent = Series::new("equivalent".into(), &[result.equivalent]);
    let alpha = Series::new("alpha".into(), &[result.alpha]);
    let n = Series::new("n".into(), &[result.n as u32]);

    let df = StructChunked::from_series(
        name.into(),
        1,
        [
            &estimate,
            &ci_lower,
            &ci_upper,
            &bound_lower,
            &bound_upper,
            &tost_p_value,
            &equivalent,
            &alpha,
            &n,
        ]
        .into_iter(),
    )?;
    Ok(df.into_series())
}

/// Helper to create error TostResult output
fn tost_error_output(name: &str) -> PolarsResult<Series> {
    let estimate = Series::new("estimate".into(), &[f64::NAN]);
    let ci_lower = Series::new("ci_lower".into(), &[f64::NAN]);
    let ci_upper = Series::new("ci_upper".into(), &[f64::NAN]);
    let bound_lower = Series::new("bound_lower".into(), &[f64::NAN]);
    let bound_upper = Series::new("bound_upper".into(), &[f64::NAN]);
    let tost_p_value = Series::new("tost_p_value".into(), &[f64::NAN]);
    let equivalent = Series::new("equivalent".into(), &[false]);
    let alpha = Series::new("alpha".into(), &[f64::NAN]);
    let n = Series::new("n".into(), &[0u32]);

    let df = StructChunked::from_series(
        name.into(),
        1,
        [
            &estimate,
            &ci_lower,
            &ci_upper,
            &bound_lower,
            &bound_upper,
            &tost_p_value,
            &equivalent,
            &alpha,
            &n,
        ]
        .into_iter(),
    )?;
    Ok(df.into_series())
}

/// Parse bounds from string
fn parse_bounds(bounds_type: &str, delta: f64, lower: f64, upper: f64) -> EquivalenceBounds {
    match bounds_type.to_lowercase().as_str() {
        "raw" => EquivalenceBounds::Raw { lower, upper },
        "cohen_d" | "cohend" => EquivalenceBounds::CohenD { d: delta },
        _ => EquivalenceBounds::Symmetric { delta },
    }
}

/// Public Rust-callable variant. Same input contract as the `pl_tost_t_test_one_sample` expression shim.
pub fn tost_t_test_one_sample_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let x = super::inputs::as_f64(&inputs[0])?;
    let mu = super::inputs::as_f64(&inputs[1])?.get(0).unwrap_or(0.0);
    let bounds_type = inputs[2].str()?.get(0).unwrap_or("symmetric");
    let delta = super::inputs::as_f64(&inputs[3])?.get(0).unwrap_or(0.5);
    let lower = super::inputs::as_f64(&inputs[4])?.get(0).unwrap_or(-0.5);
    let upper = super::inputs::as_f64(&inputs[5])?.get(0).unwrap_or(0.5);
    let alpha = super::inputs::as_f64(&inputs[6])?.get(0).unwrap_or(0.05);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let bounds = parse_bounds(bounds_type, delta, lower, upper);

    match tost_t_test_one_sample(&x_vec, mu, &bounds, alpha) {
        Ok(result) => tost_output(&result, "tost_one_sample"),
        Err(_) => tost_error_output("tost_one_sample"),
    }
}

/// One-sample TOST equivalence test
#[cfg(feature = "python")]
#[polars_expr(output_type_func=tost_output_dtype)]
fn pl_tost_t_test_one_sample(inputs: &[Series]) -> PolarsResult<Series> {
    tost_t_test_one_sample_fit(inputs)
}

/// Public Rust-callable variant. Same input contract as the `pl_tost_t_test_two_sample` expression shim.
pub fn tost_t_test_two_sample_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let x = super::inputs::as_f64(&inputs[0])?;
    let y = super::inputs::as_f64(&inputs[1])?;
    let bounds_type = inputs[2].str()?.get(0).unwrap_or("symmetric");
    let delta = super::inputs::as_f64(&inputs[3])?.get(0).unwrap_or(0.5);
    let lower = super::inputs::as_f64(&inputs[4])?.get(0).unwrap_or(-0.5);
    let upper = super::inputs::as_f64(&inputs[5])?.get(0).unwrap_or(0.5);
    let alpha = super::inputs::as_f64(&inputs[6])?.get(0).unwrap_or(0.05);
    let pooled = inputs[7].bool()?.get(0).unwrap_or(false);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();
    let bounds = parse_bounds(bounds_type, delta, lower, upper);

    match tost_t_test_two_sample(&x_vec, &y_vec, &bounds, alpha, pooled) {
        Ok(result) => tost_output(&result, "tost_two_sample"),
        Err(_) => tost_error_output("tost_two_sample"),
    }
}

/// Two-sample TOST equivalence test
#[cfg(feature = "python")]
#[polars_expr(output_type_func=tost_output_dtype)]
fn pl_tost_t_test_two_sample(inputs: &[Series]) -> PolarsResult<Series> {
    tost_t_test_two_sample_fit(inputs)
}

/// Public Rust-callable variant. Same input contract as the `pl_tost_t_test_paired` expression shim.
pub fn tost_t_test_paired_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let x = super::inputs::as_f64(&inputs[0])?;
    let y = super::inputs::as_f64(&inputs[1])?;
    let bounds_type = inputs[2].str()?.get(0).unwrap_or("symmetric");
    let delta = super::inputs::as_f64(&inputs[3])?.get(0).unwrap_or(0.5);
    let lower = super::inputs::as_f64(&inputs[4])?.get(0).unwrap_or(-0.5);
    let upper = super::inputs::as_f64(&inputs[5])?.get(0).unwrap_or(0.5);
    let alpha = super::inputs::as_f64(&inputs[6])?.get(0).unwrap_or(0.05);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();
    let bounds = parse_bounds(bounds_type, delta, lower, upper);

    match tost_t_test_paired(&x_vec, &y_vec, &bounds, alpha) {
        Ok(result) => tost_output(&result, "tost_paired"),
        Err(_) => tost_error_output("tost_paired"),
    }
}

/// Paired-samples TOST equivalence test
#[cfg(feature = "python")]
#[polars_expr(output_type_func=tost_output_dtype)]
fn pl_tost_t_test_paired(inputs: &[Series]) -> PolarsResult<Series> {
    tost_t_test_paired_fit(inputs)
}

/// Public Rust-callable variant. Same input contract as the `pl_tost_correlation` expression shim.
pub fn tost_correlation_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let x = super::inputs::as_f64(&inputs[0])?;
    let y = super::inputs::as_f64(&inputs[1])?;
    let method_str = inputs[2].str()?.get(0).unwrap_or("pearson");
    let rho_null = super::inputs::as_f64(&inputs[3])?.get(0).unwrap_or(0.0);
    let bounds_type = inputs[4].str()?.get(0).unwrap_or("symmetric");
    let delta = super::inputs::as_f64(&inputs[5])?.get(0).unwrap_or(0.3);
    let lower = super::inputs::as_f64(&inputs[6])?.get(0).unwrap_or(-0.3);
    let upper = super::inputs::as_f64(&inputs[7])?.get(0).unwrap_or(0.3);
    let alpha = super::inputs::as_f64(&inputs[8])?.get(0).unwrap_or(0.05);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();

    let method = match method_str.to_lowercase().as_str() {
        "spearman" => CorrelationTostMethod::Spearman,
        _ => CorrelationTostMethod::Pearson,
    };

    let bounds = parse_bounds(bounds_type, delta, lower, upper);

    match tost_correlation(&x_vec, &y_vec, rho_null, &bounds, alpha, method) {
        Ok(result) => tost_output(&result, "tost_correlation"),
        Err(_) => tost_error_output("tost_correlation"),
    }
}

/// Correlation TOST equivalence test
#[cfg(feature = "python")]
#[polars_expr(output_type_func=tost_output_dtype)]
fn pl_tost_correlation(inputs: &[Series]) -> PolarsResult<Series> {
    tost_correlation_fit(inputs)
}

/// Public Rust-callable variant. Same input contract as the `pl_tost_prop_one` expression shim.
pub fn tost_prop_one_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let successes = super::inputs::as_u32(&inputs[0])?.get(0).unwrap_or(0) as usize;
    let n = super::inputs::as_u32(&inputs[1])?.get(0).unwrap_or(1) as usize;
    let p0 = super::inputs::as_f64(&inputs[2])?.get(0).unwrap_or(0.5);
    let bounds_type = inputs[3].str()?.get(0).unwrap_or("symmetric");
    let delta = super::inputs::as_f64(&inputs[4])?.get(0).unwrap_or(0.1);
    let lower = super::inputs::as_f64(&inputs[5])?.get(0).unwrap_or(-0.1);
    let upper = super::inputs::as_f64(&inputs[6])?.get(0).unwrap_or(0.1);
    let alpha = super::inputs::as_f64(&inputs[7])?.get(0).unwrap_or(0.05);

    let bounds = parse_bounds(bounds_type, delta, lower, upper);

    match tost_prop_one(successes, n, p0, &bounds, alpha) {
        Ok(result) => tost_output(&result, "tost_prop_one"),
        Err(_) => tost_error_output("tost_prop_one"),
    }
}

/// One-proportion TOST equivalence test
#[cfg(feature = "python")]
#[polars_expr(output_type_func=tost_output_dtype)]
fn pl_tost_prop_one(inputs: &[Series]) -> PolarsResult<Series> {
    tost_prop_one_fit(inputs)
}

/// Public Rust-callable variant. Same input contract as the `pl_tost_prop_two` expression shim.
pub fn tost_prop_two_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let successes1 = super::inputs::as_u32(&inputs[0])?.get(0).unwrap_or(0) as usize;
    let n1 = super::inputs::as_u32(&inputs[1])?.get(0).unwrap_or(1) as usize;
    let successes2 = super::inputs::as_u32(&inputs[2])?.get(0).unwrap_or(0) as usize;
    let n2 = super::inputs::as_u32(&inputs[3])?.get(0).unwrap_or(1) as usize;
    let bounds_type = inputs[4].str()?.get(0).unwrap_or("symmetric");
    let delta = super::inputs::as_f64(&inputs[5])?.get(0).unwrap_or(0.1);
    let lower = super::inputs::as_f64(&inputs[6])?.get(0).unwrap_or(-0.1);
    let upper = super::inputs::as_f64(&inputs[7])?.get(0).unwrap_or(0.1);
    let alpha = super::inputs::as_f64(&inputs[8])?.get(0).unwrap_or(0.05);

    let bounds = parse_bounds(bounds_type, delta, lower, upper);

    match tost_prop_two(successes1, n1, successes2, n2, &bounds, alpha) {
        Ok(result) => tost_output(&result, "tost_prop_two"),
        Err(_) => tost_error_output("tost_prop_two"),
    }
}

/// Two-proportion TOST equivalence test
#[cfg(feature = "python")]
#[polars_expr(output_type_func=tost_output_dtype)]
fn pl_tost_prop_two(inputs: &[Series]) -> PolarsResult<Series> {
    tost_prop_two_fit(inputs)
}

/// Public Rust-callable variant. Same input contract as the `pl_tost_wilcoxon_paired` expression shim.
pub fn tost_wilcoxon_paired_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let x = super::inputs::as_f64(&inputs[0])?;
    let y = super::inputs::as_f64(&inputs[1])?;
    let bounds_type = inputs[2].str()?.get(0).unwrap_or("symmetric");
    let delta = super::inputs::as_f64(&inputs[3])?.get(0).unwrap_or(0.5);
    let lower = super::inputs::as_f64(&inputs[4])?.get(0).unwrap_or(-0.5);
    let upper = super::inputs::as_f64(&inputs[5])?.get(0).unwrap_or(0.5);
    let alpha = super::inputs::as_f64(&inputs[6])?.get(0).unwrap_or(0.05);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();
    let bounds = parse_bounds(bounds_type, delta, lower, upper);

    match tost_wilcoxon_paired(&x_vec, &y_vec, &bounds, alpha) {
        Ok(result) => tost_output(&result, "tost_wilcoxon_paired"),
        Err(_) => tost_error_output("tost_wilcoxon_paired"),
    }
}

/// Wilcoxon paired-samples TOST equivalence test
#[cfg(feature = "python")]
#[polars_expr(output_type_func=tost_output_dtype)]
fn pl_tost_wilcoxon_paired(inputs: &[Series]) -> PolarsResult<Series> {
    tost_wilcoxon_paired_fit(inputs)
}

/// Public Rust-callable variant. Same input contract as the `pl_tost_wilcoxon_two_sample` expression shim.
pub fn tost_wilcoxon_two_sample_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let x = super::inputs::as_f64(&inputs[0])?;
    let y = super::inputs::as_f64(&inputs[1])?;
    let bounds_type = inputs[2].str()?.get(0).unwrap_or("symmetric");
    let delta = super::inputs::as_f64(&inputs[3])?.get(0).unwrap_or(0.5);
    let lower = super::inputs::as_f64(&inputs[4])?.get(0).unwrap_or(-0.5);
    let upper = super::inputs::as_f64(&inputs[5])?.get(0).unwrap_or(0.5);
    let alpha = super::inputs::as_f64(&inputs[6])?.get(0).unwrap_or(0.05);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();
    let bounds = parse_bounds(bounds_type, delta, lower, upper);

    match tost_wilcoxon_two_sample(&x_vec, &y_vec, &bounds, alpha) {
        Ok(result) => tost_output(&result, "tost_wilcoxon_two_sample"),
        Err(_) => tost_error_output("tost_wilcoxon_two_sample"),
    }
}

/// Wilcoxon two-sample TOST equivalence test
#[cfg(feature = "python")]
#[polars_expr(output_type_func=tost_output_dtype)]
fn pl_tost_wilcoxon_two_sample(inputs: &[Series]) -> PolarsResult<Series> {
    tost_wilcoxon_two_sample_fit(inputs)
}

/// Public Rust-callable variant. Same input contract as the `pl_tost_bootstrap` expression shim.
pub fn tost_bootstrap_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let x = super::inputs::as_f64(&inputs[0])?;
    let y = super::inputs::as_f64(&inputs[1])?;
    let bounds_type = inputs[2].str()?.get(0).unwrap_or("symmetric");
    let delta = super::inputs::as_f64(&inputs[3])?.get(0).unwrap_or(0.5);
    let lower = super::inputs::as_f64(&inputs[4])?.get(0).unwrap_or(-0.5);
    let upper = super::inputs::as_f64(&inputs[5])?.get(0).unwrap_or(0.5);
    let alpha = super::inputs::as_f64(&inputs[6])?.get(0).unwrap_or(0.05);
    let n_bootstrap = super::inputs::as_u32(&inputs[7])?.get(0).unwrap_or(1000) as usize;
    let seed = inputs[8].u64()?.get(0);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();
    let bounds = parse_bounds(bounds_type, delta, lower, upper);

    match tost_bootstrap(&x_vec, &y_vec, &bounds, alpha, n_bootstrap, seed) {
        Ok(result) => tost_output(&result, "tost_bootstrap"),
        Err(_) => tost_error_output("tost_bootstrap"),
    }
}

/// Bootstrap TOST equivalence test
#[cfg(feature = "python")]
#[polars_expr(output_type_func=tost_output_dtype)]
fn pl_tost_bootstrap(inputs: &[Series]) -> PolarsResult<Series> {
    tost_bootstrap_fit(inputs)
}

/// Public Rust-callable variant. Same input contract as the `pl_tost_yuen` expression shim.
pub fn tost_yuen_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let x = super::inputs::as_f64(&inputs[0])?;
    let y = super::inputs::as_f64(&inputs[1])?;
    let trim = super::inputs::as_f64(&inputs[2])?.get(0).unwrap_or(0.2);
    let bounds_type = inputs[3].str()?.get(0).unwrap_or("symmetric");
    let delta = super::inputs::as_f64(&inputs[4])?.get(0).unwrap_or(0.5);
    let lower = super::inputs::as_f64(&inputs[5])?.get(0).unwrap_or(-0.5);
    let upper = super::inputs::as_f64(&inputs[6])?.get(0).unwrap_or(0.5);
    let alpha = super::inputs::as_f64(&inputs[7])?.get(0).unwrap_or(0.05);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();
    let bounds = parse_bounds(bounds_type, delta, lower, upper);

    match tost_yuen(&x_vec, &y_vec, &bounds, alpha, trim) {
        Ok(result) => tost_output(&result, "tost_yuen"),
        Err(_) => tost_error_output("tost_yuen"),
    }
}

/// Yuen TOST equivalence test (trimmed means)
#[cfg(feature = "python")]
#[polars_expr(output_type_func=tost_output_dtype)]
fn pl_tost_yuen(inputs: &[Series]) -> PolarsResult<Series> {
    tost_yuen_fit(inputs)
}
