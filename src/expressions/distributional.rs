//! Distributional test expressions (normality tests, etc.).

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use anofox_statistics::{dagostino_k_squared, shapiro_wilk};

use crate::expressions::output_types::{generic_stats_output, stats_output_dtype};

/// Shapiro-Wilk test for normality
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_shapiro_wilk(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();

    match shapiro_wilk(&x_vec) {
        Ok(result) => generic_stats_output(result.statistic, result.p_value, "shapiro_wilk"),
        Err(_) => generic_stats_output(f64::NAN, f64::NAN, "shapiro_wilk"),
    }
}

/// D'Agostino K-squared test for normality
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_dagostino(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();

    match dagostino_k_squared(&x_vec) {
        Ok(result) => generic_stats_output(result.statistic, result.p_value, "dagostino"),
        Err(_) => generic_stats_output(f64::NAN, f64::NAN, "dagostino"),
    }
}
