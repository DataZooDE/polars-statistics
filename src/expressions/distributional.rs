//! Distributional test expressions (normality tests, etc.).

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use anofox_statistics::{dagostino_k_squared, jarque_bera, shapiro_wilk};

use crate::expressions::output_types::{
    generic_stats_output, jarque_bera_output, jarque_bera_output_dtype, stats_output_dtype,
};

/// Public Rust-callable variant. Same input contract as the `pl_shapiro_wilk` expression shim.
pub fn shapiro_wilk_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();

    match shapiro_wilk(&x_vec) {
        Ok(result) => generic_stats_output(result.statistic, result.p_value, "shapiro_wilk"),
        Err(_) => generic_stats_output(f64::NAN, f64::NAN, "shapiro_wilk"),
    }
}

/// Shapiro-Wilk test for normality
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_shapiro_wilk(inputs: &[Series]) -> PolarsResult<Series> {
    shapiro_wilk_fit(inputs)
}

/// Public Rust-callable variant. Same input contract as the `pl_dagostino` expression shim.
pub fn dagostino_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();

    match dagostino_k_squared(&x_vec) {
        Ok(result) => generic_stats_output(result.statistic, result.p_value, "dagostino"),
        Err(_) => generic_stats_output(f64::NAN, f64::NAN, "dagostino"),
    }
}

/// D'Agostino K-squared test for normality
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_dagostino(inputs: &[Series]) -> PolarsResult<Series> {
    dagostino_fit(inputs)
}

/// Public Rust-callable variant. Same input contract as the `pl_jarque_bera` expression shim.
///
/// # Input contract
/// `inputs[0]`: x (f64 Series).
pub fn jarque_bera_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();

    match jarque_bera(&x_vec) {
        Ok(result) => jarque_bera_output(
            result.statistic,
            result.p_value,
            result.skewness,
            result.kurtosis,
            result.n as u32,
            "jarque_bera",
        ),
        Err(_) => jarque_bera_output(f64::NAN, f64::NAN, f64::NAN, f64::NAN, 0, "jarque_bera"),
    }
}

/// Jarque-Bera test for normality: asymptotic chi-squared(2) test on sample
/// skewness and excess kurtosis (population moments), matching R
/// `tseries::jarque.bera.test`. Requires at least 3 finite, non-constant values.
#[polars_expr(output_type_func=jarque_bera_output_dtype)]
fn pl_jarque_bera(inputs: &[Series]) -> PolarsResult<Series> {
    jarque_bera_fit(inputs)
}
