//! Forecast comparison test expressions.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use anofox_statistics::{diebold_mariano, permutation_t_test, Alternative, LossFunction};

use crate::expressions::output_types::{generic_stats_output, stats_output_dtype};

/// Helper to parse loss function from string
fn parse_loss_function(s: &str) -> LossFunction {
    match s.to_lowercase().as_str() {
        "absolute" | "ae" | "mae" => LossFunction::AbsoluteError,
        _ => LossFunction::SquaredError,
    }
}

/// Helper to parse alternative hypothesis from string
fn parse_alternative(s: &str) -> Alternative {
    match s.to_lowercase().as_str() {
        "less" => Alternative::Less,
        "greater" => Alternative::Greater,
        _ => Alternative::TwoSided,
    }
}

/// Diebold-Mariano test for comparing forecast accuracy.
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_diebold_mariano(inputs: &[Series]) -> PolarsResult<Series> {
    let e1 = inputs[0].f64()?;
    let e2 = inputs[1].f64()?;
    let loss_str = inputs[2].str()?.get(0).unwrap_or("squared");
    let h = inputs[3].u32()?.get(0).unwrap_or(1) as usize;

    let e1_vec: Vec<f64> = e1.into_no_null_iter().collect();
    let e2_vec: Vec<f64> = e2.into_no_null_iter().collect();

    let loss = parse_loss_function(loss_str);

    match diebold_mariano(&e1_vec, &e2_vec, loss, h) {
        Ok(result) => generic_stats_output(result.statistic, result.p_value, "diebold_mariano"),
        Err(_) => generic_stats_output(f64::NAN, f64::NAN, "diebold_mariano"),
    }
}

/// Permutation t-test for comparing two samples.
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_permutation_t_test(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;
    let y = inputs[1].f64()?;
    let alt_str = inputs[2].str()?.get(0).unwrap_or("two-sided");
    let n_perm = inputs[3].u32()?.get(0).unwrap_or(999) as usize;
    let seed = inputs[4].u64()?.get(0);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();

    let alternative = parse_alternative(alt_str);

    match permutation_t_test(&x_vec, &y_vec, alternative, n_perm, seed) {
        Ok(result) => generic_stats_output(result.statistic, result.p_value, "permutation_t_test"),
        Err(_) => generic_stats_output(f64::NAN, f64::NAN, "permutation_t_test"),
    }
}
