//! Modern statistical test expressions (Energy Distance, MMD).

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use anofox_statistics::{energy_distance_test_1d, mmd_test_1d};

use crate::expressions::output_types::{generic_stats_output, stats_output_dtype};

/// Energy Distance test for comparing distributions.
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_energy_distance(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;
    let y = inputs[1].f64()?;
    let n_perm = inputs[2].u32()?.get(0).unwrap_or(999) as usize;
    let seed = inputs[3].u64()?.get(0);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();

    match energy_distance_test_1d(&x_vec, &y_vec, n_perm, seed) {
        Ok(result) => generic_stats_output(result.statistic, result.p_value, "energy_distance"),
        Err(_) => generic_stats_output(f64::NAN, f64::NAN, "energy_distance"),
    }
}

/// Maximum Mean Discrepancy (MMD) test for comparing distributions.
/// Uses Gaussian kernel with median heuristic bandwidth.
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_mmd_test(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;
    let y = inputs[1].f64()?;
    let n_perm = inputs[2].u32()?.get(0).unwrap_or(999) as usize;
    let seed = inputs[3].u64()?.get(0);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();

    match mmd_test_1d(&x_vec, &y_vec, n_perm, seed) {
        Ok(result) => generic_stats_output(result.statistic, result.p_value, "mmd_test"),
        Err(_) => generic_stats_output(f64::NAN, f64::NAN, "mmd_test"),
    }
}
