//! Modern statistical test expressions (Energy Distance, MMD).

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use anofox_statistics::{energy_distance_test, energy_distance_test_1d, mmd_test_1d};

use crate::expressions::output_types::{generic_stats_output, stats_output_dtype};

/// Public Rust-callable variant. Same input contract as the `pl_energy_distance` expression shim.
pub fn energy_distance_fit(inputs: &[Series]) -> PolarsResult<Series> {
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

/// Energy Distance test for comparing distributions.
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_energy_distance(inputs: &[Series]) -> PolarsResult<Series> {
    energy_distance_fit(inputs)
}

/// Public Rust-callable variant. Same input contract as the `pl_mmd_test` expression shim.
pub fn mmd_test_fit(inputs: &[Series]) -> PolarsResult<Series> {
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

/// Maximum Mean Discrepancy (MMD) test for comparing distributions.
/// Uses Gaussian kernel with median heuristic bandwidth.
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_mmd_test(inputs: &[Series]) -> PolarsResult<Series> {
    mmd_test_fit(inputs)
}

/// Multi-dimensional Energy Distance test for comparing multivariate distributions.
///
/// Input contract:
/// - inputs[0]: UInt32 scalar — d (number of feature dimensions)
/// - inputs[1]: UInt32 scalar — n_permutations
/// - inputs[2]: UInt64 scalar, nullable — seed
/// - inputs[3..3+d]: f64 Series — X sample, one Series per dimension
/// - inputs[3+d..3+2d]: f64 Series — Y sample, one Series per dimension
///
/// Each observation in X and Y is represented as a row across the d dimension columns.
pub fn energy_distance_nd_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let d = inputs[0].u32()?.get(0).unwrap_or(1) as usize;
    let n_perm = inputs[1].u32()?.get(0).unwrap_or(999) as usize;
    let seed = inputs[2].u64()?.get(0);

    // Collect X feature columns: inputs[3..3+d], each column is dimension j
    let mut x_cols: Vec<Vec<f64>> = Vec::with_capacity(d);
    for j in 0..d {
        if let Some(col_series) = inputs.get(3 + j) {
            let col: Vec<f64> = col_series.f64()?.into_no_null_iter().collect();
            x_cols.push(col);
        }
    }

    // Collect Y feature columns: inputs[3+d..3+2d]
    let mut y_cols: Vec<Vec<f64>> = Vec::with_capacity(d);
    for j in 0..d {
        if let Some(col_series) = inputs.get(3 + d + j) {
            let col: Vec<f64> = col_series.f64()?.into_no_null_iter().collect();
            y_cols.push(col);
        }
    }

    // Transpose column-major (dim × obs) to row-major (obs × dim) for the crate API
    let n_x = x_cols.first().map(|c| c.len()).unwrap_or(0);
    let x: Vec<Vec<f64>> = (0..n_x)
        .map(|obs| (0..x_cols.len()).map(|dim| x_cols[dim][obs]).collect())
        .collect();

    let n_y = y_cols.first().map(|c| c.len()).unwrap_or(0);
    let y: Vec<Vec<f64>> = (0..n_y)
        .map(|obs| (0..y_cols.len()).map(|dim| y_cols[dim][obs]).collect())
        .collect();

    match energy_distance_test(&x, &y, n_perm, seed) {
        Ok(result) => generic_stats_output(result.statistic, result.p_value, "energy_distance_nd"),
        Err(_) => generic_stats_output(f64::NAN, f64::NAN, "energy_distance_nd"),
    }
}

/// Multi-dimensional Energy Distance test expression shim.
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_energy_distance_nd(inputs: &[Series]) -> PolarsResult<Series> {
    energy_distance_nd_fit(inputs)
}
