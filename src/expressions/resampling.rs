//! Percentile bootstrap confidence intervals wrapping `anofox_statistics::resampling`.
//!
//! `bootstrap_mean_ci` exposes the crate's IID / stationary / circular-block mean
//! bootstrap directly. `bootstrap_ci` exposes a fixed `statistic` dispatch
//! ("mean" | "median" | "std") built from native Rust closures, since the
//! upstream `bootstrap_ci<F: Fn(&[f64]) -> f64>` takes a generic Rust closure
//! that cannot cross the Python FFI boundary (see the phase's threat model and
//! CONTEXT.md for the scope-limit rationale). The "mean" arm reuses exactly the
//! same closure expression `anofox_statistics::bootstrap_mean_ci` uses
//! internally, so the two builders produce bit-identical replicates for the
//! same seed.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use anofox_statistics::{bootstrap_ci, bootstrap_mean_ci};

use crate::expressions::output_types::{bootstrap_ci_output, bootstrap_ci_output_dtype};

/// Sample median via `f64::total_cmp` sort (NaN-safe ordering); middle value for
/// odd length, mean of the two middle values for even length.
fn median_of(s: &[f64]) -> f64 {
    if s.is_empty() {
        return f64::NAN;
    }
    let mut v = s.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Sample standard deviation (denominator `n - 1`); NaN for fewer than 2 values.
fn std_of(s: &[f64]) -> f64 {
    let n = s.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = s.iter().sum::<f64>() / n as f64;
    let var = s.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    var.sqrt()
}

/// Public Rust-callable variant. Same input contract as the `pl_bootstrap_mean_ci`
/// expression shim.
///
/// # Input contract
/// `inputs[0]`: x (f64 Series). `inputs[1]`: n_bootstrap (u32, default 1000).
/// `inputs[2]`: conf_level (f64, default 0.95). `inputs[3]`: block_length (u32,
/// default 0 = IID; 1 = stationary bootstrap, expected block length 1; >1 =
/// circular block bootstrap with that block length). `inputs[4]`: seed (u64,
/// nullable).
pub fn bootstrap_mean_ci_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;
    let n_bootstrap = inputs[1].u32()?.get(0).unwrap_or(1000) as usize;
    let conf_level = inputs[2].f64()?.get(0).unwrap_or(0.95);
    let block_length = inputs[3].u32()?.get(0).unwrap_or(0) as usize;
    let seed = inputs[4].u64()?.get(0);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();

    match bootstrap_mean_ci(&x_vec, n_bootstrap, conf_level, block_length, seed) {
        Ok(r) => bootstrap_ci_output(
            r.estimate,
            r.se,
            r.conf_int_lower,
            r.conf_int_upper,
            r.conf_level,
            r.n_bootstrap as u32,
            "bootstrap_mean_ci",
        ),
        Err(_) => bootstrap_ci_output(
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            conf_level,
            n_bootstrap as u32,
            "bootstrap_mean_ci",
        ),
    }
}

/// Percentile bootstrap confidence interval for the mean, with IID / stationary /
/// circular-block resampling selected via `block_length`.
#[polars_expr(output_type_func=bootstrap_ci_output_dtype)]
fn pl_bootstrap_mean_ci(inputs: &[Series]) -> PolarsResult<Series> {
    bootstrap_mean_ci_fit(inputs)
}

/// Public Rust-callable variant. Same input contract as the `pl_bootstrap_ci`
/// expression shim.
///
/// # Input contract
/// `inputs[0]`: x (f64 Series). `inputs[1]`: statistic ("mean" | "median" |
/// "std"; any other value yields an all-NaN struct). `inputs[2]`: n_bootstrap
/// (u32, default 1000). `inputs[3]`: conf_level (f64, default 0.95).
/// `inputs[4]`: seed (u64, nullable).
pub fn bootstrap_ci_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;
    let statistic_str = inputs[1].str()?.get(0).unwrap_or("mean");
    let n_bootstrap = inputs[2].u32()?.get(0).unwrap_or(1000) as usize;
    let conf_level = inputs[3].f64()?.get(0).unwrap_or(0.95);
    let seed = inputs[4].u64()?.get(0);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();

    let result = match statistic_str {
        // Exactly upstream's own mean closure (anofox_statistics::bootstrap_mean_ci),
        // so this arm is bit-identical to bootstrap_mean_ci(block_length=0) for the
        // same seed.
        "mean" => {
            let stat = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
            bootstrap_ci(&x_vec, stat, n_bootstrap, conf_level, seed)
        }
        "median" => bootstrap_ci(&x_vec, median_of, n_bootstrap, conf_level, seed),
        "std" => bootstrap_ci(&x_vec, std_of, n_bootstrap, conf_level, seed),
        _ => {
            return bootstrap_ci_output(
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                conf_level,
                n_bootstrap as u32,
                "bootstrap_ci",
            );
        }
    };

    match result {
        Ok(r) => bootstrap_ci_output(
            r.estimate,
            r.se,
            r.conf_int_lower,
            r.conf_int_upper,
            r.conf_level,
            r.n_bootstrap as u32,
            "bootstrap_ci",
        ),
        Err(_) => bootstrap_ci_output(
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            conf_level,
            n_bootstrap as u32,
            "bootstrap_ci",
        ),
    }
}

/// Percentile bootstrap confidence interval for a fixed statistic ("mean" |
/// "median" | "std"). A deliberate scope limit (see the phase threat model):
/// upstream's generic `bootstrap_ci<F: Fn(&[f64]) -> f64>` takes a native Rust
/// closure and cannot accept an arbitrary Python callable across the FFI
/// boundary without a GIL-reentrancy hazard inside Polars' multithreaded
/// group_by.
#[polars_expr(output_type_func=bootstrap_ci_output_dtype)]
fn pl_bootstrap_ci(inputs: &[Series]) -> PolarsResult<Series> {
    bootstrap_ci_fit(inputs)
}
