//! Parametric statistical test expressions.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use anofox_statistics::{t_test, yuen_test, Alternative, TTestKind};

use crate::expressions::output_types::{generic_stats_output, stats_output_dtype};

/// Helper to parse alternative hypothesis from string
fn parse_alternative(s: &str) -> Alternative {
    match s.to_lowercase().as_str() {
        "less" => Alternative::Less,
        "greater" => Alternative::Greater,
        _ => Alternative::TwoSided,
    }
}

/// Independent samples t-test from raw data
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_ttest_ind(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;
    let y = inputs[1].f64()?;
    let alt_str = inputs[2].str()?.get(0).unwrap_or("two-sided");
    let equal_var = inputs[3].bool()?.get(0).unwrap_or(false);
    let mu = inputs[4].f64()?.get(0).unwrap_or(0.0);
    let conf_level = inputs[5].f64()?.get(0).unwrap_or(0.95);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();

    let alternative = parse_alternative(alt_str);
    let kind = if equal_var {
        TTestKind::Student
    } else {
        TTestKind::Welch
    };

    match t_test(&x_vec, &y_vec, kind, alternative, mu, Some(conf_level)) {
        Ok(result) => generic_stats_output(result.statistic, result.p_value, "ttest_ind"),
        Err(_) => generic_stats_output(f64::NAN, f64::NAN, "ttest_ind"),
    }
}

/// Paired samples t-test
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_ttest_paired(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;
    let y = inputs[1].f64()?;
    let alt_str = inputs[2].str()?.get(0).unwrap_or("two-sided");
    let mu = inputs[3].f64()?.get(0).unwrap_or(0.0);
    let conf_level = inputs[4].f64()?.get(0).unwrap_or(0.95);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();

    let alternative = parse_alternative(alt_str);

    match t_test(
        &x_vec,
        &y_vec,
        TTestKind::Paired,
        alternative,
        mu,
        Some(conf_level),
    ) {
        Ok(result) => generic_stats_output(result.statistic, result.p_value, "ttest_paired"),
        Err(_) => generic_stats_output(f64::NAN, f64::NAN, "ttest_paired"),
    }
}

/// Brown-Forsythe test for equal variances
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_brown_forsythe(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;
    let y = inputs[1].f64()?;

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();

    // brown_forsythe takes a slice of slices
    let groups: [&[f64]; 2] = [x_vec.as_slice(), y_vec.as_slice()];

    match anofox_statistics::brown_forsythe(&groups) {
        Ok(result) => generic_stats_output(result.statistic, result.p_value, "brown_forsythe"),
        Err(_) => generic_stats_output(f64::NAN, f64::NAN, "brown_forsythe"),
    }
}

/// Yuen's test for trimmed means (robust alternative to t-test)
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_yuen_test(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;
    let y = inputs[1].f64()?;
    let trim = inputs[2].f64()?.get(0).unwrap_or(0.2);
    let alt_str = inputs[3].str()?.get(0).unwrap_or("two-sided");
    let conf_level = inputs[4].f64()?.get(0).unwrap_or(0.95);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();

    let alternative = parse_alternative(alt_str);

    match yuen_test(&x_vec, &y_vec, trim, alternative, Some(conf_level)) {
        Ok(result) => generic_stats_output(result.statistic, result.p_value, "yuen_test"),
        Err(_) => generic_stats_output(f64::NAN, f64::NAN, "yuen_test"),
    }
}
