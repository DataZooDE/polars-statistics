//! Non-parametric statistical test expressions.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use anofox_statistics::{
    brunner_munzel, kruskal_wallis, mann_whitney_u, wilcoxon_signed_rank, Alternative,
};

use crate::expressions::output_types::{generic_stats_output, stats_output_dtype};

/// Helper to parse alternative hypothesis from string
fn parse_alternative(s: &str) -> Alternative {
    match s.to_lowercase().as_str() {
        "less" => Alternative::Less,
        "greater" => Alternative::Greater,
        _ => Alternative::TwoSided,
    }
}

/// Mann-Whitney U test
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_mann_whitney_u(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;
    let y = inputs[1].f64()?;
    let alt_str = inputs[2].str()?.get(0).unwrap_or("two-sided");
    let continuity_correction = inputs[3].bool()?.get(0).unwrap_or(true);
    let exact = inputs[4].bool()?.get(0).unwrap_or(false);
    let conf_level = inputs[5].f64()?.get(0);
    let mu = inputs[6].f64()?.get(0);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();

    let alternative = parse_alternative(alt_str);

    match mann_whitney_u(
        &x_vec,
        &y_vec,
        alternative,
        continuity_correction,
        exact,
        conf_level,
        mu,
    ) {
        Ok(result) => generic_stats_output(result.statistic, result.p_value, "mann_whitney_u"),
        Err(_) => generic_stats_output(f64::NAN, f64::NAN, "mann_whitney_u"),
    }
}

/// Wilcoxon signed-rank test
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_wilcoxon_signed_rank(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;
    let y = inputs[1].f64()?;
    let alt_str = inputs[2].str()?.get(0).unwrap_or("two-sided");
    let continuity_correction = inputs[3].bool()?.get(0).unwrap_or(true);
    let exact = inputs[4].bool()?.get(0).unwrap_or(false);
    let conf_level = inputs[5].f64()?.get(0);
    let mu = inputs[6].f64()?.get(0);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();

    let alternative = parse_alternative(alt_str);

    match wilcoxon_signed_rank(
        &x_vec,
        &y_vec,
        alternative,
        continuity_correction,
        exact,
        conf_level,
        mu,
    ) {
        Ok(result) => {
            generic_stats_output(result.statistic, result.p_value, "wilcoxon_signed_rank")
        }
        Err(_) => generic_stats_output(f64::NAN, f64::NAN, "wilcoxon_signed_rank"),
    }
}

/// Kruskal-Wallis H test
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_kruskal_wallis(inputs: &[Series]) -> PolarsResult<Series> {
    // This takes multiple groups as separate series
    let groups: Vec<Vec<f64>> = inputs
        .iter()
        .map(|s| s.f64().map(|ca| ca.into_no_null_iter().collect()))
        .collect::<PolarsResult<Vec<_>>>()?;

    let group_refs: Vec<&[f64]> = groups.iter().map(|g| g.as_slice()).collect();

    match kruskal_wallis(&group_refs) {
        Ok(result) => generic_stats_output(result.statistic, result.p_value, "kruskal_wallis"),
        Err(_) => generic_stats_output(f64::NAN, f64::NAN, "kruskal_wallis"),
    }
}

/// Brunner-Munzel test (robust alternative to Mann-Whitney)
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_brunner_munzel(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;
    let y = inputs[1].f64()?;
    let alt_str = inputs[2].str()?.get(0).unwrap_or("two-sided");
    let alpha = inputs[3].f64()?.get(0);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();

    let alternative = parse_alternative(alt_str);

    match brunner_munzel(&x_vec, &y_vec, alternative, alpha) {
        Ok(result) => generic_stats_output(result.statistic, result.p_value, "brunner_munzel"),
        Err(_) => generic_stats_output(f64::NAN, f64::NAN, "brunner_munzel"),
    }
}
