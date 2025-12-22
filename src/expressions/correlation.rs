//! Correlation test expressions.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use anofox_statistics::{
    distance_cor_test, kendall, partial_cor, pearson, semi_partial_cor, spearman,
    CorrelationResult, KendallVariant,
};

use crate::expressions::output_types::correlation_output_dtype;

/// Helper to create correlation output
fn correlation_output(result: &CorrelationResult, name: &str) -> PolarsResult<Series> {
    let estimate = Series::new("estimate".into(), &[result.estimate]);
    let statistic = Series::new("statistic".into(), &[result.statistic]);
    let p_value = Series::new("p_value".into(), &[result.p_value]);
    let ci_lower = Series::new(
        "ci_lower".into(),
        &[result
            .conf_int
            .as_ref()
            .map(|ci| ci.lower)
            .unwrap_or(f64::NAN)],
    );
    let ci_upper = Series::new(
        "ci_upper".into(),
        &[result
            .conf_int
            .as_ref()
            .map(|ci| ci.upper)
            .unwrap_or(f64::NAN)],
    );
    let n = Series::new("n".into(), &[result.n as u32]);

    let df = StructChunked::from_series(
        name.into(),
        1,
        [&estimate, &statistic, &p_value, &ci_lower, &ci_upper, &n].into_iter(),
    )?;
    Ok(df.into_series())
}

/// Helper to create error correlation output
fn correlation_error_output(name: &str) -> PolarsResult<Series> {
    let estimate = Series::new("estimate".into(), &[f64::NAN]);
    let statistic = Series::new("statistic".into(), &[f64::NAN]);
    let p_value = Series::new("p_value".into(), &[f64::NAN]);
    let ci_lower = Series::new("ci_lower".into(), &[f64::NAN]);
    let ci_upper = Series::new("ci_upper".into(), &[f64::NAN]);
    let n = Series::new("n".into(), &[0u32]);

    let df = StructChunked::from_series(
        name.into(),
        1,
        [&estimate, &statistic, &p_value, &ci_lower, &ci_upper, &n].into_iter(),
    )?;
    Ok(df.into_series())
}

/// Pearson correlation coefficient
#[polars_expr(output_type_func=correlation_output_dtype)]
fn pl_pearson(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;
    let y = inputs[1].f64()?;
    let conf_level = inputs[2].f64()?.get(0);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();

    match pearson(&x_vec, &y_vec, conf_level) {
        Ok(result) => correlation_output(&result, "pearson"),
        Err(_) => correlation_error_output("pearson"),
    }
}

/// Spearman rank correlation coefficient
#[polars_expr(output_type_func=correlation_output_dtype)]
fn pl_spearman(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;
    let y = inputs[1].f64()?;
    let conf_level = inputs[2].f64()?.get(0);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();

    match spearman(&x_vec, &y_vec, conf_level) {
        Ok(result) => correlation_output(&result, "spearman"),
        Err(_) => correlation_error_output("spearman"),
    }
}

/// Kendall's tau correlation coefficient
#[polars_expr(output_type_func=correlation_output_dtype)]
fn pl_kendall(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;
    let y = inputs[1].f64()?;
    let variant_str = inputs[2].str()?.get(0).unwrap_or("b");

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();

    let variant = match variant_str.to_lowercase().as_str() {
        "a" => KendallVariant::TauA,
        "c" => KendallVariant::TauC,
        _ => KendallVariant::TauB,
    };

    match kendall(&x_vec, &y_vec, variant) {
        Ok(result) => correlation_output(&result, "kendall"),
        Err(_) => correlation_error_output("kendall"),
    }
}

/// Distance correlation test
#[polars_expr(output_type_func=correlation_output_dtype)]
fn pl_distance_cor(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;
    let y = inputs[1].f64()?;
    let n_permutations = inputs[2].u32()?.get(0).unwrap_or(999) as usize;
    let seed = inputs[3].u64()?.get(0);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();

    match distance_cor_test(&x_vec, &y_vec, n_permutations, seed) {
        Ok(result) => {
            let estimate = Series::new("estimate".into(), &[result.dcor]);
            let statistic = Series::new("statistic".into(), &[result.statistic]);
            let p_value = Series::new("p_value".into(), &[result.p_value]);
            let ci_lower = Series::new("ci_lower".into(), &[f64::NAN]);
            let ci_upper = Series::new("ci_upper".into(), &[f64::NAN]);
            let n = Series::new("n".into(), &[result.n as u32]);

            let df = StructChunked::from_series(
                "distance_cor".into(),
                1,
                [&estimate, &statistic, &p_value, &ci_lower, &ci_upper, &n].into_iter(),
            )?;
            Ok(df.into_series())
        }
        Err(_) => correlation_error_output("distance_cor"),
    }
}

/// Partial correlation
#[polars_expr(output_type_func=correlation_output_dtype)]
fn pl_partial_cor(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;
    let y = inputs[1].f64()?;
    // Covariates are passed as separate series after x and y
    let n_covariates = inputs[2].u32()?.get(0).unwrap_or(1) as usize;

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();

    // Collect covariates
    let mut covariates: Vec<Vec<f64>> = Vec::new();
    for i in 0..n_covariates {
        if let Some(cov_series) = inputs.get(3 + i) {
            if let Ok(cov) = cov_series.f64() {
                covariates.push(cov.into_no_null_iter().collect());
            }
        }
    }

    if covariates.is_empty() {
        return correlation_error_output("partial_cor");
    }

    // Convert to slice of slices for the API
    let cov_refs: Vec<&[f64]> = covariates.iter().map(|v| v.as_slice()).collect();

    match partial_cor(&x_vec, &y_vec, &cov_refs) {
        Ok(result) => {
            let estimate = Series::new("estimate".into(), &[result.estimate]);
            let statistic = Series::new("statistic".into(), &[result.statistic]);
            let p_value = Series::new("p_value".into(), &[result.p_value]);
            let ci_lower = Series::new("ci_lower".into(), &[f64::NAN]);
            let ci_upper = Series::new("ci_upper".into(), &[f64::NAN]);
            let n = Series::new("n".into(), &[result.n as u32]);

            let df = StructChunked::from_series(
                "partial_cor".into(),
                1,
                [&estimate, &statistic, &p_value, &ci_lower, &ci_upper, &n].into_iter(),
            )?;
            Ok(df.into_series())
        }
        Err(_) => correlation_error_output("partial_cor"),
    }
}

/// Semi-partial correlation
#[polars_expr(output_type_func=correlation_output_dtype)]
fn pl_semi_partial_cor(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;
    let y = inputs[1].f64()?;
    let n_covariates = inputs[2].u32()?.get(0).unwrap_or(1) as usize;

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();

    let mut covariates: Vec<Vec<f64>> = Vec::new();
    for i in 0..n_covariates {
        if let Some(cov_series) = inputs.get(3 + i) {
            if let Ok(cov) = cov_series.f64() {
                covariates.push(cov.into_no_null_iter().collect());
            }
        }
    }

    if covariates.is_empty() {
        return correlation_error_output("semi_partial_cor");
    }

    let cov_refs: Vec<&[f64]> = covariates.iter().map(|v| v.as_slice()).collect();

    match semi_partial_cor(&x_vec, &y_vec, &cov_refs) {
        Ok(result) => {
            let estimate = Series::new("estimate".into(), &[result.estimate]);
            let statistic = Series::new("statistic".into(), &[result.statistic]);
            let p_value = Series::new("p_value".into(), &[result.p_value]);
            let ci_lower = Series::new("ci_lower".into(), &[f64::NAN]);
            let ci_upper = Series::new("ci_upper".into(), &[f64::NAN]);
            let n = Series::new("n".into(), &[result.n as u32]);

            let df = StructChunked::from_series(
                "semi_partial_cor".into(),
                1,
                [&estimate, &statistic, &p_value, &ci_lower, &ci_upper, &n].into_iter(),
            )?;
            Ok(df.into_series())
        }
        Err(_) => correlation_error_output("semi_partial_cor"),
    }
}

/// Intraclass correlation coefficient (ICC)
/// Note: ICC requires a 2D matrix structure (subjects x raters).
/// This is a placeholder that requires proper matrix input handling.
#[polars_expr(output_type_func=correlation_output_dtype)]
fn pl_icc(inputs: &[Series]) -> PolarsResult<Series> {
    // Data is passed as a matrix where each column is a rater
    // For simplicity, we expect groups and values columns
    let values = inputs[0].f64()?;
    let _icc_type_str = inputs[1].str()?.get(0).unwrap_or("icc1");
    let _conf_level = inputs[2].f64()?.get(0).unwrap_or(0.95);

    let values_vec: Vec<f64> = values.into_no_null_iter().collect();

    // ICC requires a 2D matrix structure - this is a simplified placeholder
    // The actual implementation would need proper matrix handling
    // TODO: Implement proper ICC with matrix input
    let estimate = Series::new("estimate".into(), &[f64::NAN]);
    let statistic = Series::new("statistic".into(), &[f64::NAN]);
    let p_value = Series::new("p_value".into(), &[f64::NAN]);
    let ci_lower = Series::new("ci_lower".into(), &[f64::NAN]);
    let ci_upper = Series::new("ci_upper".into(), &[f64::NAN]);
    let n = Series::new("n".into(), &[values_vec.len() as u32]);

    let df = StructChunked::from_series(
        "icc".into(),
        1,
        [&estimate, &statistic, &p_value, &ci_lower, &ci_upper, &n].into_iter(),
    )?;
    Ok(df.into_series())
}
