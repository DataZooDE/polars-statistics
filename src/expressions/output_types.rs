//! Common output types for statistical test expressions.

use polars::prelude::*;

/// Standard output dtype for statistical tests: struct{statistic: f64, p_value: f64}
pub fn stats_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("statistic".into(), DataType::Float64),
        Field::new("p_value".into(), DataType::Float64),
    ];
    Ok(Field::new("stats".into(), DataType::Struct(fields)))
}

/// Create output Series from statistic and p-value
pub fn generic_stats_output(statistic: f64, p_value: f64, name: &str) -> PolarsResult<Series> {
    let stat_series = Series::new("statistic".into(), vec![statistic]);
    let pval_series = Series::new("p_value".into(), vec![p_value]);

    StructChunked::from_series(
        name.into(),
        stat_series.len(),
        [&stat_series, &pval_series].into_iter(),
    )
    .map(|ca| ca.into_series())
}

/// Output dtype for TOST equivalence tests
pub fn tost_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("estimate".into(), DataType::Float64),
        Field::new("ci_lower".into(), DataType::Float64),
        Field::new("ci_upper".into(), DataType::Float64),
        Field::new("bound_lower".into(), DataType::Float64),
        Field::new("bound_upper".into(), DataType::Float64),
        Field::new("tost_p_value".into(), DataType::Float64),
        Field::new("equivalent".into(), DataType::Boolean),
        Field::new("alpha".into(), DataType::Float64),
        Field::new("n".into(), DataType::UInt32),
    ];
    Ok(Field::new("tost".into(), DataType::Struct(fields)))
}

/// Output dtype for correlation tests
pub fn correlation_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("estimate".into(), DataType::Float64),
        Field::new("statistic".into(), DataType::Float64),
        Field::new("p_value".into(), DataType::Float64),
        Field::new("ci_lower".into(), DataType::Float64),
        Field::new("ci_upper".into(), DataType::Float64),
        Field::new("n".into(), DataType::UInt32),
    ];
    Ok(Field::new("correlation".into(), DataType::Struct(fields)))
}

/// Output dtype for proportion tests
pub fn proportion_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("estimate".into(), DataType::Float64),
        Field::new("statistic".into(), DataType::Float64),
        Field::new("p_value".into(), DataType::Float64),
        Field::new("ci_lower".into(), DataType::Float64),
        Field::new("ci_upper".into(), DataType::Float64),
        Field::new("n".into(), DataType::UInt32),
    ];
    Ok(Field::new("proportion".into(), DataType::Struct(fields)))
}

/// Output dtype for chi-square tests
pub fn chisq_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("statistic".into(), DataType::Float64),
        Field::new("p_value".into(), DataType::Float64),
        Field::new("df".into(), DataType::Float64),
        Field::new("n".into(), DataType::UInt32),
    ];
    Ok(Field::new("chisq".into(), DataType::Struct(fields)))
}

/// Output dtype for association measures
pub fn association_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("estimate".into(), DataType::Float64),
        Field::new("statistic".into(), DataType::Float64),
        Field::new("p_value".into(), DataType::Float64),
    ];
    Ok(Field::new("association".into(), DataType::Struct(fields)))
}
