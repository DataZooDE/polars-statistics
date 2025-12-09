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
