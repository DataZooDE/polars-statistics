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

/// Output dtype for one-way ANOVA (Fisher and Welch).
///
/// Welch variant returns NaN for ss_between, ss_within, ms_between, ms_within, eta_squared.
pub fn one_way_anova_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("statistic".into(), DataType::Float64),
        Field::new("df_between".into(), DataType::Float64),
        Field::new("df_within".into(), DataType::Float64),
        Field::new("p_value".into(), DataType::Float64),
        Field::new("ss_between".into(), DataType::Float64),
        Field::new("ss_within".into(), DataType::Float64),
        Field::new("ms_between".into(), DataType::Float64),
        Field::new("ms_within".into(), DataType::Float64),
        Field::new("eta_squared".into(), DataType::Float64),
        Field::new("n_groups".into(), DataType::UInt32),
    ];
    Ok(Field::new("one_way_anova".into(), DataType::Struct(fields)))
}

/// Output dtype for two-way ANOVA.
///
/// Flattens AnovaTableRow structs one level deep with prefixed field names.
/// Variable-length Vec fields (cell_means, marginal_means) are excluded.
pub fn two_way_anova_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        // Factor A main effect
        Field::new("a_ss".into(), DataType::Float64),
        Field::new("a_df".into(), DataType::Float64),
        Field::new("a_ms".into(), DataType::Float64),
        Field::new("a_f".into(), DataType::Float64),
        Field::new("a_p_value".into(), DataType::Float64),
        // Factor B main effect
        Field::new("b_ss".into(), DataType::Float64),
        Field::new("b_df".into(), DataType::Float64),
        Field::new("b_ms".into(), DataType::Float64),
        Field::new("b_f".into(), DataType::Float64),
        Field::new("b_p_value".into(), DataType::Float64),
        // A×B interaction
        Field::new("ab_ss".into(), DataType::Float64),
        Field::new("ab_df".into(), DataType::Float64),
        Field::new("ab_ms".into(), DataType::Float64),
        Field::new("ab_f".into(), DataType::Float64),
        Field::new("ab_p_value".into(), DataType::Float64),
        // Residual
        Field::new("residual_ss".into(), DataType::Float64),
        Field::new("residual_df".into(), DataType::Float64),
        Field::new("residual_ms".into(), DataType::Float64),
        // Summary
        Field::new("grand_mean".into(), DataType::Float64),
        Field::new("n".into(), DataType::UInt32),
    ];
    Ok(Field::new("two_way_anova".into(), DataType::Struct(fields)))
}

/// Output dtype for repeated-measures ANOVA.
///
/// Flattens within-subjects, error, sphericity, and correction structs.
/// Variable-length Vec fields (condition_means, subject_means) are excluded.
pub fn repeated_measures_anova_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        // Within-subjects (condition) effect
        Field::new("ws_f".into(), DataType::Float64),
        Field::new("ws_df".into(), DataType::Float64),
        Field::new("ws_ss".into(), DataType::Float64),
        Field::new("ws_ms".into(), DataType::Float64),
        Field::new("ws_p_value".into(), DataType::Float64),
        // Error term
        Field::new("error_df".into(), DataType::Float64),
        Field::new("error_ss".into(), DataType::Float64),
        Field::new("error_ms".into(), DataType::Float64),
        // Mauchly's sphericity test (NaN if k < 3 or compute_sphericity=false)
        Field::new("mauchly_w".into(), DataType::Float64),
        Field::new("mauchly_p_value".into(), DataType::Float64),
        // Greenhouse-Geisser correction (NaN if k < 3)
        Field::new("gg_epsilon".into(), DataType::Float64),
        Field::new("gg_p_value".into(), DataType::Float64),
        // Huynh-Feldt correction (NaN if k < 3)
        Field::new("hf_epsilon".into(), DataType::Float64),
        Field::new("hf_p_value".into(), DataType::Float64),
        // Summary
        Field::new("grand_mean".into(), DataType::Float64),
    ];
    Ok(Field::new(
        "repeated_measures_anova".into(),
        DataType::Struct(fields),
    ))
}

/// Output dtype for ICC (intra-class correlation coefficient).
///
/// Replaces the all-NaN stub. Variable-type fields (icc_type enum, method String) are excluded.
pub fn icc_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("icc".into(), DataType::Float64),
        Field::new("f_value".into(), DataType::Float64),
        Field::new("df1".into(), DataType::Float64),
        Field::new("df2".into(), DataType::Float64),
        Field::new("p_value".into(), DataType::Float64),
        Field::new("ci_lower".into(), DataType::Float64),
        Field::new("ci_upper".into(), DataType::Float64),
        Field::new("n_subjects".into(), DataType::UInt32),
        Field::new("n_raters".into(), DataType::UInt32),
    ];
    Ok(Field::new("icc".into(), DataType::Struct(fields)))
}
