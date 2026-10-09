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

/// Output dtype for effect-size expressions (Cohen's d, rank-biserial family):
/// struct{estimate: f64}
pub fn effect_size_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![Field::new("estimate".into(), DataType::Float64)];
    Ok(Field::new("effect_size".into(), DataType::Struct(fields)))
}

/// Create output Series for an effect-size expression from a single estimate.
pub fn effect_size_output(estimate: f64, name: &str) -> PolarsResult<Series> {
    let estimate_series = Series::new("estimate".into(), vec![estimate]);

    StructChunked::from_series(
        name.into(),
        estimate_series.len(),
        [&estimate_series].into_iter(),
    )
    .map(|ca| ca.into_series())
}

/// Output dtype for Fisher's exact test (with confidence interval):
/// struct{statistic: f64, p_value: f64, ci_lower: f64, ci_upper: f64}
pub fn fisher_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("statistic".into(), DataType::Float64),
        Field::new("p_value".into(), DataType::Float64),
        Field::new("ci_lower".into(), DataType::Float64),
        Field::new("ci_upper".into(), DataType::Float64),
    ];
    Ok(Field::new("fisher_exact".into(), DataType::Struct(fields)))
}

/// Create output Series for Fisher's exact test: odds ratio (as `statistic`),
/// p-value, and the confidence interval bounds.
pub fn fisher_output(
    odds_ratio: f64,
    p_value: f64,
    ci_lower: f64,
    ci_upper: f64,
    name: &str,
) -> PolarsResult<Series> {
    let stat_series = Series::new("statistic".into(), vec![odds_ratio]);
    let pval_series = Series::new("p_value".into(), vec![p_value]);
    let ci_lower_series = Series::new("ci_lower".into(), vec![ci_lower]);
    let ci_upper_series = Series::new("ci_upper".into(), vec![ci_upper]);

    StructChunked::from_series(
        name.into(),
        stat_series.len(),
        [
            &stat_series,
            &pval_series,
            &ci_lower_series,
            &ci_upper_series,
        ]
        .into_iter(),
    )
    .map(|ca| ca.into_series())
}

/// Output dtype for the Jarque-Bera normality test:
/// struct{statistic: f64, p_value: f64, skewness: f64, kurtosis: f64, n: u32}
pub fn jarque_bera_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("statistic".into(), DataType::Float64),
        Field::new("p_value".into(), DataType::Float64),
        Field::new("skewness".into(), DataType::Float64),
        Field::new("kurtosis".into(), DataType::Float64),
        Field::new("n".into(), DataType::UInt32),
    ];
    Ok(Field::new("jarque_bera".into(), DataType::Struct(fields)))
}

/// Create output Series for the Jarque-Bera normality test.
pub fn jarque_bera_output(
    statistic: f64,
    p_value: f64,
    skewness: f64,
    kurtosis: f64,
    n: u32,
    name: &str,
) -> PolarsResult<Series> {
    let stat_series = Series::new("statistic".into(), vec![statistic]);
    let pval_series = Series::new("p_value".into(), vec![p_value]);
    let skew_series = Series::new("skewness".into(), vec![skewness]);
    let kurt_series = Series::new("kurtosis".into(), vec![kurtosis]);
    let n_series = Series::new("n".into(), vec![n]);

    StructChunked::from_series(
        name.into(),
        stat_series.len(),
        [
            &stat_series,
            &pval_series,
            &skew_series,
            &kurt_series,
            &n_series,
        ]
        .into_iter(),
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
