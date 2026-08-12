//! Parametric statistical test expressions.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use anofox_statistics::{
    one_way_anova, repeated_measures_anova, t_test, two_way_anova, yuen_test, Alternative,
    AnovaKind, TTestKind,
};

use crate::expressions::output_types::{
    generic_stats_output, one_way_anova_output_dtype, repeated_measures_anova_output_dtype,
    stats_output_dtype, two_way_anova_output_dtype,
};

/// Helper to parse alternative hypothesis from string
fn parse_alternative(s: &str) -> Alternative {
    match s.to_lowercase().as_str() {
        "less" => Alternative::Less,
        "greater" => Alternative::Greater,
        _ => Alternative::TwoSided,
    }
}

/// Public Rust-callable variant. Same input contract as the `pl_ttest_ind` expression shim.
pub fn ttest_ind_fit(inputs: &[Series]) -> PolarsResult<Series> {
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

/// Independent samples t-test from raw data
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_ttest_ind(inputs: &[Series]) -> PolarsResult<Series> {
    ttest_ind_fit(inputs)
}

/// Public Rust-callable variant. Same input contract as the `pl_ttest_paired` expression shim.
pub fn ttest_paired_fit(inputs: &[Series]) -> PolarsResult<Series> {
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

/// Paired samples t-test
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_ttest_paired(inputs: &[Series]) -> PolarsResult<Series> {
    ttest_paired_fit(inputs)
}

/// Public Rust-callable variant. Same input contract as the `pl_brown_forsythe` expression shim.
pub fn brown_forsythe_fit(inputs: &[Series]) -> PolarsResult<Series> {
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

/// Brown-Forsythe test for equal variances
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_brown_forsythe(inputs: &[Series]) -> PolarsResult<Series> {
    brown_forsythe_fit(inputs)
}

/// Public Rust-callable variant. Same input contract as the `pl_yuen_test` expression shim.
pub fn yuen_test_fit(inputs: &[Series]) -> PolarsResult<Series> {
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

/// Yuen's test for trimmed means (robust alternative to t-test)
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_yuen_test(inputs: &[Series]) -> PolarsResult<Series> {
    yuen_test_fit(inputs)
}

// ─── ANOVA family ────────────────────────────────────────────────────────────

/// Helper to parse ANOVA kind from string
fn parse_anova_kind(s: &str) -> AnovaKind {
    match s.to_lowercase().as_str() {
        "welch" => AnovaKind::Welch,
        _ => AnovaKind::Fisher,
    }
}

/// Error-output helper for one_way_anova: all-NaN struct matching the output schema.
fn one_way_anova_error_output() -> PolarsResult<Series> {
    let statistic = Series::new("statistic".into(), &[f64::NAN]);
    let df_between = Series::new("df_between".into(), &[f64::NAN]);
    let df_within = Series::new("df_within".into(), &[f64::NAN]);
    let p_value = Series::new("p_value".into(), &[f64::NAN]);
    let ss_between = Series::new("ss_between".into(), &[f64::NAN]);
    let ss_within = Series::new("ss_within".into(), &[f64::NAN]);
    let ms_between = Series::new("ms_between".into(), &[f64::NAN]);
    let ms_within = Series::new("ms_within".into(), &[f64::NAN]);
    let eta_squared = Series::new("eta_squared".into(), &[f64::NAN]);
    let n_groups = Series::new("n_groups".into(), &[0u32]);
    StructChunked::from_series(
        "one_way_anova".into(),
        1,
        [
            &statistic,
            &df_between,
            &df_within,
            &p_value,
            &ss_between,
            &ss_within,
            &ms_between,
            &ms_within,
            &eta_squared,
            &n_groups,
        ]
        .into_iter(),
    )
    .map(|ca| ca.into_series())
}

/// One-way ANOVA (Fisher or Welch) from separate group Series.
///
/// Input contract:
///   inputs[0..n-1] — f64 Series, one per group (variable arity, ≥ 2 groups)
///   inputs[n-1]    — String literal: "fisher" | "welch"
///
/// The LAST input is always the kind literal; all preceding inputs are group data.
pub fn one_way_anova_fit(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() < 3 {
        // Need at least 2 groups + the kind literal
        return one_way_anova_error_output();
    }
    let kind_series = &inputs[inputs.len() - 1];
    let kind_str = kind_series.str()?.get(0).unwrap_or("fisher");
    let kind = parse_anova_kind(kind_str);

    // Collect all group Series except the last (kind literal)
    let group_series = &inputs[..inputs.len() - 1];
    let groups: Vec<Vec<f64>> = group_series
        .iter()
        .map(|s| s.f64().map(|ca| ca.into_no_null_iter().collect()))
        .collect::<PolarsResult<Vec<_>>>()?;

    let group_refs: Vec<&[f64]> = groups.iter().map(|g| g.as_slice()).collect();

    match one_way_anova(&group_refs, kind) {
        Ok(r) => {
            // Prefer ss_between / ss_total when the crate supplies ss_total; otherwise
            // fall back to ss_between / (ss_between + ss_within), which equals eta-squared
            // for a one-way design (ss_total = ss_between + ss_within). This keeps eta²
            // finite for Fisher fits where ss_total is not populated (WR-02).
            let eta_sq = match (r.ss_between, r.ss_total) {
                (Some(ssb), Some(sst)) if sst > 0.0 => ssb / sst,
                (Some(ssb), _) => match r.ss_within {
                    Some(ssw) if (ssb + ssw) > 0.0 => ssb / (ssb + ssw),
                    _ => f64::NAN,
                },
                _ => f64::NAN,
            };
            let statistic = Series::new("statistic".into(), &[r.statistic]);
            let df_between = Series::new("df_between".into(), &[r.df_between]);
            let df_within = Series::new("df_within".into(), &[r.df_within]);
            let p_value = Series::new("p_value".into(), &[r.p_value]);
            let ss_between = Series::new("ss_between".into(), &[r.ss_between.unwrap_or(f64::NAN)]);
            let ss_within = Series::new("ss_within".into(), &[r.ss_within.unwrap_or(f64::NAN)]);
            let ms_between = Series::new("ms_between".into(), &[r.ms_between.unwrap_or(f64::NAN)]);
            let ms_within = Series::new("ms_within".into(), &[r.ms_within.unwrap_or(f64::NAN)]);
            let eta_squared = Series::new("eta_squared".into(), &[eta_sq]);
            let n_groups = Series::new("n_groups".into(), &[r.n_groups as u32]);
            StructChunked::from_series(
                "one_way_anova".into(),
                1,
                [
                    &statistic,
                    &df_between,
                    &df_within,
                    &p_value,
                    &ss_between,
                    &ss_within,
                    &ms_between,
                    &ms_within,
                    &eta_squared,
                    &n_groups,
                ]
                .into_iter(),
            )
            .map(|ca| ca.into_series())
        }
        Err(_) => one_way_anova_error_output(),
    }
}

/// One-way ANOVA expression shim (FFI entry point).
#[polars_expr(output_type_func=one_way_anova_output_dtype)]
fn pl_one_way_anova(inputs: &[Series]) -> PolarsResult<Series> {
    one_way_anova_fit(inputs)
}

/// Error-output helper for two_way_anova: all-NaN struct matching the output schema.
fn two_way_anova_error_output() -> PolarsResult<Series> {
    let a_ss = Series::new("a_ss".into(), &[f64::NAN]);
    let a_df = Series::new("a_df".into(), &[f64::NAN]);
    let a_ms = Series::new("a_ms".into(), &[f64::NAN]);
    let a_f = Series::new("a_f".into(), &[f64::NAN]);
    let a_p_value = Series::new("a_p_value".into(), &[f64::NAN]);
    let b_ss = Series::new("b_ss".into(), &[f64::NAN]);
    let b_df = Series::new("b_df".into(), &[f64::NAN]);
    let b_ms = Series::new("b_ms".into(), &[f64::NAN]);
    let b_f = Series::new("b_f".into(), &[f64::NAN]);
    let b_p_value = Series::new("b_p_value".into(), &[f64::NAN]);
    let ab_ss = Series::new("ab_ss".into(), &[f64::NAN]);
    let ab_df = Series::new("ab_df".into(), &[f64::NAN]);
    let ab_ms = Series::new("ab_ms".into(), &[f64::NAN]);
    let ab_f = Series::new("ab_f".into(), &[f64::NAN]);
    let ab_p_value = Series::new("ab_p_value".into(), &[f64::NAN]);
    let residual_ss = Series::new("residual_ss".into(), &[f64::NAN]);
    let residual_df = Series::new("residual_df".into(), &[f64::NAN]);
    let residual_ms = Series::new("residual_ms".into(), &[f64::NAN]);
    let grand_mean = Series::new("grand_mean".into(), &[f64::NAN]);
    let n = Series::new("n".into(), &[0u32]);
    StructChunked::from_series(
        "two_way_anova".into(),
        1,
        [
            &a_ss,
            &a_df,
            &a_ms,
            &a_f,
            &a_p_value,
            &b_ss,
            &b_df,
            &b_ms,
            &b_f,
            &b_p_value,
            &ab_ss,
            &ab_df,
            &ab_ms,
            &ab_f,
            &ab_p_value,
            &residual_ss,
            &residual_df,
            &residual_ms,
            &grand_mean,
            &n,
        ]
        .into_iter(),
    )
    .map(|ca| ca.into_series())
}

/// Two-way ANOVA from value + two integer-coded factor Series.
///
/// Input contract:
///   inputs[0] — f64 Series (dependent variable values)
///   inputs[1] — u32 Series (factor A levels, 0-indexed; encoded by Python builder)
///   inputs[2] — u32 Series (factor B levels, 0-indexed; encoded by Python builder)
pub fn two_way_anova_fit(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() < 3 {
        return two_way_anova_error_output();
    }
    let values: Vec<f64> = inputs[0].f64()?.into_no_null_iter().collect();
    let factor_a: Vec<usize> = inputs[1]
        .u32()?
        .into_no_null_iter()
        .map(|v| v as usize)
        .collect();
    let factor_b: Vec<usize> = inputs[2]
        .u32()?
        .into_no_null_iter()
        .map(|v| v as usize)
        .collect();

    match two_way_anova(&values, &factor_a, &factor_b) {
        Ok(r) => {
            let a_ss = Series::new("a_ss".into(), &[r.factor_a.ss]);
            let a_df = Series::new("a_df".into(), &[r.factor_a.df]);
            let a_ms = Series::new("a_ms".into(), &[r.factor_a.ms]);
            let a_f = Series::new("a_f".into(), &[r.factor_a.f_statistic.unwrap_or(f64::NAN)]);
            let a_p_value = Series::new(
                "a_p_value".into(),
                &[r.factor_a.p_value.unwrap_or(f64::NAN)],
            );
            let b_ss = Series::new("b_ss".into(), &[r.factor_b.ss]);
            let b_df = Series::new("b_df".into(), &[r.factor_b.df]);
            let b_ms = Series::new("b_ms".into(), &[r.factor_b.ms]);
            let b_f = Series::new("b_f".into(), &[r.factor_b.f_statistic.unwrap_or(f64::NAN)]);
            let b_p_value = Series::new(
                "b_p_value".into(),
                &[r.factor_b.p_value.unwrap_or(f64::NAN)],
            );
            let ab_ss = Series::new("ab_ss".into(), &[r.interaction.ss]);
            let ab_df = Series::new("ab_df".into(), &[r.interaction.df]);
            let ab_ms = Series::new("ab_ms".into(), &[r.interaction.ms]);
            let ab_f = Series::new(
                "ab_f".into(),
                &[r.interaction.f_statistic.unwrap_or(f64::NAN)],
            );
            let ab_p_value = Series::new(
                "ab_p_value".into(),
                &[r.interaction.p_value.unwrap_or(f64::NAN)],
            );
            let residual_ss = Series::new("residual_ss".into(), &[r.residual.ss]);
            let residual_df = Series::new("residual_df".into(), &[r.residual.df]);
            let residual_ms = Series::new("residual_ms".into(), &[r.residual.ms]);
            let grand_mean = Series::new("grand_mean".into(), &[r.grand_mean]);
            let n = Series::new("n".into(), &[r.n as u32]);
            StructChunked::from_series(
                "two_way_anova".into(),
                1,
                [
                    &a_ss,
                    &a_df,
                    &a_ms,
                    &a_f,
                    &a_p_value,
                    &b_ss,
                    &b_df,
                    &b_ms,
                    &b_f,
                    &b_p_value,
                    &ab_ss,
                    &ab_df,
                    &ab_ms,
                    &ab_f,
                    &ab_p_value,
                    &residual_ss,
                    &residual_df,
                    &residual_ms,
                    &grand_mean,
                    &n,
                ]
                .into_iter(),
            )
            .map(|ca| ca.into_series())
        }
        Err(_) => two_way_anova_error_output(),
    }
}

/// Two-way ANOVA expression shim (FFI entry point).
#[polars_expr(output_type_func=two_way_anova_output_dtype)]
fn pl_two_way_anova(inputs: &[Series]) -> PolarsResult<Series> {
    two_way_anova_fit(inputs)
}

/// Error-output helper for repeated_measures_anova: all-NaN struct matching the output schema.
fn repeated_measures_anova_error_output() -> PolarsResult<Series> {
    let ws_f = Series::new("ws_f".into(), &[f64::NAN]);
    let ws_df = Series::new("ws_df".into(), &[f64::NAN]);
    let ws_ss = Series::new("ws_ss".into(), &[f64::NAN]);
    let ws_ms = Series::new("ws_ms".into(), &[f64::NAN]);
    let ws_p_value = Series::new("ws_p_value".into(), &[f64::NAN]);
    let error_df = Series::new("error_df".into(), &[f64::NAN]);
    let error_ss = Series::new("error_ss".into(), &[f64::NAN]);
    let error_ms = Series::new("error_ms".into(), &[f64::NAN]);
    let mauchly_w = Series::new("mauchly_w".into(), &[f64::NAN]);
    let mauchly_p_value = Series::new("mauchly_p_value".into(), &[f64::NAN]);
    let gg_epsilon = Series::new("gg_epsilon".into(), &[f64::NAN]);
    let gg_p_value = Series::new("gg_p_value".into(), &[f64::NAN]);
    let hf_epsilon = Series::new("hf_epsilon".into(), &[f64::NAN]);
    let hf_p_value = Series::new("hf_p_value".into(), &[f64::NAN]);
    let grand_mean = Series::new("grand_mean".into(), &[f64::NAN]);
    StructChunked::from_series(
        "repeated_measures_anova".into(),
        1,
        [
            &ws_f,
            &ws_df,
            &ws_ss,
            &ws_ms,
            &ws_p_value,
            &error_df,
            &error_ss,
            &error_ms,
            &mauchly_w,
            &mauchly_p_value,
            &gg_epsilon,
            &gg_p_value,
            &hf_epsilon,
            &hf_p_value,
            &grand_mean,
        ]
        .into_iter(),
    )
    .map(|ca| ca.into_series())
}

/// Repeated-measures ANOVA from long-format data (value + subject + condition).
///
/// Input contract:
///   inputs[0] — f64 Series (observed value)
///   inputs[1] — u32 Series (subject ID, 0-indexed; encoded by Python builder)
///   inputs[2] — u32 Series (condition label, 0-indexed; encoded by Python builder)
///   inputs[3] — bool literal (compute_sphericity)
///
/// The long-format data is pivoted to a subjects × conditions matrix internally.
/// Returns the all-NaN error struct if the design is unbalanced (pitfall 3).
pub fn repeated_measures_anova_fit(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() < 4 {
        return repeated_measures_anova_error_output();
    }
    let values: Vec<f64> = inputs[0].f64()?.into_no_null_iter().collect();
    let subjects: Vec<u32> = inputs[1].u32()?.into_no_null_iter().collect();
    let conditions: Vec<u32> = inputs[2].u32()?.into_no_null_iter().collect();
    let compute_sphericity = inputs[3].bool()?.get(0).unwrap_or(true);

    if values.is_empty() {
        return repeated_measures_anova_error_output();
    }

    // Collect unique subjects and conditions (preserving insertion order)
    let mut unique_subjects: Vec<u32> = Vec::new();
    for &s in &subjects {
        if !unique_subjects.contains(&s) {
            unique_subjects.push(s);
        }
    }
    let mut unique_conditions: Vec<u32> = Vec::new();
    for &c in &conditions {
        if !unique_conditions.contains(&c) {
            unique_conditions.push(c);
        }
    }
    let n_subjects = unique_subjects.len();
    let n_conditions = unique_conditions.len();

    // Build subject × condition lookup
    use std::collections::HashMap;
    let mut cell_map: HashMap<(u32, u32), f64> = HashMap::new();
    for i in 0..values.len() {
        cell_map.insert((subjects[i], conditions[i]), values[i]);
    }

    // Validate balanced design: every subject must have exactly n_conditions entries
    for &subj in &unique_subjects {
        let count = subjects.iter().filter(|&&s| s == subj).count();
        if count != n_conditions {
            return repeated_measures_anova_error_output();
        }
    }

    // Build matrix: data[subject_i] = Vec<f64> across conditions in order
    let mut matrix: Vec<Vec<f64>> = Vec::with_capacity(n_subjects);
    for &subj in &unique_subjects {
        let row: Vec<f64> = unique_conditions
            .iter()
            .map(|&cond| *cell_map.get(&(subj, cond)).unwrap_or(&f64::NAN))
            .collect();
        matrix.push(row);
    }

    let matrix_refs: Vec<&[f64]> = matrix.iter().map(|r| r.as_slice()).collect();

    match repeated_measures_anova(&matrix_refs, compute_sphericity) {
        Ok(r) => {
            let ws = &r.within_subjects;
            let err = &r.error;
            let ws_f = Series::new("ws_f".into(), &[ws.f_statistic.unwrap_or(f64::NAN)]);
            let ws_df = Series::new("ws_df".into(), &[ws.df]);
            let ws_ss = Series::new("ws_ss".into(), &[ws.ss]);
            let ws_ms = Series::new("ws_ms".into(), &[ws.ms]);
            let ws_p_value = Series::new("ws_p_value".into(), &[ws.p_value.unwrap_or(f64::NAN)]);
            let error_df = Series::new("error_df".into(), &[err.df]);
            let error_ss = Series::new("error_ss".into(), &[err.ss]);
            let error_ms = Series::new("error_ms".into(), &[err.ms]);
            let (mauchly_w_v, mauchly_p_v) = r
                .sphericity
                .as_ref()
                .map(|s| (s.w, s.p_value))
                .unwrap_or((f64::NAN, f64::NAN));
            let (gg_eps, gg_p) = r
                .greenhouse_geisser
                .as_ref()
                .map(|g| (g.epsilon, g.p_value))
                .unwrap_or((f64::NAN, f64::NAN));
            let (hf_eps, hf_p) = r
                .huynh_feldt
                .as_ref()
                .map(|h| (h.epsilon, h.p_value))
                .unwrap_or((f64::NAN, f64::NAN));
            let mauchly_w = Series::new("mauchly_w".into(), &[mauchly_w_v]);
            let mauchly_p_value = Series::new("mauchly_p_value".into(), &[mauchly_p_v]);
            let gg_epsilon = Series::new("gg_epsilon".into(), &[gg_eps]);
            let gg_p_value = Series::new("gg_p_value".into(), &[gg_p]);
            let hf_epsilon = Series::new("hf_epsilon".into(), &[hf_eps]);
            let hf_p_value = Series::new("hf_p_value".into(), &[hf_p]);
            let grand_mean = Series::new("grand_mean".into(), &[r.grand_mean]);
            StructChunked::from_series(
                "repeated_measures_anova".into(),
                1,
                [
                    &ws_f,
                    &ws_df,
                    &ws_ss,
                    &ws_ms,
                    &ws_p_value,
                    &error_df,
                    &error_ss,
                    &error_ms,
                    &mauchly_w,
                    &mauchly_p_value,
                    &gg_epsilon,
                    &gg_p_value,
                    &hf_epsilon,
                    &hf_p_value,
                    &grand_mean,
                ]
                .into_iter(),
            )
            .map(|ca| ca.into_series())
        }
        Err(_) => repeated_measures_anova_error_output(),
    }
}

/// Repeated-measures ANOVA expression shim (FFI entry point).
#[polars_expr(output_type_func=repeated_measures_anova_output_dtype)]
fn pl_repeated_measures_anova(inputs: &[Series]) -> PolarsResult<Series> {
    repeated_measures_anova_fit(inputs)
}
