//! Non-parametric statistical test expressions.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use anofox_statistics::{
    brunner_munzel, kruskal_wallis, mann_whitney_u, matched_pairs_rank_biserial, rank_biserial,
    rank_biserial_from_u, wilcoxon_signed_rank, Alternative,
};

use crate::expressions::output_types::{
    effect_size_output, effect_size_output_dtype, generic_stats_output, rank_location_output,
    rank_location_output_dtype, stats_output_dtype,
};

/// Helper to parse alternative hypothesis from string
fn parse_alternative(s: &str) -> Alternative {
    match s.to_lowercase().as_str() {
        "less" => Alternative::Less,
        "greater" => Alternative::Greater,
        _ => Alternative::TwoSided,
    }
}

/// Public Rust-callable variant. Same input contract as the `pl_mann_whitney_u` expression shim.
///
/// Returns `struct{statistic: f64, p_value: f64, estimate: f64, ci_lower: f64, ci_upper: f64}`.
/// `estimate`/`ci_lower`/`ci_upper` are NaN unless `conf_level` (inputs[5]) is `Some`; a
/// one-sided `alternative` may give an infinite bound, preserved as such.
pub fn mann_whitney_u_fit(inputs: &[Series]) -> PolarsResult<Series> {
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
        Ok(result) => rank_location_output(
            result.statistic,
            result.p_value,
            result.estimate,
            result.conf_int.as_ref().map(|ci| (ci.lower, ci.upper)),
            "mann_whitney_u",
        ),
        Err(_) => rank_location_output(f64::NAN, f64::NAN, None, None, "mann_whitney_u"),
    }
}

/// Mann-Whitney U test
#[polars_expr(output_type_func=rank_location_output_dtype)]
fn pl_mann_whitney_u(inputs: &[Series]) -> PolarsResult<Series> {
    mann_whitney_u_fit(inputs)
}

/// Public Rust-callable variant. Same input contract as the `pl_wilcoxon_signed_rank` expression shim.
///
/// Returns `struct{statistic: f64, p_value: f64, estimate: f64, ci_lower: f64, ci_upper: f64}`.
/// `estimate`/`ci_lower`/`ci_upper` are NaN unless `conf_level` (inputs[5]) is `Some`; a
/// one-sided `alternative` may give an infinite bound, preserved as such.
pub fn wilcoxon_signed_rank_fit(inputs: &[Series]) -> PolarsResult<Series> {
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
        Ok(result) => rank_location_output(
            result.statistic,
            result.p_value,
            result.estimate,
            result.conf_int.as_ref().map(|ci| (ci.lower, ci.upper)),
            "wilcoxon_signed_rank",
        ),
        Err(_) => rank_location_output(f64::NAN, f64::NAN, None, None, "wilcoxon_signed_rank"),
    }
}

/// Wilcoxon signed-rank test
#[polars_expr(output_type_func=rank_location_output_dtype)]
fn pl_wilcoxon_signed_rank(inputs: &[Series]) -> PolarsResult<Series> {
    wilcoxon_signed_rank_fit(inputs)
}

/// Public Rust-callable variant. Same input contract as the `pl_kruskal_wallis` expression shim.
pub fn kruskal_wallis_fit(inputs: &[Series]) -> PolarsResult<Series> {
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

/// Kruskal-Wallis H test
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_kruskal_wallis(inputs: &[Series]) -> PolarsResult<Series> {
    kruskal_wallis_fit(inputs)
}

/// Public Rust-callable variant. Same input contract as the `pl_brunner_munzel` expression shim.
pub fn brunner_munzel_fit(inputs: &[Series]) -> PolarsResult<Series> {
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

/// Brunner-Munzel test (robust alternative to Mann-Whitney)
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_brunner_munzel(inputs: &[Series]) -> PolarsResult<Series> {
    brunner_munzel_fit(inputs)
}

// ─── Rank-biserial effect sizes ──────────────────────────────────────────────

/// Public Rust-callable variant. Same input contract as the `pl_rank_biserial` expression shim.
///
/// # Input contract
/// - `inputs[0]` — `f64` Series (first sample `x`).
/// - `inputs[1]` — `f64` Series (second sample `y`).
/// - `inputs[2]` — `f64` literal, nullable: `mu` (location shift under the null; `None` means 0.0).
pub fn rank_biserial_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;
    let y = inputs[1].f64()?;
    let mu = inputs[2].f64()?.get(0);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();

    match rank_biserial(&x_vec, &y_vec, mu) {
        Ok(r) => effect_size_output(r, "rank_biserial"),
        Err(_) => effect_size_output(f64::NAN, "rank_biserial"),
    }
}

/// Rank-biserial correlation effect size for the Mann-Whitney U test.
#[polars_expr(output_type_func=effect_size_output_dtype)]
fn pl_rank_biserial(inputs: &[Series]) -> PolarsResult<Series> {
    rank_biserial_fit(inputs)
}

/// Public Rust-callable variant. Same input contract as the `pl_matched_pairs_rank_biserial`
/// expression shim.
///
/// # Input contract
/// - `inputs[0]` — `f64` Series (first sample `x`).
/// - `inputs[1]` — `f64` Series (second sample `y`).
/// - `inputs[2]` — `f64` literal, nullable: `mu` (location shift under the null; `None` means 0.0).
pub fn matched_pairs_rank_biserial_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;
    let y = inputs[1].f64()?;
    let mu = inputs[2].f64()?.get(0);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();

    match matched_pairs_rank_biserial(&x_vec, &y_vec, mu) {
        Ok(r) => effect_size_output(r, "matched_pairs_rank_biserial"),
        Err(_) => effect_size_output(f64::NAN, "matched_pairs_rank_biserial"),
    }
}

/// Matched-pairs rank-biserial correlation effect size for the Wilcoxon signed-rank test.
#[polars_expr(output_type_func=effect_size_output_dtype)]
fn pl_matched_pairs_rank_biserial(inputs: &[Series]) -> PolarsResult<Series> {
    matched_pairs_rank_biserial_fit(inputs)
}

/// Public Rust-callable variant. Same input contract as the `pl_rank_biserial_from_u` expression
/// shim.
///
/// # Input contract
/// - `inputs[0]` — `f64` literal: `u1`, the Mann-Whitney U statistic of the first sample.
/// - `inputs[1]` — `u32` literal: `n1`, the size of the first sample.
/// - `inputs[2]` — `u32` literal: `n2`, the size of the second sample.
///
/// `rank_biserial_from_u` is infallible upstream (returns NaN when `n1 * n2 == 0`), so this
/// never takes the error branch.
pub fn rank_biserial_from_u_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let u1 = inputs[0].f64()?.get(0).unwrap_or(f64::NAN);
    let n1 = inputs[1].u32()?.get(0).unwrap_or(0) as usize;
    let n2 = inputs[2].u32()?.get(0).unwrap_or(0) as usize;

    let r = rank_biserial_from_u(u1, n1, n2);
    effect_size_output(r, "rank_biserial_from_u")
}

/// Rank-biserial correlation computed directly from a Mann-Whitney U statistic.
#[polars_expr(output_type_func=effect_size_output_dtype)]
fn pl_rank_biserial_from_u(inputs: &[Series]) -> PolarsResult<Series> {
    rank_biserial_from_u_fit(inputs)
}
