//! Post-hoc multiple-comparison tests wrapping `anofox_statistics::posthoc`
//! (crate version 0.4.7).
//!
//! - [`p_adjust_fit`] / `pl_p_adjust`: R's `p.adjust` (holm, hochberg, hommel,
//!   bonferroni, BH / fdr, BY, none).
//! - `ptukey_fit` / `pl_ptukey`, `qtukey_fit` / `pl_qtukey` (Task 2): R's
//!   studentized-range distribution `ptukey` / `qtukey`.
//! - `tukey_hsd_fit` / `pl_tukey_hsd`, `pairwise_t_test_fit` /
//!   `pl_pairwise_t_test`, `dunn_test_fit` / `pl_dunn_test` (Task 3): R's
//!   `TukeyHSD(aov(...))`, `pairwise.t.test` and `dunn.test`.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use anofox_statistics::{
    dunn_test, p_adjust, pairwise_t_test, ptukey, qtukey, tukey_hsd, Alternative, PAdjustMethod,
    PairwiseResult,
};

/// Parse an R `p.adjust` method name (case-insensitive; `"fdr"` aliases `"BH"`).
///
/// Returns a `ComputeError` naming every allowed method when `s` does not
/// match any of them.
fn parse_p_adjust_method(s: &str) -> PolarsResult<PAdjustMethod> {
    s.parse::<PAdjustMethod>()
        .map_err(|_| polars_err!(ComputeError: "unknown p_adjust method '{}' (expected one of: holm, hochberg, hommel, bonferroni, BH, fdr, BY, none)", s))
}

/// Public Rust-callable variant. Same input contract as the `pl_p_adjust`
/// expression shim.
///
/// # Input contract
/// `inputs[0]`: p (f64 Series, nullable). `inputs[1]`: method (String
/// scalar, default "holm").
///
/// Null positions in `inputs[0]` are treated as NaN for the adjustment (R's
/// `NA` semantics: kept as NaN, excluded from the comparison count `n`) and
/// are null again in the output; non-null NaN inputs stay NaN in the output.
pub fn p_adjust_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let p_ca = inputs[0].f64()?;
    let method_str = inputs[1].str()?.get(0).unwrap_or("holm");
    let method = parse_p_adjust_method(method_str)?;

    let is_null: Vec<bool> = p_ca.iter().map(|v| v.is_none()).collect();
    let p_vec: Vec<f64> = p_ca.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
    let adjusted = p_adjust(&p_vec, method);

    let out: Vec<Option<f64>> = is_null
        .iter()
        .zip(adjusted.iter())
        .map(|(&null, &v)| if null { None } else { Some(v) })
        .collect();

    Ok(Series::new(inputs[0].name().clone(), out))
}

/// R `p.adjust`: multiple-comparison p-value adjustment.
#[polars_expr(output_type=Float64)]
fn pl_p_adjust(inputs: &[Series]) -> PolarsResult<Series> {
    p_adjust_fit(inputs)
}

/// Elementwise application of a 4-argument scalar function over `inputs`
/// (exactly 4 Series), broadcasting any length-1 argument against the
/// others. A null in any argument gives a null output row; the output takes
/// the name of `inputs[0]`.
///
/// # Errors
/// If any argument's length is neither 1 nor the maximum length among all
/// four arguments.
fn elementwise4(inputs: &[Series], f: impl Fn(f64, f64, f64, f64) -> f64) -> PolarsResult<Series> {
    let name = inputs[0].name().clone();
    let cols = [
        inputs[0].f64()?,
        inputs[1].f64()?,
        inputs[2].f64()?,
        inputs[3].f64()?,
    ];
    let n = cols.iter().map(|c| c.len()).max().unwrap_or(0);
    for c in &cols {
        if c.len() != 1 && c.len() != n {
            return Err(
                polars_err!(ComputeError: "elementwise4: every argument must have length 1 or {} (got {})", n, c.len()),
            );
        }
    }
    let get = |c: &Float64Chunked, i: usize| -> Option<f64> {
        if c.len() == 1 {
            c.get(0)
        } else {
            c.get(i)
        }
    };
    let out: Vec<Option<f64>> = (0..n)
        .map(|i| {
            let a = get(cols[0], i)?;
            let b = get(cols[1], i)?;
            let c = get(cols[2], i)?;
            let d = get(cols[3], i)?;
            Some(f(a, b, c, d))
        })
        .collect();
    Ok(Series::new(name, out))
}

/// Public Rust-callable variant. Same input contract as the `pl_ptukey`
/// expression shim.
///
/// # Input contract
/// `inputs[0..4]`: q, nmeans, df, nranges (f64, each length 1 or n).
/// `inputs[4]`: lower_tail (bool scalar, default true).
pub fn ptukey_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let lower_tail = inputs[4].bool()?.get(0).unwrap_or(true);
    elementwise4(&inputs[0..4], |q, nmeans, df, nranges| {
        ptukey(q, nmeans, df, nranges, lower_tail)
    })
}

/// R `ptukey`: studentized-range distribution CDF.
#[polars_expr(output_type=Float64)]
fn pl_ptukey(inputs: &[Series]) -> PolarsResult<Series> {
    ptukey_fit(inputs)
}

/// Public Rust-callable variant. Same input contract as the `pl_qtukey`
/// expression shim.
///
/// # Input contract
/// `inputs[0..4]`: p, nmeans, df, nranges (f64, each length 1 or n).
/// `inputs[4]`: lower_tail (bool scalar, default true).
pub fn qtukey_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let lower_tail = inputs[4].bool()?.get(0).unwrap_or(true);
    elementwise4(&inputs[0..4], |p, nmeans, df, nranges| {
        qtukey(p, nmeans, df, nranges, lower_tail)
    })
}

/// R `qtukey`: studentized-range distribution quantile function.
#[polars_expr(output_type=Float64)]
fn pl_qtukey(inputs: &[Series]) -> PolarsResult<Series> {
    qtukey_fit(inputs)
}

/// Parse an alternative-hypothesis string, accepting R-style ("two.sided"),
/// Python-style ("two_sided" / "two-sided") and "less" / "greater";
/// anything else falls back to two-sided (mirrors `parametric.rs`'s
/// `parse_alternative`, extended to accept the dotted/underscored spellings
/// post-hoc callers may pass through).
fn parse_alternative(s: &str) -> Alternative {
    match s.to_lowercase().replace(['_', '.'], "-").as_str() {
        "less" => Alternative::Less,
        "greater" => Alternative::Greater,
        _ => Alternative::TwoSided,
    }
}

/// Output dtype for the three pairwise post-hoc tests: a List[Struct] of
/// pairwise comparison rows.
fn pairwise_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("group1".into(), DataType::String),
        Field::new("group2".into(), DataType::String),
        Field::new("n1".into(), DataType::UInt32),
        Field::new("n2".into(), DataType::UInt32),
        Field::new("estimate".into(), DataType::Float64),
        Field::new("statistic".into(), DataType::Float64),
        Field::new("df".into(), DataType::Float64),
        Field::new("p_value".into(), DataType::Float64),
        Field::new("p_adj".into(), DataType::Float64),
        Field::new("ci_lower".into(), DataType::Float64),
        Field::new("ci_upper".into(), DataType::Float64),
    ];
    Ok(Field::new(
        "pairwise".into(),
        DataType::List(Box::new(DataType::Struct(fields))),
    ))
}

/// Build the one-row List[Struct] output from raw pairwise-comparison
/// columns, mirroring `regression.rs`'s `summary_output`.
#[allow(clippy::too_many_arguments)]
fn pairwise_output(
    group1: Vec<String>,
    group2: Vec<String>,
    n1: Vec<u32>,
    n2: Vec<u32>,
    estimate: Vec<f64>,
    statistic: Vec<f64>,
    df: Vec<f64>,
    p_value: Vec<f64>,
    p_adj: Vec<f64>,
    ci_lower: Vec<f64>,
    ci_upper: Vec<f64>,
) -> PolarsResult<Series> {
    let n = group1.len();

    let group1_s = Series::new("group1".into(), group1);
    let group2_s = Series::new("group2".into(), group2);
    let n1_s = Series::new("n1".into(), n1);
    let n2_s = Series::new("n2".into(), n2);
    let estimate_s = Series::new("estimate".into(), estimate);
    let statistic_s = Series::new("statistic".into(), statistic);
    let df_s = Series::new("df".into(), df);
    let p_value_s = Series::new("p_value".into(), p_value);
    let p_adj_s = Series::new("p_adj".into(), p_adj);
    let ci_lower_s = Series::new("ci_lower".into(), ci_lower);
    let ci_upper_s = Series::new("ci_upper".into(), ci_upper);

    let struct_ca = StructChunked::from_series(
        "item".into(),
        n,
        [
            &group1_s,
            &group2_s,
            &n1_s,
            &n2_s,
            &estimate_s,
            &statistic_s,
            &df_s,
            &p_value_s,
            &p_adj_s,
            &ci_lower_s,
            &ci_upper_s,
        ]
        .into_iter(),
    )?;

    let struct_series = struct_ca.into_series();
    Ok(Series::new("pairwise".into(), &[struct_series]))
}

/// Empty pairwise output (degenerate input: fewer than 2 usable groups, or
/// no rows at all) — an empty list, never an exception.
fn pairwise_empty_output() -> PolarsResult<Series> {
    pairwise_output(
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
    )
}

/// Convert an upstream [`PairwiseResult<String>`] into the pairwise output
/// Series. `n1`/`n2` are looked up per comparison from `result.groups` /
/// `result.group_sizes`; `Option<f64>` fields (`df`, `conf_low`,
/// `conf_high`) become NaN when `None`.
fn from_pairwise_result(result: &PairwiseResult<String>) -> PolarsResult<Series> {
    let size_of = |g: &str| -> u32 {
        result
            .groups
            .iter()
            .position(|x| x == g)
            .map(|i| result.group_sizes[i] as u32)
            .unwrap_or(0)
    };

    let group1: Vec<String> = result
        .comparisons
        .iter()
        .map(|c| c.group1.clone())
        .collect();
    let group2: Vec<String> = result
        .comparisons
        .iter()
        .map(|c| c.group2.clone())
        .collect();
    let n1: Vec<u32> = result
        .comparisons
        .iter()
        .map(|c| size_of(&c.group1))
        .collect();
    let n2: Vec<u32> = result
        .comparisons
        .iter()
        .map(|c| size_of(&c.group2))
        .collect();
    let estimate: Vec<f64> = result.comparisons.iter().map(|c| c.estimate).collect();
    let statistic: Vec<f64> = result.comparisons.iter().map(|c| c.statistic).collect();
    let df: Vec<f64> = result
        .comparisons
        .iter()
        .map(|c| c.df.unwrap_or(f64::NAN))
        .collect();
    let p_value: Vec<f64> = result.comparisons.iter().map(|c| c.p_value).collect();
    let p_adj: Vec<f64> = result.comparisons.iter().map(|c| c.p_adj).collect();
    let ci_lower: Vec<f64> = result
        .comparisons
        .iter()
        .map(|c| c.conf_low.unwrap_or(f64::NAN))
        .collect();
    let ci_upper: Vec<f64> = result
        .comparisons
        .iter()
        .map(|c| c.conf_high.unwrap_or(f64::NAN))
        .collect();

    pairwise_output(
        group1, group2, n1, n2, estimate, statistic, df, p_value, p_adj, ci_lower, ci_upper,
    )
}

/// Build `(values, groups)` from a values column (f64, nullable) and a
/// groups column (String, nullable), dropping rows where the value is null,
/// NaN, or ±Inf, or the group is null (R's `NA` row-drop semantics for the
/// pairwise tests). Dropping non-finite values (not just NaN) keeps a single
/// `Inf`/`-Inf` row from reaching `anofox_statistics::{tukey_hsd,
/// pairwise_t_test, dunn_test}`'s `ensure_finite` check, which otherwise
/// rejects the whole call and silently empties every pairwise comparison,
/// not just the ones touching the offending row.
fn long_format(values: &Series, groups: &Series) -> PolarsResult<(Vec<f64>, Vec<String>)> {
    let v_ca = values.f64()?;
    let g_ca = groups.str()?;
    let mut out_values = Vec::new();
    let mut out_groups = Vec::new();
    for i in 0..v_ca.len() {
        if let (Some(v), Some(g)) = (v_ca.get(i), g_ca.get(i)) {
            if v.is_finite() {
                out_values.push(v);
                out_groups.push(g.to_string());
            }
        }
    }
    Ok((out_values, out_groups))
}

/// Public Rust-callable variant. Same input contract as the `pl_tukey_hsd`
/// expression shim.
///
/// # Input contract
/// `inputs[0]`: values (f64, nullable). `inputs[1]`: groups (String,
/// nullable). `inputs[2]`: conf_level (f64 scalar, default 0.95).
///
/// Rows with a null, NaN, or non-finite (±Inf) value, or a null group, are
/// dropped first (see [`long_format`]). Fewer than 2 usable groups, or an
/// invalid `conf_level`, give an empty list — never an exception.
pub fn tukey_hsd_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let conf_level = inputs[2].f64()?.get(0).unwrap_or(0.95);
    let (values, groups) = long_format(&inputs[0], &inputs[1])?;
    match tukey_hsd(&values, &groups, conf_level) {
        Ok(result) => from_pairwise_result(&result),
        Err(_) => pairwise_empty_output(),
    }
}

/// Tukey's honest significant difference: all pairwise mean differences,
/// matching R's `TukeyHSD(aov(values ~ groups))`.
#[polars_expr(output_type_func=pairwise_output_dtype)]
fn pl_tukey_hsd(inputs: &[Series]) -> PolarsResult<Series> {
    tukey_hsd_fit(inputs)
}

/// Public Rust-callable variant. Same input contract as the
/// `pl_pairwise_t_test` expression shim.
///
/// # Input contract
/// `inputs[0]`: values (f64, nullable). `inputs[1]`: groups (String,
/// nullable). `inputs[2]`: pooled_sd (bool scalar, default true).
/// `inputs[3]`: p_adjust_method (String scalar, default "holm").
/// `inputs[4]`: alternative (String scalar, default "two-sided").
///
/// Rows with a null, NaN, or non-finite (±Inf) value, or a null group, are
/// dropped first (see [`long_format`]).
pub fn pairwise_t_test_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let pooled_sd = inputs[2].bool()?.get(0).unwrap_or(true);
    let method_str = inputs[3].str()?.get(0).unwrap_or("holm");
    let method = parse_p_adjust_method(method_str)?;
    let alt_str = inputs[4].str()?.get(0).unwrap_or("two-sided");
    let alternative = parse_alternative(alt_str);

    let (values, groups) = long_format(&inputs[0], &inputs[1])?;
    match pairwise_t_test(&values, &groups, pooled_sd, method, alternative) {
        Ok(result) => from_pairwise_result(&result),
        Err(_) => pairwise_empty_output(),
    }
}

/// Pairwise t-tests with p-value adjustment, matching R's
/// `pairwise.t.test(values, groups, p.adjust.method, pool.sd, alternative)`.
#[polars_expr(output_type_func=pairwise_output_dtype)]
fn pl_pairwise_t_test(inputs: &[Series]) -> PolarsResult<Series> {
    pairwise_t_test_fit(inputs)
}

/// Public Rust-callable variant. Same input contract as the `pl_dunn_test`
/// expression shim.
///
/// # Input contract
/// `inputs[0]`: values (f64, nullable). `inputs[1]`: groups (String,
/// nullable). `inputs[2]`: p_adjust_method (String scalar, default "holm").
///
/// Rows with a null, NaN, or non-finite (±Inf) value, or a null group, are
/// dropped first (see [`long_format`]).
pub fn dunn_test_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let method_str = inputs[2].str()?.get(0).unwrap_or("holm");
    let method = parse_p_adjust_method(method_str)?;

    let (values, groups) = long_format(&inputs[0], &inputs[1])?;
    match dunn_test(&values, &groups, method) {
        Ok(result) => from_pairwise_result(&result),
        Err(_) => pairwise_empty_output(),
    }
}

/// Dunn's (1964) pairwise rank-mean comparison after Kruskal-Wallis,
/// matching `dunn.test::dunn.test(values, groups, method, altp = TRUE)`.
#[polars_expr(output_type_func=pairwise_output_dtype)]
fn pl_dunn_test(inputs: &[Series]) -> PolarsResult<Series> {
    dunn_test_fit(inputs)
}
