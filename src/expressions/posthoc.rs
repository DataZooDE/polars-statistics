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

use anofox_statistics::{p_adjust, ptukey, qtukey, PAdjustMethod};

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
