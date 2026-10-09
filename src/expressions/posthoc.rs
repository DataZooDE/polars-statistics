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

use anofox_statistics::{p_adjust, PAdjustMethod};

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
