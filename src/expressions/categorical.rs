//! Categorical statistical test expressions.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use anofox_statistics::{
    binom_test, chisq_goodness_of_fit, chisq_test, cohen_kappa, contingency_coef, cramers_v,
    fisher_exact, g_test, mcnemar_exact, mcnemar_test, phi_coefficient, prop_test_one,
    prop_test_two, Alternative,
};

use crate::expressions::output_types::{
    association_output_dtype, chisq_output_dtype, proportion_output_dtype, stats_output_dtype,
};

/// Helper to parse alternative hypothesis
fn parse_alternative(s: &str) -> Alternative {
    match s.to_lowercase().as_str() {
        "less" => Alternative::Less,
        "greater" => Alternative::Greater,
        _ => Alternative::TwoSided,
    }
}

/// Helper to create proportion output
fn proportion_output(
    estimate: f64,
    statistic: f64,
    p_value: f64,
    ci_lower: f64,
    ci_upper: f64,
    n: u32,
    name: &str,
) -> PolarsResult<Series> {
    let estimate_s = Series::new("estimate".into(), &[estimate]);
    let statistic_s = Series::new("statistic".into(), &[statistic]);
    let p_value_s = Series::new("p_value".into(), &[p_value]);
    let ci_lower_s = Series::new("ci_lower".into(), &[ci_lower]);
    let ci_upper_s = Series::new("ci_upper".into(), &[ci_upper]);
    let n_s = Series::new("n".into(), &[n]);

    let df = StructChunked::from_series(
        name.into(),
        1,
        [
            &estimate_s,
            &statistic_s,
            &p_value_s,
            &ci_lower_s,
            &ci_upper_s,
            &n_s,
        ]
        .into_iter(),
    )?;
    Ok(df.into_series())
}

/// Helper to create chi-square output
fn chisq_output(statistic: f64, p_value: f64, df: f64, n: u32, name: &str) -> PolarsResult<Series> {
    let statistic_s = Series::new("statistic".into(), &[statistic]);
    let p_value_s = Series::new("p_value".into(), &[p_value]);
    let df_s = Series::new("df".into(), &[df]);
    let n_s = Series::new("n".into(), &[n]);

    let result = StructChunked::from_series(
        name.into(),
        1,
        [&statistic_s, &p_value_s, &df_s, &n_s].into_iter(),
    )?;
    Ok(result.into_series())
}

/// Helper to create association output
fn association_output(
    estimate: f64,
    statistic: f64,
    p_value: f64,
    name: &str,
) -> PolarsResult<Series> {
    let estimate_s = Series::new("estimate".into(), &[estimate]);
    let statistic_s = Series::new("statistic".into(), &[statistic]);
    let p_value_s = Series::new("p_value".into(), &[p_value]);

    let result = StructChunked::from_series(
        name.into(),
        1,
        [&estimate_s, &statistic_s, &p_value_s].into_iter(),
    )?;
    Ok(result.into_series())
}

/// Exact binomial test
#[polars_expr(output_type_func=proportion_output_dtype)]
fn pl_binom_test(inputs: &[Series]) -> PolarsResult<Series> {
    let successes = inputs[0].u32()?.get(0).unwrap_or(0) as usize;
    let n = inputs[1].u32()?.get(0).unwrap_or(1) as usize;
    let p0 = inputs[2].f64()?.get(0).unwrap_or(0.5);
    let alt_str = inputs[3].str()?.get(0).unwrap_or("two-sided");

    let alternative = parse_alternative(alt_str);

    match binom_test(successes, n, p0, alternative) {
        Ok(result) => proportion_output(
            result.estimate,
            result.estimate, // For binom test, estimate is the test value
            result.p_value,
            result.conf_int_lower,
            result.conf_int_upper,
            n as u32,
            "binom_test",
        ),
        Err(_) => proportion_output(
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            0,
            "binom_test",
        ),
    }
}

/// One-sample proportion test
#[polars_expr(output_type_func=proportion_output_dtype)]
fn pl_prop_test_one(inputs: &[Series]) -> PolarsResult<Series> {
    let successes = inputs[0].u32()?.get(0).unwrap_or(0) as usize;
    let n = inputs[1].u32()?.get(0).unwrap_or(1) as usize;
    let p0 = inputs[2].f64()?.get(0).unwrap_or(0.5);
    let alt_str = inputs[3].str()?.get(0).unwrap_or("two-sided");

    let alternative = parse_alternative(alt_str);

    match prop_test_one(successes, n, p0, alternative) {
        Ok(result) => proportion_output(
            result.estimate[0],
            result.statistic,
            result.p_value,
            result.conf_int_lower,
            result.conf_int_upper,
            n as u32,
            "prop_test_one",
        ),
        Err(_) => proportion_output(
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            0,
            "prop_test_one",
        ),
    }
}

/// Two-sample proportion test
#[polars_expr(output_type_func=proportion_output_dtype)]
fn pl_prop_test_two(inputs: &[Series]) -> PolarsResult<Series> {
    let successes1 = inputs[0].u32()?.get(0).unwrap_or(0) as usize;
    let n1 = inputs[1].u32()?.get(0).unwrap_or(1) as usize;
    let successes2 = inputs[2].u32()?.get(0).unwrap_or(0) as usize;
    let n2 = inputs[3].u32()?.get(0).unwrap_or(1) as usize;
    let alt_str = inputs[4].str()?.get(0).unwrap_or("two-sided");
    let correction = inputs[5].bool()?.get(0).unwrap_or(false);

    let alternative = parse_alternative(alt_str);

    match prop_test_two([successes1, successes2], [n1, n2], alternative, correction) {
        Ok(result) => {
            // Difference in proportions
            let estimate = result.estimate[0] - result.estimate[1];
            proportion_output(
                estimate,
                result.statistic,
                result.p_value,
                result.conf_int_lower,
                result.conf_int_upper,
                (n1 + n2) as u32,
                "prop_test_two",
            )
        }
        Err(_) => proportion_output(
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            0,
            "prop_test_two",
        ),
    }
}

/// Chi-square test for contingency table
#[polars_expr(output_type_func=chisq_output_dtype)]
fn pl_chisq_test(inputs: &[Series]) -> PolarsResult<Series> {
    // Expects a flattened contingency table with dimensions
    let data = inputs[0].u32()?;
    let n_rows = inputs[1].u32()?.get(0).unwrap_or(2) as usize;
    let n_cols = inputs[2].u32()?.get(0).unwrap_or(2) as usize;
    let correction = inputs[3].bool()?.get(0).unwrap_or(false);

    let data_vec: Vec<usize> = data.into_no_null_iter().map(|v| v as usize).collect();

    // Reshape to 2D
    let mut observed: Vec<Vec<usize>> = Vec::new();
    for i in 0..n_rows {
        let mut row = Vec::new();
        for j in 0..n_cols {
            let idx = i * n_cols + j;
            if idx < data_vec.len() {
                row.push(data_vec[idx]);
            } else {
                row.push(0);
            }
        }
        observed.push(row);
    }

    match chisq_test(&observed, correction) {
        Ok(result) => {
            let total: usize = observed.iter().map(|r| r.iter().sum::<usize>()).sum();
            chisq_output(
                result.statistic,
                result.p_value,
                result.df,
                total as u32,
                "chisq_test",
            )
        }
        Err(_) => chisq_output(f64::NAN, f64::NAN, f64::NAN, 0, "chisq_test"),
    }
}

/// Chi-square goodness-of-fit test
#[polars_expr(output_type_func=chisq_output_dtype)]
fn pl_chisq_goodness_of_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let observed = inputs[0].u32()?;
    let has_expected = inputs[1].bool()?.get(0).unwrap_or(false);

    let observed_vec: Vec<usize> = observed.into_no_null_iter().map(|v| v as usize).collect();

    let expected_opt: Option<Vec<f64>> = if has_expected && inputs.len() > 2 {
        Some(inputs[2].f64()?.into_no_null_iter().collect())
    } else {
        None
    };

    let expected_slice = expected_opt.as_deref();

    match chisq_goodness_of_fit(&observed_vec, expected_slice) {
        Ok(result) => {
            let total: usize = observed_vec.iter().sum();
            chisq_output(
                result.statistic,
                result.p_value,
                result.df,
                total as u32,
                "chisq_gof",
            )
        }
        Err(_) => chisq_output(f64::NAN, f64::NAN, f64::NAN, 0, "chisq_gof"),
    }
}

/// G-test (likelihood ratio test)
#[polars_expr(output_type_func=chisq_output_dtype)]
fn pl_g_test(inputs: &[Series]) -> PolarsResult<Series> {
    let data = inputs[0].u32()?;
    let n_rows = inputs[1].u32()?.get(0).unwrap_or(2) as usize;
    let n_cols = inputs[2].u32()?.get(0).unwrap_or(2) as usize;

    let data_vec: Vec<usize> = data.into_no_null_iter().map(|v| v as usize).collect();

    let mut observed: Vec<Vec<usize>> = Vec::new();
    for i in 0..n_rows {
        let mut row = Vec::new();
        for j in 0..n_cols {
            let idx = i * n_cols + j;
            if idx < data_vec.len() {
                row.push(data_vec[idx]);
            } else {
                row.push(0);
            }
        }
        observed.push(row);
    }

    match g_test(&observed) {
        Ok(result) => {
            let total: usize = observed.iter().map(|r| r.iter().sum::<usize>()).sum();
            chisq_output(
                result.statistic,
                result.p_value,
                result.df,
                total as u32,
                "g_test",
            )
        }
        Err(_) => chisq_output(f64::NAN, f64::NAN, f64::NAN, 0, "g_test"),
    }
}

/// Fisher's exact test for 2x2 tables
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_fisher_exact(inputs: &[Series]) -> PolarsResult<Series> {
    let a = inputs[0].u32()?.get(0).unwrap_or(0) as usize;
    let b = inputs[1].u32()?.get(0).unwrap_or(0) as usize;
    let c = inputs[2].u32()?.get(0).unwrap_or(0) as usize;
    let d = inputs[3].u32()?.get(0).unwrap_or(0) as usize;
    let alt_str = inputs[4].str()?.get(0).unwrap_or("two-sided");

    let table = [[a, b], [c, d]];
    let alternative = parse_alternative(alt_str);

    match fisher_exact(&table, alternative) {
        Ok(result) => {
            let stat_series = Series::new("statistic".into(), vec![result.odds_ratio]);
            let pval_series = Series::new("p_value".into(), vec![result.p_value]);

            StructChunked::from_series(
                "fisher_exact".into(),
                1,
                [&stat_series, &pval_series].into_iter(),
            )
            .map(|ca| ca.into_series())
        }
        Err(_) => {
            let stat_series = Series::new("statistic".into(), vec![f64::NAN]);
            let pval_series = Series::new("p_value".into(), vec![f64::NAN]);

            StructChunked::from_series(
                "fisher_exact".into(),
                1,
                [&stat_series, &pval_series].into_iter(),
            )
            .map(|ca| ca.into_series())
        }
    }
}

/// McNemar's test for paired proportions
#[polars_expr(output_type_func=chisq_output_dtype)]
fn pl_mcnemar_test(inputs: &[Series]) -> PolarsResult<Series> {
    let a = inputs[0].u32()?.get(0).unwrap_or(0) as usize;
    let b = inputs[1].u32()?.get(0).unwrap_or(0) as usize;
    let c = inputs[2].u32()?.get(0).unwrap_or(0) as usize;
    let d = inputs[3].u32()?.get(0).unwrap_or(0) as usize;
    let correction = inputs[4].bool()?.get(0).unwrap_or(false);

    let table = [[a, b], [c, d]];

    match mcnemar_test(&table, correction) {
        Ok(result) => {
            let total = (a + b + c + d) as u32;
            chisq_output(result.statistic, result.p_value, 1.0, total, "mcnemar_test")
        }
        Err(_) => chisq_output(f64::NAN, f64::NAN, f64::NAN, 0, "mcnemar_test"),
    }
}

/// McNemar's exact test
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_mcnemar_exact(inputs: &[Series]) -> PolarsResult<Series> {
    let a = inputs[0].u32()?.get(0).unwrap_or(0) as usize;
    let b = inputs[1].u32()?.get(0).unwrap_or(0) as usize;
    let c = inputs[2].u32()?.get(0).unwrap_or(0) as usize;
    let d = inputs[3].u32()?.get(0).unwrap_or(0) as usize;

    let table = [[a, b], [c, d]];

    match mcnemar_exact(&table) {
        Ok(result) => {
            // Use b and c as the discordant cells statistic
            let stat_series = Series::new("statistic".into(), vec![(result.b + result.c) as f64]);
            let pval_series = Series::new("p_value".into(), vec![result.p_value]);

            StructChunked::from_series(
                "mcnemar_exact".into(),
                1,
                [&stat_series, &pval_series].into_iter(),
            )
            .map(|ca| ca.into_series())
        }
        Err(_) => {
            let stat_series = Series::new("statistic".into(), vec![f64::NAN]);
            let pval_series = Series::new("p_value".into(), vec![f64::NAN]);

            StructChunked::from_series(
                "mcnemar_exact".into(),
                1,
                [&stat_series, &pval_series].into_iter(),
            )
            .map(|ca| ca.into_series())
        }
    }
}

/// Cohen's Kappa for inter-rater agreement
#[polars_expr(output_type_func=association_output_dtype)]
fn pl_cohen_kappa(inputs: &[Series]) -> PolarsResult<Series> {
    let data = inputs[0].u32()?;
    let n_categories = inputs[1].u32()?.get(0).unwrap_or(2) as usize;
    let weighted = inputs[2].bool()?.get(0).unwrap_or(false);

    let data_vec: Vec<usize> = data.into_no_null_iter().map(|v| v as usize).collect();

    // Build confusion matrix
    let mut matrix: Vec<Vec<usize>> = vec![vec![0; n_categories]; n_categories];
    for (idx, &value) in data_vec.iter().enumerate() {
        let i = idx / n_categories;
        let j = idx % n_categories;
        if i < n_categories && j < n_categories {
            matrix[i][j] = value;
        }
    }

    match cohen_kappa(&matrix, weighted) {
        Ok(result) => association_output(result.kappa, result.se, result.p_value, "cohen_kappa"),
        Err(_) => association_output(f64::NAN, f64::NAN, f64::NAN, "cohen_kappa"),
    }
}

/// Cramér's V for association strength
#[polars_expr(output_type_func=association_output_dtype)]
fn pl_cramers_v(inputs: &[Series]) -> PolarsResult<Series> {
    let data = inputs[0].u32()?;
    let n_rows = inputs[1].u32()?.get(0).unwrap_or(2) as usize;
    let n_cols = inputs[2].u32()?.get(0).unwrap_or(2) as usize;

    let data_vec: Vec<usize> = data.into_no_null_iter().map(|v| v as usize).collect();

    let mut observed: Vec<Vec<usize>> = Vec::new();
    for i in 0..n_rows {
        let mut row = Vec::new();
        for j in 0..n_cols {
            let idx = i * n_cols + j;
            if idx < data_vec.len() {
                row.push(data_vec[idx]);
            } else {
                row.push(0);
            }
        }
        observed.push(row);
    }

    match cramers_v(&observed) {
        Ok(result) => association_output(
            result.estimate,
            result.se.unwrap_or(f64::NAN),
            f64::NAN,
            "cramers_v",
        ),
        Err(_) => association_output(f64::NAN, f64::NAN, f64::NAN, "cramers_v"),
    }
}

/// Phi coefficient for 2x2 tables
#[polars_expr(output_type_func=association_output_dtype)]
fn pl_phi_coefficient(inputs: &[Series]) -> PolarsResult<Series> {
    let a = inputs[0].u32()?.get(0).unwrap_or(0) as usize;
    let b = inputs[1].u32()?.get(0).unwrap_or(0) as usize;
    let c = inputs[2].u32()?.get(0).unwrap_or(0) as usize;
    let d = inputs[3].u32()?.get(0).unwrap_or(0) as usize;

    let table = [[a, b], [c, d]];

    match phi_coefficient(&table) {
        Ok(result) => association_output(
            result.estimate,
            result.se.unwrap_or(f64::NAN),
            f64::NAN,
            "phi",
        ),
        Err(_) => association_output(f64::NAN, f64::NAN, f64::NAN, "phi"),
    }
}

/// Contingency coefficient
#[polars_expr(output_type_func=association_output_dtype)]
fn pl_contingency_coef(inputs: &[Series]) -> PolarsResult<Series> {
    let data = inputs[0].u32()?;
    let n_rows = inputs[1].u32()?.get(0).unwrap_or(2) as usize;
    let n_cols = inputs[2].u32()?.get(0).unwrap_or(2) as usize;

    let data_vec: Vec<usize> = data.into_no_null_iter().map(|v| v as usize).collect();

    let mut observed: Vec<Vec<usize>> = Vec::new();
    for i in 0..n_rows {
        let mut row = Vec::new();
        for j in 0..n_cols {
            let idx = i * n_cols + j;
            if idx < data_vec.len() {
                row.push(data_vec[idx]);
            } else {
                row.push(0);
            }
        }
        observed.push(row);
    }

    match contingency_coef(&observed) {
        Ok(result) => association_output(
            result.estimate,
            result.se.unwrap_or(f64::NAN),
            f64::NAN,
            "contingency_coef",
        ),
        Err(_) => association_output(f64::NAN, f64::NAN, f64::NAN, "contingency_coef"),
    }
}
