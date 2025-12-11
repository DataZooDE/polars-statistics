//! Forecast comparison test expressions.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use anofox_statistics::{
    clark_west, diebold_mariano, model_confidence_set, mspe_adjusted_spa, permutation_t_test,
    spa_test, Alternative, LossFunction, MCSStatistic, VarEstimator,
};

use crate::expressions::output_types::{generic_stats_output, stats_output_dtype};

/// Helper to parse loss function from string
fn parse_loss_function(s: &str) -> LossFunction {
    match s.to_lowercase().as_str() {
        "absolute" | "ae" | "mae" => LossFunction::AbsoluteError,
        _ => LossFunction::SquaredError,
    }
}

/// Helper to parse alternative hypothesis from string
fn parse_alternative(s: &str) -> Alternative {
    match s.to_lowercase().as_str() {
        "less" => Alternative::Less,
        "greater" => Alternative::Greater,
        _ => Alternative::TwoSided,
    }
}

/// Helper to parse variance estimator from string
fn parse_var_estimator(s: &str) -> VarEstimator {
    match s.to_lowercase().as_str() {
        "bartlett" => VarEstimator::Bartlett,
        _ => VarEstimator::Acf,
    }
}

/// Diebold-Mariano test for comparing forecast accuracy.
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_diebold_mariano(inputs: &[Series]) -> PolarsResult<Series> {
    let e1 = inputs[0].f64()?;
    let e2 = inputs[1].f64()?;
    let loss_str = inputs[2].str()?.get(0).unwrap_or("squared");
    let h = inputs[3].u32()?.get(0).unwrap_or(1) as usize;
    let alt_str = inputs[4].str()?.get(0).unwrap_or("two-sided");
    let var_est_str = inputs[5].str()?.get(0).unwrap_or("acf");

    let e1_vec: Vec<f64> = e1.into_no_null_iter().collect();
    let e2_vec: Vec<f64> = e2.into_no_null_iter().collect();

    let loss = parse_loss_function(loss_str);
    let alternative = parse_alternative(alt_str);
    let varestimator = parse_var_estimator(var_est_str);

    match diebold_mariano(&e1_vec, &e2_vec, loss, h, alternative, varestimator) {
        Ok(result) => generic_stats_output(result.statistic, result.p_value, "diebold_mariano"),
        Err(_) => generic_stats_output(f64::NAN, f64::NAN, "diebold_mariano"),
    }
}

/// Permutation t-test for comparing two samples.
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_permutation_t_test(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;
    let y = inputs[1].f64()?;
    let alt_str = inputs[2].str()?.get(0).unwrap_or("two-sided");
    let n_perm = inputs[3].u32()?.get(0).unwrap_or(999) as usize;
    let seed = inputs[4].u64()?.get(0);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();

    let alternative = parse_alternative(alt_str);

    match permutation_t_test(&x_vec, &y_vec, alternative, n_perm, seed) {
        Ok(result) => generic_stats_output(result.statistic, result.p_value, "permutation_t_test"),
        Err(_) => generic_stats_output(f64::NAN, f64::NAN, "permutation_t_test"),
    }
}

/// Clark-West test for nested model comparison.
#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_clark_west(inputs: &[Series]) -> PolarsResult<Series> {
    let e1 = inputs[0].f64()?;
    let e2 = inputs[1].f64()?;
    let h = inputs[2].u32()?.get(0).unwrap_or(1) as usize;

    let e1_vec: Vec<f64> = e1.into_no_null_iter().collect();
    let e2_vec: Vec<f64> = e2.into_no_null_iter().collect();

    match clark_west(&e1_vec, &e2_vec, h) {
        Ok(result) => generic_stats_output(result.statistic, result.p_value, "clark_west"),
        Err(_) => generic_stats_output(f64::NAN, f64::NAN, "clark_west"),
    }
}

/// SPA output type with additional fields
fn spa_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("statistic".into(), DataType::Float64),
        Field::new("p_value_consistent".into(), DataType::Float64),
        Field::new("p_value_upper".into(), DataType::Float64),
        Field::new("best_model_idx".into(), DataType::UInt32),
    ];
    Ok(Field::new("spa_result".into(), DataType::Struct(fields)))
}

/// Superior Predictive Ability (SPA) test.
#[polars_expr(output_type_func=spa_output_dtype)]
fn pl_spa_test(inputs: &[Series]) -> PolarsResult<Series> {
    let benchmark = inputs[0].f64()?;
    let n_bootstrap = inputs[1].u32()?.get(0).unwrap_or(999) as usize;
    let block_length = inputs[2].f64()?.get(0).unwrap_or(5.0);
    let seed = inputs[3].u64()?.get(0);

    let benchmark_vec: Vec<f64> = benchmark.into_no_null_iter().collect();

    // Remaining inputs are model losses
    let model_losses: Vec<Vec<f64>> = inputs[4..]
        .iter()
        .map(|s| s.f64().map(|ca| ca.into_no_null_iter().collect()))
        .collect::<PolarsResult<Vec<_>>>()?;

    match spa_test(
        &benchmark_vec,
        &model_losses,
        n_bootstrap,
        block_length,
        seed,
    ) {
        Ok(result) => {
            let statistic = Series::new("statistic".into(), &[result.statistic]);
            let p_consistent =
                Series::new("p_value_consistent".into(), &[result.p_value_consistent]);
            let p_upper = Series::new("p_value_upper".into(), &[result.p_value_upper]);
            let best_idx = Series::new(
                "best_model_idx".into(),
                &[result.best_model_idx.map(|i| i as u32)],
            );

            StructChunked::from_series(
                "spa_result".into(),
                1,
                [&statistic, &p_consistent, &p_upper, &best_idx].into_iter(),
            )
            .map(|ca| ca.into_series())
        }
        Err(_) => {
            let statistic = Series::new("statistic".into(), &[f64::NAN]);
            let p_consistent = Series::new("p_value_consistent".into(), &[f64::NAN]);
            let p_upper = Series::new("p_value_upper".into(), &[f64::NAN]);
            let best_idx: Series = Series::new("best_model_idx".into(), &[None::<u32>]);

            StructChunked::from_series(
                "spa_result".into(),
                1,
                [&statistic, &p_consistent, &p_upper, &best_idx].into_iter(),
            )
            .map(|ca| ca.into_series())
        }
    }
}

/// MCS output type
fn mcs_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new(
            "included_models".into(),
            DataType::List(Box::new(DataType::UInt32)),
        ),
        Field::new("mcs_p_value".into(), DataType::Float64),
    ];
    Ok(Field::new("mcs_result".into(), DataType::Struct(fields)))
}

/// Model Confidence Set (MCS) test.
#[polars_expr(output_type_func=mcs_output_dtype)]
fn pl_model_confidence_set(inputs: &[Series]) -> PolarsResult<Series> {
    let alpha = inputs[0].f64()?.get(0).unwrap_or(0.1);
    let stat_str = inputs[1].str()?.get(0).unwrap_or("range");
    let n_bootstrap = inputs[2].u32()?.get(0).unwrap_or(999) as usize;
    let block_length = inputs[3].f64()?.get(0).unwrap_or(5.0);
    let seed = inputs[4].u64()?.get(0);

    let statistic = match stat_str.to_lowercase().as_str() {
        "max" => MCSStatistic::Max,
        _ => MCSStatistic::Range,
    };

    // Remaining inputs are model losses
    let model_losses: Vec<Vec<f64>> = inputs[5..]
        .iter()
        .map(|s| s.f64().map(|ca| ca.into_no_null_iter().collect()))
        .collect::<PolarsResult<Vec<_>>>()?;

    match model_confidence_set(
        &model_losses,
        alpha,
        statistic,
        n_bootstrap,
        block_length,
        seed,
    ) {
        Ok(result) => {
            let included: Vec<u32> = result.included_models.iter().map(|&i| i as u32).collect();
            // Create list series by wrapping inner Series
            let inner_series = Series::new("".into(), &included);
            let included_series = Series::new("included_models".into(), [inner_series]);
            let p_value = Series::new("mcs_p_value".into(), &[result.mcs_p_value]);

            StructChunked::from_series(
                "mcs_result".into(),
                1,
                [&included_series, &p_value].into_iter(),
            )
            .map(|ca| ca.into_series())
        }
        Err(_) => {
            let inner_series = Series::new("".into(), Vec::<u32>::new());
            let included_series = Series::new("included_models".into(), [inner_series]);
            let p_value = Series::new("mcs_p_value".into(), &[f64::NAN]);

            StructChunked::from_series(
                "mcs_result".into(),
                1,
                [&included_series, &p_value].into_iter(),
            )
            .map(|ca| ca.into_series())
        }
    }
}

/// MSPE-Adjusted SPA test for nested models.
#[polars_expr(output_type_func=spa_output_dtype)]
fn pl_mspe_adjusted(inputs: &[Series]) -> PolarsResult<Series> {
    let benchmark = inputs[0].f64()?;
    let n_bootstrap = inputs[1].u32()?.get(0).unwrap_or(999) as usize;
    let block_length = inputs[2].f64()?.get(0).unwrap_or(5.0);
    let seed = inputs[3].u64()?.get(0);

    let benchmark_vec: Vec<f64> = benchmark.into_no_null_iter().collect();

    // Remaining inputs are model errors
    let model_errors: Vec<Vec<f64>> = inputs[4..]
        .iter()
        .map(|s| s.f64().map(|ca| ca.into_no_null_iter().collect()))
        .collect::<PolarsResult<Vec<_>>>()?;

    match mspe_adjusted_spa(
        &benchmark_vec,
        &model_errors,
        n_bootstrap,
        block_length,
        seed,
    ) {
        Ok(result) => {
            let statistic = Series::new("statistic".into(), &[result.statistic]);
            let p_consistent =
                Series::new("p_value_consistent".into(), &[result.p_value_consistent]);
            let p_upper = Series::new("p_value_upper".into(), &[result.p_value_upper]);
            let best_idx = Series::new(
                "best_model_idx".into(),
                &[result.best_model_idx.map(|i| i as u32)],
            );

            StructChunked::from_series(
                "spa_result".into(),
                1,
                [&statistic, &p_consistent, &p_upper, &best_idx].into_iter(),
            )
            .map(|ca| ca.into_series())
        }
        Err(_) => {
            let statistic = Series::new("statistic".into(), &[f64::NAN]);
            let p_consistent = Series::new("p_value_consistent".into(), &[f64::NAN]);
            let p_upper = Series::new("p_value_upper".into(), &[f64::NAN]);
            let best_idx: Series = Series::new("best_model_idx".into(), &[None::<u32>]);

            StructChunked::from_series(
                "spa_result".into(),
                1,
                [&statistic, &p_consistent, &p_upper, &best_idx].into_iter(),
            )
            .map(|ca| ca.into_series())
        }
    }
}
