//! Regression model expressions for Polars.
//!
//! These expressions allow fitting regression models within group_by and over operations.

use faer::{Col, Mat};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use anofox_regression::diagnostics::{
    check_binary_separation, check_count_sparsity, condition_diagnostic, ConditionSeverity,
    SeparationCheck, SeparationType,
};
use anofox_regression::solvers::{
    AidClassifier, AlmDistribution, AlmRegressor, AnomalyType, BinomialRegressor, BlsRegressor,
    DemandDistribution, DemandType, ElasticNetRegressor, FittedRegressor, InformationCriterion,
    IsotonicRegressor, LmDynamicRegressor, NegativeBinomialRegressor, OlsRegressor,
    PoissonRegressor, QuantileRegressor, Regressor, RidgeRegressor, RlsRegressor, TweedieRegressor,
    WlsRegressor,
};
use anofox_regression::IntervalType;

/// Result type for build_xy_with_null_policy: (X_fit, y, valid_mask, X_pred)
type XyNullPolicyResult = (Mat<f64>, Col<f64>, Vec<bool>, Mat<f64>);

// ============================================================================
// Output Type Definitions
// ============================================================================

/// Output type for linear regression models (OLS, Ridge, ElasticNet, WLS, RLS, BLS)
fn linear_regression_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("intercept".into(), DataType::Float64),
        Field::new(
            "coefficients".into(),
            DataType::List(Box::new(DataType::Float64)),
        ),
        Field::new("r_squared".into(), DataType::Float64),
        Field::new("adj_r_squared".into(), DataType::Float64),
        Field::new("mse".into(), DataType::Float64),
        Field::new("rmse".into(), DataType::Float64),
        Field::new("f_statistic".into(), DataType::Float64),
        Field::new("f_pvalue".into(), DataType::Float64),
        Field::new("aic".into(), DataType::Float64),
        Field::new("bic".into(), DataType::Float64),
        Field::new("n_observations".into(), DataType::UInt32),
    ];
    Ok(Field::new("regression".into(), DataType::Struct(fields)))
}

/// Output type for GLM models (Logistic, Poisson, NegBin, Tweedie, Probit, Cloglog)
fn glm_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("intercept".into(), DataType::Float64),
        Field::new(
            "coefficients".into(),
            DataType::List(Box::new(DataType::Float64)),
        ),
        Field::new("aic".into(), DataType::Float64),
        Field::new("bic".into(), DataType::Float64),
        Field::new("n_observations".into(), DataType::UInt32),
    ];
    Ok(Field::new("glm".into(), DataType::Struct(fields)))
}

/// Output type for Quantile regression
fn quantile_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("intercept".into(), DataType::Float64),
        Field::new(
            "coefficients".into(),
            DataType::List(Box::new(DataType::Float64)),
        ),
        Field::new("tau".into(), DataType::Float64),
        Field::new("pseudo_r_squared".into(), DataType::Float64),
        Field::new("check_loss".into(), DataType::Float64),
        Field::new("n_observations".into(), DataType::UInt32),
    ];
    Ok(Field::new("quantile".into(), DataType::Struct(fields)))
}

/// Output type for Isotonic regression
fn isotonic_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("r_squared".into(), DataType::Float64),
        Field::new("increasing".into(), DataType::Boolean),
        Field::new(
            "fitted_values".into(),
            DataType::List(Box::new(DataType::Float64)),
        ),
        Field::new("n_observations".into(), DataType::UInt32),
    ];
    Ok(Field::new("isotonic".into(), DataType::Struct(fields)))
}

/// Output type for condition number diagnostics
fn condition_number_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("condition_number".into(), DataType::Float64),
        Field::new("condition_number_xtx".into(), DataType::Float64),
        Field::new(
            "singular_values".into(),
            DataType::List(Box::new(DataType::Float64)),
        ),
        Field::new(
            "condition_indices".into(),
            DataType::List(Box::new(DataType::Float64)),
        ),
        Field::new("severity".into(), DataType::String),
        Field::new("warning".into(), DataType::String),
    ];
    Ok(Field::new("condition".into(), DataType::Struct(fields)))
}

/// Output type for separation check diagnostics
fn separation_check_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("has_separation".into(), DataType::Boolean),
        Field::new(
            "separated_predictors".into(),
            DataType::List(Box::new(DataType::UInt32)),
        ),
        Field::new(
            "separation_types".into(),
            DataType::List(Box::new(DataType::String)),
        ),
        Field::new("warning".into(), DataType::String),
    ];
    Ok(Field::new("separation".into(), DataType::Struct(fields)))
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Build X matrix and y vector from input Series
fn build_xy_data(
    inputs: &[Series],
    y_idx: usize,
    x_start_idx: usize,
) -> PolarsResult<(Mat<f64>, Col<f64>)> {
    let y_series = inputs[y_idx].f64()?;
    let n_rows = y_series.len();
    let n_features = inputs.len() - x_start_idx;

    // Build y vector
    let y_vec: Vec<f64> = y_series.into_no_null_iter().collect();
    let y = Col::from_fn(y_vec.len(), |i| y_vec[i]);

    // Build X matrix (row-major order for faer)
    let x = Mat::from_fn(n_rows, n_features, |row, col| {
        inputs[x_start_idx + col]
            .f64()
            .ok()
            .and_then(|ca| ca.get(row))
            .unwrap_or(f64::NAN)
    });

    Ok((x, y))
}

/// Create linear regression output struct
#[allow(clippy::too_many_arguments)]
fn linear_output(
    intercept: Option<f64>,
    coefficients: &[f64],
    r_squared: f64,
    adj_r_squared: f64,
    mse: f64,
    rmse: f64,
    f_statistic: f64,
    f_pvalue: f64,
    aic: f64,
    bic: f64,
    n_obs: usize,
) -> PolarsResult<Series> {
    let intercept_s = Series::new("intercept".into(), &[intercept.unwrap_or(f64::NAN)]);
    // Create List series for coefficients
    let coef_inner = Series::new("item".into(), coefficients);
    let coef_s = Series::new("coefficients".into(), &[coef_inner]);
    let r2_s = Series::new("r_squared".into(), &[r_squared]);
    let adj_r2_s = Series::new("adj_r_squared".into(), &[adj_r_squared]);
    let mse_s = Series::new("mse".into(), &[mse]);
    let rmse_s = Series::new("rmse".into(), &[rmse]);
    let f_s = Series::new("f_statistic".into(), &[f_statistic]);
    let fp_s = Series::new("f_pvalue".into(), &[f_pvalue]);
    let aic_s = Series::new("aic".into(), &[aic]);
    let bic_s = Series::new("bic".into(), &[bic]);
    let n_s = Series::new("n_observations".into(), &[n_obs as u32]);

    StructChunked::from_series(
        "regression".into(),
        1,
        [
            &intercept_s,
            &coef_s,
            &r2_s,
            &adj_r2_s,
            &mse_s,
            &rmse_s,
            &f_s,
            &fp_s,
            &aic_s,
            &bic_s,
            &n_s,
        ]
        .into_iter(),
    )
    .map(|ca| ca.into_series())
}

/// Create GLM output struct
fn glm_output(
    intercept: Option<f64>,
    coefficients: &[f64],
    aic: f64,
    bic: f64,
    n_obs: usize,
) -> PolarsResult<Series> {
    let intercept_s = Series::new("intercept".into(), &[intercept.unwrap_or(f64::NAN)]);
    // Create List series for coefficients
    let coef_inner = Series::new("item".into(), coefficients);
    let coef_s = Series::new("coefficients".into(), &[coef_inner]);
    let aic_s = Series::new("aic".into(), &[aic]);
    let bic_s = Series::new("bic".into(), &[bic]);
    let n_s = Series::new("n_observations".into(), &[n_obs as u32]);

    StructChunked::from_series(
        "glm".into(),
        1,
        [&intercept_s, &coef_s, &aic_s, &bic_s, &n_s].into_iter(),
    )
    .map(|ca| ca.into_series())
}

/// Create NaN output for linear models on error
fn linear_nan_output() -> PolarsResult<Series> {
    linear_output(
        None,
        &[],
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        0,
    )
}

/// Create NaN output for GLM models on error
fn glm_nan_output() -> PolarsResult<Series> {
    glm_output(None, &[], f64::NAN, f64::NAN, 0)
}

/// Create Quantile regression output struct
fn quantile_output(
    intercept: Option<f64>,
    coefficients: &[f64],
    tau: f64,
    pseudo_r_squared: f64,
    check_loss: f64,
    n_obs: usize,
) -> PolarsResult<Series> {
    let intercept_s = Series::new("intercept".into(), &[intercept.unwrap_or(f64::NAN)]);
    let coef_inner = Series::new("item".into(), coefficients);
    let coef_s = Series::new("coefficients".into(), &[coef_inner]);
    let tau_s = Series::new("tau".into(), &[tau]);
    let pseudo_r2_s = Series::new("pseudo_r_squared".into(), &[pseudo_r_squared]);
    let check_loss_s = Series::new("check_loss".into(), &[check_loss]);
    let n_s = Series::new("n_observations".into(), &[n_obs as u32]);

    StructChunked::from_series(
        "quantile".into(),
        1,
        [
            &intercept_s,
            &coef_s,
            &tau_s,
            &pseudo_r2_s,
            &check_loss_s,
            &n_s,
        ]
        .into_iter(),
    )
    .map(|ca| ca.into_series())
}

/// Create NaN output for Quantile models on error
fn quantile_nan_output(tau: f64) -> PolarsResult<Series> {
    quantile_output(None, &[], tau, f64::NAN, f64::NAN, 0)
}

/// Create Isotonic regression output struct
fn isotonic_output(
    r_squared: f64,
    increasing: bool,
    fitted_values: &[f64],
    n_obs: usize,
) -> PolarsResult<Series> {
    let r2_s = Series::new("r_squared".into(), &[r_squared]);
    let inc_s = Series::new("increasing".into(), &[increasing]);
    let fitted_inner = Series::new("item".into(), fitted_values);
    let fitted_s = Series::new("fitted_values".into(), &[fitted_inner]);
    let n_s = Series::new("n_observations".into(), &[n_obs as u32]);

    StructChunked::from_series(
        "isotonic".into(),
        1,
        [&r2_s, &inc_s, &fitted_s, &n_s].into_iter(),
    )
    .map(|ca| ca.into_series())
}

/// Create NaN output for Isotonic models on error
fn isotonic_nan_output(increasing: bool) -> PolarsResult<Series> {
    isotonic_output(f64::NAN, increasing, &[], 0)
}

/// Extract coefficients as Vec<f64>
fn col_to_vec(col: &Col<f64>) -> Vec<f64> {
    (0..col.nrows()).map(|i| col[i]).collect()
}

/// Output type for coefficient summary (tidy format)
fn summary_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("term".into(), DataType::String),
        Field::new("estimate".into(), DataType::Float64),
        Field::new("std_error".into(), DataType::Float64),
        Field::new("statistic".into(), DataType::Float64),
        Field::new("p_value".into(), DataType::Float64),
    ];
    Ok(Field::new(
        "summary".into(),
        DataType::List(Box::new(DataType::Struct(fields))),
    ))
}

/// Create coefficient summary output as List[Struct] (tidy format)
fn summary_output(
    terms: Vec<String>,
    estimates: Vec<f64>,
    std_errors: Vec<f64>,
    statistics: Vec<f64>,
    p_values: Vec<f64>,
) -> PolarsResult<Series> {
    let n = terms.len();

    // Build individual Series for struct fields
    let term_s = Series::new("term".into(), terms);
    let estimate_s = Series::new("estimate".into(), estimates);
    let std_error_s = Series::new("std_error".into(), std_errors);
    let statistic_s = Series::new("statistic".into(), statistics);
    let p_value_s = Series::new("p_value".into(), p_values);

    // Create struct from series
    let struct_ca = StructChunked::from_series(
        "item".into(),
        n,
        [&term_s, &estimate_s, &std_error_s, &statistic_s, &p_value_s].into_iter(),
    )?;

    // Wrap in a list (single element containing all coefficient rows)
    let struct_series = struct_ca.into_series();
    let list_s = Series::new("summary".into(), &[struct_series]);

    Ok(list_s)
}

/// Create empty summary output on error
fn summary_nan_output() -> PolarsResult<Series> {
    summary_output(vec![], vec![], vec![], vec![], vec![])
}

// ============================================================================
// Linear Regression Expressions
// ============================================================================

/// OLS regression expression.
/// inputs[0] = y, inputs[1] = with_intercept (bool), inputs[2..] = x columns
#[polars_expr(output_type_func=linear_regression_output_dtype)]
fn pl_ols(inputs: &[Series]) -> PolarsResult<Series> {
    let with_intercept = inputs[1].bool()?.get(0).unwrap_or(true);

    let (x, y) = match build_xy_data(inputs, 0, 2) {
        Ok(data) => data,
        Err(_) => return linear_nan_output(),
    };

    let model = OlsRegressor::builder()
        .with_intercept(with_intercept)
        .build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            linear_output(
                fitted.intercept(),
                &col_to_vec(fitted.coefficients()),
                result.r_squared,
                result.adj_r_squared,
                result.mse,
                result.rmse,
                result.f_statistic,
                result.f_pvalue,
                result.aic,
                result.bic,
                result.n_observations,
            )
        }
        Err(_) => linear_nan_output(),
    }
}

/// Ridge regression expression.
/// inputs[0] = y, inputs[1] = lambda, inputs[2] = with_intercept, inputs[3..] = x columns
#[polars_expr(output_type_func=linear_regression_output_dtype)]
fn pl_ridge(inputs: &[Series]) -> PolarsResult<Series> {
    let lambda = inputs[1].f64()?.get(0).unwrap_or(1.0);
    let with_intercept = inputs[2].bool()?.get(0).unwrap_or(true);

    let (x, y) = match build_xy_data(inputs, 0, 3) {
        Ok(data) => data,
        Err(_) => return linear_nan_output(),
    };

    let model = RidgeRegressor::builder()
        .lambda(lambda)
        .with_intercept(with_intercept)
        .build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            linear_output(
                fitted.intercept(),
                &col_to_vec(fitted.coefficients()),
                result.r_squared,
                result.adj_r_squared,
                result.mse,
                result.rmse,
                result.f_statistic,
                result.f_pvalue,
                result.aic,
                result.bic,
                result.n_observations,
            )
        }
        Err(_) => linear_nan_output(),
    }
}

/// Elastic Net regression expression.
/// inputs[0] = y, inputs[1] = lambda, inputs[2] = alpha (L1 ratio), inputs[3] = with_intercept, inputs[4..] = x columns
#[polars_expr(output_type_func=linear_regression_output_dtype)]
fn pl_elastic_net(inputs: &[Series]) -> PolarsResult<Series> {
    let lambda = inputs[1].f64()?.get(0).unwrap_or(1.0);
    let alpha = inputs[2].f64()?.get(0).unwrap_or(0.5);
    let with_intercept = inputs[3].bool()?.get(0).unwrap_or(true);

    let (x, y) = match build_xy_data(inputs, 0, 4) {
        Ok(data) => data,
        Err(_) => return linear_nan_output(),
    };

    let model = ElasticNetRegressor::builder()
        .lambda(lambda)
        .alpha(alpha)
        .with_intercept(with_intercept)
        .build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            linear_output(
                fitted.intercept(),
                &col_to_vec(fitted.coefficients()),
                result.r_squared,
                result.adj_r_squared,
                result.mse,
                result.rmse,
                result.f_statistic,
                result.f_pvalue,
                result.aic,
                result.bic,
                result.n_observations,
            )
        }
        Err(_) => linear_nan_output(),
    }
}

/// WLS regression expression.
/// inputs[0] = y, inputs[1] = weights, inputs[2] = with_intercept, inputs[3..] = x columns
#[polars_expr(output_type_func=linear_regression_output_dtype)]
fn pl_wls(inputs: &[Series]) -> PolarsResult<Series> {
    let weights_series = inputs[1].f64()?;
    let with_intercept = inputs[2].bool()?.get(0).unwrap_or(true);

    let (x, y) = match build_xy_data(inputs, 0, 3) {
        Ok(data) => data,
        Err(_) => return linear_nan_output(),
    };

    let weights_vec: Vec<f64> = weights_series.into_no_null_iter().collect();
    let weights = Col::from_fn(weights_vec.len(), |i| weights_vec[i]);

    let model = WlsRegressor::builder()
        .with_intercept(with_intercept)
        .weights(weights)
        .build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            linear_output(
                fitted.intercept(),
                &col_to_vec(fitted.coefficients()),
                result.r_squared,
                result.adj_r_squared,
                result.mse,
                result.rmse,
                result.f_statistic,
                result.f_pvalue,
                result.aic,
                result.bic,
                result.n_observations,
            )
        }
        Err(_) => linear_nan_output(),
    }
}

/// RLS regression expression.
/// inputs[0] = y, inputs[1] = forgetting_factor, inputs[2] = with_intercept, inputs[3..] = x columns
#[polars_expr(output_type_func=linear_regression_output_dtype)]
fn pl_rls(inputs: &[Series]) -> PolarsResult<Series> {
    let forgetting_factor = inputs[1].f64()?.get(0).unwrap_or(0.99);
    let with_intercept = inputs[2].bool()?.get(0).unwrap_or(true);

    let (x, y) = match build_xy_data(inputs, 0, 3) {
        Ok(data) => data,
        Err(_) => return linear_nan_output(),
    };

    let model = RlsRegressor::builder()
        .forgetting_factor(forgetting_factor)
        .with_intercept(with_intercept)
        .build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            linear_output(
                fitted.intercept(),
                &col_to_vec(fitted.coefficients()),
                result.r_squared,
                result.adj_r_squared,
                result.mse,
                result.rmse,
                result.f_statistic,
                result.f_pvalue,
                result.aic,
                result.bic,
                result.n_observations,
            )
        }
        Err(_) => linear_nan_output(),
    }
}

/// BLS (Bounded Least Squares) regression expression.
/// inputs[0] = y, inputs[1] = lower_bound, inputs[2] = upper_bound, inputs[3] = with_intercept, inputs[4..] = x columns
#[polars_expr(output_type_func=linear_regression_output_dtype)]
fn pl_bls(inputs: &[Series]) -> PolarsResult<Series> {
    let lower_bound = inputs[1].f64()?.get(0);
    let upper_bound = inputs[2].f64()?.get(0);
    let with_intercept = inputs[3].bool()?.get(0).unwrap_or(true);

    let (x, y) = match build_xy_data(inputs, 0, 4) {
        Ok(data) => data,
        Err(_) => return linear_nan_output(),
    };

    let mut builder = BlsRegressor::builder().with_intercept(with_intercept);

    if let Some(lb) = lower_bound {
        builder = builder.lower_bound_all(lb);
    }
    if let Some(ub) = upper_bound {
        builder = builder.upper_bound_all(ub);
    }

    let model = builder.build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            linear_output(
                fitted.intercept(),
                &col_to_vec(fitted.coefficients()),
                result.r_squared,
                result.adj_r_squared,
                result.mse,
                result.rmse,
                result.f_statistic,
                result.f_pvalue,
                result.aic,
                result.bic,
                result.n_observations,
            )
        }
        Err(_) => linear_nan_output(),
    }
}

// ============================================================================
// Robust Regression Expressions
// ============================================================================

/// Quantile regression expression.
/// inputs[0] = y, inputs[1] = tau, inputs[2] = with_intercept, inputs[3..] = x columns
#[polars_expr(output_type_func=quantile_output_dtype)]
fn pl_quantile(inputs: &[Series]) -> PolarsResult<Series> {
    let tau = inputs[1].f64()?.get(0).unwrap_or(0.5);
    let with_intercept = inputs[2].bool()?.get(0).unwrap_or(true);

    let (x, y) = match build_xy_data(inputs, 0, 3) {
        Ok(data) => data,
        Err(_) => return quantile_nan_output(tau),
    };

    let model = QuantileRegressor::builder()
        .tau(tau)
        .with_intercept(with_intercept)
        .build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            quantile_output(
                fitted.intercept(),
                &col_to_vec(fitted.coefficients()),
                fitted.tau(),
                fitted.pseudo_r_squared(),
                fitted.check_loss(),
                result.n_observations,
            )
        }
        Err(_) => quantile_nan_output(tau),
    }
}

/// Isotonic regression expression.
/// inputs[0] = y, inputs[1] = x (single feature), inputs[2] = increasing
#[polars_expr(output_type_func=isotonic_output_dtype)]
fn pl_isotonic(inputs: &[Series]) -> PolarsResult<Series> {
    let increasing = inputs[2].bool()?.get(0).unwrap_or(true);

    // Extract y values
    let y_series = inputs[0].f64()?;
    let y_vec: Vec<f64> = y_series.into_no_null_iter().collect();
    let y = Col::from_fn(y_vec.len(), |i| y_vec[i]);

    // Extract x values (single column)
    let x_series = inputs[1].f64()?;
    let x_vec: Vec<f64> = x_series.into_no_null_iter().collect();
    let x = Col::from_fn(x_vec.len(), |i| x_vec[i]);

    if x.nrows() != y.nrows() || x.nrows() < 2 {
        return isotonic_nan_output(increasing);
    }

    let model = IsotonicRegressor::builder().increasing(increasing).build();

    match model.fit_1d(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            isotonic_output(
                result.r_squared,
                fitted.is_increasing(),
                &col_to_vec(fitted.fitted_values()),
                result.n_observations,
            )
        }
        Err(_) => isotonic_nan_output(increasing),
    }
}

// ============================================================================
// Diagnostics Expressions
// ============================================================================

/// Condition number diagnostic output helper.
fn condition_output(
    condition_number: f64,
    condition_number_xtx: f64,
    singular_values: &[f64],
    condition_indices: &[f64],
    severity: &str,
    warning: Option<&str>,
) -> PolarsResult<Series> {
    let cond_s = Series::new("condition_number".into(), &[condition_number]);
    let cond_xtx_s = Series::new("condition_number_xtx".into(), &[condition_number_xtx]);
    let sv_inner = Series::new("item".into(), singular_values);
    let sv_s = Series::new("singular_values".into(), &[sv_inner]);
    let ci_inner = Series::new("item".into(), condition_indices);
    let ci_s = Series::new("condition_indices".into(), &[ci_inner]);
    let severity_s = Series::new("severity".into(), &[severity]);
    let warning_s = Series::new("warning".into(), &[warning.unwrap_or("")]);

    StructChunked::from_series(
        "condition".into(),
        1,
        [&cond_s, &cond_xtx_s, &sv_s, &ci_s, &severity_s, &warning_s].into_iter(),
    )
    .map(|ca| ca.into_series())
}

/// NaN output for condition number diagnostics on error.
fn condition_nan_output() -> PolarsResult<Series> {
    condition_output(f64::NAN, f64::NAN, &[], &[], "Unknown", None)
}

/// Condition number diagnostics expression.
/// inputs[0] = with_intercept, inputs[1..] = x columns
#[polars_expr(output_type_func=condition_number_output_dtype)]
fn pl_condition_number(inputs: &[Series]) -> PolarsResult<Series> {
    let with_intercept = inputs[0].bool()?.get(0).unwrap_or(true);

    let n_cols = inputs.len() - 1;
    if n_cols == 0 {
        return condition_nan_output();
    }

    let n_rows = inputs[1].len();
    if n_rows < 2 {
        return condition_nan_output();
    }

    // Build X matrix
    let x = Mat::from_fn(n_rows, n_cols, |row, col| {
        inputs[1 + col]
            .f64()
            .ok()
            .and_then(|ca| ca.get(row))
            .unwrap_or(f64::NAN)
    });

    // Check for NaN values
    for i in 0..n_rows {
        for j in 0..n_cols {
            if x[(i, j)].is_nan() {
                return condition_nan_output();
            }
        }
    }

    let diag = condition_diagnostic(&x, with_intercept);

    let severity_str = match diag.severity {
        ConditionSeverity::WellConditioned => "WellConditioned",
        ConditionSeverity::Moderate => "Moderate",
        ConditionSeverity::High => "High",
        ConditionSeverity::Severe => "Severe",
    };

    condition_output(
        diag.condition_number,
        diag.condition_number_xtx,
        &diag.singular_values,
        &diag.condition_indices,
        severity_str,
        diag.warning.as_deref(),
    )
}

/// Convert SeparationType to string.
fn separation_type_to_string(sep_type: &SeparationType) -> &'static str {
    match sep_type {
        SeparationType::None => "None",
        SeparationType::Complete => "Complete",
        SeparationType::Quasi => "Quasi",
        SeparationType::MonotonicResponse => "MonotonicResponse",
    }
}

/// Separation check output helper.
fn separation_output(check: &SeparationCheck) -> PolarsResult<Series> {
    let has_separation_s = Series::new("has_separation".into(), &[check.has_separation]);
    let predictors_u32: Vec<u32> = check
        .separated_predictors
        .iter()
        .map(|&p| p as u32)
        .collect();
    let predictors_inner = Series::new("item".into(), &predictors_u32);
    let predictors_s = Series::new("separated_predictors".into(), &[predictors_inner]);
    let types_strs: Vec<&str> = check
        .separation_types
        .iter()
        .map(separation_type_to_string)
        .collect();
    let types_inner = Series::new("item".into(), &types_strs);
    let types_s = Series::new("separation_types".into(), &[types_inner]);
    let warning_s = Series::new(
        "warning".into(),
        &[check.warning_message.as_deref().unwrap_or("")],
    );

    StructChunked::from_series(
        "separation".into(),
        1,
        [&has_separation_s, &predictors_s, &types_s, &warning_s].into_iter(),
    )
    .map(|ca| ca.into_series())
}

/// Empty/default output for separation check.
fn separation_default_output() -> PolarsResult<Series> {
    separation_output(&SeparationCheck::default())
}

/// Check binary response (logistic/probit) data for quasi-separation.
/// inputs[0] = y (binary), inputs[1..] = x columns
#[polars_expr(output_type_func=separation_check_output_dtype)]
fn pl_check_binary_separation(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.is_empty() {
        return separation_default_output();
    }

    let y_series = inputs[0].f64()?;
    let n_rows = y_series.len();
    let n_cols = inputs.len() - 1;

    if n_cols == 0 || n_rows < 2 {
        return separation_default_output();
    }

    // Build y vector
    let y_vec: Vec<f64> = y_series.into_no_null_iter().collect();
    let y = Col::from_fn(y_vec.len(), |i| y_vec[i]);

    // Build X matrix
    let x = Mat::from_fn(n_rows, n_cols, |row, col| {
        inputs[1 + col]
            .f64()
            .ok()
            .and_then(|ca| ca.get(row))
            .unwrap_or(f64::NAN)
    });

    // Check for NaN values
    for i in 0..n_rows {
        for j in 0..n_cols {
            if x[(i, j)].is_nan() {
                return separation_default_output();
            }
        }
    }
    for i in 0..y_vec.len() {
        if y[i].is_nan() {
            return separation_default_output();
        }
    }

    let check = check_binary_separation(&x, &y);
    separation_output(&check)
}

/// Check count data (Poisson/NegBin) for sparsity-induced separation.
/// inputs[0] = y (counts), inputs[1..] = x columns
#[polars_expr(output_type_func=separation_check_output_dtype)]
fn pl_check_count_sparsity(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.is_empty() {
        return separation_default_output();
    }

    let y_series = inputs[0].f64()?;
    let n_rows = y_series.len();
    let n_cols = inputs.len() - 1;

    if n_cols == 0 || n_rows < 2 {
        return separation_default_output();
    }

    // Build y vector
    let y_vec: Vec<f64> = y_series.into_no_null_iter().collect();
    let y = Col::from_fn(y_vec.len(), |i| y_vec[i]);

    // Build X matrix
    let x = Mat::from_fn(n_rows, n_cols, |row, col| {
        inputs[1 + col]
            .f64()
            .ok()
            .and_then(|ca| ca.get(row))
            .unwrap_or(f64::NAN)
    });

    // Check for NaN values
    for i in 0..n_rows {
        for j in 0..n_cols {
            if x[(i, j)].is_nan() {
                return separation_default_output();
            }
        }
    }
    for i in 0..y_vec.len() {
        if y[i].is_nan() {
            return separation_default_output();
        }
    }

    let check = check_count_sparsity(&x, &y);
    separation_output(&check)
}

// ============================================================================
// GLM Expressions
// ============================================================================

/// Logistic regression expression.
/// inputs[0] = y, inputs[1] = lambda (L2 regularization), inputs[2] = with_intercept, inputs[3..] = x columns
#[polars_expr(output_type_func=glm_output_dtype)]
fn pl_logistic(inputs: &[Series]) -> PolarsResult<Series> {
    let lambda = inputs[1].f64()?.get(0).unwrap_or(0.0);
    let with_intercept = inputs[2].bool()?.get(0).unwrap_or(true);

    let (x, y) = match build_xy_data(inputs, 0, 3) {
        Ok(data) => data,
        Err(_) => return glm_nan_output(),
    };

    let mut builder = BinomialRegressor::logistic().with_intercept(with_intercept);
    if lambda > 0.0 {
        builder = builder.lambda(lambda);
    }
    let model = builder.build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            glm_output(
                fitted.intercept(),
                &col_to_vec(fitted.coefficients()),
                result.aic,
                result.bic,
                result.n_observations,
            )
        }
        Err(_) => glm_nan_output(),
    }
}

/// Poisson regression expression.
/// inputs[0] = y, inputs[1] = lambda (L2 regularization), inputs[2] = with_intercept, inputs[3..] = x columns
#[polars_expr(output_type_func=glm_output_dtype)]
fn pl_poisson(inputs: &[Series]) -> PolarsResult<Series> {
    let lambda = inputs[1].f64()?.get(0).unwrap_or(0.0);
    let with_intercept = inputs[2].bool()?.get(0).unwrap_or(true);

    let (x, y) = match build_xy_data(inputs, 0, 3) {
        Ok(data) => data,
        Err(_) => return glm_nan_output(),
    };

    let mut builder = PoissonRegressor::builder().with_intercept(with_intercept);
    if lambda > 0.0 {
        builder = builder.lambda(lambda);
    }
    let model = builder.build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            glm_output(
                fitted.intercept(),
                &col_to_vec(fitted.coefficients()),
                result.aic,
                result.bic,
                result.n_observations,
            )
        }
        Err(_) => glm_nan_output(),
    }
}

/// Negative Binomial regression expression.
/// inputs[0] = y, inputs[1] = theta (optional), inputs[2] = lambda (L2 regularization), inputs[3] = with_intercept, inputs[4..] = x columns
#[polars_expr(output_type_func=glm_output_dtype)]
fn pl_negative_binomial(inputs: &[Series]) -> PolarsResult<Series> {
    let theta = inputs[1].f64()?.get(0);
    let lambda = inputs[2].f64()?.get(0).unwrap_or(0.0);
    let with_intercept = inputs[3].bool()?.get(0).unwrap_or(true);

    let (x, y) = match build_xy_data(inputs, 0, 4) {
        Ok(data) => data,
        Err(_) => return glm_nan_output(),
    };

    let mut builder = NegativeBinomialRegressor::builder().with_intercept(with_intercept);

    if let Some(t) = theta {
        builder = builder.theta(t);
    } else {
        builder = builder.estimate_theta(true);
    }

    if lambda > 0.0 {
        builder = builder.lambda(lambda);
    }

    let model = builder.build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            glm_output(
                fitted.intercept(),
                &col_to_vec(fitted.coefficients()),
                result.aic,
                result.bic,
                result.n_observations,
            )
        }
        Err(_) => glm_nan_output(),
    }
}

/// Tweedie regression expression.
/// inputs[0] = y, inputs[1] = var_power, inputs[2] = lambda (L2 regularization), inputs[3] = with_intercept, inputs[4..] = x columns
#[polars_expr(output_type_func=glm_output_dtype)]
fn pl_tweedie(inputs: &[Series]) -> PolarsResult<Series> {
    let var_power = inputs[1].f64()?.get(0).unwrap_or(1.5);
    let lambda = inputs[2].f64()?.get(0).unwrap_or(0.0);
    let with_intercept = inputs[3].bool()?.get(0).unwrap_or(true);

    let (x, y) = match build_xy_data(inputs, 0, 4) {
        Ok(data) => data,
        Err(_) => return glm_nan_output(),
    };

    let mut builder = TweedieRegressor::builder()
        .var_power(var_power)
        .with_intercept(with_intercept);

    if lambda > 0.0 {
        builder = builder.lambda(lambda);
    }

    let model = builder.build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            glm_output(
                fitted.intercept(),
                &col_to_vec(fitted.coefficients()),
                result.aic,
                result.bic,
                result.n_observations,
            )
        }
        Err(_) => glm_nan_output(),
    }
}

/// Probit regression expression.
/// inputs[0] = y, inputs[1] = lambda (L2 regularization), inputs[2] = with_intercept, inputs[3..] = x columns
#[polars_expr(output_type_func=glm_output_dtype)]
fn pl_probit(inputs: &[Series]) -> PolarsResult<Series> {
    let lambda = inputs[1].f64()?.get(0).unwrap_or(0.0);
    let with_intercept = inputs[2].bool()?.get(0).unwrap_or(true);

    let (x, y) = match build_xy_data(inputs, 0, 3) {
        Ok(data) => data,
        Err(_) => return glm_nan_output(),
    };

    let mut builder = BinomialRegressor::probit().with_intercept(with_intercept);
    if lambda > 0.0 {
        builder = builder.lambda(lambda);
    }
    let model = builder.build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            glm_output(
                fitted.intercept(),
                &col_to_vec(fitted.coefficients()),
                result.aic,
                result.bic,
                result.n_observations,
            )
        }
        Err(_) => glm_nan_output(),
    }
}

/// Complementary log-log regression expression.
/// inputs[0] = y, inputs[1] = lambda (L2 regularization), inputs[2] = with_intercept, inputs[3..] = x columns
#[polars_expr(output_type_func=glm_output_dtype)]
fn pl_cloglog(inputs: &[Series]) -> PolarsResult<Series> {
    let lambda = inputs[1].f64()?.get(0).unwrap_or(0.0);
    let with_intercept = inputs[2].bool()?.get(0).unwrap_or(true);

    let (x, y) = match build_xy_data(inputs, 0, 3) {
        Ok(data) => data,
        Err(_) => return glm_nan_output(),
    };

    let mut builder = BinomialRegressor::cloglog().with_intercept(with_intercept);
    if lambda > 0.0 {
        builder = builder.lambda(lambda);
    }
    let model = builder.build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            glm_output(
                fitted.intercept(),
                &col_to_vec(fitted.coefficients()),
                result.aic,
                result.bic,
                result.n_observations,
            )
        }
        Err(_) => glm_nan_output(),
    }
}

// ============================================================================
// ALM Expression
// ============================================================================

/// Parse distribution string to AlmDistribution enum.
fn parse_alm_distribution(s: &str) -> Option<AlmDistribution> {
    match s.to_lowercase().as_str() {
        "normal" | "gaussian" => Some(AlmDistribution::Normal),
        "laplace" => Some(AlmDistribution::Laplace),
        "student_t" | "studentt" | "t" => Some(AlmDistribution::StudentT),
        "logistic" => Some(AlmDistribution::Logistic),
        "gamma" => Some(AlmDistribution::Gamma),
        "inverse_gaussian" | "inversegaussian" => Some(AlmDistribution::InverseGaussian),
        "exponential" => Some(AlmDistribution::Exponential),
        "beta" => Some(AlmDistribution::Beta),
        "poisson" => Some(AlmDistribution::Poisson),
        "negative_binomial" | "negativebinomial" | "negbin" => {
            Some(AlmDistribution::NegativeBinomial)
        }
        "binomial" => Some(AlmDistribution::Binomial),
        "geometric" => Some(AlmDistribution::Geometric),
        "lognormal" | "log_normal" => Some(AlmDistribution::LogNormal),
        "loglaplace" | "log_laplace" => Some(AlmDistribution::LogLaplace),
        _ => None,
    }
}

/// ALM (Augmented Linear Model) expression.
/// inputs[0] = y, inputs[1] = distribution (string), inputs[2] = with_intercept, inputs[3..] = x columns
#[polars_expr(output_type_func=glm_output_dtype)]
fn pl_alm(inputs: &[Series]) -> PolarsResult<Series> {
    let dist_str = inputs[1].str()?.get(0).unwrap_or("normal");
    let with_intercept = inputs[2].bool()?.get(0).unwrap_or(true);

    let (x, y) = match build_xy_data(inputs, 0, 3) {
        Ok(data) => data,
        Err(_) => return glm_nan_output(),
    };

    let distribution = match parse_alm_distribution(dist_str) {
        Some(d) => d,
        None => return glm_nan_output(),
    };

    let model = AlmRegressor::builder()
        .distribution(distribution)
        .with_intercept(with_intercept)
        .build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            glm_output(
                fitted.intercept(),
                &col_to_vec(fitted.coefficients()),
                result.aic,
                result.bic,
                result.n_observations,
            )
        }
        Err(_) => glm_nan_output(),
    }
}

// ============================================================================
// Summary Expressions (Tidy Coefficient Output)
// ============================================================================

/// Helper to build term names for coefficients
fn build_term_names(n_features: usize, with_intercept: bool) -> Vec<String> {
    let mut terms = Vec::with_capacity(n_features + if with_intercept { 1 } else { 0 });
    if with_intercept {
        terms.push("intercept".to_string());
    }
    for i in 1..=n_features {
        terms.push(format!("x{}", i));
    }
    terms
}

/// OLS summary expression - returns tidy coefficient table
/// inputs[0] = y, inputs[1] = with_intercept (bool), inputs[2..] = x columns
#[polars_expr(output_type_func=summary_output_dtype)]
fn pl_ols_summary(inputs: &[Series]) -> PolarsResult<Series> {
    let with_intercept = inputs[1].bool()?.get(0).unwrap_or(true);
    let n_features = inputs.len() - 2;

    let (x, y) = match build_xy_data(inputs, 0, 2) {
        Ok(data) => data,
        Err(_) => return summary_nan_output(),
    };

    let model = OlsRegressor::builder()
        .with_intercept(with_intercept)
        .build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            let terms = build_term_names(n_features, with_intercept);

            // Build coefficient vectors including intercept
            let mut estimates = Vec::new();
            let mut std_errors = Vec::new();
            let mut statistics = Vec::new();
            let mut p_values = Vec::new();

            if with_intercept {
                estimates.push(fitted.intercept().unwrap_or(f64::NAN));
                std_errors.push(result.intercept_std_error.unwrap_or(f64::NAN));
                statistics.push(result.intercept_t_statistic.unwrap_or(f64::NAN));
                p_values.push(result.intercept_p_value.unwrap_or(f64::NAN));
            }

            // Add feature coefficients
            let coefs = col_to_vec(fitted.coefficients());
            estimates.extend(coefs);

            if let Some(ref se) = result.std_errors {
                std_errors.extend(col_to_vec(se));
            } else {
                std_errors.extend(vec![f64::NAN; n_features]);
            }

            if let Some(ref ts) = result.t_statistics {
                statistics.extend(col_to_vec(ts));
            } else {
                statistics.extend(vec![f64::NAN; n_features]);
            }

            if let Some(ref pv) = result.p_values {
                p_values.extend(col_to_vec(pv));
            } else {
                p_values.extend(vec![f64::NAN; n_features]);
            }

            summary_output(terms, estimates, std_errors, statistics, p_values)
        }
        Err(_) => summary_nan_output(),
    }
}

/// Ridge summary expression
/// inputs[0] = y, inputs[1] = lambda, inputs[2] = with_intercept, inputs[3..] = x columns
#[polars_expr(output_type_func=summary_output_dtype)]
fn pl_ridge_summary(inputs: &[Series]) -> PolarsResult<Series> {
    let lambda = inputs[1].f64()?.get(0).unwrap_or(1.0);
    let with_intercept = inputs[2].bool()?.get(0).unwrap_or(true);
    let n_features = inputs.len() - 3;

    let (x, y) = match build_xy_data(inputs, 0, 3) {
        Ok(data) => data,
        Err(_) => return summary_nan_output(),
    };

    let model = RidgeRegressor::builder()
        .lambda(lambda)
        .with_intercept(with_intercept)
        .build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            let terms = build_term_names(n_features, with_intercept);

            let mut estimates = Vec::new();
            let mut std_errors = Vec::new();
            let mut statistics = Vec::new();
            let mut p_values = Vec::new();

            if with_intercept {
                estimates.push(fitted.intercept().unwrap_or(f64::NAN));
                std_errors.push(result.intercept_std_error.unwrap_or(f64::NAN));
                statistics.push(result.intercept_t_statistic.unwrap_or(f64::NAN));
                p_values.push(result.intercept_p_value.unwrap_or(f64::NAN));
            }

            let coefs = col_to_vec(fitted.coefficients());
            estimates.extend(coefs);

            if let Some(ref se) = result.std_errors {
                std_errors.extend(col_to_vec(se));
            } else {
                std_errors.extend(vec![f64::NAN; n_features]);
            }

            if let Some(ref ts) = result.t_statistics {
                statistics.extend(col_to_vec(ts));
            } else {
                statistics.extend(vec![f64::NAN; n_features]);
            }

            if let Some(ref pv) = result.p_values {
                p_values.extend(col_to_vec(pv));
            } else {
                p_values.extend(vec![f64::NAN; n_features]);
            }

            summary_output(terms, estimates, std_errors, statistics, p_values)
        }
        Err(_) => summary_nan_output(),
    }
}

/// Elastic Net summary expression
/// inputs[0] = y, inputs[1] = lambda, inputs[2] = alpha, inputs[3] = with_intercept, inputs[4..] = x columns
#[polars_expr(output_type_func=summary_output_dtype)]
fn pl_elastic_net_summary(inputs: &[Series]) -> PolarsResult<Series> {
    let lambda = inputs[1].f64()?.get(0).unwrap_or(1.0);
    let alpha = inputs[2].f64()?.get(0).unwrap_or(0.5);
    let with_intercept = inputs[3].bool()?.get(0).unwrap_or(true);
    let n_features = inputs.len() - 4;

    let (x, y) = match build_xy_data(inputs, 0, 4) {
        Ok(data) => data,
        Err(_) => return summary_nan_output(),
    };

    let model = ElasticNetRegressor::builder()
        .lambda(lambda)
        .alpha(alpha)
        .with_intercept(with_intercept)
        .build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            let terms = build_term_names(n_features, with_intercept);

            let mut estimates = Vec::new();
            let mut std_errors = Vec::new();
            let mut statistics = Vec::new();
            let mut p_values = Vec::new();

            if with_intercept {
                estimates.push(fitted.intercept().unwrap_or(f64::NAN));
                std_errors.push(result.intercept_std_error.unwrap_or(f64::NAN));
                statistics.push(result.intercept_t_statistic.unwrap_or(f64::NAN));
                p_values.push(result.intercept_p_value.unwrap_or(f64::NAN));
            }

            let coefs = col_to_vec(fitted.coefficients());
            estimates.extend(coefs);

            if let Some(ref se) = result.std_errors {
                std_errors.extend(col_to_vec(se));
            } else {
                std_errors.extend(vec![f64::NAN; n_features]);
            }

            if let Some(ref ts) = result.t_statistics {
                statistics.extend(col_to_vec(ts));
            } else {
                statistics.extend(vec![f64::NAN; n_features]);
            }

            if let Some(ref pv) = result.p_values {
                p_values.extend(col_to_vec(pv));
            } else {
                p_values.extend(vec![f64::NAN; n_features]);
            }

            summary_output(terms, estimates, std_errors, statistics, p_values)
        }
        Err(_) => summary_nan_output(),
    }
}

/// WLS summary expression
/// inputs[0] = y, inputs[1] = weights, inputs[2] = with_intercept, inputs[3..] = x columns
#[polars_expr(output_type_func=summary_output_dtype)]
fn pl_wls_summary(inputs: &[Series]) -> PolarsResult<Series> {
    let weights_series = inputs[1].f64()?;
    let with_intercept = inputs[2].bool()?.get(0).unwrap_or(true);
    let n_features = inputs.len() - 3;

    let (x, y) = match build_xy_data(inputs, 0, 3) {
        Ok(data) => data,
        Err(_) => return summary_nan_output(),
    };

    let weights_vec: Vec<f64> = weights_series.into_no_null_iter().collect();
    let weights = Col::from_fn(weights_vec.len(), |i| weights_vec[i]);

    let model = WlsRegressor::builder()
        .with_intercept(with_intercept)
        .weights(weights)
        .build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            let terms = build_term_names(n_features, with_intercept);

            let mut estimates = Vec::new();
            let mut std_errors = Vec::new();
            let mut statistics = Vec::new();
            let mut p_values = Vec::new();

            if with_intercept {
                estimates.push(fitted.intercept().unwrap_or(f64::NAN));
                std_errors.push(result.intercept_std_error.unwrap_or(f64::NAN));
                statistics.push(result.intercept_t_statistic.unwrap_or(f64::NAN));
                p_values.push(result.intercept_p_value.unwrap_or(f64::NAN));
            }

            let coefs = col_to_vec(fitted.coefficients());
            estimates.extend(coefs);

            if let Some(ref se) = result.std_errors {
                std_errors.extend(col_to_vec(se));
            } else {
                std_errors.extend(vec![f64::NAN; n_features]);
            }

            if let Some(ref ts) = result.t_statistics {
                statistics.extend(col_to_vec(ts));
            } else {
                statistics.extend(vec![f64::NAN; n_features]);
            }

            if let Some(ref pv) = result.p_values {
                p_values.extend(col_to_vec(pv));
            } else {
                p_values.extend(vec![f64::NAN; n_features]);
            }

            summary_output(terms, estimates, std_errors, statistics, p_values)
        }
        Err(_) => summary_nan_output(),
    }
}

/// RLS summary expression
/// inputs[0] = y, inputs[1] = forgetting_factor, inputs[2] = with_intercept, inputs[3..] = x columns
#[polars_expr(output_type_func=summary_output_dtype)]
fn pl_rls_summary(inputs: &[Series]) -> PolarsResult<Series> {
    let forgetting_factor = inputs[1].f64()?.get(0).unwrap_or(0.99);
    let with_intercept = inputs[2].bool()?.get(0).unwrap_or(true);
    let n_features = inputs.len() - 3;

    let (x, y) = match build_xy_data(inputs, 0, 3) {
        Ok(data) => data,
        Err(_) => return summary_nan_output(),
    };

    let model = RlsRegressor::builder()
        .forgetting_factor(forgetting_factor)
        .with_intercept(with_intercept)
        .build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            let terms = build_term_names(n_features, with_intercept);

            let mut estimates = Vec::new();
            let mut std_errors = Vec::new();
            let mut statistics = Vec::new();
            let mut p_values = Vec::new();

            if with_intercept {
                estimates.push(fitted.intercept().unwrap_or(f64::NAN));
                std_errors.push(result.intercept_std_error.unwrap_or(f64::NAN));
                statistics.push(result.intercept_t_statistic.unwrap_or(f64::NAN));
                p_values.push(result.intercept_p_value.unwrap_or(f64::NAN));
            }

            let coefs = col_to_vec(fitted.coefficients());
            estimates.extend(coefs);

            if let Some(ref se) = result.std_errors {
                std_errors.extend(col_to_vec(se));
            } else {
                std_errors.extend(vec![f64::NAN; n_features]);
            }

            if let Some(ref ts) = result.t_statistics {
                statistics.extend(col_to_vec(ts));
            } else {
                statistics.extend(vec![f64::NAN; n_features]);
            }

            if let Some(ref pv) = result.p_values {
                p_values.extend(col_to_vec(pv));
            } else {
                p_values.extend(vec![f64::NAN; n_features]);
            }

            summary_output(terms, estimates, std_errors, statistics, p_values)
        }
        Err(_) => summary_nan_output(),
    }
}

/// BLS summary expression
/// inputs[0] = y, inputs[1] = lower_bound, inputs[2] = upper_bound, inputs[3] = with_intercept, inputs[4..] = x columns
#[polars_expr(output_type_func=summary_output_dtype)]
fn pl_bls_summary(inputs: &[Series]) -> PolarsResult<Series> {
    let lower_bound = inputs[1].f64()?.get(0);
    let upper_bound = inputs[2].f64()?.get(0);
    let with_intercept = inputs[3].bool()?.get(0).unwrap_or(true);
    let n_features = inputs.len() - 4;

    let (x, y) = match build_xy_data(inputs, 0, 4) {
        Ok(data) => data,
        Err(_) => return summary_nan_output(),
    };

    let mut builder = BlsRegressor::builder().with_intercept(with_intercept);

    if let Some(lb) = lower_bound {
        builder = builder.lower_bound_all(lb);
    }
    if let Some(ub) = upper_bound {
        builder = builder.upper_bound_all(ub);
    }

    let model = builder.build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            let terms = build_term_names(n_features, with_intercept);

            let mut estimates = Vec::new();
            let mut std_errors = Vec::new();
            let mut statistics = Vec::new();
            let mut p_values = Vec::new();

            if with_intercept {
                estimates.push(fitted.intercept().unwrap_or(f64::NAN));
                std_errors.push(result.intercept_std_error.unwrap_or(f64::NAN));
                statistics.push(result.intercept_t_statistic.unwrap_or(f64::NAN));
                p_values.push(result.intercept_p_value.unwrap_or(f64::NAN));
            }

            let coefs = col_to_vec(fitted.coefficients());
            estimates.extend(coefs);

            if let Some(ref se) = result.std_errors {
                std_errors.extend(col_to_vec(se));
            } else {
                std_errors.extend(vec![f64::NAN; n_features]);
            }

            if let Some(ref ts) = result.t_statistics {
                statistics.extend(col_to_vec(ts));
            } else {
                statistics.extend(vec![f64::NAN; n_features]);
            }

            if let Some(ref pv) = result.p_values {
                p_values.extend(col_to_vec(pv));
            } else {
                p_values.extend(vec![f64::NAN; n_features]);
            }

            summary_output(terms, estimates, std_errors, statistics, p_values)
        }
        Err(_) => summary_nan_output(),
    }
}

// ============================================================================
// GLM Summary Expressions
// ============================================================================

/// Logistic summary expression
/// inputs[0] = y, inputs[1] = lambda (L2 regularization), inputs[2] = with_intercept, inputs[3..] = x columns
#[polars_expr(output_type_func=summary_output_dtype)]
fn pl_logistic_summary(inputs: &[Series]) -> PolarsResult<Series> {
    let lambda = inputs[1].f64()?.get(0).unwrap_or(0.0);
    let with_intercept = inputs[2].bool()?.get(0).unwrap_or(true);
    let n_features = inputs.len() - 3;

    let (x, y) = match build_xy_data(inputs, 0, 3) {
        Ok(data) => data,
        Err(_) => return summary_nan_output(),
    };

    let mut builder = BinomialRegressor::logistic().with_intercept(with_intercept);
    if lambda > 0.0 {
        builder = builder.lambda(lambda);
    }
    let model = builder.build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            let terms = build_term_names(n_features, with_intercept);

            let mut estimates = Vec::new();
            let mut std_errors = Vec::new();
            let mut statistics = Vec::new();
            let mut p_values = Vec::new();

            if with_intercept {
                estimates.push(fitted.intercept().unwrap_or(f64::NAN));
                std_errors.push(result.intercept_std_error.unwrap_or(f64::NAN));
                statistics.push(result.intercept_t_statistic.unwrap_or(f64::NAN));
                p_values.push(result.intercept_p_value.unwrap_or(f64::NAN));
            }

            let coefs = col_to_vec(fitted.coefficients());
            estimates.extend(coefs);

            if let Some(ref se) = result.std_errors {
                std_errors.extend(col_to_vec(se));
            } else {
                std_errors.extend(vec![f64::NAN; n_features]);
            }

            if let Some(ref zs) = result.t_statistics {
                statistics.extend(col_to_vec(zs));
            } else {
                statistics.extend(vec![f64::NAN; n_features]);
            }

            if let Some(ref pv) = result.p_values {
                p_values.extend(col_to_vec(pv));
            } else {
                p_values.extend(vec![f64::NAN; n_features]);
            }

            summary_output(terms, estimates, std_errors, statistics, p_values)
        }
        Err(_) => summary_nan_output(),
    }
}

/// Poisson summary expression
/// inputs[0] = y, inputs[1] = lambda (L2 regularization), inputs[2] = with_intercept, inputs[3..] = x columns
#[polars_expr(output_type_func=summary_output_dtype)]
fn pl_poisson_summary(inputs: &[Series]) -> PolarsResult<Series> {
    let lambda = inputs[1].f64()?.get(0).unwrap_or(0.0);
    let with_intercept = inputs[2].bool()?.get(0).unwrap_or(true);
    let n_features = inputs.len() - 3;

    let (x, y) = match build_xy_data(inputs, 0, 3) {
        Ok(data) => data,
        Err(_) => return summary_nan_output(),
    };

    let mut builder = PoissonRegressor::builder().with_intercept(with_intercept);
    if lambda > 0.0 {
        builder = builder.lambda(lambda);
    }
    let model = builder.build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            let terms = build_term_names(n_features, with_intercept);

            let mut estimates = Vec::new();
            let mut std_errors = Vec::new();
            let mut statistics = Vec::new();
            let mut p_values = Vec::new();

            if with_intercept {
                estimates.push(fitted.intercept().unwrap_or(f64::NAN));
                std_errors.push(result.intercept_std_error.unwrap_or(f64::NAN));
                statistics.push(result.intercept_t_statistic.unwrap_or(f64::NAN));
                p_values.push(result.intercept_p_value.unwrap_or(f64::NAN));
            }

            let coefs = col_to_vec(fitted.coefficients());
            estimates.extend(coefs);

            if let Some(ref se) = result.std_errors {
                std_errors.extend(col_to_vec(se));
            } else {
                std_errors.extend(vec![f64::NAN; n_features]);
            }

            if let Some(ref zs) = result.t_statistics {
                statistics.extend(col_to_vec(zs));
            } else {
                statistics.extend(vec![f64::NAN; n_features]);
            }

            if let Some(ref pv) = result.p_values {
                p_values.extend(col_to_vec(pv));
            } else {
                p_values.extend(vec![f64::NAN; n_features]);
            }

            summary_output(terms, estimates, std_errors, statistics, p_values)
        }
        Err(_) => summary_nan_output(),
    }
}

/// Negative Binomial summary expression
/// inputs[0] = y, inputs[1] = theta, inputs[2] = lambda (L2 regularization), inputs[3] = with_intercept, inputs[4..] = x columns
#[polars_expr(output_type_func=summary_output_dtype)]
fn pl_negative_binomial_summary(inputs: &[Series]) -> PolarsResult<Series> {
    let theta = inputs[1].f64()?.get(0);
    let lambda = inputs[2].f64()?.get(0).unwrap_or(0.0);
    let with_intercept = inputs[3].bool()?.get(0).unwrap_or(true);
    let n_features = inputs.len() - 4;

    let (x, y) = match build_xy_data(inputs, 0, 4) {
        Ok(data) => data,
        Err(_) => return summary_nan_output(),
    };

    let mut builder = NegativeBinomialRegressor::builder().with_intercept(with_intercept);

    if let Some(t) = theta {
        builder = builder.theta(t);
    } else {
        builder = builder.estimate_theta(true);
    }

    if lambda > 0.0 {
        builder = builder.lambda(lambda);
    }

    let model = builder.build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            let terms = build_term_names(n_features, with_intercept);

            let mut estimates = Vec::new();
            let mut std_errors = Vec::new();
            let mut statistics = Vec::new();
            let mut p_values = Vec::new();

            if with_intercept {
                estimates.push(fitted.intercept().unwrap_or(f64::NAN));
                std_errors.push(result.intercept_std_error.unwrap_or(f64::NAN));
                statistics.push(result.intercept_t_statistic.unwrap_or(f64::NAN));
                p_values.push(result.intercept_p_value.unwrap_or(f64::NAN));
            }

            let coefs = col_to_vec(fitted.coefficients());
            estimates.extend(coefs);

            if let Some(ref se) = result.std_errors {
                std_errors.extend(col_to_vec(se));
            } else {
                std_errors.extend(vec![f64::NAN; n_features]);
            }

            if let Some(ref zs) = result.t_statistics {
                statistics.extend(col_to_vec(zs));
            } else {
                statistics.extend(vec![f64::NAN; n_features]);
            }

            if let Some(ref pv) = result.p_values {
                p_values.extend(col_to_vec(pv));
            } else {
                p_values.extend(vec![f64::NAN; n_features]);
            }

            summary_output(terms, estimates, std_errors, statistics, p_values)
        }
        Err(_) => summary_nan_output(),
    }
}

/// Tweedie summary expression
/// inputs[0] = y, inputs[1] = var_power, inputs[2] = lambda (L2 regularization), inputs[3] = with_intercept, inputs[4..] = x columns
#[polars_expr(output_type_func=summary_output_dtype)]
fn pl_tweedie_summary(inputs: &[Series]) -> PolarsResult<Series> {
    let var_power = inputs[1].f64()?.get(0).unwrap_or(1.5);
    let lambda = inputs[2].f64()?.get(0).unwrap_or(0.0);
    let with_intercept = inputs[3].bool()?.get(0).unwrap_or(true);
    let n_features = inputs.len() - 4;

    let (x, y) = match build_xy_data(inputs, 0, 4) {
        Ok(data) => data,
        Err(_) => return summary_nan_output(),
    };

    let mut builder = TweedieRegressor::builder()
        .var_power(var_power)
        .with_intercept(with_intercept);

    if lambda > 0.0 {
        builder = builder.lambda(lambda);
    }

    let model = builder.build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            let terms = build_term_names(n_features, with_intercept);

            let mut estimates = Vec::new();
            let mut std_errors = Vec::new();
            let mut statistics = Vec::new();
            let mut p_values = Vec::new();

            if with_intercept {
                estimates.push(fitted.intercept().unwrap_or(f64::NAN));
                std_errors.push(result.intercept_std_error.unwrap_or(f64::NAN));
                statistics.push(result.intercept_t_statistic.unwrap_or(f64::NAN));
                p_values.push(result.intercept_p_value.unwrap_or(f64::NAN));
            }

            let coefs = col_to_vec(fitted.coefficients());
            estimates.extend(coefs);

            if let Some(ref se) = result.std_errors {
                std_errors.extend(col_to_vec(se));
            } else {
                std_errors.extend(vec![f64::NAN; n_features]);
            }

            if let Some(ref zs) = result.t_statistics {
                statistics.extend(col_to_vec(zs));
            } else {
                statistics.extend(vec![f64::NAN; n_features]);
            }

            if let Some(ref pv) = result.p_values {
                p_values.extend(col_to_vec(pv));
            } else {
                p_values.extend(vec![f64::NAN; n_features]);
            }

            summary_output(terms, estimates, std_errors, statistics, p_values)
        }
        Err(_) => summary_nan_output(),
    }
}

/// Probit summary expression
/// inputs[0] = y, inputs[1] = lambda (L2 regularization), inputs[2] = with_intercept, inputs[3..] = x columns
#[polars_expr(output_type_func=summary_output_dtype)]
fn pl_probit_summary(inputs: &[Series]) -> PolarsResult<Series> {
    let lambda = inputs[1].f64()?.get(0).unwrap_or(0.0);
    let with_intercept = inputs[2].bool()?.get(0).unwrap_or(true);
    let n_features = inputs.len() - 3;

    let (x, y) = match build_xy_data(inputs, 0, 3) {
        Ok(data) => data,
        Err(_) => return summary_nan_output(),
    };

    let mut builder = BinomialRegressor::probit().with_intercept(with_intercept);
    if lambda > 0.0 {
        builder = builder.lambda(lambda);
    }
    let model = builder.build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            let terms = build_term_names(n_features, with_intercept);

            let mut estimates = Vec::new();
            let mut std_errors = Vec::new();
            let mut statistics = Vec::new();
            let mut p_values = Vec::new();

            if with_intercept {
                estimates.push(fitted.intercept().unwrap_or(f64::NAN));
                std_errors.push(result.intercept_std_error.unwrap_or(f64::NAN));
                statistics.push(result.intercept_t_statistic.unwrap_or(f64::NAN));
                p_values.push(result.intercept_p_value.unwrap_or(f64::NAN));
            }

            let coefs = col_to_vec(fitted.coefficients());
            estimates.extend(coefs);

            if let Some(ref se) = result.std_errors {
                std_errors.extend(col_to_vec(se));
            } else {
                std_errors.extend(vec![f64::NAN; n_features]);
            }

            if let Some(ref zs) = result.t_statistics {
                statistics.extend(col_to_vec(zs));
            } else {
                statistics.extend(vec![f64::NAN; n_features]);
            }

            if let Some(ref pv) = result.p_values {
                p_values.extend(col_to_vec(pv));
            } else {
                p_values.extend(vec![f64::NAN; n_features]);
            }

            summary_output(terms, estimates, std_errors, statistics, p_values)
        }
        Err(_) => summary_nan_output(),
    }
}

/// Cloglog summary expression
/// inputs[0] = y, inputs[1] = lambda (L2 regularization), inputs[2] = with_intercept, inputs[3..] = x columns
#[polars_expr(output_type_func=summary_output_dtype)]
fn pl_cloglog_summary(inputs: &[Series]) -> PolarsResult<Series> {
    let lambda = inputs[1].f64()?.get(0).unwrap_or(0.0);
    let with_intercept = inputs[2].bool()?.get(0).unwrap_or(true);
    let n_features = inputs.len() - 3;

    let (x, y) = match build_xy_data(inputs, 0, 3) {
        Ok(data) => data,
        Err(_) => return summary_nan_output(),
    };

    let mut builder = BinomialRegressor::cloglog().with_intercept(with_intercept);
    if lambda > 0.0 {
        builder = builder.lambda(lambda);
    }
    let model = builder.build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            let terms = build_term_names(n_features, with_intercept);

            let mut estimates = Vec::new();
            let mut std_errors = Vec::new();
            let mut statistics = Vec::new();
            let mut p_values = Vec::new();

            if with_intercept {
                estimates.push(fitted.intercept().unwrap_or(f64::NAN));
                std_errors.push(result.intercept_std_error.unwrap_or(f64::NAN));
                statistics.push(result.intercept_t_statistic.unwrap_or(f64::NAN));
                p_values.push(result.intercept_p_value.unwrap_or(f64::NAN));
            }

            let coefs = col_to_vec(fitted.coefficients());
            estimates.extend(coefs);

            if let Some(ref se) = result.std_errors {
                std_errors.extend(col_to_vec(se));
            } else {
                std_errors.extend(vec![f64::NAN; n_features]);
            }

            if let Some(ref zs) = result.t_statistics {
                statistics.extend(col_to_vec(zs));
            } else {
                statistics.extend(vec![f64::NAN; n_features]);
            }

            if let Some(ref pv) = result.p_values {
                p_values.extend(col_to_vec(pv));
            } else {
                p_values.extend(vec![f64::NAN; n_features]);
            }

            summary_output(terms, estimates, std_errors, statistics, p_values)
        }
        Err(_) => summary_nan_output(),
    }
}

// ============================================================================
// Prediction Expressions
// ============================================================================

/// Kwargs for prediction functions - contains the prefix for output column names
#[derive(Deserialize)]
pub struct PredictKwargs {
    /// Prefix for output column names (e.g., "ols", "ridge", or user-provided)
    pub prefix: String,
}

/// Output type for prediction with intervals - uses prefix from kwargs
fn predict_output_dtype_with_prefix(
    _input_fields: &[Field],
    kwargs: PredictKwargs,
) -> PolarsResult<Field> {
    let prefix = &kwargs.prefix;
    let fields = vec![
        Field::new(format!("{}_prediction", prefix).into(), DataType::Float64),
        Field::new(format!("{}_lower", prefix).into(), DataType::Float64),
        Field::new(format!("{}_upper", prefix).into(), DataType::Float64),
    ];
    Ok(Field::new(
        format!("{}_predictions", prefix).into(),
        DataType::Struct(fields),
    ))
}

/// Create prediction output with prefixed field names
fn prediction_output_with_prefix(
    predictions: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    prefix: &str,
) -> PolarsResult<Series> {
    let n = predictions.len();

    let pred_s = Series::new(format!("{}_prediction", prefix).into(), predictions);
    let lower_s = Series::new(format!("{}_lower", prefix).into(), lower);
    let upper_s = Series::new(format!("{}_upper", prefix).into(), upper);

    let struct_ca = StructChunked::from_series(
        format!("{}_predictions", prefix).into(),
        n,
        [&pred_s, &lower_s, &upper_s].into_iter(),
    )?;

    Ok(struct_ca.into_series())
}

/// Create empty prediction output with prefixed field names
fn prediction_nan_output_with_prefix(n: usize, prefix: &str) -> PolarsResult<Series> {
    prediction_output_with_prefix(
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        vec![f64::NAN; n],
        prefix,
    )
}

/// Build X matrix and y vector with null handling based on policy.
/// Returns (X, y, valid_mask) where valid_mask indicates which original rows were used.
fn build_xy_with_null_policy(
    inputs: &[Series],
    y_idx: usize,
    x_start_idx: usize,
    null_policy: &str,
) -> PolarsResult<XyNullPolicyResult> {
    let y_series = inputs[y_idx].f64()?;
    let n_rows = y_series.len();
    let n_features = inputs.len() - x_start_idx;

    match null_policy {
        "drop_y_zero_x" => {
            // Drop rows with null y, zero fill X
            let mut valid_mask = vec![true; n_rows];

            // Check y for nulls only
            for (i, mask) in valid_mask.iter_mut().enumerate() {
                if y_series.get(i).is_none() {
                    *mask = false;
                }
            }

            let valid_indices: Vec<usize> = valid_mask
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| if v { Some(i) } else { None })
                .collect();

            let n_valid = valid_indices.len();

            if n_valid == 0 {
                return Err(polars_err!(ComputeError: "No valid rows after dropping null targets"));
            }

            // Build y for fitting (only valid rows)
            let y = Col::from_fn(n_valid, |i| y_series.get(valid_indices[i]).unwrap());

            // Build X for fitting (valid rows, zero fill nulls)
            let x_fit = Mat::from_fn(n_valid, n_features, |row, col| {
                inputs[x_start_idx + col]
                    .f64()
                    .ok()
                    .and_then(|ca| ca.get(valid_indices[row]))
                    .unwrap_or(0.0)
            });

            // Build X for prediction (all rows, zero fill nulls)
            let x_pred = Mat::from_fn(n_rows, n_features, |row, col| {
                inputs[x_start_idx + col]
                    .f64()
                    .ok()
                    .and_then(|ca| ca.get(row))
                    .unwrap_or(0.0)
            });

            Ok((x_fit, y, valid_mask, x_pred))
        }
        // Default: "drop" - drop rows with any nulls
        _ => {
            // Drop rows with any nulls - create mask of valid rows
            let mut valid_mask = vec![true; n_rows];

            // Check y for nulls
            for (i, mask) in valid_mask.iter_mut().enumerate() {
                if y_series.get(i).is_none() {
                    *mask = false;
                }
            }

            // Check X columns for nulls
            for col_idx in 0..n_features {
                if let Ok(col) = inputs[x_start_idx + col_idx].f64() {
                    for (i, mask) in valid_mask.iter_mut().enumerate() {
                        if col.get(i).is_none() {
                            *mask = false;
                        }
                    }
                }
            }

            let valid_indices: Vec<usize> = valid_mask
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| if v { Some(i) } else { None })
                .collect();

            let n_valid = valid_indices.len();

            if n_valid == 0 {
                return Err(polars_err!(ComputeError: "No valid rows after dropping nulls"));
            }

            // Build y and X for fitting (only valid rows)
            let y = Col::from_fn(n_valid, |i| y_series.get(valid_indices[i]).unwrap());
            let x_fit = Mat::from_fn(n_valid, n_features, |row, col| {
                inputs[x_start_idx + col]
                    .f64()
                    .ok()
                    .and_then(|ca| ca.get(valid_indices[row]))
                    .unwrap()
            });

            // X for prediction - same as fitting data (predictions will be masked later)
            // But we need full X for prediction with NaN where nulls exist
            let x_pred = Mat::from_fn(n_rows, n_features, |row, col| {
                inputs[x_start_idx + col]
                    .f64()
                    .ok()
                    .and_then(|ca| ca.get(row))
                    .unwrap_or(f64::NAN)
            });

            Ok((x_fit, y, valid_mask, x_pred))
        }
    }
}

/// OLS prediction expression - returns predictions with optional intervals.
/// inputs[0] = y, inputs[1] = with_intercept, inputs[2] = interval (string or null),
/// inputs[3] = level, inputs[4] = null_policy, inputs[5..] = x columns
#[polars_expr(output_type_func_with_kwargs=predict_output_dtype_with_prefix)]
fn pl_ols_predict(inputs: &[Series], kwargs: PredictKwargs) -> PolarsResult<Series> {
    let with_intercept = inputs[1].bool()?.get(0).unwrap_or(true);
    let interval = inputs[2].str()?.get(0); // None, "confidence", or "prediction"
    let level = inputs[3].f64()?.get(0).unwrap_or(0.95);
    let null_policy = inputs[4].str()?.get(0).unwrap_or("drop");
    let prefix = &kwargs.prefix;

    // Get number of rows from the first series
    let n_rows = inputs[0].len();

    // Build data with null handling
    let (x_fit, y, valid_mask, x_pred) = match build_xy_with_null_policy(inputs, 0, 5, null_policy)
    {
        Ok(data) => data,
        Err(_) => return prediction_nan_output_with_prefix(n_rows, prefix),
    };

    let model = OlsRegressor::builder()
        .with_intercept(with_intercept)
        .build();

    match model.fit(&x_fit, &y) {
        Ok(fitted) => {
            // Determine interval type
            let interval_type = match interval {
                Some("confidence") => Some(IntervalType::Confidence),
                Some("prediction") => Some(IntervalType::Prediction),
                _ => None,
            };

            // Get predictions on full data (x_pred)
            let pred_result = fitted.predict_with_interval(&x_pred, interval_type, level);

            // Apply mask for "drop" policy - set predictions to NaN where data was invalid
            let predictions: Vec<f64> = (0..n_rows)
                .map(|i| {
                    if null_policy == "drop" && !valid_mask[i] {
                        f64::NAN
                    } else {
                        pred_result.fit[i]
                    }
                })
                .collect();

            let (lower, upper) = if interval_type.is_some() {
                let lower: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        if null_policy == "drop" && !valid_mask[i] {
                            f64::NAN
                        } else {
                            pred_result.lower[i]
                        }
                    })
                    .collect();
                let upper: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        if null_policy == "drop" && !valid_mask[i] {
                            f64::NAN
                        } else {
                            pred_result.upper[i]
                        }
                    })
                    .collect();
                (lower, upper)
            } else {
                (vec![f64::NAN; n_rows], vec![f64::NAN; n_rows])
            };

            prediction_output_with_prefix(predictions, lower, upper, prefix)
        }
        Err(_) => prediction_nan_output_with_prefix(n_rows, prefix),
    }
}

/// Ridge prediction expression
/// inputs[0] = y, inputs[1] = with_intercept, inputs[2] = interval, inputs[3] = level,
/// inputs[4] = null_policy, inputs[5] = lambda, inputs[6..] = x columns
#[polars_expr(output_type_func_with_kwargs=predict_output_dtype_with_prefix)]
fn pl_ridge_predict(inputs: &[Series], kwargs: PredictKwargs) -> PolarsResult<Series> {
    let with_intercept = inputs[1].bool()?.get(0).unwrap_or(true);
    let interval = inputs[2].str()?.get(0);
    let level = inputs[3].f64()?.get(0).unwrap_or(0.95);
    let null_policy = inputs[4].str()?.get(0).unwrap_or("drop");
    let lambda = inputs[5].f64()?.get(0).unwrap_or(1.0);
    let prefix = &kwargs.prefix;

    let n_rows = inputs[0].len();

    let (x_fit, y, valid_mask, x_pred) = match build_xy_with_null_policy(inputs, 0, 6, null_policy)
    {
        Ok(data) => data,
        Err(_) => return prediction_nan_output_with_prefix(n_rows, prefix),
    };

    let model = RidgeRegressor::builder()
        .lambda(lambda)
        .with_intercept(with_intercept)
        .build();

    match model.fit(&x_fit, &y) {
        Ok(fitted) => {
            let interval_type = match interval {
                Some("confidence") => Some(IntervalType::Confidence),
                Some("prediction") => Some(IntervalType::Prediction),
                _ => None,
            };

            let pred_result = fitted.predict_with_interval(&x_pred, interval_type, level);

            let predictions: Vec<f64> = (0..n_rows)
                .map(|i| {
                    if null_policy == "drop" && !valid_mask[i] {
                        f64::NAN
                    } else {
                        pred_result.fit[i]
                    }
                })
                .collect();

            let (lower, upper) = if interval_type.is_some() {
                let lower: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        if null_policy == "drop" && !valid_mask[i] {
                            f64::NAN
                        } else {
                            pred_result.lower[i]
                        }
                    })
                    .collect();
                let upper: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        if null_policy == "drop" && !valid_mask[i] {
                            f64::NAN
                        } else {
                            pred_result.upper[i]
                        }
                    })
                    .collect();
                (lower, upper)
            } else {
                (vec![f64::NAN; n_rows], vec![f64::NAN; n_rows])
            };

            prediction_output_with_prefix(predictions, lower, upper, prefix)
        }
        Err(_) => prediction_nan_output_with_prefix(n_rows, prefix),
    }
}

/// Elastic Net prediction expression
/// inputs[0] = y, inputs[1] = with_intercept, inputs[2] = interval, inputs[3] = level,
/// inputs[4] = null_policy, inputs[5] = lambda, inputs[6] = alpha, inputs[7..] = x columns
#[polars_expr(output_type_func_with_kwargs=predict_output_dtype_with_prefix)]
fn pl_elastic_net_predict(inputs: &[Series], kwargs: PredictKwargs) -> PolarsResult<Series> {
    let with_intercept = inputs[1].bool()?.get(0).unwrap_or(true);
    let interval = inputs[2].str()?.get(0);
    let level = inputs[3].f64()?.get(0).unwrap_or(0.95);
    let null_policy = inputs[4].str()?.get(0).unwrap_or("drop");
    let lambda = inputs[5].f64()?.get(0).unwrap_or(1.0);
    let alpha = inputs[6].f64()?.get(0).unwrap_or(0.5);
    let prefix = &kwargs.prefix;

    let n_rows = inputs[0].len();

    let (x_fit, y, valid_mask, x_pred) = match build_xy_with_null_policy(inputs, 0, 7, null_policy)
    {
        Ok(data) => data,
        Err(_) => return prediction_nan_output_with_prefix(n_rows, prefix),
    };

    let model = ElasticNetRegressor::builder()
        .lambda(lambda)
        .alpha(alpha)
        .with_intercept(with_intercept)
        .build();

    match model.fit(&x_fit, &y) {
        Ok(fitted) => {
            let interval_type = match interval {
                Some("confidence") => Some(IntervalType::Confidence),
                Some("prediction") => Some(IntervalType::Prediction),
                _ => None,
            };

            let pred_result = fitted.predict_with_interval(&x_pred, interval_type, level);

            let predictions: Vec<f64> = (0..n_rows)
                .map(|i| {
                    if null_policy == "drop" && !valid_mask[i] {
                        f64::NAN
                    } else {
                        pred_result.fit[i]
                    }
                })
                .collect();

            let (lower, upper) = if interval_type.is_some() {
                let lower: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        if null_policy == "drop" && !valid_mask[i] {
                            f64::NAN
                        } else {
                            pred_result.lower[i]
                        }
                    })
                    .collect();
                let upper: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        if null_policy == "drop" && !valid_mask[i] {
                            f64::NAN
                        } else {
                            pred_result.upper[i]
                        }
                    })
                    .collect();
                (lower, upper)
            } else {
                (vec![f64::NAN; n_rows], vec![f64::NAN; n_rows])
            };

            prediction_output_with_prefix(predictions, lower, upper, prefix)
        }
        Err(_) => prediction_nan_output_with_prefix(n_rows, prefix),
    }
}

/// WLS prediction expression (weights column is inputs[6])
/// inputs[0] = y, inputs[1] = with_intercept, inputs[2] = interval, inputs[3] = level,
/// inputs[4] = null_policy, inputs[5] = weights, inputs[6..] = x columns
#[polars_expr(output_type_func_with_kwargs=predict_output_dtype_with_prefix)]
fn pl_wls_predict(inputs: &[Series], kwargs: PredictKwargs) -> PolarsResult<Series> {
    let with_intercept = inputs[1].bool()?.get(0).unwrap_or(true);
    let interval = inputs[2].str()?.get(0);
    let level = inputs[3].f64()?.get(0).unwrap_or(0.95);
    let null_policy = inputs[4].str()?.get(0).unwrap_or("drop");
    let prefix = &kwargs.prefix;

    let n_rows = inputs[0].len();

    // Build weights
    let weights_series = inputs[5].f64()?;
    let weights = Col::from_fn(n_rows, |i| weights_series.get(i).unwrap_or(1.0));

    let (x_fit, y, valid_mask, x_pred) = match build_xy_with_null_policy(inputs, 0, 6, null_policy)
    {
        Ok(data) => data,
        Err(_) => return prediction_nan_output_with_prefix(n_rows, prefix),
    };

    // Build weights for fitting (only valid rows)
    let valid_indices: Vec<usize> = valid_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v { Some(i) } else { None })
        .collect();
    let weights_fit = Col::from_fn(valid_indices.len(), |i| weights[valid_indices[i]]);

    let model = WlsRegressor::builder()
        .with_intercept(with_intercept)
        .weights(weights_fit)
        .build();

    match model.fit(&x_fit, &y) {
        Ok(fitted) => {
            let interval_type = match interval {
                Some("confidence") => Some(IntervalType::Confidence),
                Some("prediction") => Some(IntervalType::Prediction),
                _ => None,
            };

            let pred_result = fitted.predict_with_interval(&x_pred, interval_type, level);

            let predictions: Vec<f64> = (0..n_rows)
                .map(|i| {
                    if null_policy == "drop" && !valid_mask[i] {
                        f64::NAN
                    } else {
                        pred_result.fit[i]
                    }
                })
                .collect();

            let (lower, upper) = if interval_type.is_some() {
                let lower: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        if null_policy == "drop" && !valid_mask[i] {
                            f64::NAN
                        } else {
                            pred_result.lower[i]
                        }
                    })
                    .collect();
                let upper: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        if null_policy == "drop" && !valid_mask[i] {
                            f64::NAN
                        } else {
                            pred_result.upper[i]
                        }
                    })
                    .collect();
                (lower, upper)
            } else {
                (vec![f64::NAN; n_rows], vec![f64::NAN; n_rows])
            };

            prediction_output_with_prefix(predictions, lower, upper, prefix)
        }
        Err(_) => prediction_nan_output_with_prefix(n_rows, prefix),
    }
}

/// RLS prediction expression
/// inputs[0] = y, inputs[1] = with_intercept, inputs[2] = interval, inputs[3] = level,
/// inputs[4] = null_policy, inputs[5] = forgetting_factor, inputs[6..] = x columns
#[polars_expr(output_type_func_with_kwargs=predict_output_dtype_with_prefix)]
fn pl_rls_predict(inputs: &[Series], kwargs: PredictKwargs) -> PolarsResult<Series> {
    let with_intercept = inputs[1].bool()?.get(0).unwrap_or(true);
    let interval = inputs[2].str()?.get(0);
    let level = inputs[3].f64()?.get(0).unwrap_or(0.95);
    let null_policy = inputs[4].str()?.get(0).unwrap_or("drop");
    let forgetting_factor = inputs[5].f64()?.get(0).unwrap_or(0.99);
    let prefix = &kwargs.prefix;

    let n_rows = inputs[0].len();

    let (x_fit, y, valid_mask, x_pred) = match build_xy_with_null_policy(inputs, 0, 6, null_policy)
    {
        Ok(data) => data,
        Err(_) => return prediction_nan_output_with_prefix(n_rows, prefix),
    };

    let model = RlsRegressor::builder()
        .forgetting_factor(forgetting_factor)
        .with_intercept(with_intercept)
        .build();

    match model.fit(&x_fit, &y) {
        Ok(fitted) => {
            let interval_type = match interval {
                Some("confidence") => Some(IntervalType::Confidence),
                Some("prediction") => Some(IntervalType::Prediction),
                _ => None,
            };

            let pred_result = fitted.predict_with_interval(&x_pred, interval_type, level);

            let predictions: Vec<f64> = (0..n_rows)
                .map(|i| {
                    if null_policy == "drop" && !valid_mask[i] {
                        f64::NAN
                    } else {
                        pred_result.fit[i]
                    }
                })
                .collect();

            let (lower, upper) = if interval_type.is_some() {
                let lower: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        if null_policy == "drop" && !valid_mask[i] {
                            f64::NAN
                        } else {
                            pred_result.lower[i]
                        }
                    })
                    .collect();
                let upper: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        if null_policy == "drop" && !valid_mask[i] {
                            f64::NAN
                        } else {
                            pred_result.upper[i]
                        }
                    })
                    .collect();
                (lower, upper)
            } else {
                (vec![f64::NAN; n_rows], vec![f64::NAN; n_rows])
            };

            prediction_output_with_prefix(predictions, lower, upper, prefix)
        }
        Err(_) => prediction_nan_output_with_prefix(n_rows, prefix),
    }
}

/// BLS prediction expression
/// inputs[0] = y, inputs[1] = with_intercept, inputs[2] = interval, inputs[3] = level,
/// inputs[4] = null_policy, inputs[5] = lower_bound (or null), inputs[6] = upper_bound (or null),
/// inputs[7..] = x columns
#[polars_expr(output_type_func_with_kwargs=predict_output_dtype_with_prefix)]
fn pl_bls_predict(inputs: &[Series], kwargs: PredictKwargs) -> PolarsResult<Series> {
    let with_intercept = inputs[1].bool()?.get(0).unwrap_or(true);
    let interval = inputs[2].str()?.get(0);
    let level = inputs[3].f64()?.get(0).unwrap_or(0.95);
    let null_policy = inputs[4].str()?.get(0).unwrap_or("drop");
    let lower_bound = inputs[5].f64()?.get(0);
    let upper_bound = inputs[6].f64()?.get(0);
    let prefix = &kwargs.prefix;

    let n_rows = inputs[0].len();

    let (x_fit, y, valid_mask, x_pred) = match build_xy_with_null_policy(inputs, 0, 7, null_policy)
    {
        Ok(data) => data,
        Err(_) => return prediction_nan_output_with_prefix(n_rows, prefix),
    };

    let mut builder = BlsRegressor::builder().with_intercept(with_intercept);
    if let Some(lb) = lower_bound {
        builder = builder.lower_bound_all(lb);
    }
    if let Some(ub) = upper_bound {
        builder = builder.upper_bound_all(ub);
    }
    let model = builder.build();

    match model.fit(&x_fit, &y) {
        Ok(fitted) => {
            let interval_type = match interval {
                Some("confidence") => Some(IntervalType::Confidence),
                Some("prediction") => Some(IntervalType::Prediction),
                _ => None,
            };

            let pred_result = fitted.predict_with_interval(&x_pred, interval_type, level);

            let predictions: Vec<f64> = (0..n_rows)
                .map(|i| {
                    if null_policy == "drop" && !valid_mask[i] {
                        f64::NAN
                    } else {
                        pred_result.fit[i]
                    }
                })
                .collect();

            let (lower, upper) = if interval_type.is_some() {
                let lower: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        if null_policy == "drop" && !valid_mask[i] {
                            f64::NAN
                        } else {
                            pred_result.lower[i]
                        }
                    })
                    .collect();
                let upper: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        if null_policy == "drop" && !valid_mask[i] {
                            f64::NAN
                        } else {
                            pred_result.upper[i]
                        }
                    })
                    .collect();
                (lower, upper)
            } else {
                (vec![f64::NAN; n_rows], vec![f64::NAN; n_rows])
            };

            prediction_output_with_prefix(predictions, lower, upper, prefix)
        }
        Err(_) => prediction_nan_output_with_prefix(n_rows, prefix),
    }
}

/// Logistic prediction expression
/// inputs[0] = y, inputs[1] = lambda (L2 regularization), inputs[2] = with_intercept, inputs[3] = interval, inputs[4] = level,
/// inputs[5] = null_policy, inputs[6..] = x columns
#[polars_expr(output_type_func_with_kwargs=predict_output_dtype_with_prefix)]
fn pl_logistic_predict(inputs: &[Series], kwargs: PredictKwargs) -> PolarsResult<Series> {
    let lambda = inputs[1].f64()?.get(0).unwrap_or(0.0);
    let with_intercept = inputs[2].bool()?.get(0).unwrap_or(true);
    let interval = inputs[3].str()?.get(0);
    let level = inputs[4].f64()?.get(0).unwrap_or(0.95);
    let null_policy = inputs[5].str()?.get(0).unwrap_or("drop");
    let prefix = &kwargs.prefix;

    let n_rows = inputs[0].len();

    let (x_fit, y, valid_mask, x_pred) = match build_xy_with_null_policy(inputs, 0, 6, null_policy)
    {
        Ok(data) => data,
        Err(_) => return prediction_nan_output_with_prefix(n_rows, prefix),
    };

    let mut builder = BinomialRegressor::builder().with_intercept(with_intercept);
    if lambda > 0.0 {
        builder = builder.lambda(lambda);
    }
    let model = builder.build();

    match model.fit(&x_fit, &y) {
        Ok(fitted) => {
            let interval_type = match interval {
                Some("confidence") => Some(IntervalType::Confidence),
                Some("prediction") => Some(IntervalType::Prediction),
                _ => None,
            };

            let pred_result = fitted.predict_with_interval(&x_pred, interval_type, level);

            let predictions: Vec<f64> = (0..n_rows)
                .map(|i| {
                    if null_policy == "drop" && !valid_mask[i] {
                        f64::NAN
                    } else {
                        pred_result.fit[i]
                    }
                })
                .collect();

            let (lower, upper) = if interval_type.is_some() {
                let lower: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        if null_policy == "drop" && !valid_mask[i] {
                            f64::NAN
                        } else {
                            pred_result.lower[i]
                        }
                    })
                    .collect();
                let upper: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        if null_policy == "drop" && !valid_mask[i] {
                            f64::NAN
                        } else {
                            pred_result.upper[i]
                        }
                    })
                    .collect();
                (lower, upper)
            } else {
                (vec![f64::NAN; n_rows], vec![f64::NAN; n_rows])
            };

            prediction_output_with_prefix(predictions, lower, upper, prefix)
        }
        Err(_) => prediction_nan_output_with_prefix(n_rows, prefix),
    }
}

/// Poisson prediction expression
/// inputs[0] = y, inputs[1] = lambda (L2 regularization), inputs[2] = with_intercept, inputs[3] = interval, inputs[4] = level,
/// inputs[5] = null_policy, inputs[6..] = x columns
#[polars_expr(output_type_func_with_kwargs=predict_output_dtype_with_prefix)]
fn pl_poisson_predict(inputs: &[Series], kwargs: PredictKwargs) -> PolarsResult<Series> {
    let lambda = inputs[1].f64()?.get(0).unwrap_or(0.0);
    let with_intercept = inputs[2].bool()?.get(0).unwrap_or(true);
    let interval = inputs[3].str()?.get(0);
    let level = inputs[4].f64()?.get(0).unwrap_or(0.95);
    let null_policy = inputs[5].str()?.get(0).unwrap_or("drop");
    let prefix = &kwargs.prefix;

    let n_rows = inputs[0].len();

    let (x_fit, y, valid_mask, x_pred) = match build_xy_with_null_policy(inputs, 0, 6, null_policy)
    {
        Ok(data) => data,
        Err(_) => return prediction_nan_output_with_prefix(n_rows, prefix),
    };

    let mut builder = PoissonRegressor::builder().with_intercept(with_intercept);
    if lambda > 0.0 {
        builder = builder.lambda(lambda);
    }
    let model = builder.build();

    match model.fit(&x_fit, &y) {
        Ok(fitted) => {
            let interval_type = match interval {
                Some("confidence") => Some(IntervalType::Confidence),
                Some("prediction") => Some(IntervalType::Prediction),
                _ => None,
            };

            let pred_result = fitted.predict_with_interval(&x_pred, interval_type, level);

            let predictions: Vec<f64> = (0..n_rows)
                .map(|i| {
                    if null_policy == "drop" && !valid_mask[i] {
                        f64::NAN
                    } else {
                        pred_result.fit[i]
                    }
                })
                .collect();

            let (lower, upper) = if interval_type.is_some() {
                let lower: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        if null_policy == "drop" && !valid_mask[i] {
                            f64::NAN
                        } else {
                            pred_result.lower[i]
                        }
                    })
                    .collect();
                let upper: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        if null_policy == "drop" && !valid_mask[i] {
                            f64::NAN
                        } else {
                            pred_result.upper[i]
                        }
                    })
                    .collect();
                (lower, upper)
            } else {
                (vec![f64::NAN; n_rows], vec![f64::NAN; n_rows])
            };

            prediction_output_with_prefix(predictions, lower, upper, prefix)
        }
        Err(_) => prediction_nan_output_with_prefix(n_rows, prefix),
    }
}

/// Negative Binomial prediction expression
/// inputs[0] = y, inputs[1] = lambda (L2 regularization), inputs[2] = with_intercept, inputs[3] = interval, inputs[4] = level,
/// inputs[5] = null_policy, inputs[6..] = x columns
#[polars_expr(output_type_func_with_kwargs=predict_output_dtype_with_prefix)]
fn pl_negative_binomial_predict(inputs: &[Series], kwargs: PredictKwargs) -> PolarsResult<Series> {
    let lambda = inputs[1].f64()?.get(0).unwrap_or(0.0);
    let with_intercept = inputs[2].bool()?.get(0).unwrap_or(true);
    let interval = inputs[3].str()?.get(0);
    let level = inputs[4].f64()?.get(0).unwrap_or(0.95);
    let null_policy = inputs[5].str()?.get(0).unwrap_or("drop");
    let prefix = &kwargs.prefix;

    let n_rows = inputs[0].len();

    let (x_fit, y, valid_mask, x_pred) = match build_xy_with_null_policy(inputs, 0, 6, null_policy)
    {
        Ok(data) => data,
        Err(_) => return prediction_nan_output_with_prefix(n_rows, prefix),
    };

    let mut builder = NegativeBinomialRegressor::builder().with_intercept(with_intercept);
    if lambda > 0.0 {
        builder = builder.lambda(lambda);
    }
    let model = builder.build();

    match model.fit(&x_fit, &y) {
        Ok(fitted) => {
            let interval_type = match interval {
                Some("confidence") => Some(IntervalType::Confidence),
                Some("prediction") => Some(IntervalType::Prediction),
                _ => None,
            };

            let pred_result = fitted.predict_with_interval(&x_pred, interval_type, level);

            let predictions: Vec<f64> = (0..n_rows)
                .map(|i| {
                    if null_policy == "drop" && !valid_mask[i] {
                        f64::NAN
                    } else {
                        pred_result.fit[i]
                    }
                })
                .collect();

            let (lower, upper) = if interval_type.is_some() {
                let lower: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        if null_policy == "drop" && !valid_mask[i] {
                            f64::NAN
                        } else {
                            pred_result.lower[i]
                        }
                    })
                    .collect();
                let upper: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        if null_policy == "drop" && !valid_mask[i] {
                            f64::NAN
                        } else {
                            pred_result.upper[i]
                        }
                    })
                    .collect();
                (lower, upper)
            } else {
                (vec![f64::NAN; n_rows], vec![f64::NAN; n_rows])
            };

            prediction_output_with_prefix(predictions, lower, upper, prefix)
        }
        Err(_) => prediction_nan_output_with_prefix(n_rows, prefix),
    }
}

/// Tweedie prediction expression
/// inputs[0] = y, inputs[1] = lambda (L2 regularization), inputs[2] = with_intercept, inputs[3] = interval, inputs[4] = level,
/// inputs[5] = null_policy, inputs[6] = var_power, inputs[7..] = x columns
#[polars_expr(output_type_func_with_kwargs=predict_output_dtype_with_prefix)]
fn pl_tweedie_predict(inputs: &[Series], kwargs: PredictKwargs) -> PolarsResult<Series> {
    let lambda = inputs[1].f64()?.get(0).unwrap_or(0.0);
    let with_intercept = inputs[2].bool()?.get(0).unwrap_or(true);
    let interval = inputs[3].str()?.get(0);
    let level = inputs[4].f64()?.get(0).unwrap_or(0.95);
    let null_policy = inputs[5].str()?.get(0).unwrap_or("drop");
    let var_power = inputs[6].f64()?.get(0).unwrap_or(1.5);
    let prefix = &kwargs.prefix;

    let n_rows = inputs[0].len();

    let (x_fit, y, valid_mask, x_pred) = match build_xy_with_null_policy(inputs, 0, 7, null_policy)
    {
        Ok(data) => data,
        Err(_) => return prediction_nan_output_with_prefix(n_rows, prefix),
    };

    let mut builder = TweedieRegressor::builder()
        .var_power(var_power)
        .with_intercept(with_intercept);

    if lambda > 0.0 {
        builder = builder.lambda(lambda);
    }

    let model = builder.build();

    match model.fit(&x_fit, &y) {
        Ok(fitted) => {
            let interval_type = match interval {
                Some("confidence") => Some(IntervalType::Confidence),
                Some("prediction") => Some(IntervalType::Prediction),
                _ => None,
            };

            let pred_result = fitted.predict_with_interval(&x_pred, interval_type, level);

            let predictions: Vec<f64> = (0..n_rows)
                .map(|i| {
                    if null_policy == "drop" && !valid_mask[i] {
                        f64::NAN
                    } else {
                        pred_result.fit[i]
                    }
                })
                .collect();

            let (lower, upper) = if interval_type.is_some() {
                let lower: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        if null_policy == "drop" && !valid_mask[i] {
                            f64::NAN
                        } else {
                            pred_result.lower[i]
                        }
                    })
                    .collect();
                let upper: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        if null_policy == "drop" && !valid_mask[i] {
                            f64::NAN
                        } else {
                            pred_result.upper[i]
                        }
                    })
                    .collect();
                (lower, upper)
            } else {
                (vec![f64::NAN; n_rows], vec![f64::NAN; n_rows])
            };

            prediction_output_with_prefix(predictions, lower, upper, prefix)
        }
        Err(_) => prediction_nan_output_with_prefix(n_rows, prefix),
    }
}

/// Probit prediction expression
/// inputs[0] = y, inputs[1] = lambda (L2 regularization), inputs[2] = with_intercept, inputs[3] = interval, inputs[4] = level,
/// inputs[5] = null_policy, inputs[6..] = x columns
#[polars_expr(output_type_func_with_kwargs=predict_output_dtype_with_prefix)]
fn pl_probit_predict(inputs: &[Series], kwargs: PredictKwargs) -> PolarsResult<Series> {
    let lambda = inputs[1].f64()?.get(0).unwrap_or(0.0);
    let with_intercept = inputs[2].bool()?.get(0).unwrap_or(true);
    let interval = inputs[3].str()?.get(0);
    let level = inputs[4].f64()?.get(0).unwrap_or(0.95);
    let null_policy = inputs[5].str()?.get(0).unwrap_or("drop");
    let prefix = &kwargs.prefix;

    let n_rows = inputs[0].len();

    let (x_fit, y, valid_mask, x_pred) = match build_xy_with_null_policy(inputs, 0, 6, null_policy)
    {
        Ok(data) => data,
        Err(_) => return prediction_nan_output_with_prefix(n_rows, prefix),
    };

    let mut builder = BinomialRegressor::probit().with_intercept(with_intercept);
    if lambda > 0.0 {
        builder = builder.lambda(lambda);
    }
    let model = builder.build();

    match model.fit(&x_fit, &y) {
        Ok(fitted) => {
            let interval_type = match interval {
                Some("confidence") => Some(IntervalType::Confidence),
                Some("prediction") => Some(IntervalType::Prediction),
                _ => None,
            };

            let pred_result = fitted.predict_with_interval(&x_pred, interval_type, level);

            let predictions: Vec<f64> = (0..n_rows)
                .map(|i| {
                    if null_policy == "drop" && !valid_mask[i] {
                        f64::NAN
                    } else {
                        pred_result.fit[i]
                    }
                })
                .collect();

            let (lower, upper) = if interval_type.is_some() {
                let lower: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        if null_policy == "drop" && !valid_mask[i] {
                            f64::NAN
                        } else {
                            pred_result.lower[i]
                        }
                    })
                    .collect();
                let upper: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        if null_policy == "drop" && !valid_mask[i] {
                            f64::NAN
                        } else {
                            pred_result.upper[i]
                        }
                    })
                    .collect();
                (lower, upper)
            } else {
                (vec![f64::NAN; n_rows], vec![f64::NAN; n_rows])
            };

            prediction_output_with_prefix(predictions, lower, upper, prefix)
        }
        Err(_) => prediction_nan_output_with_prefix(n_rows, prefix),
    }
}

/// Cloglog prediction expression
/// inputs[0] = y, inputs[1] = lambda (L2 regularization), inputs[2] = with_intercept, inputs[3] = interval, inputs[4] = level,
/// inputs[5] = null_policy, inputs[6..] = x columns
#[polars_expr(output_type_func_with_kwargs=predict_output_dtype_with_prefix)]
fn pl_cloglog_predict(inputs: &[Series], kwargs: PredictKwargs) -> PolarsResult<Series> {
    let lambda = inputs[1].f64()?.get(0).unwrap_or(0.0);
    let with_intercept = inputs[2].bool()?.get(0).unwrap_or(true);
    let interval = inputs[3].str()?.get(0);
    let level = inputs[4].f64()?.get(0).unwrap_or(0.95);
    let null_policy = inputs[5].str()?.get(0).unwrap_or("drop");
    let prefix = &kwargs.prefix;

    let n_rows = inputs[0].len();

    let (x_fit, y, valid_mask, x_pred) = match build_xy_with_null_policy(inputs, 0, 6, null_policy)
    {
        Ok(data) => data,
        Err(_) => return prediction_nan_output_with_prefix(n_rows, prefix),
    };

    let mut builder = BinomialRegressor::cloglog().with_intercept(with_intercept);
    if lambda > 0.0 {
        builder = builder.lambda(lambda);
    }
    let model = builder.build();

    match model.fit(&x_fit, &y) {
        Ok(fitted) => {
            let interval_type = match interval {
                Some("confidence") => Some(IntervalType::Confidence),
                Some("prediction") => Some(IntervalType::Prediction),
                _ => None,
            };

            let pred_result = fitted.predict_with_interval(&x_pred, interval_type, level);

            let predictions: Vec<f64> = (0..n_rows)
                .map(|i| {
                    if null_policy == "drop" && !valid_mask[i] {
                        f64::NAN
                    } else {
                        pred_result.fit[i]
                    }
                })
                .collect();

            let (lower, upper) = if interval_type.is_some() {
                let lower: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        if null_policy == "drop" && !valid_mask[i] {
                            f64::NAN
                        } else {
                            pred_result.lower[i]
                        }
                    })
                    .collect();
                let upper: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        if null_policy == "drop" && !valid_mask[i] {
                            f64::NAN
                        } else {
                            pred_result.upper[i]
                        }
                    })
                    .collect();
                (lower, upper)
            } else {
                (vec![f64::NAN; n_rows], vec![f64::NAN; n_rows])
            };

            prediction_output_with_prefix(predictions, lower, upper, prefix)
        }
        Err(_) => prediction_nan_output_with_prefix(n_rows, prefix),
    }
}

/// ALM prediction expression
/// inputs[0] = y, inputs[1] = with_intercept, inputs[2] = interval, inputs[3] = level,
/// inputs[4] = null_policy, inputs[5] = distribution, inputs[6..] = x columns
#[polars_expr(output_type_func_with_kwargs=predict_output_dtype_with_prefix)]
fn pl_alm_predict(inputs: &[Series], kwargs: PredictKwargs) -> PolarsResult<Series> {
    let with_intercept = inputs[1].bool()?.get(0).unwrap_or(true);
    let interval = inputs[2].str()?.get(0);
    let level = inputs[3].f64()?.get(0).unwrap_or(0.95);
    let null_policy = inputs[4].str()?.get(0).unwrap_or("drop");
    let dist_str = inputs[5].str()?.get(0).unwrap_or("normal");
    let prefix = &kwargs.prefix;

    let n_rows = inputs[0].len();

    let (x_fit, y, valid_mask, x_pred) = match build_xy_with_null_policy(inputs, 0, 6, null_policy)
    {
        Ok(data) => data,
        Err(_) => return prediction_nan_output_with_prefix(n_rows, prefix),
    };

    let distribution = match parse_alm_distribution(dist_str) {
        Some(d) => d,
        None => return prediction_nan_output_with_prefix(n_rows, prefix),
    };

    let model = AlmRegressor::builder()
        .distribution(distribution)
        .with_intercept(with_intercept)
        .build();

    match model.fit(&x_fit, &y) {
        Ok(fitted) => {
            let interval_type = match interval {
                Some("confidence") => Some(IntervalType::Confidence),
                Some("prediction") => Some(IntervalType::Prediction),
                _ => None,
            };

            let pred_result = fitted.predict_with_interval(&x_pred, interval_type, level);

            let predictions: Vec<f64> = (0..n_rows)
                .map(|i| {
                    if null_policy == "drop" && !valid_mask[i] {
                        f64::NAN
                    } else {
                        pred_result.fit[i]
                    }
                })
                .collect();

            let (lower, upper) = if interval_type.is_some() {
                let lower: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        if null_policy == "drop" && !valid_mask[i] {
                            f64::NAN
                        } else {
                            pred_result.lower[i]
                        }
                    })
                    .collect();
                let upper: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        if null_policy == "drop" && !valid_mask[i] {
                            f64::NAN
                        } else {
                            pred_result.upper[i]
                        }
                    })
                    .collect();
                (lower, upper)
            } else {
                (vec![f64::NAN; n_rows], vec![f64::NAN; n_rows])
            };

            prediction_output_with_prefix(predictions, lower, upper, prefix)
        }
        Err(_) => prediction_nan_output_with_prefix(n_rows, prefix),
    }
}

// ============================================================================
// GLM Summary Expressions
// ============================================================================

/// ALM summary expression
/// inputs[0] = y, inputs[1] = distribution, inputs[2] = with_intercept, inputs[3..] = x columns
#[polars_expr(output_type_func=summary_output_dtype)]
fn pl_alm_summary(inputs: &[Series]) -> PolarsResult<Series> {
    let dist_str = inputs[1].str()?.get(0).unwrap_or("normal");
    let with_intercept = inputs[2].bool()?.get(0).unwrap_or(true);
    let n_features = inputs.len() - 3;

    let (x, y) = match build_xy_data(inputs, 0, 3) {
        Ok(data) => data,
        Err(_) => return summary_nan_output(),
    };

    let distribution = match parse_alm_distribution(dist_str) {
        Some(d) => d,
        None => return summary_nan_output(),
    };

    let model = AlmRegressor::builder()
        .distribution(distribution)
        .with_intercept(with_intercept)
        .build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            let terms = build_term_names(n_features, with_intercept);

            let mut estimates = Vec::new();
            let mut std_errors = Vec::new();
            let mut statistics = Vec::new();
            let mut p_values = Vec::new();

            if with_intercept {
                estimates.push(fitted.intercept().unwrap_or(f64::NAN));
                std_errors.push(result.intercept_std_error.unwrap_or(f64::NAN));
                statistics.push(result.intercept_t_statistic.unwrap_or(f64::NAN));
                p_values.push(result.intercept_p_value.unwrap_or(f64::NAN));
            }

            let coefs = col_to_vec(fitted.coefficients());
            estimates.extend(coefs);

            if let Some(ref se) = result.std_errors {
                std_errors.extend(col_to_vec(se));
            } else {
                std_errors.extend(vec![f64::NAN; n_features]);
            }

            if let Some(ref zs) = result.t_statistics {
                statistics.extend(col_to_vec(zs));
            } else {
                statistics.extend(vec![f64::NAN; n_features]);
            }

            if let Some(ref pv) = result.p_values {
                p_values.extend(col_to_vec(pv));
            } else {
                p_values.extend(vec![f64::NAN; n_features]);
            }

            summary_output(terms, estimates, std_errors, statistics, p_values)
        }
        Err(_) => summary_nan_output(),
    }
}

// ============================================================================
// AID (Automatic Identification of Demand) Expression
// ============================================================================

/// Output type for AID classification
fn aid_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("demand_type".into(), DataType::String),
        Field::new("is_intermittent".into(), DataType::Boolean),
        Field::new("is_fractional".into(), DataType::Boolean),
        Field::new("distribution".into(), DataType::String),
        Field::new("mean".into(), DataType::Float64),
        Field::new("variance".into(), DataType::Float64),
        Field::new("zero_proportion".into(), DataType::Float64),
        Field::new("n_observations".into(), DataType::UInt32),
        // Anomaly summary fields
        Field::new("has_stockouts".into(), DataType::Boolean),
        Field::new("is_new_product".into(), DataType::Boolean),
        Field::new("is_obsolete_product".into(), DataType::Boolean),
        Field::new("stockout_count".into(), DataType::UInt32),
        Field::new("new_product_count".into(), DataType::UInt32),
        Field::new("obsolete_product_count".into(), DataType::UInt32),
        Field::new("high_outlier_count".into(), DataType::UInt32),
        Field::new("low_outlier_count".into(), DataType::UInt32),
    ];
    Ok(Field::new("aid".into(), DataType::Struct(fields)))
}

/// Convert DemandType to string
fn demand_type_str(dt: DemandType) -> &'static str {
    match dt {
        DemandType::Regular => "regular",
        DemandType::Intermittent => "intermittent",
    }
}

/// Convert DemandDistribution to string
fn demand_distribution_str(dist: DemandDistribution) -> &'static str {
    match dist {
        DemandDistribution::Poisson => "poisson",
        DemandDistribution::NegativeBinomial => "negative_binomial",
        DemandDistribution::Geometric => "geometric",
        DemandDistribution::Normal => "normal",
        DemandDistribution::Gamma => "gamma",
        DemandDistribution::LogNormal => "lognormal",
        DemandDistribution::RectifiedNormal => "rectified_normal",
    }
}

/// Anomaly counts structure
struct AnomalyCounts {
    stockout: u32,
    new_product: u32,
    obsolete_product: u32,
    high_outlier: u32,
    low_outlier: u32,
}

impl AnomalyCounts {
    fn from_anomalies(anomalies: &[AnomalyType]) -> Self {
        let mut counts = AnomalyCounts {
            stockout: 0,
            new_product: 0,
            obsolete_product: 0,
            high_outlier: 0,
            low_outlier: 0,
        };
        for a in anomalies {
            match a {
                AnomalyType::Stockout => counts.stockout += 1,
                AnomalyType::NewProduct => counts.new_product += 1,
                AnomalyType::ObsoleteProduct => counts.obsolete_product += 1,
                AnomalyType::HighOutlier => counts.high_outlier += 1,
                AnomalyType::LowOutlier => counts.low_outlier += 1,
                AnomalyType::None => {}
            }
        }
        counts
    }

    fn has_stockouts(&self) -> bool {
        self.stockout > 0
    }

    fn is_new_product(&self) -> bool {
        self.new_product > 0
    }

    fn is_obsolete_product(&self) -> bool {
        self.obsolete_product > 0
    }
}

/// Create AID output struct
#[allow(clippy::too_many_arguments)]
fn aid_output(
    demand_type: &str,
    is_intermittent: bool,
    is_fractional: bool,
    distribution: &str,
    mean: f64,
    variance: f64,
    zero_proportion: f64,
    n_observations: u32,
    anomaly_counts: &AnomalyCounts,
) -> PolarsResult<Series> {
    let fields = vec![
        Series::new("demand_type".into(), vec![demand_type]),
        Series::new("is_intermittent".into(), vec![is_intermittent]),
        Series::new("is_fractional".into(), vec![is_fractional]),
        Series::new("distribution".into(), vec![distribution]),
        Series::new("mean".into(), vec![mean]),
        Series::new("variance".into(), vec![variance]),
        Series::new("zero_proportion".into(), vec![zero_proportion]),
        Series::new("n_observations".into(), vec![n_observations]),
        // Anomaly summary fields
        Series::new("has_stockouts".into(), vec![anomaly_counts.has_stockouts()]),
        Series::new(
            "is_new_product".into(),
            vec![anomaly_counts.is_new_product()],
        ),
        Series::new(
            "is_obsolete_product".into(),
            vec![anomaly_counts.is_obsolete_product()],
        ),
        Series::new("stockout_count".into(), vec![anomaly_counts.stockout]),
        Series::new("new_product_count".into(), vec![anomaly_counts.new_product]),
        Series::new(
            "obsolete_product_count".into(),
            vec![anomaly_counts.obsolete_product],
        ),
        Series::new(
            "high_outlier_count".into(),
            vec![anomaly_counts.high_outlier],
        ),
        Series::new("low_outlier_count".into(), vec![anomaly_counts.low_outlier]),
    ];

    StructChunked::from_series("aid".into(), 1, fields.iter()).map(|ca| ca.into_series())
}

/// Create AID NaN output
fn aid_nan_output() -> PolarsResult<Series> {
    let empty_counts = AnomalyCounts {
        stockout: 0,
        new_product: 0,
        obsolete_product: 0,
        high_outlier: 0,
        low_outlier: 0,
    };
    aid_output(
        "unknown",
        false,
        false,
        "unknown",
        f64::NAN,
        f64::NAN,
        f64::NAN,
        0,
        &empty_counts,
    )
}

/// Automatic Identification of Demand (AID) classifier.
///
/// Classifies demand patterns as regular or intermittent and selects the
/// best-fitting distribution.
///
/// # Arguments
/// * inputs[0] - y: demand time series
/// * inputs[1] - intermittent_threshold: threshold for classifying as intermittent (0.0 to 1.0)
/// * inputs[2] - detect_anomalies: whether to detect anomalies (boolean)
#[polars_expr(output_type_func=aid_output_dtype)]
fn pl_aid(inputs: &[Series]) -> PolarsResult<Series> {
    let intermittent_threshold = inputs[1].f64()?.get(0).unwrap_or(0.3);
    let detect_anomalies = inputs[2].bool()?.get(0).unwrap_or(true);

    let y_series = inputs[0].f64()?;
    let n_rows = y_series.len();

    if n_rows == 0 {
        return aid_nan_output();
    }

    // Build y vector
    let y_vec: Vec<f64> = y_series.into_no_null_iter().collect();
    let y = Col::from_fn(y_vec.len(), |i| y_vec[i]);

    let classifier = AidClassifier::builder()
        .intermittent_threshold(intermittent_threshold)
        .detect_anomalies(detect_anomalies)
        .build();

    let result = classifier.classify(&y);
    let anomaly_counts = AnomalyCounts::from_anomalies(&result.anomalies);

    aid_output(
        demand_type_str(result.demand_type),
        result.demand_type == DemandType::Intermittent,
        result.is_fractional,
        demand_distribution_str(result.distribution),
        result.parameters.mean,
        result.parameters.variance,
        result.zero_proportion,
        result.n_observations as u32,
        &anomaly_counts,
    )
}

// ============================================================================
// AID Anomalies Expression
// ============================================================================

/// Output type for AID anomalies (per-observation)
fn aid_anomalies_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("stockout".into(), DataType::Boolean),
        Field::new("new_product".into(), DataType::Boolean),
        Field::new("obsolete_product".into(), DataType::Boolean),
        Field::new("high_outlier".into(), DataType::Boolean),
        Field::new("low_outlier".into(), DataType::Boolean),
    ];
    Ok(Field::new("aid_anomalies".into(), DataType::Struct(fields)))
}

/// Create AID anomalies output struct (per-row, not aggregated)
fn aid_anomalies_output(
    stockout: Vec<bool>,
    new_product: Vec<bool>,
    obsolete_product: Vec<bool>,
    high_outlier: Vec<bool>,
    low_outlier: Vec<bool>,
) -> PolarsResult<Series> {
    let n = stockout.len();

    let stockout_s = Series::new("stockout".into(), stockout);
    let new_product_s = Series::new("new_product".into(), new_product);
    let obsolete_product_s = Series::new("obsolete_product".into(), obsolete_product);
    let high_outlier_s = Series::new("high_outlier".into(), high_outlier);
    let low_outlier_s = Series::new("low_outlier".into(), low_outlier);

    let struct_ca = StructChunked::from_series(
        "aid_anomalies".into(),
        n,
        [
            &stockout_s,
            &new_product_s,
            &obsolete_product_s,
            &high_outlier_s,
            &low_outlier_s,
        ]
        .into_iter(),
    )?;

    Ok(struct_ca.into_series())
}

/// Create AID anomalies empty output
fn aid_anomalies_nan_output(n_rows: usize) -> PolarsResult<Series> {
    aid_anomalies_output(
        vec![false; n_rows],
        vec![false; n_rows],
        vec![false; n_rows],
        vec![false; n_rows],
        vec![false; n_rows],
    )
}

/// AID Anomalies expression - returns per-observation anomaly flags.
///
/// # Arguments
/// * inputs[0] - y: demand time series
/// * inputs[1] - intermittent_threshold: threshold for classifying as intermittent (0.0 to 1.0)
#[polars_expr(output_type_func=aid_anomalies_output_dtype)]
fn pl_aid_anomalies(inputs: &[Series]) -> PolarsResult<Series> {
    let intermittent_threshold = inputs[1].f64()?.get(0).unwrap_or(0.3);

    let y_series = inputs[0].f64()?;
    let n_rows = y_series.len();

    if n_rows == 0 {
        return aid_anomalies_nan_output(0);
    }

    // Build y vector
    let y_vec: Vec<f64> = y_series.into_no_null_iter().collect();
    let y = Col::from_fn(y_vec.len(), |i| y_vec[i]);

    let classifier = AidClassifier::builder()
        .intermittent_threshold(intermittent_threshold)
        .detect_anomalies(true)
        .build();

    let result = classifier.classify(&y);

    // Convert anomalies to per-type boolean vectors
    let mut stockout = vec![false; n_rows];
    let mut new_product = vec![false; n_rows];
    let mut obsolete_product = vec![false; n_rows];
    let mut high_outlier = vec![false; n_rows];
    let mut low_outlier = vec![false; n_rows];

    for (i, anomaly) in result.anomalies.iter().enumerate() {
        match anomaly {
            AnomalyType::Stockout => stockout[i] = true,
            AnomalyType::NewProduct => new_product[i] = true,
            AnomalyType::ObsoleteProduct => obsolete_product[i] = true,
            AnomalyType::HighOutlier => high_outlier[i] = true,
            AnomalyType::LowOutlier => low_outlier[i] = true,
            AnomalyType::None => {}
        }
    }

    aid_anomalies_output(
        stockout,
        new_product,
        obsolete_product,
        high_outlier,
        low_outlier,
    )
}

// ============================================================================
// Dynamic Linear Model (LmDynamic) Expression
// ============================================================================

/// Parse information criterion string
fn parse_ic(s: &str) -> Option<InformationCriterion> {
    match s.to_lowercase().as_str() {
        "aic" => Some(InformationCriterion::AIC),
        "aicc" | "aic_c" => Some(InformationCriterion::AICc),
        "bic" => Some(InformationCriterion::BIC),
        _ => None,
    }
}

/// Output type for dynamic linear model
fn lm_dynamic_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("intercept".into(), DataType::Float64),
        Field::new(
            "coefficients".into(),
            DataType::List(Box::new(DataType::Float64)),
        ),
        Field::new("r_squared".into(), DataType::Float64),
        Field::new("adj_r_squared".into(), DataType::Float64),
        Field::new("mse".into(), DataType::Float64),
        Field::new("rmse".into(), DataType::Float64),
        Field::new("n_observations".into(), DataType::UInt32),
    ];
    Ok(Field::new("lm_dynamic".into(), DataType::Struct(fields)))
}

/// Create dynamic linear model output struct
fn lm_dynamic_output(
    intercept: Option<f64>,
    coefficients: &[f64],
    r_squared: f64,
    adj_r_squared: f64,
    mse: f64,
    rmse: f64,
    n_observations: u32,
) -> PolarsResult<Series> {
    let coef_series = Series::new("".into(), coefficients);

    let fields = [
        Series::new("intercept".into(), vec![intercept.unwrap_or(f64::NAN)]),
        Series::new("coefficients".into(), vec![coef_series]),
        Series::new("r_squared".into(), vec![r_squared]),
        Series::new("adj_r_squared".into(), vec![adj_r_squared]),
        Series::new("mse".into(), vec![mse]),
        Series::new("rmse".into(), vec![rmse]),
        Series::new("n_observations".into(), vec![n_observations]),
    ];

    StructChunked::from_series("lm_dynamic".into(), 1, fields.iter()).map(|ca| ca.into_series())
}

/// Create dynamic linear model NaN output
fn lm_dynamic_nan_output() -> PolarsResult<Series> {
    lm_dynamic_output(None, &[], f64::NAN, f64::NAN, f64::NAN, f64::NAN, 0)
}

/// Dynamic Linear Model regression.
///
/// A time-varying parameter model that combines multiple candidate regression
/// models using pointwise information criteria weighting.
///
/// # Arguments
/// * inputs[0] - y: target variable
/// * inputs[1] - ic: information criterion ("aic", "aicc", "bic")
/// * inputs[2] - distribution: error distribution
/// * inputs[3] - lowess_span: LOWESS smoothing span (0.0 to disable, 0.05-1.0 to enable)
/// * inputs[4] - max_models: maximum number of candidate models
/// * inputs[5] - with_intercept: whether to include intercept
/// * inputs[6..] - x: feature variables
#[polars_expr(output_type_func=lm_dynamic_output_dtype)]
fn pl_lm_dynamic(inputs: &[Series]) -> PolarsResult<Series> {
    let ic_str = inputs[1].str()?.get(0).unwrap_or("aicc");
    let dist_str = inputs[2].str()?.get(0).unwrap_or("normal");
    let lowess_span = inputs[3].f64()?.get(0).unwrap_or(0.3);
    let max_models = inputs[4].u32()?.get(0).unwrap_or(64) as usize;
    let with_intercept = inputs[5].bool()?.get(0).unwrap_or(true);

    let (x, y) = match build_xy_data(inputs, 0, 6) {
        Ok(data) => data,
        Err(_) => return lm_dynamic_nan_output(),
    };

    let ic_type = match parse_ic(ic_str) {
        Some(ic) => ic,
        None => return lm_dynamic_nan_output(),
    };

    let distribution = match parse_alm_distribution(dist_str) {
        Some(d) => d,
        None => return lm_dynamic_nan_output(),
    };

    let mut builder = LmDynamicRegressor::builder()
        .ic(ic_type)
        .distribution(distribution)
        .with_intercept(with_intercept)
        .max_models(max_models);

    if lowess_span > 0.0 {
        builder = builder.lowess_span(lowess_span);
    } else {
        builder = builder.no_smoothing();
    }

    let model = builder.build();

    match model.fit(&x, &y) {
        Ok(fitted) => {
            let result = fitted.result();
            let coefficients = col_to_vec(&result.coefficients);

            lm_dynamic_output(
                result.intercept,
                &coefficients,
                result.r_squared,
                result.adj_r_squared,
                result.mse,
                result.rmse,
                result.n_observations as u32,
            )
        }
        Err(_) => lm_dynamic_nan_output(),
    }
}
