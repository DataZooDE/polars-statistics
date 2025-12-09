//! Regression model expressions for Polars.
//!
//! These expressions allow fitting regression models within group_by and over operations.

use faer::{Col, Mat};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use anofox_regression::solvers::{
    AlmDistribution, AlmRegressor, BinomialRegressor, BlsRegressor, ElasticNetRegressor,
    FittedRegressor, NegativeBinomialRegressor, OlsRegressor, PoissonRegressor, Regressor,
    RidgeRegressor, RlsRegressor, TweedieRegressor, WlsRegressor,
};

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

/// Extract coefficients as Vec<f64>
fn col_to_vec(col: &Col<f64>) -> Vec<f64> {
    (0..col.nrows()).map(|i| col[i]).collect()
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
        .compute_inference(true)
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
// GLM Expressions
// ============================================================================

/// Logistic regression expression.
/// inputs[0] = y, inputs[1] = with_intercept, inputs[2..] = x columns
#[polars_expr(output_type_func=glm_output_dtype)]
fn pl_logistic(inputs: &[Series]) -> PolarsResult<Series> {
    let with_intercept = inputs[1].bool()?.get(0).unwrap_or(true);

    let (x, y) = match build_xy_data(inputs, 0, 2) {
        Ok(data) => data,
        Err(_) => return glm_nan_output(),
    };

    let model = BinomialRegressor::logistic()
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

/// Poisson regression expression.
/// inputs[0] = y, inputs[1] = with_intercept, inputs[2..] = x columns
#[polars_expr(output_type_func=glm_output_dtype)]
fn pl_poisson(inputs: &[Series]) -> PolarsResult<Series> {
    let with_intercept = inputs[1].bool()?.get(0).unwrap_or(true);

    let (x, y) = match build_xy_data(inputs, 0, 2) {
        Ok(data) => data,
        Err(_) => return glm_nan_output(),
    };

    let model = PoissonRegressor::builder()
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

/// Negative Binomial regression expression.
/// inputs[0] = y, inputs[1] = theta (optional), inputs[2] = with_intercept, inputs[3..] = x columns
#[polars_expr(output_type_func=glm_output_dtype)]
fn pl_negative_binomial(inputs: &[Series]) -> PolarsResult<Series> {
    let theta = inputs[1].f64()?.get(0);
    let with_intercept = inputs[2].bool()?.get(0).unwrap_or(true);

    let (x, y) = match build_xy_data(inputs, 0, 3) {
        Ok(data) => data,
        Err(_) => return glm_nan_output(),
    };

    let mut builder = NegativeBinomialRegressor::builder().with_intercept(with_intercept);

    if let Some(t) = theta {
        builder = builder.theta(t);
    } else {
        builder = builder.estimate_theta(true);
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
/// inputs[0] = y, inputs[1] = var_power, inputs[2] = with_intercept, inputs[3..] = x columns
#[polars_expr(output_type_func=glm_output_dtype)]
fn pl_tweedie(inputs: &[Series]) -> PolarsResult<Series> {
    let var_power = inputs[1].f64()?.get(0).unwrap_or(1.5);
    let with_intercept = inputs[2].bool()?.get(0).unwrap_or(true);

    let (x, y) = match build_xy_data(inputs, 0, 3) {
        Ok(data) => data,
        Err(_) => return glm_nan_output(),
    };

    let model = TweedieRegressor::builder()
        .var_power(var_power)
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

/// Probit regression expression.
/// inputs[0] = y, inputs[1] = with_intercept, inputs[2..] = x columns
#[polars_expr(output_type_func=glm_output_dtype)]
fn pl_probit(inputs: &[Series]) -> PolarsResult<Series> {
    let with_intercept = inputs[1].bool()?.get(0).unwrap_or(true);

    let (x, y) = match build_xy_data(inputs, 0, 2) {
        Ok(data) => data,
        Err(_) => return glm_nan_output(),
    };

    let model = BinomialRegressor::probit()
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

/// Complementary log-log regression expression.
/// inputs[0] = y, inputs[1] = with_intercept, inputs[2..] = x columns
#[polars_expr(output_type_func=glm_output_dtype)]
fn pl_cloglog(inputs: &[Series]) -> PolarsResult<Series> {
    let with_intercept = inputs[1].bool()?.get(0).unwrap_or(true);

    let (x, y) = match build_xy_data(inputs, 0, 2) {
        Ok(data) => data,
        Err(_) => return glm_nan_output(),
    };

    let model = BinomialRegressor::cloglog()
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
