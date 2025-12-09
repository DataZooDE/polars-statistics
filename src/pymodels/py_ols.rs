//! PyO3 wrapper for OLS regression.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::prelude::*;
use anofox_regression::solvers::{FittedOls, OlsRegressor, Regressor};

use crate::utils::{IntoNumpy, ToFaer};

/// Ordinary Least Squares regression model.
///
/// Fits a linear model with coefficients w = (w1, ..., wp) to minimize
/// the residual sum of squares between the observed targets and the
/// predictions.
///
/// Parameters
/// ----------
/// with_intercept : bool, default True
///     Whether to include an intercept term in the model.
/// compute_inference : bool, default True
///     Whether to compute statistical inference (std errors, t-stats, p-values).
/// confidence_level : float, default 0.95
///     Confidence level for confidence intervals.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from polars_statistics import OLS
/// >>>
/// >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
/// >>> y = np.array([6, 8, 9, 11])
/// >>>
/// >>> model = OLS()
/// >>> model.fit(X, y)
/// >>> model.coefficients
/// array([1., 2.])
/// >>> model.intercept
/// 3.0
#[pyclass(name = "OLS")]
pub struct PyOLS {
    with_intercept: bool,
    compute_inference: bool,
    confidence_level: f64,
    fitted: Option<FittedOls>,
}

#[pymethods]
impl PyOLS {
    /// Create a new OLS model.
    #[new]
    #[pyo3(signature = (with_intercept=true, compute_inference=true, confidence_level=0.95))]
    fn new(with_intercept: bool, compute_inference: bool, confidence_level: f64) -> Self {
        Self {
            with_intercept,
            compute_inference,
            confidence_level,
            fitted: None,
        }
    }

    /// Fit the model to the data.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data.
    /// y : array-like of shape (n_samples,)
    ///     Target values.
    ///
    /// Returns
    /// -------
    /// self
    ///     Fitted model.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_mat = x.to_faer();
        let y_col = y.to_faer();

        let model = OlsRegressor::builder()
            .with_intercept(slf.with_intercept)
            .compute_inference(slf.compute_inference)
            .confidence_level(slf.confidence_level)
            .build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        slf.fitted = Some(fitted);
        Ok(slf)
    }

    /// Predict using the linear model.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Samples to predict.
    ///
    /// Returns
    /// -------
    /// array of shape (n_samples,)
    ///     Predicted values.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        let x_mat = x.to_faer();
        let predictions = fitted.predict(&x_mat);

        Ok(predictions.into_numpy(py))
    }

    /// Check if the model has been fitted.
    fn is_fitted(&self) -> bool {
        self.fitted.is_some()
    }

    /// Get the model coefficients (excluding intercept).
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.coefficients().into_numpy(py))
    }

    /// Get the intercept term.
    #[getter]
    fn intercept(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.intercept())
    }

    /// Get R-squared value.
    #[getter]
    fn r_squared(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.r_squared())
    }

    /// Get adjusted R-squared value.
    #[getter]
    fn adj_r_squared(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.result().adj_r_squared)
    }

    /// Get root mean squared error.
    #[getter]
    fn rmse(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.result().rmse)
    }

    /// Get mean squared error.
    #[getter]
    fn mse(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.result().mse)
    }

    /// Get F-statistic for overall model significance.
    #[getter]
    fn f_statistic(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.result().f_statistic)
    }

    /// Get p-value for F-statistic.
    #[getter]
    fn f_pvalue(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.result().f_pvalue)
    }

    /// Get Akaike Information Criterion.
    #[getter]
    fn aic(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.result().aic)
    }

    /// Get Bayesian Information Criterion.
    #[getter]
    fn bic(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.result().bic)
    }

    /// Get log-likelihood.
    #[getter]
    fn log_likelihood(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.result().log_likelihood)
    }

    /// Get standard errors of coefficients.
    #[getter]
    fn std_errors<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted
            .result()
            .std_errors
            .as_ref()
            .map(|se| se.into_numpy(py)))
    }

    /// Get t-statistics for coefficients.
    #[getter]
    fn t_statistics<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted
            .result()
            .t_statistics
            .as_ref()
            .map(|ts| ts.into_numpy(py)))
    }

    /// Get p-values for coefficient significance tests.
    #[getter]
    fn p_values<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted
            .result()
            .p_values
            .as_ref()
            .map(|pv| pv.into_numpy(py)))
    }

    /// Get residuals.
    #[getter]
    fn residuals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok((&fitted.result().residuals).into_numpy(py))
    }

    /// Get fitted values.
    #[getter]
    fn fitted_values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok((&fitted.result().fitted_values).into_numpy(py))
    }

    /// Get number of observations.
    #[getter]
    fn n_observations(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.result().n_observations)
    }

    /// Get number of parameters (including intercept).
    #[getter]
    fn n_parameters(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.result().n_parameters)
    }

    /// Get a formatted summary of the regression results.
    fn summary(&self) -> PyResult<String> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        let result = fitted.result();
        let mut summary = String::new();

        summary.push_str("OLS Regression Results\n");
        summary.push_str("=======================\n\n");

        summary.push_str(&format!(
            "Observations:     {:>10}\n",
            result.n_observations
        ));
        summary.push_str(&format!("Df Residuals:     {:>10}\n", result.residual_df()));
        summary.push_str(&format!("Df Model:         {:>10}\n", result.model_df()));
        summary.push('\n');

        summary.push_str(&format!("R-squared:        {:>10.6}\n", result.r_squared));
        summary.push_str(&format!(
            "Adj. R-squared:   {:>10.6}\n",
            result.adj_r_squared
        ));
        summary.push_str(&format!("F-statistic:      {:>10.4}\n", result.f_statistic));
        summary.push_str(&format!("Prob (F):         {:>10.4e}\n", result.f_pvalue));
        summary.push('\n');

        summary.push_str(&format!("AIC:              {:>10.2}\n", result.aic));
        summary.push_str(&format!("BIC:              {:>10.2}\n", result.bic));
        summary.push_str(&format!(
            "Log-Likelihood:   {:>10.2}\n",
            result.log_likelihood
        ));
        summary.push('\n');

        if let Some(ref se) = result.std_errors {
            summary.push_str("Coefficients:\n");
            summary.push_str("-------------------------------------------------\n");
            summary.push_str("            coef    std err          t    P>|t|\n");
            summary.push_str("-------------------------------------------------\n");

            // Intercept
            if let Some(intercept) = result.intercept {
                let int_se = result.intercept_std_error.unwrap_or(f64::NAN);
                let int_t = result.intercept_t_statistic.unwrap_or(f64::NAN);
                let int_p = result.intercept_p_value.unwrap_or(f64::NAN);
                summary.push_str(&format!(
                    "const   {:>8.4}  {:>8.4}  {:>9.3}  {:>8.4}\n",
                    intercept, int_se, int_t, int_p
                ));
            }

            // Coefficients
            for i in 0..result.coefficients.nrows() {
                let coef = result.coefficients[i];
                let coef_se = se[i];
                let coef_t = result.t_statistics.as_ref().map_or(f64::NAN, |t| t[i]);
                let coef_p = result.p_values.as_ref().map_or(f64::NAN, |p| p[i]);

                summary.push_str(&format!(
                    "x{}      {:>8.4}  {:>8.4}  {:>9.3}  {:>8.4}\n",
                    i + 1,
                    coef,
                    coef_se,
                    coef_t,
                    coef_p
                ));
            }
            summary.push_str("-------------------------------------------------\n");
        }

        Ok(summary)
    }
}
