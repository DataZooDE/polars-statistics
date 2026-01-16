//! PyO3 wrapper for Quantile regression.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::solvers::{FittedQuantile, FittedRegressor, QuantileRegressor, Regressor};

use crate::utils::{IntoNumpy, ToFaer};

/// Quantile Regression model.
///
/// Estimates conditional quantiles of the response variable using IRLS
/// (Iteratively Reweighted Least Squares). Provides robust regression
/// that is less sensitive to outliers than OLS.
///
/// Parameters
/// ----------
/// tau : float, default 0.5
///     The quantile to estimate (must be between 0 and 1).
///     0.5 corresponds to median regression.
/// with_intercept : bool, default True
///     Whether to include an intercept term in the model.
/// max_iterations : int, default 100
///     Maximum number of IRLS iterations.
/// tolerance : float, default 1e-6
///     Convergence tolerance for IRLS.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from polars_statistics import Quantile
/// >>>
/// >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
/// >>> y = np.array([6, 8, 9, 11])
/// >>>
/// >>> # Median regression (tau=0.5)
/// >>> model = Quantile(tau=0.5)
/// >>> model.fit(X, y)
/// >>> model.coefficients
/// array([1., 2.])
#[pyclass(name = "Quantile")]
pub struct PyQuantile {
    tau: f64,
    with_intercept: bool,
    max_iterations: usize,
    tolerance: f64,
    fitted: Option<FittedQuantile>,
}

#[pymethods]
impl PyQuantile {
    /// Create a new Quantile regression model.
    #[new]
    #[pyo3(signature = (tau=0.5, with_intercept=true, max_iterations=100, tolerance=1e-6))]
    fn new(tau: f64, with_intercept: bool, max_iterations: usize, tolerance: f64) -> Self {
        Self {
            tau,
            with_intercept,
            max_iterations,
            tolerance,
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

        let model = QuantileRegressor::builder()
            .tau(slf.tau)
            .with_intercept(slf.with_intercept)
            .max_iterations(slf.max_iterations)
            .tolerance(slf.tolerance)
            .build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        slf.fitted = Some(fitted);
        Ok(slf)
    }

    /// Predict using the quantile regression model.
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

    /// Get the quantile (tau) that was estimated.
    #[getter]
    fn tau(&self) -> f64 {
        self.tau
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

    /// Get the pseudo R-squared (Koenker-Machado).
    #[getter]
    fn pseudo_r_squared(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.pseudo_r_squared())
    }

    /// Get the check function loss (quantile loss).
    #[getter]
    fn check_loss(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.check_loss())
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

    /// Get a formatted summary of the regression results.
    fn summary(&self) -> PyResult<String> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        let result = fitted.result();
        let mut summary = String::new();

        summary.push_str(&format!(
            "Quantile Regression Results (tau = {:.2})\n",
            self.tau
        ));
        summary.push_str("==========================================\n\n");

        summary.push_str(&format!(
            "Observations:       {:>10}\n",
            result.n_observations
        ));
        summary.push_str(&format!(
            "Pseudo R-squared:   {:>10.6}\n",
            fitted.pseudo_r_squared()
        ));
        summary.push_str(&format!(
            "Check loss:         {:>10.4}\n",
            fitted.check_loss()
        ));
        summary.push('\n');

        summary.push_str("Coefficients:\n");
        summary.push_str("-------------------------------------------------\n");

        // Intercept
        if let Some(intercept) = result.intercept {
            summary.push_str(&format!("const   {:>12.6}\n", intercept));
        }

        // Coefficients
        for i in 0..result.coefficients.nrows() {
            summary.push_str(&format!(
                "x{}      {:>12.6}\n",
                i + 1,
                result.coefficients[i]
            ));
        }
        summary.push_str("-------------------------------------------------\n");

        Ok(summary)
    }
}
