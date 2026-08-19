//! PyO3 wrapper for Gamma regression.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::solvers::{FittedGamma, FittedRegressor, GammaRegressor, Regressor};

use crate::utils::{IntoNumpy, ToFaer};

/// Gamma GLM regression model.
///
/// Fits a Gamma generalized linear model with a log link via IRLS. The Gamma
/// distribution is appropriate for continuous, strictly positive response variables
/// where the variance scales as the square of the mean (Var[Y] = φ · μ²).
///
/// Parameters
/// ----------
/// with_intercept : bool, default True
///     Whether to include an intercept term in the model.
/// max_iter : int, default 25
///     Maximum number of IRLS iterations.
/// tol : float, default 1e-8
///     Convergence tolerance for the IRLS algorithm.
/// lambda_ : float, default 0.0
///     L2 regularisation strength (0.0 = no regularisation).
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from polars_statistics import Gamma
/// >>> X = np.random.randn(100, 2)
/// >>> y = np.exp(X @ np.array([1.0, -0.5]) + 0.1 * np.random.randn(100))
/// >>> model = Gamma().fit(X, y)
/// >>> model.is_fitted()
/// True
/// >>> preds = model.predict(X)
/// >>> preds.shape
/// (100,)
#[pyclass(name = "Gamma")]
pub struct PyGamma {
    with_intercept: bool,
    max_iter: usize,
    tol: f64,
    lambda_: f64,
    fitted: Option<FittedGamma>,
}

#[pymethods]
impl PyGamma {
    #[new]
    #[pyo3(signature = (with_intercept=true, max_iter=25, tol=1e-8, lambda_=0.0))]
    fn new(with_intercept: bool, max_iter: usize, tol: f64, lambda_: f64) -> Self {
        Self {
            with_intercept,
            max_iter,
            tol,
            lambda_,
            fitted: None,
        }
    }

    /// Fit the Gamma GLM to training data.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray of shape (n_samples, n_features)
    ///     Feature matrix. Must be float64.
    /// y : numpy.ndarray of shape (n_samples,)
    ///     Response vector. All values must be strictly positive.
    ///
    /// Returns
    /// -------
    /// self
    ///     The fitted model (enables method chaining).
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_mat = x.to_faer();
        let y_col = y.to_faer();

        let model = GammaRegressor::builder()
            .with_intercept(slf.with_intercept)
            .max_iterations(slf.max_iter)
            .tolerance(slf.tol)
            .lambda(slf.lambda_)
            .build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        slf.fitted = Some(fitted);
        Ok(slf)
    }

    /// Predict response-scale values (μ = exp(η)) for new data.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray of shape (n_samples, n_features)
    ///     Feature matrix. Must be float64.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray of shape (n_samples,)
    ///     Predicted values on the response scale.
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

    /// Predict the linear predictor η = X·β + intercept.
    ///
    /// For a Gamma GLM with log link: μ = exp(η).
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray of shape (n_samples, n_features)
    ///     Feature matrix. Must be float64.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray of shape (n_samples,)
    ///     Linear predictor values on the link scale.
    fn predict_eta<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        let x_mat = x.to_faer();
        Ok(fitted.predict_eta(&x_mat).into_numpy(py))
    }

    /// Whether the model has been fitted.
    fn is_fitted(&self) -> bool {
        self.fitted.is_some()
    }

    /// Fitted slope coefficients (excludes intercept).
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.coefficients().into_numpy(py))
    }

    /// Fitted intercept, or None when with_intercept=False.
    #[getter]
    fn intercept(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.intercept())
    }

    /// Standard errors of the slope coefficients, or None when inference is disabled.
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

    /// P-values for the slope coefficients, or None when inference is disabled.
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

    /// Akaike Information Criterion.
    #[getter]
    fn aic(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(Some(fitted.result().aic))
    }

    /// Bayesian Information Criterion.
    #[getter]
    fn bic(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(Some(fitted.result().bic))
    }

    /// Whether the IRLS algorithm converged within max_iter iterations.
    ///
    /// Returns False only when error_on_non_convergence=False and IRLS
    /// exhausted max_iter without meeting the tolerance threshold.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.converged())
    }
}
