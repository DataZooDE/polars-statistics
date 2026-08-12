//! PyO3 wrapper for Bayesian Ridge regression (empirical-Bayes regularization).

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::solvers::{BayesianRidge, FittedBayesianRidge, FittedRegressor, Regressor};

use crate::utils::{IntoNumpy, ToFaer};

/// Bayesian Ridge regression with automatic regularization via empirical Bayes.
///
/// Estimates the precision hyperparameters `alpha` (noise) and `lambda` (weights)
/// from the data using SVD-based EM updates. Robust to singular designs
/// (returns `ValueError` on degeneracy). Inference fields (`alpha_`, `lambda_`,
/// `sigma_diag`) are accessible after fitting.
///
/// Parameters
/// ----------
/// fit_intercept : bool, default True
///     Whether to fit an intercept. The design matrix is centered before
///     optimization when True.
/// max_iter : int, default 300
///     Maximum EM iterations.
/// tol : float, default 1e-3
///     Convergence tolerance on the max coefficient change.
/// alpha_1 : float, default 1e-6
///     Shape parameter of the Gamma prior over noise precision.
/// alpha_2 : float, default 1e-6
///     Rate parameter of the Gamma prior over noise precision.
/// lambda_1 : float, default 1e-6
///     Shape parameter of the Gamma prior over weight precision.
/// lambda_2 : float, default 1e-6
///     Rate parameter of the Gamma prior over weight precision.
/// alpha_init : float or None, default None
///     Initial value for noise precision. None → 1/Var(y).
/// lambda_init : float or None, default None
///     Initial value for weight precision. None → 1.0.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from polars_statistics import BayesianRidge
/// >>> X = np.random.randn(80, 3)
/// >>> y = X @ [1.0, -0.5, 0.2] + 0.3 * np.random.randn(80)
/// >>> model = BayesianRidge().fit(X, y)
/// >>> model.is_fitted()
/// True
/// >>> model.coefficients
/// array([...])
/// >>> model.alpha_  # estimated noise precision
/// ...
#[pyclass(name = "BayesianRidge")]
pub struct PyBayesianRidge {
    fit_intercept: bool,
    max_iter: usize,
    tol: f64,
    alpha_1: f64,
    alpha_2: f64,
    lambda_1: f64,
    lambda_2: f64,
    alpha_init: Option<f64>,
    lambda_init: Option<f64>,
    fitted: Option<FittedBayesianRidge>,
}

#[pymethods]
impl PyBayesianRidge {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (fit_intercept=true, max_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6, alpha_init=None, lambda_init=None))]
    fn new(
        fit_intercept: bool,
        max_iter: usize,
        tol: f64,
        alpha_1: f64,
        alpha_2: f64,
        lambda_1: f64,
        lambda_2: f64,
        alpha_init: Option<f64>,
        lambda_init: Option<f64>,
    ) -> Self {
        Self {
            fit_intercept,
            max_iter,
            tol,
            alpha_1,
            alpha_2,
            lambda_1,
            lambda_2,
            alpha_init,
            lambda_init,
            fitted: None,
        }
    }

    /// Fit the Bayesian Ridge regression model.
    ///
    /// Raises `ValueError` on singular design matrix or dimension mismatch.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_mat = x.to_faer();
        let y_col = y.to_faer();

        let mut b = BayesianRidge::builder()
            .fit_intercept(slf.fit_intercept)
            .max_iter(slf.max_iter)
            .tolerance(slf.tol)
            .alpha_1(slf.alpha_1)
            .alpha_2(slf.alpha_2)
            .lambda_1(slf.lambda_1)
            .lambda_2(slf.lambda_2);
        if let Some(ai) = slf.alpha_init {
            b = b.alpha_init(ai);
        }
        if let Some(li) = slf.lambda_init {
            b = b.lambda_init(li);
        }
        let model = b.build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        slf.fitted = Some(fitted);
        Ok(slf)
    }

    /// Predict response values for new data.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray of shape (n_samples, n_features)
    ///     Feature matrix. Must be float64.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray of shape (n_samples,)
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
        Ok(fitted.predict(&x_mat).into_numpy(py))
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

    /// Fitted intercept, or None when fit_intercept=False.
    #[getter]
    fn intercept(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.intercept())
    }

    /// Coefficient of determination R².
    #[getter]
    fn r_squared(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.result().r_squared)
    }

    /// Residuals (y − ŷ) for the training data.
    #[getter]
    fn residuals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok((&fitted.result().residuals).into_numpy(py))
    }

    /// Estimated noise precision hyperparameter (alpha).
    #[getter]
    fn alpha_(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.alpha())
    }

    /// Estimated weight precision hyperparameter (lambda).
    #[getter]
    fn lambda_(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.lambda())
    }

    /// Diagonal of the posterior covariance matrix (useful for prediction intervals).
    #[getter]
    fn sigma_diag<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(PyArray1::from_slice(py, fitted.sigma_diag()))
    }

    /// Number of observations used in the fit.
    #[getter]
    fn n_observations(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.result().n_observations)
    }
}
