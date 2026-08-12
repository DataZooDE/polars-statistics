//! PyO3 wrapper for ARD (Automatic Relevance Determination) regression.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::solvers::{ArdRegression, FittedArd, FittedRegressor, Regressor};

use crate::utils::{IntoNumpy, ToFaer};

/// Automatic Relevance Determination (ARD) regression.
///
/// Extends Bayesian Ridge with per-feature precision hyperparameters `lambda_j`.
/// Features whose precision exceeds `threshold_lambda` are pruned (coefficient
/// set to zero). The `lambdas` getter returns all per-feature precisions
/// including pruned ones.
///
/// Parameters
/// ----------
/// fit_intercept : bool, default True
///     Whether to fit an intercept. The design matrix is centered when True.
/// max_iter : int, default 300
///     Maximum EM iterations.
/// tol : float, default 1e-3
///     Convergence tolerance on the max coefficient change.
/// alpha_1 : float, default 1e-6
///     Shape parameter of the Gamma prior over noise precision.
/// alpha_2 : float, default 1e-6
///     Rate parameter of the Gamma prior over noise precision.
/// lambda_1 : float, default 1e-6
///     Shape parameter of the Gamma prior over per-feature weight precision.
/// lambda_2 : float, default 1e-6
///     Rate parameter of the Gamma prior over per-feature weight precision.
/// threshold_lambda : float, default 10000.0
///     Features with precision above this threshold are pruned.
#[pyclass(name = "ARD")]
pub struct PyARD {
    fit_intercept: bool,
    max_iter: usize,
    tol: f64,
    alpha_1: f64,
    alpha_2: f64,
    lambda_1: f64,
    lambda_2: f64,
    threshold_lambda: f64,
    fitted: Option<FittedArd>,
}

#[pymethods]
impl PyARD {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (fit_intercept=true, max_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6, threshold_lambda=10000.0))]
    fn new(
        fit_intercept: bool,
        max_iter: usize,
        tol: f64,
        alpha_1: f64,
        alpha_2: f64,
        lambda_1: f64,
        lambda_2: f64,
        threshold_lambda: f64,
    ) -> Self {
        Self {
            fit_intercept,
            max_iter,
            tol,
            alpha_1,
            alpha_2,
            lambda_1,
            lambda_2,
            threshold_lambda,
            fitted: None,
        }
    }

    /// Fit the ARD regression model.
    ///
    /// Raises `ValueError` on singular design matrix or dimension mismatch.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_mat = x.to_faer();
        let y_col = y.to_faer();

        let model = ArdRegression::builder()
            .fit_intercept(slf.fit_intercept)
            .max_iter(slf.max_iter)
            .tolerance(slf.tol)
            .alpha_1(slf.alpha_1)
            .alpha_2(slf.alpha_2)
            .lambda_1(slf.lambda_1)
            .lambda_2(slf.lambda_2)
            .threshold_lambda(slf.threshold_lambda)
            .build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        slf.fitted = Some(fitted);
        Ok(slf)
    }

    /// Predict response values.
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

    fn is_fitted(&self) -> bool {
        self.fitted.is_some()
    }

    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.coefficients().into_numpy(py))
    }

    #[getter]
    fn intercept(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.intercept())
    }

    #[getter]
    fn r_squared(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.result().r_squared)
    }

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

    /// Per-feature precision hyperparameters (lambda_j).
    ///
    /// Length equals the number of input features. Features with
    /// `lambda_j > threshold_lambda` have been pruned (coefficient set to 0).
    #[getter]
    fn lambdas<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(PyArray1::from_slice(py, fitted.lambdas()))
    }

    #[getter]
    fn n_observations(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.result().n_observations)
    }
}
