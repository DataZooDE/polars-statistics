//! PyO3 wrapper for Elastic Net regression.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::solvers::{ElasticNetRegressor, FittedRegressor, Regressor};

use crate::utils::{IntoNumpy, ToFaer};

/// Elastic Net regression model (L1 + L2 regularization).
///
/// Combines L1 (Lasso) and L2 (Ridge) regularization.
/// The objective function is: ||y - Xw||² + lambda * alpha * ||w||₁ + lambda * (1 - alpha) * ||w||²
///
/// Parameters
/// ----------
/// lambda_ : float, default 1.0
///     Overall regularization strength.
/// alpha : float, default 0.5
///     The ratio of L1 regularization. 0 = Ridge, 1 = Lasso.
/// with_intercept : bool, default True
///     Whether to include an intercept term.
/// max_iter : int, default 1000
///     Maximum number of iterations for coordinate descent.
/// tol : float, default 1e-4
///     Tolerance for convergence.
#[pyclass(name = "ElasticNet")]
pub struct PyElasticNet {
    lambda_: f64,
    alpha: f64,
    with_intercept: bool,
    max_iter: usize,
    tol: f64,
    fitted: Option<Box<dyn FittedRegressor + Send + Sync>>,
}

#[pymethods]
impl PyElasticNet {
    #[new]
    #[pyo3(signature = (lambda_=1.0, alpha=0.5, with_intercept=true, max_iter=1000, tol=1e-4))]
    fn new(lambda_: f64, alpha: f64, with_intercept: bool, max_iter: usize, tol: f64) -> Self {
        Self {
            lambda_,
            alpha,
            with_intercept,
            max_iter,
            tol,
            fitted: None,
        }
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_mat = x.to_faer();
        let y_col = y.to_faer();

        let model = ElasticNetRegressor::builder()
            .lambda(slf.lambda_)
            .alpha(slf.alpha)
            .with_intercept(slf.with_intercept)
            .max_iterations(slf.max_iter)
            .tolerance(slf.tol)
            .build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        slf.fitted = Some(Box::new(fitted));
        Ok(slf)
    }

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

        Ok(fitted.r_squared())
    }

    #[getter]
    fn lambda_value(&self) -> f64 {
        self.lambda_
    }

    #[getter]
    fn alpha_value(&self) -> f64 {
        self.alpha
    }
}
