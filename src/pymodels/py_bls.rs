//! PyO3 wrapper for Bounded Least Squares regression.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::solvers::{BlsRegressor, FittedRegressor, Regressor};

use crate::utils::{IntoNumpy, ToFaer};

/// Bounded Least Squares regression model.
///
/// Fits a linear model with box constraints on the coefficients.
/// Can be used for non-negative least squares (NNLS) by setting lower_bound=0.
///
/// Parameters
/// ----------
/// lower_bound : float, optional
///     Lower bound for all coefficients. Use None for no lower bound.
/// upper_bound : float, optional
///     Upper bound for all coefficients. Use None for no upper bound.
/// with_intercept : bool, default True
///     Whether to include an intercept term in the model.
/// max_iter : int, default 1000
///     Maximum number of iterations for the active set algorithm.
/// tol : float, default 1e-6
///     Convergence tolerance.
#[pyclass(name = "BLS")]
pub struct PyBLS {
    lower_bound: Option<f64>,
    upper_bound: Option<f64>,
    with_intercept: bool,
    max_iter: usize,
    tol: f64,
    fitted: Option<Box<dyn FittedRegressor + Send + Sync>>,
}

#[pymethods]
impl PyBLS {
    #[new]
    #[pyo3(signature = (lower_bound=None, upper_bound=None, with_intercept=true, max_iter=1000, tol=1e-6))]
    fn new(
        lower_bound: Option<f64>,
        upper_bound: Option<f64>,
        with_intercept: bool,
        max_iter: usize,
        tol: f64,
    ) -> Self {
        Self {
            lower_bound,
            upper_bound,
            with_intercept,
            max_iter,
            tol,
            fitted: None,
        }
    }

    /// Create a Non-Negative Least Squares model (coefficients >= 0).
    #[staticmethod]
    #[pyo3(signature = (with_intercept=true, max_iter=1000, tol=1e-6))]
    fn nnls(with_intercept: bool, max_iter: usize, tol: f64) -> Self {
        Self {
            lower_bound: Some(0.0),
            upper_bound: None,
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

        let mut builder = BlsRegressor::builder()
            .with_intercept(slf.with_intercept)
            .max_iterations(slf.max_iter)
            .tolerance(slf.tol);

        if let Some(lb) = slf.lower_bound {
            builder = builder.lower_bound_all(lb);
        }
        if let Some(ub) = slf.upper_bound {
            builder = builder.upper_bound_all(ub);
        }

        let model = builder.build();

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
}
