//! PyO3 wrapper for Recursive Least Squares regression.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::solvers::{FittedRegressor, Regressor, RlsRegressor};

use crate::utils::{IntoNumpy, ToFaer};

/// Recursive Least Squares regression model.
///
/// Online learning algorithm that updates coefficients incrementally.
/// Useful for streaming data and adaptive filtering.
///
/// Parameters
/// ----------
/// forgetting_factor : float, default 1.0
///     Weighting factor (0 < λ ≤ 1). Value of 1.0 gives equal weight to all
///     observations (converges to OLS). Values < 1.0 weight recent data more heavily.
/// with_intercept : bool, default True
///     Whether to include an intercept term in the model.
#[pyclass(name = "RLS")]
pub struct PyRLS {
    forgetting_factor: f64,
    with_intercept: bool,
    fitted: Option<Box<dyn FittedRegressor + Send + Sync>>,
}

#[pymethods]
impl PyRLS {
    #[new]
    #[pyo3(signature = (forgetting_factor=1.0, with_intercept=true))]
    fn new(forgetting_factor: f64, with_intercept: bool) -> Self {
        Self {
            forgetting_factor,
            with_intercept,
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

        let model = RlsRegressor::builder()
            .forgetting_factor(slf.forgetting_factor)
            .with_intercept(slf.with_intercept)
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
    fn forgetting_factor_value(&self) -> f64 {
        self.forgetting_factor
    }
}
