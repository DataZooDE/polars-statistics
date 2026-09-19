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
    #[pyo3(signature = (forgetting_factor=1.0, add_intercept=None, with_intercept=None))]
    fn new(
        py: Python<'_>,
        forgetting_factor: f64,
        add_intercept: Option<bool>,
        with_intercept: Option<bool>,
    ) -> PyResult<Self> {
        let with_intercept =
            crate::pymodels::errors::resolve_intercept(py, add_intercept, with_intercept, true)?;
        Ok(Self {
            forgetting_factor,
            with_intercept,
            fitted: None,
        })
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        crate::pymodels::errors::validate_xy("RLS", &x, &y)?;
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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RLS"))?;

        let x_mat = x.to_faer();
        let predictions = fitted.predict(&x_mat);

        Ok(predictions.into_numpy(py))
    }

    /// R² (coefficient of determination) of the prediction on ``(X, y)``.
    ///
    /// Defined as ``1 - SS_res / SS_tot``; ``1.0`` is a perfect fit. Consistent
    /// with :meth:`sklearn.base.RegressorMixin.score`.
    fn score<'py>(
        &self,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RLS"))?;
        let x_mat = x.to_faer();
        let y_col = y.to_faer();
        Ok(fitted.score(&x_mat, &y_col))
    }

    fn is_fitted(&self) -> bool {
        self.fitted.is_some()
    }

    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RLS"))?;

        Ok(fitted.coefficients().into_numpy(py))
    }

    #[getter]
    fn intercept(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RLS"))?;

        Ok(fitted.intercept())
    }

    #[getter]
    fn r_squared(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RLS"))?;

        Ok(fitted.r_squared())
    }

    #[getter]
    fn forgetting_factor_value(&self) -> f64 {
        self.forgetting_factor
    }
    /// Informative repr: class name plus key state; never panics if unfitted.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        crate::pymodels::ergonomics::repr(slf.as_any())
    }

    /// Return the results as a plain Python ``dict`` (``{"fitted": False}`` if unfitted).
    fn to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        crate::pymodels::ergonomics::to_dict(slf.as_any())
    }

    /// Readable multi-line summary of the results.
    fn summary(slf: &Bound<'_, Self>) -> PyResult<String> {
        crate::pymodels::ergonomics::generic_summary(slf.as_any())
    }
}
