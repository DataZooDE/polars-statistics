//! PyO3 wrapper for Recursive Least Squares regression.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::solvers::{FittedRegressor, Regressor, RlsRegressor};
use anofox_regression::{HasModelInfo, ModelInfo};

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
/// initial_p_diagonal : float, default 1e6
///     Diagonal delta of the initial covariance P0 = delta*I. Large values
///     express a vague prior (approaches batch OLS with forgetting_factor=1);
///     small values shrink early estimates toward zero and are numerically
///     more stable. Must be positive and finite.
#[pyclass(name = "RLS")]
pub struct PyRLS {
    forgetting_factor: f64,
    with_intercept: bool,
    initial_p_diagonal: f64,
    fitted: Option<Box<dyn FittedRegressor + Send + Sync>>,
    /// Captured from the concrete fit at `fit` time — `HasModelInfo` is not a
    /// supertrait of `FittedRegressor`, so it cannot be read back from the
    /// boxed `fitted` field once type-erased.
    model_info: Option<ModelInfo>,
}

#[pymethods]
impl PyRLS {
    #[new]
    #[pyo3(signature = (forgetting_factor=1.0, add_intercept=None, with_intercept=None, initial_p_diagonal=1e6))]
    fn new(
        py: Python<'_>,
        forgetting_factor: f64,
        add_intercept: Option<bool>,
        with_intercept: Option<bool>,
        initial_p_diagonal: f64,
    ) -> PyResult<Self> {
        let with_intercept =
            crate::pymodels::errors::resolve_intercept(py, add_intercept, with_intercept, true)?;
        Ok(Self {
            forgetting_factor,
            with_intercept,
            initial_p_diagonal,
            fitted: None,
            model_info: None,
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
            .initial_p_diagonal(slf.initial_p_diagonal)
            .build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        slf.model_info = Some(fitted.model_info());
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

    #[getter]
    fn initial_p_diagonal_value(&self) -> f64 {
        self.initial_p_diagonal
    }

    /// Model type, error family (R naming; None for estimators without a
    /// likelihood) and link of the fit (anofox-regression ModelInfo).
    #[getter]
    fn model_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let info = self
            .model_info
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RLS"))?;
        crate::pymodels::model_info::model_info_dict(py, info)
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
