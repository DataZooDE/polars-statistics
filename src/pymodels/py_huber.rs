//! PyO3 wrapper for Huber M-estimator regression (robust to outliers).

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::solvers::{FittedHuber, FittedRegressor, HuberRegressor, Regressor};

use crate::utils::{IntoNumpy, ToFaer};

/// Huber M-estimator regression (robust to outliers).
///
/// Fits a regression that down-weights observations whose scaled residual
/// magnitude exceeds `epsilon`. Useful when the response has heavy-tailed
/// noise or contains a small fraction of outliers.
///
/// Parameters
/// ----------
/// epsilon : float, default 1.35
///     Huber threshold parameter; must be > 1.0. Smaller values
///     down-weight more observations.
/// alpha : float, default 0.0001
///     L2 ridge penalty applied during the weighted least-squares update.
/// with_intercept : bool, default True
///     Whether to include an intercept term.
/// max_iter : int, default 100
///     Maximum IRLS iterations.
/// tol : float, default 1e-5
///     Convergence tolerance on the max coefficient change between iterations.
#[pyclass(name = "Huber")]
pub struct PyHuber {
    epsilon: f64,
    alpha: f64,
    with_intercept: bool,
    max_iter: usize,
    tol: f64,
    fitted: Option<FittedHuber>,
}

#[pymethods]
impl PyHuber {
    #[new]
    #[pyo3(signature = (epsilon=1.35, alpha=0.0001, add_intercept=None, with_intercept=None, max_iter=100, tol=1e-5))]
    fn new(
        py: Python<'_>,
        epsilon: f64,
        alpha: f64,
        add_intercept: Option<bool>,
        with_intercept: Option<bool>,
        max_iter: usize,
        tol: f64,
    ) -> PyResult<Self> {
        let with_intercept =
            crate::pymodels::errors::resolve_intercept(py, add_intercept, with_intercept, true)?;
        Ok(Self {
            epsilon,
            alpha,
            with_intercept,
            max_iter,
            tol,
            fitted: None,
        })
    }

    /// Fit the Huber regression model.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        crate::pymodels::errors::validate_xy("Huber", &x, &y)?;
        let x_mat = x.to_faer();
        let y_col = y.to_faer();

        let model = HuberRegressor::builder()
            .epsilon(slf.epsilon)
            .alpha(slf.alpha)
            .with_intercept(slf.with_intercept)
            .max_iterations(slf.max_iter)
            .tolerance(slf.tol)
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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Huber"))?;

        let x_mat = x.to_faer();
        Ok(fitted.predict(&x_mat).into_numpy(py))
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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Huber"))?;
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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Huber"))?;

        Ok(fitted.coefficients().into_numpy(py))
    }

    #[getter]
    fn intercept(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Huber"))?;

        Ok(fitted.intercept())
    }

    /// Robust scale estimate (sigma) from the MAD of residuals.
    #[getter]
    fn scale(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Huber"))?;

        Ok(fitted.scale())
    }

    /// Huber threshold parameter used during fitting.
    #[getter]
    fn epsilon(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Huber"))?;

        Ok(fitted.epsilon())
    }

    /// Boolean outlier mask: True where |residual| > epsilon * scale.
    #[getter]
    fn outliers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Huber"))?;

        Ok(PyArray1::from_slice(py, fitted.outliers()))
    }

    /// Number of observations flagged as outliers.
    #[getter]
    fn n_outliers(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Huber"))?;

        Ok(fitted.n_outliers())
    }

    #[getter]
    fn r_squared(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Huber"))?;

        Ok(fitted.result().r_squared)
    }

    #[getter]
    fn mse(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Huber"))?;

        Ok(fitted.result().mse)
    }

    #[getter]
    fn rmse(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Huber"))?;

        Ok(fitted.result().rmse)
    }

    #[getter]
    fn n_observations(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Huber"))?;

        Ok(fitted.result().n_observations)
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
