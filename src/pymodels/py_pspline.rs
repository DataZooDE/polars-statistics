//! PyO3 wrapper for P-spline (penalized B-spline) smoother regression.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::solvers::{FittedPSpline, FittedRegressor, PSplineRegressor, Regressor};

use crate::utils::{IntoNumpy, ToFaer};

/// Penalized B-spline (P-spline) smoother regression.
///
/// Fits a smooth function of a single continuous predictor using a penalized
/// B-spline basis. The penalty order and smoothing parameter λ control the
/// degree of smoothing; λ is selected by GCV when not specified.
///
/// Parameters
/// ----------
/// n_basis : int, default 0
///     Number of B-spline basis functions. 0 means automatic (≈ n/4, clamped
///     to the range [6, 40]).
/// penalty_order : int, default 2
///     Order of the difference penalty (2 = cubic penalty on second
///     differences, matching the default in `mgcv`).
/// lambda_ : float or None, default None
///     Smoothing parameter. None means automatic GCV selection over a grid
///     spanning {1e-8, …, 1e8} × scale.
///
/// Notes
/// -----
/// The input matrix ``x`` must have exactly one column. Pass a design matrix
/// of shape ``(n, 1)`` — e.g. ``x.reshape(-1, 1)`` for a 1-D array.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from polars_statistics import PSpline
/// >>> x = np.linspace(0, 1, 60).reshape(-1, 1)
/// >>> y = np.sin(2 * np.pi * x.ravel()) + 0.1 * np.random.randn(60)
/// >>> model = PSpline().fit(x, y)
/// >>> model.is_fitted()
/// True
/// >>> model.edf
/// 5.3...
/// >>> preds = model.predict(x)
/// >>> preds.shape
/// (60,)
#[pyclass(name = "PSpline")]
pub struct PyPSpline {
    n_basis: usize,
    penalty_order: usize,
    lambda: Option<f64>,
    fitted: Option<FittedPSpline>,
}

#[pymethods]
impl PyPSpline {
    #[new]
    #[pyo3(signature = (n_basis=0, penalty_order=2, lambda_=None))]
    fn new(n_basis: usize, penalty_order: usize, lambda_: Option<f64>) -> Self {
        Self {
            n_basis,
            penalty_order,
            lambda: lambda_,
            fitted: None,
        }
    }

    /// Fit the P-spline smoother to data.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray of shape (n_samples, 1)
    ///     Single-column predictor matrix. Must be float64.
    ///     Multi-column input raises ``ValueError``.
    /// y : numpy.ndarray of shape (n_samples,)
    ///     Response vector.
    ///
    /// Returns
    /// -------
    /// self
    ///     The fitted model (enables method chaining).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``x`` has more than one column.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_mat = x.to_faer();
        let y_col = y.to_faer();

        // Validate single-column requirement (per RESEARCH Pitfall 2)
        if x_mat.ncols() != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "PSpline requires a single-column predictor matrix (n, 1).",
            ));
        }

        let mut model = PSplineRegressor::new()
            .with_n_basis(slf.n_basis)
            .with_penalty_order(slf.penalty_order);
        if let Some(lam) = slf.lambda {
            model = model.with_lambda(lam);
        }
        let model = model.build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        slf.fitted = Some(fitted);
        Ok(slf)
    }

    /// Predict smoothed response values for new data.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray of shape (n_samples, 1)
    ///     Single-column predictor matrix. Must be float64.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray of shape (n_samples,)
    ///     Predicted (smoothed) values.
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

    /// Effective degrees of freedom (trace of the smoother matrix).
    ///
    /// For a linear signal, ``edf ≈ 2``; for a smoothly varying function,
    /// ``edf`` reflects the complexity of the fitted curve.
    #[getter]
    fn edf(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(fitted.edf())
    }

    /// Residual variance estimate σ².
    #[getter]
    fn sigma2(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(fitted.sigma2())
    }

    /// B-spline basis coefficients (not user-interpretable directly).
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(fitted.coefficients().into_numpy(py))
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

    /// Root mean squared error of residuals.
    #[getter]
    fn rmse(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(fitted.result().rmse)
    }
}
