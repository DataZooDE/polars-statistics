//! PyO3 wrapper for Theil-Sen robust regression.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::solvers::{FittedRegressor, FittedTheilSen, Regressor, TheilSenRegressor};

use crate::utils::{IntoNumpy, ToFaer};

/// Theil-Sen robust regression.
///
/// Computes the per-subsample ordinary least-squares estimate and takes the
/// element-wise spatial median (Weiszfeld algorithm) over all subsamples.
/// Robust to up to 29.3% of outliers; no analytic sampling distribution
/// (inference fields are NaN).
///
/// Parameters
/// ----------
/// with_intercept : bool, default True
///     Whether to include an intercept term.
/// max_subpopulation : int, default 10000
///     Maximum number of subsamples drawn. Larger values improve accuracy
///     at the cost of computation time.
/// n_subsamples : int or None, default None
///     Fixed number of subsamples per iteration. None → n_features + 1.
/// max_iter : int, default 300
///     Maximum Weiszfeld iterations for spatial median computation.
/// tol : float, default 1e-3
///     Convergence tolerance for the Weiszfeld algorithm.
/// random_state : int, default 0
///     Seed for reproducible subsample draws.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from polars_statistics import TheilSen
/// >>> X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
/// >>> y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])
/// >>> model = TheilSen(random_state=0).fit(X, y)
/// >>> model.is_fitted()
/// True
/// >>> model.coefficients
/// array([...])
/// >>> model.r_squared
/// 0.99...
#[pyclass(name = "TheilSen")]
pub struct PyTheilSen {
    with_intercept: bool,
    max_subpopulation: usize,
    n_subsamples: Option<usize>,
    max_iter: usize,
    tol: f64,
    random_state: u64,
    fitted: Option<FittedTheilSen>,
}

#[pymethods]
impl PyTheilSen {
    #[new]
    #[pyo3(signature = (with_intercept=true, max_subpopulation=10000, n_subsamples=None, max_iter=300, tol=1e-3, random_state=0))]
    fn new(
        with_intercept: bool,
        max_subpopulation: usize,
        n_subsamples: Option<usize>,
        max_iter: usize,
        tol: f64,
        random_state: u64,
    ) -> Self {
        Self {
            with_intercept,
            max_subpopulation,
            n_subsamples,
            max_iter,
            tol,
            random_state,
            fitted: None,
        }
    }

    /// Fit the Theil-Sen regression model.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray of shape (n_samples, n_features)
    ///     Feature matrix. Must be float64.
    /// y : numpy.ndarray of shape (n_samples,)
    ///     Response vector.
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

        let mut b = TheilSenRegressor::builder()
            .with_intercept(slf.with_intercept)
            .max_subpopulation(slf.max_subpopulation)
            .max_iter(slf.max_iter)
            .tolerance(slf.tol)
            .random_state(slf.random_state);
        if let Some(ns) = slf.n_subsamples {
            b = b.n_subsamples(ns);
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

    /// Fitted intercept, or None when with_intercept=False.
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

    /// Mean squared error of residuals.
    #[getter]
    fn mse(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.result().mse)
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

    /// Residuals (y − ŷ) for the training data.
    #[getter]
    fn residuals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok((&fitted.result().residuals).into_numpy(py))
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
