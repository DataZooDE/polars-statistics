//! PyO3 wrapper for RANSAC (Random Sample Consensus) robust regression.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::solvers::{FittedRansac, FittedRegressor, RansacRegressor, Regressor};

use crate::utils::{IntoNumpy, ToFaer};

/// RANSAC (Random Sample Consensus) robust regression.
///
/// Iteratively fits a linear model on random subsets of observations and
/// identifies the consensus set (inliers) whose absolute residuals are below
/// `residual_threshold`. Returns `ConvergenceFailed` if no consensus set is
/// found within `max_trials` iterations.
///
/// Parameters
/// ----------
/// with_intercept : bool, default True
///     Whether to include an intercept term.
/// min_samples : int or None, default None
///     Minimum number of samples drawn per trial. None → n_features + 1.
/// residual_threshold : float or None, default None
///     Absolute residual threshold to classify inliers. None → MAD(y).
/// max_trials : int, default 100
///     Maximum number of RANSAC trials.
/// stop_probability : float, default 0.99
///     RANSAC stops early when the probability of finding a better consensus
///     set exceeds this value.
/// stop_n_inliers : int or None, default None
///     Stop early once this many inliers are found. None → no early stop.
/// random_state : int, default 0
///     Seed for reproducible subsample draws.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from polars_statistics import RANSAC
/// >>> rng = np.random.default_rng(0)
/// >>> X = rng.standard_normal((50, 2))
/// >>> y = X @ [1.0, -0.5] + 0.1 * rng.standard_normal(50)
/// >>> y[0] = 100.0  # outlier
/// >>> model = RANSAC(random_state=0).fit(X, y)
/// >>> model.is_fitted()
/// True
/// >>> model.n_inliers  # only non-outlier rows
/// 49
#[pyclass(name = "RANSAC")]
pub struct PyRANSAC {
    with_intercept: bool,
    min_samples: Option<usize>,
    residual_threshold: Option<f64>,
    max_trials: usize,
    stop_probability: f64,
    stop_n_inliers: Option<usize>,
    random_state: u64,
    fitted: Option<FittedRansac>,
}

#[pymethods]
impl PyRANSAC {
    #[new]
    #[pyo3(signature = (with_intercept=true, min_samples=None, residual_threshold=None, max_trials=100, stop_probability=0.99, stop_n_inliers=None, random_state=0))]
    fn new(
        with_intercept: bool,
        min_samples: Option<usize>,
        residual_threshold: Option<f64>,
        max_trials: usize,
        stop_probability: f64,
        stop_n_inliers: Option<usize>,
        random_state: u64,
    ) -> Self {
        Self {
            with_intercept,
            min_samples,
            residual_threshold,
            max_trials,
            stop_probability,
            stop_n_inliers,
            random_state,
            fitted: None,
        }
    }

    /// Fit the RANSAC regression model.
    ///
    /// Raises `ValueError` if no consensus set is found within `max_trials`.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_mat = x.to_faer();
        let y_col = y.to_faer();

        let mut b = RansacRegressor::builder()
            .with_intercept(slf.with_intercept)
            .max_trials(slf.max_trials)
            .stop_probability(slf.stop_probability)
            .random_state(slf.random_state);
        if let Some(ms) = slf.min_samples {
            b = b.min_samples(ms);
        }
        if let Some(rt) = slf.residual_threshold {
            b = b.residual_threshold(rt);
        }
        if let Some(si) = slf.stop_n_inliers {
            b = b.stop_n_inliers(si);
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
    ///     Predicted values (from the consensus-set model).
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

    /// Fitted slope coefficients from the consensus-set model (excludes intercept).
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

    /// Coefficient of determination R² on the full dataset (inliers + outliers).
    #[getter]
    fn r_squared(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.result().r_squared)
    }

    /// Residuals (y − ŷ) for the full training dataset.
    #[getter]
    fn residuals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok((&fitted.result().residuals).into_numpy(py))
    }

    /// Boolean mask: True where the observation is classified as an inlier.
    #[getter]
    fn inlier_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(PyArray1::from_slice(py, fitted.inlier_mask()))
    }

    /// Number of inliers in the final consensus set.
    #[getter]
    fn n_inliers(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.n_inliers())
    }

    /// Number of RANSAC trials actually run.
    #[getter]
    fn n_trials(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.n_trials())
    }

    /// Residual threshold used to classify inliers.
    #[getter]
    fn residual_threshold(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.residual_threshold())
    }

    /// Total number of observations (inliers + outliers) in the training data.
    #[getter]
    fn n_observations(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.result().n_observations)
    }
}
