//! PyO3 wrapper for Dynamic Linear Model (lmDynamic).

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::solvers::{
    FittedLmDynamic, FittedRegressor, InformationCriterion, LmDynamicRegressor, Regressor,
};

use crate::pymodels::py_alm::parse_distribution;
use crate::utils::{IntoNumpy, ToFaer};

/// Convert string to InformationCriterion enum.
fn parse_ic(s: &str) -> PyResult<InformationCriterion> {
    match s.to_lowercase().as_str() {
        "aic" => Ok(InformationCriterion::AIC),
        "aicc" | "aic_c" => Ok(InformationCriterion::AICc),
        "bic" => Ok(InformationCriterion::BIC),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown information criterion: '{}'. Available: aic, aicc, bic",
            s
        ))),
    }
}

/// Dynamic Linear Model (lmDynamic).
///
/// A time-varying parameter model that combines multiple candidate regression
/// models using pointwise information criteria weighting. Based on the
/// lmDynamic function from the greybox R package.
///
/// Parameters
/// ----------
/// ic : str, default "aicc"
///     Information criterion for model weighting.
///     Options: "aic", "aicc", "bic"
/// distribution : str, default "normal"
///     The distribution family for residuals.
///     Options: same as ALM distributions.
/// lowess_span : float, optional
///     LOWESS smoothing span (0.05 to 1.0). Use None to disable smoothing.
///     Default is 0.3.
/// max_models : int, optional
///     Maximum number of candidate models to consider (default 64).
/// with_intercept : bool, default True
///     Whether to include an intercept term.
/// confidence_level : float, default 0.95
///     Confidence level for confidence intervals.
#[pyclass(name = "LmDynamic")]
pub struct PyLmDynamic {
    ic: String,
    distribution: String,
    lowess_span: Option<f64>,
    max_models: Option<usize>,
    with_intercept: bool,
    confidence_level: f64,
    fitted: Option<FittedLmDynamic>,
}

#[pymethods]
impl PyLmDynamic {
    #[new]
    #[pyo3(signature = (ic="aicc", distribution="normal", lowess_span=Some(0.3), max_models=Some(64), with_intercept=true, confidence_level=0.95))]
    fn new(
        ic: &str,
        distribution: &str,
        lowess_span: Option<f64>,
        max_models: Option<usize>,
        with_intercept: bool,
        confidence_level: f64,
    ) -> Self {
        Self {
            ic: ic.to_string(),
            distribution: distribution.to_string(),
            lowess_span,
            max_models,
            with_intercept,
            confidence_level,
            fitted: None,
        }
    }

    /// Fit the dynamic linear model.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data.
    /// y : array-like of shape (n_samples,)
    ///     Target values.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_mat = x.to_faer();
        let y_col = y.to_faer();

        let ic_type = parse_ic(&slf.ic)?;
        let dist = parse_distribution(&slf.distribution)?;

        let mut builder = LmDynamicRegressor::builder()
            .ic(ic_type)
            .distribution(dist)
            .with_intercept(slf.with_intercept)
            .confidence_level(slf.confidence_level);

        if let Some(span) = slf.lowess_span {
            builder = builder.lowess_span(span);
        } else {
            builder = builder.no_smoothing();
        }

        if let Some(max) = slf.max_models {
            builder = builder.max_models(max);
        }

        let model = builder.build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        slf.fitted = Some(fitted);
        Ok(slf)
    }

    /// Predict target values using time-averaged coefficients.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Samples to predict.
    ///
    /// Returns
    /// -------
    /// array of shape (n_samples,)
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
        let predictions = fitted.predict(&x_mat);

        Ok(predictions.into_numpy(py))
    }

    fn is_fitted(&self) -> bool {
        self.fitted.is_some()
    }

    /// Get time-averaged coefficients (excludes intercept).
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok((&fitted.result().coefficients).into_numpy(py))
    }

    /// Get time-averaged intercept.
    #[getter]
    fn intercept(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.result().intercept)
    }

    /// Get time-varying coefficients matrix (n_obs x n_coefs).
    ///
    /// If with_intercept, column 0 is the intercept.
    #[getter]
    fn dynamic_coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        let dyn_coef = fitted.dynamic_coefficients();
        let n = dyn_coef.nrows();
        let p = dyn_coef.ncols();

        // Convert faer Mat to numpy 2D array
        let arr = PyArray2::from_vec2(
            py,
            &(0..n)
                .map(|i| (0..p).map(|j| dyn_coef[(i, j)]).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(arr)
    }

    /// Get model weights matrix (n_obs x n_models).
    #[getter]
    fn model_weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        let weights = fitted.model_weights();
        let n = weights.nrows();
        let m = weights.ncols();

        let arr = PyArray2::from_vec2(
            py,
            &(0..n)
                .map(|i| (0..m).map(|j| weights[(i, j)]).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(arr)
    }

    /// Get smoothed model weights matrix (n_obs x n_models) if LOWESS was applied.
    #[getter]
    fn smoothed_weights<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        match fitted.smoothed_weights() {
            Some(weights) => {
                let n = weights.nrows();
                let m = weights.ncols();

                let arr = PyArray2::from_vec2(
                    py,
                    &(0..n)
                        .map(|i| (0..m).map(|j| weights[(i, j)]).collect::<Vec<_>>())
                        .collect::<Vec<_>>(),
                )
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

                Ok(Some(arr))
            }
            None => Ok(None),
        }
    }

    /// Get pointwise IC values matrix (n_obs x n_models).
    #[getter]
    fn pointwise_ic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        let ic = fitted.pointwise_ic();
        let n = ic.nrows();
        let m = ic.ncols();

        let arr = PyArray2::from_vec2(
            py,
            &(0..n)
                .map(|i| (0..m).map(|j| ic[(i, j)]).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(arr)
    }

    /// Get coefficient at a specific time point.
    ///
    /// Parameters
    /// ----------
    /// obs_index : int
    ///     Observation index (0 to n-1).
    /// coef_index : int
    ///     Coefficient index (0 = intercept if with_intercept).
    ///
    /// Returns
    /// -------
    /// float or None
    ///     The coefficient value, or None if indices are out of bounds.
    fn coefficient_at(&self, obs_index: usize, coef_index: usize) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.coefficient_at(obs_index, coef_index))
    }

    /// Get all coefficients at a specific time point.
    ///
    /// Parameters
    /// ----------
    /// obs_index : int
    ///     Observation index (0 to n-1).
    ///
    /// Returns
    /// -------
    /// array or None
    ///     The coefficient values, or None if index is out of bounds.
    fn coefficients_at<'py>(
        &self,
        py: Python<'py>,
        obs_index: usize,
    ) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        match fitted.coefficients_at(obs_index) {
            Some(coefs) => Ok(Some(coefs.into_numpy(py))),
            None => Ok(None),
        }
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
    fn adj_r_squared(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.result().adj_r_squared)
    }

    #[getter]
    fn mse(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.result().mse)
    }

    #[getter]
    fn rmse(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.result().rmse)
    }

    #[getter]
    fn n_observations(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.result().n_observations)
    }

    #[getter]
    fn fitted_values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok((&fitted.result().fitted_values).into_numpy(py))
    }

    #[getter]
    fn residuals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok((&fitted.result().residuals).into_numpy(py))
    }
}
