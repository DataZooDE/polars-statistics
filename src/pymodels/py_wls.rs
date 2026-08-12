//! PyO3 wrapper for Weighted Least Squares regression.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::solvers::{FittedRegressor, Regressor, WlsRegressor};

use crate::utils::{IntoNumpy, ToFaer};

/// Weighted Least Squares regression model.
///
/// Fits a linear model where each observation is weighted differently.
/// Useful when errors have non-constant variance (heteroscedasticity).
///
/// Parameters
/// ----------
/// with_intercept : bool, default True
///     Whether to include an intercept term.
/// compute_inference : bool, default True
///     Whether to compute statistical inference.
/// confidence_level : float, default 0.95
///     Confidence level for confidence intervals.
#[pyclass(name = "WLS")]
pub struct PyWLS {
    with_intercept: bool,
    compute_inference: bool,
    confidence_level: f64,
    fitted: Option<Box<dyn FittedRegressor + Send + Sync>>,
}

#[pymethods]
impl PyWLS {
    #[new]
    #[pyo3(signature = (with_intercept=true, compute_inference=true, confidence_level=0.95))]
    fn new(with_intercept: bool, compute_inference: bool, confidence_level: f64) -> Self {
        Self {
            with_intercept,
            compute_inference,
            confidence_level,
            fitted: None,
        }
    }

    /// Fit the model with sample weights.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data.
    /// y : array-like of shape (n_samples,)
    ///     Target values.
    /// weights : array-like of shape (n_samples,)
    ///     Sample weights.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        weights: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_mat = x.to_faer();
        let y_col = y.to_faer();
        let w_col = weights.to_faer();

        let model = WlsRegressor::builder()
            .with_intercept(slf.with_intercept)
            .compute_inference(slf.compute_inference)
            .confidence_level(slf.confidence_level)
            .weights(w_col)
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

    /// Compute HC (heteroskedasticity-consistent) standard errors.
    ///
    /// Parameters
    /// ----------
    /// x : array-like of shape (n_samples, n_features)
    ///     The feature matrix used to fit the model (without intercept column).
    /// hc_type : str, default "hc1"
    ///     HC variant: "hc0", "hc1", "hc2", or "hc3".
    ///
    /// Returns
    /// -------
    /// dict
    ///     Dictionary with keys: std_errors, t_statistics, p_values,
    ///     conf_interval_lower, conf_interval_upper, and optionally
    ///     intercept_std_error, intercept_t_statistic, intercept_p_value.
    #[pyo3(signature = (x, hc_type="hc1"))]
    fn hc_inference<'py>(
        &self,
        _py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        hc_type: &str,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        // Check the model is fitted so the error is "not fitted" rather than
        // "not implemented" when neither has happened.
        if self.fitted.is_none() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Model not fitted",
            ));
        }
        // CR-03: WLS HC inference requires the weight-aware sandwich
        // (X'WX)^{-1}(X'diag(w·e²)X)(X'WX)^{-1}, but the fitted object is stored
        // as Box<dyn FittedRegressor> which cannot be downcast to FittedWls to
        // retrieve the weight vector.  Returning the unweighted OLS sandwich is
        // silently wrong for heterogeneous weights — the primary use case of WLS.
        // Raise NotImplementedError so callers are not misled.
        //
        // Workaround: scale X and y by sqrt(w_i), fit OLS on the scaled data,
        // and call OLS.hc_inference on the result.
        let _ = (x, hc_type);
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "WLS.hc_inference is not yet implemented: the weight-correct sandwich \
             (X'WX)^{-1}(X'diag(w·e²)X)(X'WX)^{-1} requires the stored observation \
             weights, which are not accessible through the FittedRegressor trait. \
             Workaround: scale X and y by sqrt(w_i), fit OLS on the scaled data, \
             and call OLS.hc_inference on the result.",
        ))
    }
}
