//! PyO3 wrapper for Weighted Least Squares regression.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::inference::compute_hc_inference;
use anofox_regression::solvers::{FittedRegressor, Regressor, WlsRegressor};
use anofox_regression::HcType;

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
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        hc_type: &str,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        let hc = match hc_type {
            "hc0" => HcType::HC0,
            "hc2" => HcType::HC2,
            "hc3" => HcType::HC3,
            _ => HcType::HC1,
        };
        let x_mat = x.to_faer();
        let result_data = fitted.result();
        let residuals = result_data.residuals.clone();
        let coef = result_data.coefficients.clone();
        let intercept = fitted.intercept();
        let aliased = vec![false; x_mat.ncols()];
        let result = compute_hc_inference(
            &x_mat, &coef, intercept, &residuals, &aliased, true, hc, 0.95,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("std_errors", result.std_errors.into_numpy(py))?;
        dict.set_item("t_statistics", result.t_statistics.into_numpy(py))?;
        dict.set_item("p_values", result.p_values.into_numpy(py))?;
        dict.set_item(
            "conf_interval_lower",
            result.conf_interval_lower.into_numpy(py),
        )?;
        dict.set_item(
            "conf_interval_upper",
            result.conf_interval_upper.into_numpy(py),
        )?;
        if let Some(ref int_inf) = result.intercept {
            dict.set_item("intercept_std_error", int_inf.std_error)?;
            dict.set_item("intercept_t_statistic", int_inf.t_statistic)?;
            dict.set_item("intercept_p_value", int_inf.p_value)?;
        }
        Ok(dict)
    }
}
