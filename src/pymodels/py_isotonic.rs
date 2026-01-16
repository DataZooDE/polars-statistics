//! PyO3 wrapper for Isotonic (Monotonic) regression.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::solvers::{FittedIsotonic, FittedRegressor, IsotonicRegressor, OutOfBounds};

use crate::utils::{IntoNumpy, ToFaer};

/// Isotonic (Monotonic) Regression model.
///
/// Fits a non-decreasing (or non-increasing) step function to the data
/// using the Pool Adjacent Violators Algorithm (PAVA). This is useful
/// when you know the relationship between X and Y should be monotonic.
///
/// Parameters
/// ----------
/// increasing : bool, default True
///     Whether to fit an increasing (True) or decreasing (False) function.
/// out_of_bounds : str, default "clip"
///     How to handle predictions outside the training range:
///     - "clip": Clip to boundary values
///     - "nan": Return NaN for out-of-bounds
///     - "extrapolate": Use nearest boundary value (same as clip for step function)
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from polars_statistics import Isotonic
/// >>>
/// >>> x = np.array([1, 2, 3, 4, 5, 6])
/// >>> y = np.array([1, 2, 1.5, 4, 3.5, 6])  # Non-monotonic
/// >>>
/// >>> model = Isotonic(increasing=True)
/// >>> model.fit(x, y)
/// >>> model.fitted_values  # Will be monotonically increasing
#[pyclass(name = "Isotonic")]
pub struct PyIsotonic {
    increasing: bool,
    out_of_bounds: String,
    fitted: Option<FittedIsotonic>,
}

#[pymethods]
impl PyIsotonic {
    /// Create a new Isotonic regression model.
    #[new]
    #[pyo3(signature = (increasing=true, out_of_bounds="clip"))]
    fn new(increasing: bool, out_of_bounds: &str) -> Self {
        Self {
            increasing,
            out_of_bounds: out_of_bounds.to_string(),
            fitted: None,
        }
    }

    /// Fit the model to 1D data.
    ///
    /// Parameters
    /// ----------
    /// x : array-like of shape (n_samples,)
    ///     Feature values (will be sorted internally).
    /// y : array-like of shape (n_samples,)
    ///     Target values.
    ///
    /// Returns
    /// -------
    /// self
    ///     Fitted model.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray1<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_col = x.to_faer();
        let y_col = y.to_faer();

        let oob = match slf.out_of_bounds.as_str() {
            "nan" => OutOfBounds::Nan,
            "extrapolate" => OutOfBounds::Extrapolate,
            _ => OutOfBounds::Clip,
        };

        let model = IsotonicRegressor::builder()
            .increasing(slf.increasing)
            .out_of_bounds(oob)
            .build();

        let fitted = model
            .fit_1d(&x_col, &y_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        slf.fitted = Some(fitted);
        Ok(slf)
    }

    /// Fit the model using 2D array (uses first column only).
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, 1)
    ///     Feature matrix (only first column is used).
    /// y : array-like of shape (n_samples,)
    ///     Target values.
    ///
    /// Returns
    /// -------
    /// self
    ///     Fitted model.
    fn fit_2d<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_mat = x.to_faer();
        let y_col = y.to_faer();

        if x_mat.ncols() != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Isotonic regression requires exactly one feature. Use first column only.",
            ));
        }

        // Extract first column
        let x_col = faer::Col::from_fn(x_mat.nrows(), |i| x_mat[(i, 0)]);

        let oob = match slf.out_of_bounds.as_str() {
            "nan" => OutOfBounds::Nan,
            "extrapolate" => OutOfBounds::Extrapolate,
            _ => OutOfBounds::Clip,
        };

        let model = IsotonicRegressor::builder()
            .increasing(slf.increasing)
            .out_of_bounds(oob)
            .build();

        let fitted = model
            .fit_1d(&x_col, &y_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        slf.fitted = Some(fitted);
        Ok(slf)
    }

    /// Predict using the isotonic regression model.
    ///
    /// Parameters
    /// ----------
    /// x : array-like of shape (n_samples,)
    ///     Samples to predict.
    ///
    /// Returns
    /// -------
    /// array of shape (n_samples,)
    ///     Predicted values.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        let x_col = x.to_faer();
        let predictions = fitted.predict_1d(&x_col);

        Ok(predictions.into_numpy(py))
    }

    /// Predict using 2D array (uses first column only).
    fn predict_2d<'py>(
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

    /// Check if the model has been fitted.
    fn is_fitted(&self) -> bool {
        self.fitted.is_some()
    }

    /// Get whether the fit is increasing or decreasing.
    #[getter]
    fn is_increasing(&self) -> PyResult<bool> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.is_increasing())
    }

    /// Get the X thresholds of the step function.
    #[getter]
    fn x_thresholds<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.x_thresholds().into_numpy(py))
    }

    /// Get the Y values at each threshold.
    #[getter]
    fn y_values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.y_values().into_numpy(py))
    }

    /// Get the fitted values for the training data.
    #[getter]
    fn fitted_values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.fitted_values().into_numpy(py))
    }

    /// Get R-squared value.
    #[getter]
    fn r_squared(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.result().r_squared)
    }

    /// Get residuals.
    #[getter]
    fn residuals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok((&fitted.result().residuals).into_numpy(py))
    }

    /// Get number of observations.
    #[getter]
    fn n_observations(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.result().n_observations)
    }

    /// Get a formatted summary of the regression results.
    fn summary(&self) -> PyResult<String> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        let result = fitted.result();
        let direction = if fitted.is_increasing() {
            "increasing"
        } else {
            "decreasing"
        };

        let mut summary = String::new();

        summary.push_str(&format!("Isotonic Regression Results ({})\n", direction));
        summary.push_str("==========================================\n\n");

        summary.push_str(&format!(
            "Observations:       {:>10}\n",
            result.n_observations
        ));
        summary.push_str(&format!("R-squared:          {:>10.6}\n", result.r_squared));
        summary.push_str(&format!(
            "Step function size: {:>10}\n",
            fitted.x_thresholds().nrows()
        ));

        Ok(summary)
    }
}
