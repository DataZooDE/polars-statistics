//! PyO3 wrapper for Weighted Least Squares regression.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use anofox_regression::solvers::{FittedRegressor, FittedWls, Regressor, WlsRegressor};

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
/// alias_tolerance : float, default 1e-7
///     Relative tolerance of R `lm`'s rank-deficiency (aliasing) test. See
///     ``OLS``'s parameter of the same name.
///
/// Notes
/// -----
/// ``hc_inference`` raises ``NotImplementedError`` for WLS because the
/// weight-correct sandwich estimator requires access to the per-observation
/// weights after fitting, which are not stored via the ``FittedRegressor``
/// trait. See ``hc_inference`` for the workaround.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from polars_statistics import WLS
/// >>> X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
/// >>> y = np.array([2.0, 5.0, 8.0, 11.0])
/// >>> w = np.array([1.0, 2.0, 1.0, 2.0])  # higher weight on rows 2 and 4
/// >>> model = WLS().fit(X, y, w)
/// >>> model.is_fitted()
/// True
/// >>> model.coefficients
/// array([...])
#[pyclass(name = "WLS")]
pub struct PyWLS {
    with_intercept: bool,
    compute_inference: bool,
    confidence_level: f64,
    alias_tolerance: f64,
    fitted: Option<FittedWls>,
}

#[pymethods]
impl PyWLS {
    #[new]
    #[pyo3(signature = (add_intercept=None, with_intercept=None, compute_inference=true, confidence_level=0.95, alias_tolerance=1e-7))]
    fn new(
        py: Python<'_>,
        add_intercept: Option<bool>,
        with_intercept: Option<bool>,
        compute_inference: bool,
        confidence_level: f64,
        alias_tolerance: f64,
    ) -> PyResult<Self> {
        if !(alias_tolerance.is_finite() && alias_tolerance >= 0.0) {
            return Err(PyValueError::new_err(format!(
                "alias_tolerance must be finite and >= 0, got {alias_tolerance}"
            )));
        }
        let with_intercept =
            crate::pymodels::errors::resolve_intercept(py, add_intercept, with_intercept, true)?;
        Ok(Self {
            with_intercept,
            compute_inference,
            confidence_level,
            alias_tolerance,
            fitted: None,
        })
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
        crate::pymodels::errors::validate_xy("WLS", &x, &y)?;
        let x_mat = x.to_faer();
        let y_col = y.to_faer();
        let w_col = weights.to_faer();

        let model = WlsRegressor::builder()
            .with_intercept(slf.with_intercept)
            .compute_inference(slf.compute_inference)
            .confidence_level(slf.confidence_level)
            .alias_tolerance(slf.alias_tolerance)
            .weights(w_col)
            .build();

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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("WLS"))?;

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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("WLS"))?;
        let x_mat = x.to_faer();
        let y_col = y.to_faer();
        Ok(fitted.score(&x_mat, &y_col))
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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("WLS"))?;

        Ok(fitted.coefficients().into_numpy(py))
    }

    /// Fitted intercept, or None when with_intercept=False.
    #[getter]
    fn intercept(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("WLS"))?;

        Ok(fitted.intercept())
    }

    /// Coefficient of determination R² (weighted).
    #[getter]
    fn r_squared(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("WLS"))?;

        Ok(fitted.r_squared())
    }

    /// Standard errors of coefficients (``None`` unless ``compute_inference=True``).
    #[getter]
    fn std_errors<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("WLS"))?;

        Ok(fitted
            .result()
            .std_errors
            .as_ref()
            .map(|se| se.into_numpy(py)))
    }

    /// t-statistics for coefficients (``None`` unless ``compute_inference=True``).
    #[getter]
    fn t_statistics<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("WLS"))?;

        Ok(fitted
            .result()
            .t_statistics
            .as_ref()
            .map(|ts| ts.into_numpy(py)))
    }

    /// P-values for coefficient significance tests (``None`` unless ``compute_inference=True``).
    #[getter]
    fn p_values<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("WLS"))?;

        Ok(fitted
            .result()
            .p_values
            .as_ref()
            .map(|pv| pv.into_numpy(py)))
    }

    /// Lower bounds of coefficient confidence intervals (``None`` unless
    /// ``compute_inference=True``).
    #[getter]
    fn conf_interval_lower<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("WLS"))?;

        Ok(fitted
            .result()
            .conf_interval_lower
            .as_ref()
            .map(|ci| ci.into_numpy(py)))
    }

    /// Upper bounds of coefficient confidence intervals (``None`` unless
    /// ``compute_inference=True``).
    #[getter]
    fn conf_interval_upper<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("WLS"))?;

        Ok(fitted
            .result()
            .conf_interval_upper
            .as_ref()
            .map(|ci| ci.into_numpy(py)))
    }

    /// Standard error of the intercept (``None`` unless ``compute_inference=True``).
    #[getter]
    fn intercept_std_error(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("WLS"))?;

        Ok(fitted.result().intercept_std_error)
    }

    /// t-statistic for the intercept (``None`` unless ``compute_inference=True``).
    #[getter]
    fn intercept_t_statistic(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("WLS"))?;

        Ok(fitted.result().intercept_t_statistic)
    }

    /// P-value for the intercept (``None`` unless ``compute_inference=True``).
    #[getter]
    fn intercept_p_value(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("WLS"))?;

        Ok(fitted.result().intercept_p_value)
    }

    /// Confidence interval of the intercept, as ``(lower, upper)`` (``None``
    /// unless ``compute_inference=True``).
    #[getter]
    fn intercept_conf_interval(&self) -> PyResult<Option<(f64, f64)>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("WLS"))?;

        Ok(fitted.result().intercept_conf_interval)
    }

    /// Compute HC (heteroskedasticity-consistent) standard errors.
    ///
    /// **Not implemented for WLS.** Always raises ``NotImplementedError``.
    ///
    /// The weight-correct sandwich estimator ``(X'WX)⁻¹(X'diag(w·e²)X)(X'WX)⁻¹``
    /// requires the stored observation weights, which are not accessible through
    /// the ``FittedRegressor`` trait after fitting.
    ///
    /// Workaround: scale ``X`` and ``y`` by ``sqrt(w_i)``, fit ``OLS`` on the scaled
    /// data, and call ``OLS.hc_inference`` on the result.
    ///
    /// Parameters
    /// ----------
    /// x : array-like of shape (n_samples, n_features)
    ///     The feature matrix used to fit the model (without intercept column).
    /// hc_type : str, default "hc1"
    ///     HC variant: "hc0", "hc1", "hc2", or "hc3". (Ignored — always raises.)
    ///
    /// Raises
    /// ------
    /// NotImplementedError
    ///     Always. WLS HC inference is not yet implemented.
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
            return Err(crate::pymodels::errors::not_fitted_err("WLS"));
        }
        // CR-03: WLS HC inference requires the weight-aware sandwich
        // (X'WX)^{-1}(X'diag(w·e²)X)(X'WX)^{-1}. `self.fitted` now stores a
        // concrete `FittedWls` (16.1-05 Task 3) so the weight vector itself is
        // reachable via `fitted.weights()`, but computing this sandwich is out
        // of this plan's scope — not implemented yet. Returning the unweighted
        // OLS sandwich would be silently wrong for heterogeneous weights — the
        // primary use case of WLS. Raise NotImplementedError so callers are not
        // misled.
        //
        // Workaround: scale X and y by sqrt(w_i), fit OLS on the scaled data,
        // and call OLS.hc_inference on the result.
        let _ = (x, hc_type);
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "WLS.hc_inference is not yet implemented: the weight-correct sandwich \
             (X'WX)^{-1}(X'diag(w·e²)X)(X'WX)^{-1} has not been wired up yet. \
             Workaround: scale X and y by sqrt(w_i), fit OLS on the scaled data, \
             and call OLS.hc_inference on the result.",
        ))
    }
    /// Variance factor ``M`` with ``Var(x0' beta) = sigma^2 x0' M x0``
    /// (``(X'WX)^-1``). ``(p+1) x (p+1)`` with an intercept (index 0), ``p x
    /// p`` without; aliased rows/columns are zero. ``None`` for fits from
    /// moments.
    fn variance_factor<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("WLS"))?;

        Ok(fitted.variance_factor().as_ref().map(|m| m.into_numpy(py)))
    }

    /// Leverage ``x0' M x0`` of new rows (one column per feature; aliased
    /// columns are ignored). ``NaN`` when no variance factor is available.
    fn leverage_new<'py>(
        &self,
        py: Python<'py>,
        x_new: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("WLS"))?;

        let x_new_mat = x_new.to_faer();
        let expected = fitted.coefficients().nrows();
        if x_new_mat.ncols() != expected {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "WLS.leverage_new: x_new has {} columns but the model was fit with {} \
                 feature(s) — they must match.",
                x_new_mat.ncols(),
                expected
            )));
        }
        Ok(fitted.leverage_new(&x_new_mat).into_numpy(py))
    }

    /// Per-observation diagnostics (R ``broom::augment``): pass the training
    /// data (without the intercept column). Weights default to the fit's own
    /// weights.
    ///
    /// Parameters
    /// ----------
    /// x : array-like of shape (n_samples, n_features)
    ///     The feature matrix used to fit the model.
    /// y : array-like of shape (n_samples,)
    ///     The target values used to fit the model.
    /// weights : array-like of shape (n_samples,), optional
    ///     Prior weights. Defaults to the fit's own weights.
    /// residual_type : str, optional
    ///     ``"response"`` (default), ``"pearson"``, or ``"deviance"``.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Keys: fitted, residual, resid_type, std_residual, stud_residual,
    ///     leverage, cooks_d, dffits.
    #[pyo3(signature = (x, y, weights=None, residual_type=None))]
    fn augment<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        weights: Option<PyReadonlyArray1<'py, f64>>,
        residual_type: Option<String>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("WLS"))?;
        crate::pymodels::augment::augment_dict(py, fitted, x, y, weights, residual_type.as_deref())
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
