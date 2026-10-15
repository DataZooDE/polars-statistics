//! PyO3 wrapper for Gamma regression.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use anofox_regression::solvers::{FittedGamma, FittedRegressor, GammaRegressor, Regressor};

use crate::utils::{IntoNumpy, ToFaer};

/// Gamma GLM regression model.
///
/// Fits a Gamma generalized linear model with a log link via IRLS. The Gamma
/// distribution is appropriate for continuous, strictly positive response variables
/// where the variance scales as the square of the mean (Var[Y] = φ · μ²).
///
/// Parameters
/// ----------
/// with_intercept : bool, default True
///     Whether to include an intercept term in the model.
/// max_iter : int, default 25
///     Maximum number of IRLS iterations.
/// tol : float, default 1e-8
///     Convergence tolerance for the IRLS algorithm.
/// lambda_ : float, default 0.0
///     L2 regularisation strength (0.0 = no regularisation).
/// compute_inference : bool, default True
///     Whether to compute statistical inference.
/// confidence_level : float, default 0.95
///     Confidence level for confidence intervals.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from polars_statistics import Gamma
/// >>> X = np.random.randn(100, 2)
/// >>> y = np.exp(X @ np.array([1.0, -0.5]) + 0.1 * np.random.randn(100))
/// >>> model = Gamma().fit(X, y)
/// >>> model.is_fitted()
/// True
/// >>> preds = model.predict(X)
/// >>> preds.shape
/// (100,)
#[pyclass(name = "Gamma")]
pub struct PyGamma {
    with_intercept: bool,
    max_iter: usize,
    tol: f64,
    lambda_: f64,
    compute_inference: bool,
    confidence_level: f64,
    fitted: Option<FittedGamma>,
}

#[pymethods]
impl PyGamma {
    #[new]
    #[pyo3(signature = (add_intercept=None, with_intercept=None, max_iter=25, tol=1e-8, lambda_=0.0, compute_inference=true, confidence_level=0.95))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        add_intercept: Option<bool>,
        with_intercept: Option<bool>,
        max_iter: usize,
        tol: f64,
        lambda_: f64,
        compute_inference: bool,
        confidence_level: f64,
    ) -> PyResult<Self> {
        if !(confidence_level > 0.0 && confidence_level < 1.0) {
            return Err(PyValueError::new_err(format!(
                "confidence_level must be in (0, 1), got {confidence_level}"
            )));
        }
        let with_intercept =
            crate::pymodels::errors::resolve_intercept(py, add_intercept, with_intercept, true)?;
        Ok(Self {
            with_intercept,
            max_iter,
            tol,
            lambda_,
            compute_inference,
            confidence_level,
            fitted: None,
        })
    }

    /// Fit the Gamma GLM to training data.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray of shape (n_samples, n_features)
    ///     Feature matrix. Must be float64.
    /// y : numpy.ndarray of shape (n_samples,)
    ///     Response vector. All values must be strictly positive.
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
        crate::pymodels::errors::validate_xy("Gamma", &x, &y)?;
        let x_mat = x.to_faer();
        let y_col = y.to_faer();

        let model = GammaRegressor::builder()
            .with_intercept(slf.with_intercept)
            .max_iterations(slf.max_iter)
            .tolerance(slf.tol)
            .lambda(slf.lambda_)
            .compute_inference(slf.compute_inference)
            .confidence_level(slf.confidence_level)
            .build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        slf.fitted = Some(fitted);
        Ok(slf)
    }

    /// Predict response-scale values (μ = exp(η)) for new data.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray of shape (n_samples, n_features)
    ///     Feature matrix. Must be float64.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray of shape (n_samples,)
    ///     Predicted values on the response scale.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Gamma"))?;

        let x_mat = x.to_faer();
        let predictions = fitted.predict(&x_mat);

        Ok(predictions.into_numpy(py))
    }

    /// Predict the linear predictor η = X·β + intercept.
    ///
    /// For a Gamma GLM with log link: μ = exp(η).
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray of shape (n_samples, n_features)
    ///     Feature matrix. Must be float64.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray of shape (n_samples,)
    ///     Linear predictor values on the link scale.
    fn predict_eta<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Gamma"))?;

        let x_mat = x.to_faer();
        Ok(fitted.predict_eta(&x_mat).into_numpy(py))
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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Gamma"))?;

        Ok(fitted.coefficients().into_numpy(py))
    }

    /// Fitted intercept, or None when with_intercept=False.
    #[getter]
    fn intercept(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Gamma"))?;

        Ok(fitted.intercept())
    }

    /// Standard errors of the slope coefficients, or None when inference is disabled.
    #[getter]
    fn std_errors<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Gamma"))?;

        Ok(fitted
            .result()
            .std_errors
            .as_ref()
            .map(|se| se.into_numpy(py)))
    }

    /// P-values for the slope coefficients, or None when inference is disabled.
    #[getter]
    fn p_values<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Gamma"))?;

        Ok(fitted
            .result()
            .p_values
            .as_ref()
            .map(|pv| pv.into_numpy(py)))
    }

    /// t statistics for the slope coefficients, or None when inference is disabled.
    #[getter]
    fn t_statistics<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Gamma"))?;

        Ok(fitted
            .result()
            .t_statistics
            .as_ref()
            .map(|ts| ts.into_numpy(py)))
    }

    /// Lower bounds of coefficient confidence intervals, or None when inference is disabled.
    #[getter]
    fn conf_interval_lower<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Gamma"))?;

        Ok(fitted
            .result()
            .conf_interval_lower
            .as_ref()
            .map(|ci| ci.into_numpy(py)))
    }

    /// Upper bounds of coefficient confidence intervals, or None when inference is disabled.
    #[getter]
    fn conf_interval_upper<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Gamma"))?;

        Ok(fitted
            .result()
            .conf_interval_upper
            .as_ref()
            .map(|ci| ci.into_numpy(py)))
    }

    /// Standard error of the intercept, or None when inference is disabled.
    #[getter]
    fn intercept_std_error(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Gamma"))?;

        Ok(fitted.result().intercept_std_error)
    }

    /// t statistic for the intercept (R ``summary.glm``).
    #[getter]
    fn intercept_t_statistic(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Gamma"))?;

        Ok(fitted.result().intercept_t_statistic)
    }

    /// P-value for the intercept, or None when inference is disabled.
    #[getter]
    fn intercept_p_value(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Gamma"))?;

        Ok(fitted.result().intercept_p_value)
    }

    /// Confidence interval of the intercept at ``confidence_level`` (R
    /// ``confint.default``), as ``(lower, upper)``, or None when inference is disabled.
    #[getter]
    fn intercept_conf_interval(&self) -> PyResult<Option<(f64, f64)>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Gamma"))?;

        Ok(fitted.result().intercept_conf_interval)
    }

    /// Akaike Information Criterion.
    #[getter]
    fn aic(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Gamma"))?;

        Ok(Some(fitted.result().aic))
    }

    /// Bayesian Information Criterion.
    #[getter]
    fn bic(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Gamma"))?;

        Ok(Some(fitted.result().bic))
    }

    /// Whether the IRLS algorithm converged within max_iter iterations.
    ///
    /// Returns False only when error_on_non_convergence=False and IRLS
    /// exhausted max_iter without meeting the tolerance threshold.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Gamma"))?;

        Ok(fitted.converged())
    }

    /// Per-observation diagnostics of the fit (R ``broom::augment``; deviance
    /// residuals by default, ``residual_type="pearson"`` / ``"response"``
    /// also available); pass the training data.
    ///
    /// Parameters
    /// ----------
    /// x : array-like of shape (n_samples, n_features)
    ///     The feature matrix used to fit the model.
    /// y : array-like of shape (n_samples,)
    ///     The target values used to fit the model.
    /// residual_type : str, optional
    ///     ``"deviance"`` (default), ``"pearson"``, or ``"response"``.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Keys: fitted, residual, resid_type, std_residual, stud_residual,
    ///     leverage, cooks_d, dffits.
    #[pyo3(signature = (x, y, residual_type=None))]
    fn augment<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        residual_type: Option<String>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Gamma"))?;
        crate::pymodels::augment::augment_dict(py, fitted, x, y, None, residual_type.as_deref())
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
