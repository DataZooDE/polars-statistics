//! PyO3 wrapper for RANSAC (Random Sample Consensus) robust regression.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::solvers::{FittedRansac, FittedRegressor, RansacRegressor, Regressor};

use super::PyOLS;
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
/// compute_inference : bool, default False
///     Whether to compute standard errors / t-statistics / p-values /
///     confidence intervals for the final OLS fit on the inliers. This
///     inference ignores the data-driven inlier selection and is therefore
///     optimistic (anofox-regression >= 0.5.14).
/// confidence_level : float, default 0.95
///     Confidence level for confidence intervals (used only when
///     ``compute_inference=True``).
///
/// Since anofox-regression 0.5.14, the fit statistics reported by
/// ``r_squared`` / ``n_observations`` / the other fit-stat getters are those
/// of the final OLS fit on the inliers (previously computed over all rows);
/// ``residuals`` / ``fitted_values`` still cover every row. A constant
/// response now fits cleanly (MAD threshold 0, inlier test ``|r| <=
/// threshold``).
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
    compute_inference: bool,
    confidence_level: f64,
    fitted: Option<FittedRansac>,
}

#[pymethods]
impl PyRANSAC {
    #[new]
    #[pyo3(signature = (add_intercept=None, with_intercept=None, min_samples=None, residual_threshold=None, max_trials=100, stop_probability=0.99, stop_n_inliers=None, random_state=0, compute_inference=false, confidence_level=0.95))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        add_intercept: Option<bool>,
        with_intercept: Option<bool>,
        min_samples: Option<usize>,
        residual_threshold: Option<f64>,
        max_trials: usize,
        stop_probability: f64,
        stop_n_inliers: Option<usize>,
        random_state: u64,
        compute_inference: bool,
        confidence_level: f64,
    ) -> PyResult<Self> {
        let with_intercept =
            crate::pymodels::errors::resolve_intercept(py, add_intercept, with_intercept, true)?;
        Ok(Self {
            with_intercept,
            min_samples,
            residual_threshold,
            max_trials,
            stop_probability,
            stop_n_inliers,
            random_state,
            compute_inference,
            confidence_level,
            fitted: None,
        })
    }

    /// Fit the RANSAC regression model.
    ///
    /// Raises `ValueError` if no consensus set is found within `max_trials`.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        crate::pymodels::errors::validate_xy("RANSAC", &x, &y)?;
        let x_mat = x.to_faer();
        let y_col = y.to_faer();

        let mut b = RansacRegressor::builder()
            .with_intercept(slf.with_intercept)
            .max_trials(slf.max_trials)
            .stop_probability(slf.stop_probability)
            .random_state(slf.random_state)
            .compute_inference(slf.compute_inference)
            .confidence_level(slf.confidence_level);
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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;
        let x_mat = x.to_faer();
        let y_col = y.to_faer();
        Ok(fitted.score(&x_mat, &y_col))
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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

        Ok(fitted.coefficients().into_numpy(py))
    }

    /// Fitted intercept, or None when with_intercept=False.
    #[getter]
    fn intercept(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

        Ok(fitted.intercept())
    }

    /// R² of the OLS fit on the inliers (anofox-regression >= 0.5.14;
    /// previously computed over all rows).
    #[getter]
    fn r_squared(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

        Ok(fitted.result().r_squared)
    }

    /// Residuals (y − ŷ) for the full training dataset.
    #[getter]
    fn residuals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

        Ok((&fitted.result().residuals).into_numpy(py))
    }

    /// Boolean mask: True where the observation is classified as an inlier.
    #[getter]
    fn inlier_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

        Ok(PyArray1::from_slice(py, fitted.inlier_mask()))
    }

    /// Number of inliers in the final consensus set.
    #[getter]
    fn n_inliers(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

        Ok(fitted.n_inliers())
    }

    /// Number of RANSAC trials actually run.
    #[getter]
    fn n_trials(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

        Ok(fitted.n_trials())
    }

    /// Residual threshold used to classify inliers.
    #[getter]
    fn residual_threshold(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

        Ok(fitted.residual_threshold())
    }

    /// Number of inlier rows used by the final OLS fit (anofox-regression
    /// >= 0.5.14; residuals and fitted values still cover every row).
    #[getter]
    fn n_observations(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

        Ok(fitted.result().n_observations)
    }

    /// Standard errors of coefficients (``None`` unless ``compute_inference=True``).
    #[getter]
    fn std_errors<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

        Ok(fitted.result().intercept_std_error)
    }

    /// t-statistic for the intercept (``None`` unless ``compute_inference=True``).
    #[getter]
    fn intercept_t_statistic(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

        Ok(fitted.result().intercept_t_statistic)
    }

    /// P-value for the intercept (``None`` unless ``compute_inference=True``).
    #[getter]
    fn intercept_p_value(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

        Ok(fitted.result().intercept_p_value)
    }

    /// Confidence interval for the intercept, as ``(lower, upper)`` (``None``
    /// unless ``compute_inference=True``).
    #[getter]
    fn intercept_conf_interval(&self) -> PyResult<Option<(f64, f64)>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

        Ok(fitted.result().intercept_conf_interval)
    }

    /// Adjusted R² of the OLS fit on the inliers.
    #[getter]
    fn adj_r_squared(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

        Ok(fitted.result().adj_r_squared)
    }

    /// Mean squared error of the OLS fit on the inliers.
    #[getter]
    fn mse(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

        Ok(fitted.result().mse)
    }

    /// Root mean squared error of the OLS fit on the inliers.
    #[getter]
    fn rmse(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

        Ok(fitted.result().rmse)
    }

    /// F-statistic for overall significance of the OLS fit on the inliers.
    #[getter]
    fn f_statistic(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

        Ok(fitted.result().f_statistic)
    }

    /// P-value for the F-statistic.
    #[getter]
    fn f_pvalue(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

        Ok(fitted.result().f_pvalue)
    }

    /// Akaike Information Criterion of the OLS fit on the inliers.
    #[getter]
    fn aic(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

        Ok(fitted.result().aic)
    }

    /// Bayesian Information Criterion of the OLS fit on the inliers.
    #[getter]
    fn bic(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

        Ok(fitted.result().bic)
    }

    /// Log-likelihood of the OLS fit on the inliers.
    #[getter]
    fn log_likelihood(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

        Ok(fitted.result().log_likelihood)
    }

    /// Ordinary least-squares fit on the inlier rows (the final RANSAC model)
    /// as a fitted OLS object; its residuals/fitted values cover only the
    /// inliers, and ``hc_inference(x)`` needs the inlier rows of ``X``.
    fn inlier_fit(&self, py: Python<'_>) -> PyResult<Py<PyOLS>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("RANSAC"))?;

        Py::new(
            py,
            PyOLS::from_fitted(
                fitted.inlier_fit().clone(),
                self.with_intercept,
                self.compute_inference,
                self.confidence_level,
            ),
        )
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
