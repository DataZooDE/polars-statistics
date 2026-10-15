//! PyO3 wrapper for Complementary Log-Log regression.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use anofox_regression::solvers::{BinomialRegressor, FittedBinomial, FittedRegressor, Regressor};

use crate::utils::{IntoNumpy, ToFaer};

/// Complementary Log-Log regression model (Binomial GLM with cloglog link).
///
/// Uses the complementary log-log link function: g(μ) = log(-log(1-μ))
///
/// Useful when the probability of an event is asymmetric, particularly
/// for modeling time-to-event data or extreme value problems.
///
/// Parameters
/// ----------
/// with_intercept : bool, default True
///     Whether to include an intercept term.
/// max_iter : int, default 100
///     Maximum number of IRLS iterations.
/// tol : float, default 1e-6
///     Convergence tolerance.
/// compute_inference : bool, default True
///     Whether to compute statistical inference.
/// confidence_level : float, default 0.95
///     Confidence level for confidence intervals.
#[pyclass(name = "Cloglog")]
pub struct PyCloglog {
    with_intercept: bool,
    max_iter: usize,
    tol: f64,
    lambda_: f64,
    compute_inference: bool,
    confidence_level: f64,
    fitted: Option<FittedBinomial>,
}

#[pymethods]
impl PyCloglog {
    #[new]
    #[pyo3(signature = (add_intercept=None, with_intercept=None, max_iter=100, tol=1e-6, lambda_=0.0, compute_inference=true, confidence_level=0.95))]
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

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        crate::pymodels::errors::validate_xy("Cloglog", &x, &y)?;
        let x_mat = x.to_faer();
        let y_col = y.to_faer();

        let model = BinomialRegressor::cloglog()
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

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Cloglog"))?;

        let x_mat = x.to_faer();
        let predictions = fitted.predict(&x_mat);

        // Convert probabilities to binary predictions
        let binary: Vec<f64> = predictions
            .iter()
            .map(|&p| if p > 0.5 { 1.0 } else { 0.0 })
            .collect();
        Ok(PyArray1::from_vec(py, binary))
    }

    /// Mean classification accuracy on ``(X, y)`` using ``threshold``.
    ///
    /// Predicts class labels (``P(y=1|x) >= threshold``) and returns the
    /// fraction that match ``y``. Consistent with
    /// :meth:`sklearn.base.ClassifierMixin.score` (accuracy for classifiers).
    #[pyo3(signature = (x, y, threshold=0.5))]
    fn score<'py>(
        &self,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        threshold: f64,
    ) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Cloglog"))?;
        let x_mat = x.to_faer();
        let y_arr = y.as_array();
        let probabilities = fitted.predict_probability(&x_mat);
        let n = probabilities.nrows();
        if n != y_arr.len() {
            return Err(crate::pymodels::errors::x_y_row_mismatch_err(
                "Cloglog",
                n,
                y_arr.len(),
            ));
        }
        if n == 0 {
            return Err(crate::pymodels::errors::empty_input_err("Cloglog"));
        }
        let correct = (0..n)
            .filter(|&i| {
                let label = if probabilities[i] >= threshold {
                    1.0
                } else {
                    0.0
                };
                (label - y_arr[i]).abs() < 0.5
            })
            .count();
        Ok(correct as f64 / n as f64)
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Cloglog"))?;

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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Cloglog"))?;

        Ok(fitted.coefficients().into_numpy(py))
    }

    #[getter]
    fn intercept(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Cloglog"))?;

        Ok(fitted.intercept())
    }

    #[getter]
    fn std_errors<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Cloglog"))?;

        Ok(fitted
            .result()
            .std_errors
            .as_ref()
            .map(|se| se.into_numpy(py)))
    }

    #[getter]
    fn p_values<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Cloglog"))?;

        Ok(fitted
            .result()
            .p_values
            .as_ref()
            .map(|pv| pv.into_numpy(py)))
    }

    /// Wald z statistics for coefficients (``None`` unless ``compute_inference=True``).
    #[getter]
    fn t_statistics<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Cloglog"))?;

        Ok(fitted
            .result()
            .t_statistics
            .as_ref()
            .map(|ts| ts.into_numpy(py)))
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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Cloglog"))?;

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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Cloglog"))?;

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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Cloglog"))?;

        Ok(fitted.result().intercept_std_error)
    }

    /// Wald z statistic of the intercept (R ``summary.glm``).
    #[getter]
    fn intercept_t_statistic(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Cloglog"))?;

        Ok(fitted.result().intercept_t_statistic)
    }

    /// P-value for the intercept (``None`` unless ``compute_inference=True``).
    #[getter]
    fn intercept_p_value(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Cloglog"))?;

        Ok(fitted.result().intercept_p_value)
    }

    /// Wald confidence interval of the intercept at ``confidence_level`` (R
    /// ``confint.default``), as ``(lower, upper)`` (``None`` unless
    /// ``compute_inference=True``).
    #[getter]
    fn intercept_conf_interval(&self) -> PyResult<Option<(f64, f64)>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Cloglog"))?;

        Ok(fitted.result().intercept_conf_interval)
    }

    #[getter]
    fn aic(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Cloglog"))?;

        Ok(Some(fitted.result().aic))
    }

    #[getter]
    fn bic(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Cloglog"))?;

        Ok(Some(fitted.result().bic))
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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Cloglog"))?;
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
