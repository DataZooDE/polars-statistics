//! PyO3 wrapper for sklearn-style LogisticRegression (added in anofox-regression 0.5.4).
//!
//! Distinct from [`crate::pymodels::PyLogistic`], which wraps the lower-level
//! `BinomialRegressor` directly. This wrapper mirrors the scikit-learn API:
//! `predict_proba`, `decision_function`, `score`, plus an explicit `Penalty`
//! (None / L2) and the standard `C` regularization-strength parameter.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use anofox_regression::solvers::{FittedLogistic, FittedRegressor, LogisticRegression, Penalty};

use crate::utils::{IntoNumpy, ToFaer};

/// Sklearn-style logistic regression classifier.
///
/// Parameters
/// ----------
/// penalty : {"none", "l2"}, default "l2"
///     Regularization scheme. "l2" applies a Ridge penalty governed by `C`.
/// C : float, default 1.0
///     Inverse of regularization strength (sklearn convention). Larger `C`
///     means weaker penalty. The internal `lambda` is `1.0 / C`. Ignored when
///     `penalty="none"`.
/// threshold : float, default 0.5
///     Classification cutoff for `predict`.
/// with_intercept : bool, default True
/// max_iter : int, default 100
/// tol : float, default 1e-8
/// compute_inference : bool, default True
/// confidence_level : float, default 0.95
#[pyclass(name = "LogisticRegression")]
pub struct PyLogisticRegression {
    penalty: String,
    c: f64,
    threshold: f64,
    with_intercept: bool,
    max_iter: usize,
    tol: f64,
    compute_inference: bool,
    confidence_level: f64,
    fitted: Option<FittedLogistic>,
}

fn parse_penalty(spec: &str, c: f64) -> PyResult<Penalty> {
    match spec.to_ascii_lowercase().as_str() {
        "none" => Ok(Penalty::None),
        "l2" => Ok(Penalty::L2(1.0 / c)),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "penalty must be 'none' or 'l2', got '{other}'"
        ))),
    }
}

#[pymethods]
impl PyLogisticRegression {
    #[new]
    #[pyo3(signature = (
        penalty="l2".to_string(),
        C=1.0,
        threshold=0.5,
        add_intercept=None, with_intercept=None,
        max_iter=100,
        tol=1e-8,
        compute_inference=true,
        confidence_level=0.95,
    ))]
    #[allow(clippy::too_many_arguments, non_snake_case)]
    fn new(
        py: Python<'_>,
        penalty: String,
        C: f64,
        threshold: f64,
        add_intercept: Option<bool>,
        with_intercept: Option<bool>,
        max_iter: usize,
        tol: f64,
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
            penalty,
            c: C,
            threshold,
            with_intercept,
            max_iter,
            tol,
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
        crate::pymodels::errors::validate_xy("LogisticRegression", &x, &y)?;
        let x_mat = x.to_faer();
        let y_col = y.to_faer();
        let penalty = parse_penalty(&slf.penalty, slf.c)?;

        let model = LogisticRegression::builder()
            .penalty(penalty)
            .threshold(slf.threshold)
            .with_intercept(slf.with_intercept)
            .max_iterations(slf.max_iter)
            .tolerance(slf.tol)
            .compute_inference(slf.compute_inference)
            .confidence_level(slf.confidence_level)
            .build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        slf.fitted = Some(fitted);
        Ok(slf)
    }

    /// Predict class labels using the configured threshold.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("LogisticRegression"))?;
        let x_mat = x.to_faer();
        Ok(fitted.predict(&x_mat).into_numpy(py))
    }

    /// Predict P(y=1 | x) for each row.
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("LogisticRegression"))?;
        let x_mat = x.to_faer();
        Ok(fitted.predict_proba(&x_mat).into_numpy(py))
    }

    /// Linear predictor (logit) value for each row.
    fn decision_function<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("LogisticRegression"))?;
        let x_mat = x.to_faer();
        Ok(fitted.decision_function(&x_mat).into_numpy(py))
    }

    /// Classification accuracy on (x, y).
    fn score<'py>(
        &self,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("LogisticRegression"))?;
        let x_mat = x.to_faer();
        let y_col = y.to_faer();
        Ok(fitted.score(&x_mat, &y_col))
    }

    fn is_fitted(&self) -> bool {
        self.fitted.is_some()
    }

    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("LogisticRegression"))?;
        Ok(fitted.coefficients().into_numpy(py))
    }

    #[getter]
    fn intercept(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("LogisticRegression"))?;
        Ok(fitted.intercept())
    }

    #[getter]
    fn n_iter(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("LogisticRegression"))?;
        Ok(fitted.n_iter())
    }

    /// Standard errors of coefficients (``None`` unless ``compute_inference=True``).
    #[getter]
    fn std_errors<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("LogisticRegression"))?;

        Ok(fitted
            .inner()
            .result()
            .std_errors
            .as_ref()
            .map(|se| se.into_numpy(py)))
    }

    /// P-values for coefficient significance tests (``None`` unless ``compute_inference=True``).
    #[getter]
    fn p_values<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("LogisticRegression"))?;

        Ok(fitted
            .inner()
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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("LogisticRegression"))?;

        Ok(fitted
            .inner()
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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("LogisticRegression"))?;

        Ok(fitted
            .inner()
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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("LogisticRegression"))?;

        Ok(fitted
            .inner()
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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("LogisticRegression"))?;

        Ok(fitted.inner().result().intercept_std_error)
    }

    /// Wald z statistic of the intercept (R ``summary.glm``).
    #[getter]
    fn intercept_t_statistic(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("LogisticRegression"))?;

        Ok(fitted.inner().result().intercept_t_statistic)
    }

    /// P-value for the intercept (``None`` unless ``compute_inference=True``).
    #[getter]
    fn intercept_p_value(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("LogisticRegression"))?;

        Ok(fitted.inner().result().intercept_p_value)
    }

    /// Wald confidence interval of the intercept at ``confidence_level`` (R
    /// ``confint.default``), as ``(lower, upper)`` (``None`` unless
    /// ``compute_inference=True``).
    #[getter]
    fn intercept_conf_interval(&self) -> PyResult<Option<(f64, f64)>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("LogisticRegression"))?;

        Ok(fitted.inner().result().intercept_conf_interval)
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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("LogisticRegression"))?;
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
