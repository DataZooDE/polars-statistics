//! PyO3 wrapper for Logistic regression (Binomial GLM with logit link).

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::solvers::{BinomialRegressor, FittedBinomial, FittedRegressor, Regressor};

use crate::utils::{IntoNumpy, ToFaer};

/// Logistic regression model (Binomial GLM with logit link).
///
/// Fits a logistic regression model for binary classification.
///
/// Parameters
/// ----------
/// with_intercept : bool, default True
///     Whether to include an intercept term.
/// compute_inference : bool, default True
///     Whether to compute statistical inference.
/// confidence_level : float, default 0.95
///     Confidence level for confidence intervals.
/// max_iter : int, default 25
///     Maximum number of IRLS iterations.
/// tol : float, default 1e-8
///     Tolerance for convergence.
#[pyclass(name = "Logistic")]
pub struct PyLogistic {
    with_intercept: bool,
    compute_inference: bool,
    confidence_level: f64,
    max_iter: usize,
    tol: f64,
    fitted: Option<FittedBinomial>,
}

#[pymethods]
impl PyLogistic {
    #[new]
    #[pyo3(signature = (with_intercept=true, compute_inference=true, confidence_level=0.95, max_iter=25, tol=1e-8))]
    fn new(
        with_intercept: bool,
        compute_inference: bool,
        confidence_level: f64,
        max_iter: usize,
        tol: f64,
    ) -> Self {
        Self {
            with_intercept,
            compute_inference,
            confidence_level,
            max_iter,
            tol,
            fitted: None,
        }
    }

    /// Fit the logistic regression model.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data.
    /// y : array-like of shape (n_samples,)
    ///     Binary target values (0 or 1).
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_mat = x.to_faer();
        let y_col = y.to_faer();

        let model = BinomialRegressor::logistic()
            .with_intercept(slf.with_intercept)
            .compute_inference(slf.compute_inference)
            .confidence_level(slf.confidence_level)
            .max_iterations(slf.max_iter)
            .tolerance(slf.tol)
            .build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        slf.fitted = Some(fitted);
        Ok(slf)
    }

    /// Predict class probabilities.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Samples to predict.
    ///
    /// Returns
    /// -------
    /// array of shape (n_samples,)
    ///     Predicted probabilities for class 1.
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        let x_mat = x.to_faer();
        let probabilities = fitted.predict_probability(&x_mat);

        Ok(probabilities.into_numpy(py))
    }

    /// Predict class labels.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Samples to predict.
    /// threshold : float, default 0.5
    ///     Classification threshold.
    ///
    /// Returns
    /// -------
    /// array of shape (n_samples,)
    ///     Predicted class labels (0 or 1).
    #[pyo3(signature = (x, threshold=0.5))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        threshold: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        let x_mat = x.to_faer();
        let probabilities = fitted.predict_probability(&x_mat);

        // Convert probabilities to class labels
        let n = probabilities.nrows();
        let labels: Vec<f64> = (0..n)
            .map(|i| {
                if probabilities[i] >= threshold {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();

        Ok(numpy::PyArray1::from_vec(py, labels))
    }

    /// Get linear predictor values.
    fn predict_linear<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        let x_mat = x.to_faer();
        let linear = fitted.predict_linear(&x_mat);

        Ok(linear.into_numpy(py))
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
    fn std_errors<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

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
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted
            .result()
            .p_values
            .as_ref()
            .map(|pv| pv.into_numpy(py)))
    }

    #[getter]
    fn aic(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.result().aic)
    }

    #[getter]
    fn bic(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.result().bic)
    }

    /// Get deviance residuals.
    #[getter]
    fn deviance_residuals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.deviance_residuals().into_numpy(py))
    }

    /// Get Pearson residuals.
    #[getter]
    fn pearson_residuals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.pearson_residuals().into_numpy(py))
    }
}
