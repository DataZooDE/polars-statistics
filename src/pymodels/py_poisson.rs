//! PyO3 wrapper for Poisson regression (Poisson GLM with log link).

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::solvers::{FittedPoisson, FittedRegressor, PoissonRegressor, Regressor};

use crate::utils::{IntoNumpy, ToFaer};

/// Poisson regression model (Poisson GLM with log link).
///
/// Fits a Poisson regression model for count data.
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
#[pyclass(name = "Poisson")]
pub struct PyPoisson {
    with_intercept: bool,
    compute_inference: bool,
    confidence_level: f64,
    max_iter: usize,
    tol: f64,
    lambda_: f64,
    fitted: Option<FittedPoisson>,
}

#[pymethods]
impl PyPoisson {
    #[new]
    #[pyo3(signature = (add_intercept=None, with_intercept=None, compute_inference=true, confidence_level=0.95, max_iter=25, tol=1e-8, lambda_=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        add_intercept: Option<bool>,
        with_intercept: Option<bool>,
        compute_inference: bool,
        confidence_level: f64,
        max_iter: usize,
        tol: f64,
        lambda_: f64,
    ) -> PyResult<Self> {
        let with_intercept =
            crate::pymodels::errors::resolve_intercept(py, add_intercept, with_intercept, true)?;
        Ok(Self {
            with_intercept,
            compute_inference,
            confidence_level,
            max_iter,
            tol,
            lambda_,
            fitted: None,
        })
    }

    /// Fit the Poisson regression model.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data.
    /// y : array-like of shape (n_samples,)
    ///     Count target values (non-negative integers).
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        crate::pymodels::errors::validate_xy("Poisson", &x, &y)?;
        let x_mat = x.to_faer();
        let y_col = y.to_faer();

        let model = PoissonRegressor::log()
            .with_intercept(slf.with_intercept)
            .compute_inference(slf.compute_inference)
            .confidence_level(slf.confidence_level)
            .max_iterations(slf.max_iter)
            .tolerance(slf.tol)
            .lambda(slf.lambda_)
            .build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        slf.fitted = Some(fitted);
        Ok(slf)
    }

    /// Predict expected counts.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Samples to predict.
    ///
    /// Returns
    /// -------
    /// array of shape (n_samples,)
    ///     Predicted expected counts.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Poisson"))?;

        let x_mat = x.to_faer();
        let counts = fitted.predict_count(&x_mat);

        Ok(counts.into_numpy(py))
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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Poisson"))?;

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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Poisson"))?;

        Ok(fitted.coefficients().into_numpy(py))
    }

    #[getter]
    fn intercept(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Poisson"))?;

        Ok(fitted.intercept())
    }

    #[getter]
    fn std_errors<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Poisson"))?;

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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Poisson"))?;

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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Poisson"))?;

        Ok(fitted.result().aic)
    }

    #[getter]
    fn bic(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Poisson"))?;

        Ok(fitted.result().bic)
    }

    /// Get deviance residuals.
    #[getter]
    fn deviance_residuals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Poisson"))?;

        Ok(fitted.deviance_residuals().into_numpy(py))
    }

    /// Get Pearson residuals.
    #[getter]
    fn pearson_residuals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Poisson"))?;

        Ok(fitted.pearson_residuals().into_numpy(py))
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
