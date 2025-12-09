//! PyO3 wrapper for Tweedie regression.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::solvers::{FittedRegressor, FittedTweedie, Regressor, TweedieRegressor};

use crate::utils::{IntoNumpy, ToFaer};

/// Tweedie regression model.
///
/// Flexible GLM that covers Gaussian, Poisson, Gamma, and Inverse Gaussian
/// distributions through the variance power parameter.
///
/// Variance function: Var[Y] = φ * μ^var_power
///
/// Special cases:
/// - var_power=0: Normal/Gaussian
/// - var_power=1: Poisson-like
/// - var_power=2: Gamma
/// - var_power=3: Inverse Gaussian
/// - 1 < var_power < 2: Compound Poisson-Gamma (good for zero-inflated data)
///
/// Parameters
/// ----------
/// var_power : float, default 1.5
///     Variance power parameter.
/// link_power : float, optional
///     Link power. If None, uses canonical link.
/// with_intercept : bool, default True
///     Whether to include an intercept term.
/// max_iter : int, default 100
///     Maximum number of IRLS iterations.
/// tol : float, default 1e-6
///     Convergence tolerance.
#[pyclass(name = "Tweedie")]
pub struct PyTweedie {
    var_power: f64,
    link_power: Option<f64>,
    with_intercept: bool,
    max_iter: usize,
    tol: f64,
    fitted: Option<FittedTweedie>,
}

#[pymethods]
impl PyTweedie {
    #[new]
    #[pyo3(signature = (var_power=1.5, link_power=None, with_intercept=true, max_iter=100, tol=1e-6))]
    fn new(
        var_power: f64,
        link_power: Option<f64>,
        with_intercept: bool,
        max_iter: usize,
        tol: f64,
    ) -> Self {
        Self {
            var_power,
            link_power,
            with_intercept,
            max_iter,
            tol,
            fitted: None,
        }
    }

    /// Create a Gaussian (Normal) Tweedie model (var_power=0).
    #[staticmethod]
    #[pyo3(signature = (with_intercept=true, max_iter=100, tol=1e-6))]
    fn gaussian(with_intercept: bool, max_iter: usize, tol: f64) -> Self {
        Self {
            var_power: 0.0,
            link_power: Some(1.0),
            with_intercept,
            max_iter,
            tol,
            fitted: None,
        }
    }

    /// Create a Gamma Tweedie model (var_power=2).
    #[staticmethod]
    #[pyo3(signature = (with_intercept=true, max_iter=100, tol=1e-6))]
    fn gamma(with_intercept: bool, max_iter: usize, tol: f64) -> Self {
        Self {
            var_power: 2.0,
            link_power: Some(0.0),
            with_intercept,
            max_iter,
            tol,
            fitted: None,
        }
    }

    /// Create an Inverse Gaussian Tweedie model (var_power=3).
    #[staticmethod]
    #[pyo3(signature = (with_intercept=true, max_iter=100, tol=1e-6))]
    fn inverse_gaussian(with_intercept: bool, max_iter: usize, tol: f64) -> Self {
        Self {
            var_power: 3.0,
            link_power: Some(0.0),
            with_intercept,
            max_iter,
            tol,
            fitted: None,
        }
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_mat = x.to_faer();
        let y_col = y.to_faer();

        let mut builder = TweedieRegressor::builder()
            .var_power(slf.var_power)
            .with_intercept(slf.with_intercept)
            .max_iterations(slf.max_iter)
            .tolerance(slf.tol);

        if let Some(lp) = slf.link_power {
            builder = builder.link_power(lp);
        }

        let model = builder.build();

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
    fn aic(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(Some(fitted.result().aic))
    }

    #[getter]
    fn bic(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(Some(fitted.result().bic))
    }

    #[getter]
    fn var_power_value(&self) -> f64 {
        self.var_power
    }
}
