//! PyO3 wrapper for Negative Binomial regression.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::solvers::{
    FittedNegativeBinomial, FittedRegressor, NegativeBinomialRegressor, Regressor,
};

use crate::utils::{IntoNumpy, ToFaer};

/// Negative Binomial regression model.
///
/// For overdispersed count data where variance exceeds the mean.
/// Variance function: Var[Y] = μ + μ²/θ
///
/// Parameters
/// ----------
/// theta : float, optional
///     Dispersion parameter. If None, theta is estimated from the data.
/// estimate_theta : bool, default True
///     Whether to estimate theta from the data.
/// with_intercept : bool, default True
///     Whether to include an intercept term.
/// max_iter : int, default 100
///     Maximum number of IRLS iterations.
/// tol : float, default 1e-6
///     Convergence tolerance.
#[pyclass(name = "NegativeBinomial")]
pub struct PyNegativeBinomial {
    theta: Option<f64>,
    estimate_theta: bool,
    with_intercept: bool,
    max_iter: usize,
    tol: f64,
    fitted: Option<FittedNegativeBinomial>,
}

#[pymethods]
impl PyNegativeBinomial {
    #[new]
    #[pyo3(signature = (theta=None, estimate_theta=true, with_intercept=true, max_iter=100, tol=1e-6))]
    fn new(
        theta: Option<f64>,
        estimate_theta: bool,
        with_intercept: bool,
        max_iter: usize,
        tol: f64,
    ) -> Self {
        Self {
            theta,
            estimate_theta,
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

        let mut builder = NegativeBinomialRegressor::builder()
            .with_intercept(slf.with_intercept)
            .estimate_theta(slf.estimate_theta)
            .max_iterations(slf.max_iter)
            .tolerance(slf.tol);

        if let Some(theta) = slf.theta {
            builder = builder.theta(theta);
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
    fn theta_estimated(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.theta)
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
}
