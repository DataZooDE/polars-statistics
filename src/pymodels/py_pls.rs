//! PyO3 wrapper for Partial Least Squares regression.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::solvers::{FittedPls, FittedRegressor, PlsRegressor, Regressor};

use crate::utils::{IntoNumpy, ToFaer};

/// Partial Least Squares (PLS) regression.
///
/// Projects `X` onto a lower-dimensional space spanned by `n_components`
/// latent variables that maximize covariance with `y`, then fits a linear
/// model in that space. Useful when features are highly correlated.
///
/// Parameters
/// ----------
/// n_components : int, default 2
///     Number of PLS latent components.
/// with_intercept : bool, default True
///     Whether to include an intercept term.
/// tol : float, default 1e-6
///     Convergence tolerance for the NIPALS iterations.
/// scale : bool, default True
///     Whether to scale `X` columns to unit variance before fitting.
#[pyclass(name = "PLS")]
pub struct PyPls {
    n_components: usize,
    with_intercept: bool,
    tol: f64,
    scale: bool,
    fitted: Option<FittedPls>,
}

#[pymethods]
impl PyPls {
    #[new]
    #[pyo3(signature = (n_components=2, with_intercept=true, tol=1e-6, scale=true))]
    fn new(n_components: usize, with_intercept: bool, tol: f64, scale: bool) -> Self {
        Self {
            n_components,
            with_intercept,
            tol,
            scale,
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

        let model = PlsRegressor::builder()
            .n_components(slf.n_components)
            .with_intercept(slf.with_intercept)
            .tolerance(slf.tol)
            .scale(slf.scale)
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
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        let x_mat = x.to_faer();
        Ok(fitted.predict(&x_mat).into_numpy(py))
    }

    /// Project `X` onto the latent component space.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        let x_mat = x.to_faer();
        let scores = fitted.transform(&x_mat);
        Ok((&scores).into_numpy(py))
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
        let coefs = &fitted.result().coefficients;
        Ok(coefs.into_numpy(py))
    }

    #[getter]
    fn intercept(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(fitted.result().intercept)
    }

    #[getter]
    fn n_components(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(fitted.n_components())
    }

    /// Fraction of variance in `y` explained by each retained component.
    #[getter]
    fn explained_variance_ratio<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        let evr = fitted.explained_variance_ratio();
        Ok((&evr).into_numpy(py))
    }

    #[getter]
    fn r_squared(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(fitted.result().r_squared)
    }

    #[getter]
    fn n_observations(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(fitted.result().n_observations)
    }
}
