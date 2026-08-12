//! PyO3 wrapper for LARS (Least-Angle Regression) regression.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::solvers::{
    FittedLars, FittedRegressor, LarsMethod, LarsRegressor, Regressor,
};

use crate::utils::{IntoNumpy, ToFaer};

/// Least-Angle Regression (LARS / LassoLARS) model.
///
/// Follows the LARS or LassoLARS path algorithm. Use `method="lar"` for the
/// classical LARS algorithm or `method="lasso"` for the LASSO path (equivalent
/// to LassoLARS with coefficient sign-changes handled at each step).
///
/// Parameters
/// ----------
/// method : str, default "lar"
///     Algorithm variant: ``"lar"`` (Least-Angle Regression) or ``"lasso"``
///     (LASSO path via LARS). Passed as a string — the underlying
///     ``LarsMethod`` enum is never exposed to Python.
/// fit_intercept : bool, default True
///     Whether to fit an intercept term.
/// n_nonzero_coefs : int or None, default None
///     Maximum number of non-zero coefficients to include in the path.
///     ``None`` means ``min(n_samples - 1, n_features)``.
/// alpha : float, default 0.0
///     Regularisation strength for LassoLARS. When non-zero the path is
///     interpolated to produce exactly ``alpha``-regularised coefficients.
/// standardize : bool, default False
///     Whether to standardize features before fitting.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from polars_statistics import LARS
/// >>>
/// >>> X = np.random.randn(50, 3)
/// >>> y = X @ [1.0, 2.0, -1.0] + 0.1 * np.random.randn(50)
/// >>> model = LARS().fit(X, y)
/// >>> model.alphas   # regularisation path alphas
/// >>> model.coefficients
#[pyclass(name = "LARS")]
pub struct PyLARS {
    method: String,
    fit_intercept: bool,
    n_nonzero_coefs: Option<usize>,
    alpha: f64,
    standardize: bool,
    fitted: Option<FittedLars>,
}

#[pymethods]
impl PyLARS {
    #[new]
    #[pyo3(signature = (method="lar", fit_intercept=true, n_nonzero_coefs=None, alpha=0.0, standardize=false))]
    fn new(
        method: &str,
        fit_intercept: bool,
        n_nonzero_coefs: Option<usize>,
        alpha: f64,
        standardize: bool,
    ) -> Self {
        Self {
            method: method.to_owned(),
            fit_intercept,
            n_nonzero_coefs,
            alpha,
            standardize,
            fitted: None,
        }
    }

    /// Fit the LARS model.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data.
    /// y : array-like of shape (n_samples,)
    ///     Target values.
    ///
    /// Returns
    /// -------
    /// self
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_mat = x.to_faer();
        let y_col = y.to_faer();

        let lars_method = match slf.method.as_str() {
            "lasso" => LarsMethod::Lasso,
            _ => LarsMethod::Lar,
        };

        let mut builder = LarsRegressor::builder()
            .method(lars_method)
            .fit_intercept(slf.fit_intercept)
            .alpha(slf.alpha)
            .standardize(slf.standardize);
        if let Some(n) = slf.n_nonzero_coefs {
            builder = builder.n_nonzero_coefs(n);
        }
        let model = builder.build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        slf.fitted = Some(fitted);
        Ok(slf)
    }

    /// Predict response values.
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

    fn is_fitted(&self) -> bool {
        self.fitted.is_some()
    }

    /// Regularisation path alpha values (max absolute correlation at each step).
    #[getter]
    fn alphas<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(PyArray1::from_slice(py, fitted.alphas()))
    }

    /// Final model coefficients (after full path traversal or alpha-interpolation).
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(fitted.coefficients().into_numpy(py))
    }

    /// Intercept term (None if fit_intercept=False).
    #[getter]
    fn intercept(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(fitted.intercept())
    }

    /// R-squared coefficient of determination.
    #[getter]
    fn r_squared(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(fitted.result().r_squared)
    }

    /// Number of observations used in the fit.
    #[getter]
    fn n_observations(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(fitted.result().n_observations)
    }
}
