//! PyO3 wrapper for Generalized Linear Mixed Model (GLMM) regression.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use anofox_regression::solvers::{FittedGlmm, GlmmRegressor};

use crate::utils::{IntoNumpy, ToFaer};

/// Generalized Linear Mixed Model (GLMM) regression.
///
/// Fits a linear mixed model with fixed and random effects via REML (Gaussian)
/// or penalized IRLS (Poisson/Binomial). Random effects are specified through
/// group membership arrays passed at fit time.
///
/// Use the class-method factories (`gaussian()`, `poisson()`, `binomial()`) to
/// construct the model with the appropriate family, then call `fit` or
/// `fit_crossed`.
///
/// Parameters (via factory methods)
/// ----------------------------------
/// with_intercept : bool, default True
///     Include a fixed-effects intercept.
/// reml : bool, default True
///     Use REML criterion (Gaussian only; ignored for Poisson/Binomial).
/// max_iter : int, default 100
///     Maximum number of EM/PIRLS iterations.
/// tol : float, default 1e-8
///     Convergence tolerance.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from polars_statistics import GLMM
/// >>> X = np.random.randn(60, 2)
/// >>> y = X @ [1.0, -0.5] + np.random.randn(60) * 0.3
/// >>> group = np.repeat(np.arange(6), 10).tolist()
/// >>> model = GLMM.gaussian().fit(X, y, group)
/// >>> model.is_fitted()
/// True
/// >>> model.fixed_effects
/// array([...])
#[pyclass(name = "GLMM")]
pub struct PyGLMM {
    family: String,
    with_intercept: bool,
    random_intercept: bool,
    random_slopes: Vec<usize>,
    reml: bool,
    max_iter: usize,
    tol: f64,
    fitted: Option<FittedGlmm>,
}

#[pymethods]
impl PyGLMM {
    /// Construct a Gaussian GLMM (REML by default).
    #[staticmethod]
    #[pyo3(signature = (with_intercept=true, reml=true, max_iter=100, tol=1e-8))]
    fn gaussian(with_intercept: bool, reml: bool, max_iter: usize, tol: f64) -> Self {
        Self {
            family: "gaussian".into(),
            with_intercept,
            random_intercept: true,
            random_slopes: vec![],
            reml,
            max_iter,
            tol,
            fitted: None,
        }
    }

    /// Construct a Poisson GLMM.
    #[staticmethod]
    #[pyo3(signature = (with_intercept=true, reml=false, max_iter=100, tol=1e-8))]
    fn poisson(with_intercept: bool, reml: bool, max_iter: usize, tol: f64) -> Self {
        Self {
            family: "poisson".into(),
            with_intercept,
            random_intercept: true,
            random_slopes: vec![],
            reml,
            max_iter,
            tol,
            fitted: None,
        }
    }

    /// Construct a Binomial (logistic) GLMM.
    #[staticmethod]
    #[pyo3(signature = (with_intercept=true, reml=false, max_iter=100, tol=1e-8))]
    fn binomial(with_intercept: bool, reml: bool, max_iter: usize, tol: f64) -> Self {
        Self {
            family: "binomial".into(),
            with_intercept,
            random_intercept: true,
            random_slopes: vec![],
            reml,
            max_iter,
            tol,
            fitted: None,
        }
    }

    /// Fit the GLMM with a single grouping factor.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray of shape (n_samples, n_features)
    ///     Feature matrix (fixed effects design). Must be float64.
    /// y : numpy.ndarray of shape (n_samples,)
    ///     Response vector.
    /// group : list[int]
    ///     Group membership per observation (0-based integer IDs).
    ///     Length must equal n_samples.
    ///
    /// Returns
    /// -------
    /// self
    ///     The fitted model (enables method chaining).
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        group: Vec<u64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_mat = x.to_faer();
        let y_col = y.to_faer();
        // Portably convert u64 → usize (safe on all wheel targets)
        let group_usize: Vec<usize> = group.iter().map(|&g| g as usize).collect();

        let model = match slf.family.as_str() {
            "poisson" => GlmmRegressor::poisson(),
            "binomial" => GlmmRegressor::binomial(),
            _ => GlmmRegressor::gaussian(),
        }
        .with_intercept(slf.with_intercept)
        .random_intercept(slf.random_intercept)
        .random_slopes(slf.random_slopes.clone())
        .reml(slf.reml)
        .max_iterations(slf.max_iter)
        .tolerance(slf.tol)
        .build();

        let fitted = model
            .fit(&x_mat, &y_col, &group_usize)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        slf.fitted = Some(fitted);
        Ok(slf)
    }

    /// Fit the GLMM with multiple crossed or nested grouping factors.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray of shape (n_samples, n_features)
    ///     Feature matrix. Must be float64.
    /// y : numpy.ndarray of shape (n_samples,)
    ///     Response vector.
    /// groups : list[list[int]]
    ///     One list of group IDs per factor. Each inner list must have length
    ///     n_samples. For crossed factors pass `[sku_ids, region_ids]`; for
    ///     nested factors pass the interaction ID as a separate factor.
    ///
    /// Returns
    /// -------
    /// self
    ///     The fitted model (enables method chaining).
    fn fit_crossed<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        groups: Vec<Vec<u64>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_mat = x.to_faer();
        let y_col = y.to_faer();
        let groups_usize: Vec<Vec<usize>> = groups
            .iter()
            .map(|g| g.iter().map(|&v| v as usize).collect())
            .collect();
        let group_refs: Vec<&[usize]> = groups_usize.iter().map(|g| g.as_slice()).collect();

        let model = match slf.family.as_str() {
            "poisson" => GlmmRegressor::poisson(),
            "binomial" => GlmmRegressor::binomial(),
            _ => GlmmRegressor::gaussian(),
        }
        .with_intercept(slf.with_intercept)
        .random_intercept(slf.random_intercept)
        .random_slopes(slf.random_slopes.clone())
        .reml(slf.reml)
        .max_iterations(slf.max_iter)
        .tolerance(slf.tol)
        .build();

        let fitted = model
            .fit_crossed(&x_mat, &y_col, &group_refs)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        slf.fitted = Some(fitted);
        Ok(slf)
    }

    /// Marginal (new-group) predictions using only the fixed effects.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray of shape (n_samples, n_features)
    ///     Feature matrix. Must be float64.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray of shape (n_samples,)
    ///     Predicted values using fixed effects only (no random effects).
    fn predict_fixed<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        let x_mat = x.to_faer();
        Ok(fitted.predict_fixed(&x_mat).into_numpy(py))
    }

    /// Per-factor summaries (non-empty only for `fit_crossed`).
    ///
    /// Returns
    /// -------
    /// list[dict]
    ///     One dict per factor with keys:
    ///     - ``n_levels`` (int): number of distinct levels.
    ///     - ``sd`` (float): random-intercept standard deviation σ_f.
    ///     - ``blups`` (numpy.ndarray): BLUP per level.
    fn factors<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        fitted
            .factors()
            .iter()
            .map(|f| {
                let d = PyDict::new(py);
                d.set_item("n_levels", f.n_levels)?;
                d.set_item("sd", f.sd)?;
                d.set_item("blups", PyArray1::from_slice(py, &f.blups))?;
                Ok(d)
            })
            .collect()
    }

    /// Whether the model has been fitted.
    fn is_fitted(&self) -> bool {
        self.fitted.is_some()
    }

    /// Fixed-effect coefficients (intercept first when with_intercept=True).
    #[getter]
    fn fixed_effects<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(PyArray1::from_slice(py, fitted.fixed_effects()))
    }

    /// Standard errors of the fixed-effect coefficients.
    #[getter]
    fn std_errors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(PyArray1::from_slice(py, fitted.std_errors()))
    }

    /// Fitted intercept, or None when with_intercept=False.
    #[getter]
    fn intercept(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(fitted.intercept())
    }

    /// Non-intercept fixed effects (slopes only).
    #[getter]
    fn slopes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(PyArray1::from_slice(py, fitted.slopes()))
    }

    /// Random intercept BLUPs per group (single-factor fits).
    #[getter]
    fn random_effects<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(PyArray1::from_slice(py, fitted.random_effects()))
    }

    /// Random effect standard deviations: sqrt(diag(Σ)).
    #[getter]
    fn random_sd<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(PyArray1::from_slice(py, &fitted.random_sd()))
    }

    /// Profiled variance-component ratio σ_b / σ.
    #[getter]
    fn theta(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(fitted.theta())
    }

    /// Residual standard deviation (1.0 for Poisson/Binomial).
    #[getter]
    fn sigma(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(fitted.sigma())
    }

    /// −2 log-likelihood (or REML criterion for Gaussian).
    #[getter]
    fn deviance(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(fitted.deviance())
    }

    /// Log-likelihood (= −deviance / 2).
    #[getter]
    fn log_likelihood(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(fitted.log_likelihood())
    }

    /// Number of distinct groups (for single-factor fits).
    #[getter]
    fn n_groups(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(fitted.n_groups())
    }

    /// Whether the EM/PIRLS algorithm converged.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(fitted.converged())
    }

    /// Number of iterations used.
    #[getter]
    fn iterations(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(fitted.iterations())
    }
}
