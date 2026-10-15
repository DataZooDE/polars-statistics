//! PyO3 wrapper for Generalized Linear Mixed Model (GLMM) regression.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use anofox_regression::solvers::{FittedGlmm, GlmmRegressor, GlmmRegressorBuilder};
use faer::Col;

use crate::utils::{IntoNumpy, ToFaer};

/// Generalized Linear Mixed Model (GLMM) regression.
///
/// Fits a linear mixed model with fixed and random effects via REML (Gaussian)
/// or penalized IRLS (Poisson/Binomial). Random effects are specified through
/// group membership arrays passed at fit time.
///
/// Use the class-method factories (`gaussian()`, `poisson()`, `binomial()`,
/// `negative_binomial()`, `gamma()`, `tweedie(power)`) to construct the model
/// with the appropriate family, then call `fit` or `fit_crossed`.
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
    /// Tweedie variance power (`1 < power < 2`); unused by every other family.
    power: f64,
    /// Fixed negative-binomial size theta; `None` estimates it (`glmer.nb`).
    /// Unused by every other family.
    nb_theta: Option<f64>,
    with_intercept: bool,
    random_intercept: bool,
    random_slopes: Vec<usize>,
    /// Random-slope columns per grouping factor for `fit_crossed`; exclusive
    /// with `random_slopes`.
    random_slopes_per_factor: Option<Vec<Vec<usize>>>,
    /// Upper bound of the profiled ratio theta = sigma_b/sigma (default 1000).
    theta_max: f64,
    reml: bool,
    max_iter: usize,
    tol: f64,
    fitted: Option<FittedGlmm>,
}

#[pymethods]
impl PyGLMM {
    /// Construct a Gaussian GLMM (REML by default).
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (add_intercept=None, with_intercept=None, reml=true, max_iter=100, tol=1e-8, random_intercept=true, random_slopes=None, random_slopes_per_factor=None, theta_max=1000.0))]
    fn gaussian(
        py: Python<'_>,
        add_intercept: Option<bool>,
        with_intercept: Option<bool>,
        reml: bool,
        max_iter: usize,
        tol: f64,
        random_intercept: bool,
        random_slopes: Option<Vec<usize>>,
        random_slopes_per_factor: Option<Vec<Vec<usize>>>,
        theta_max: f64,
    ) -> PyResult<Self> {
        let with_intercept =
            crate::pymodels::errors::resolve_intercept(py, add_intercept, with_intercept, true)?;
        Self::construct(
            "gaussian",
            1.5,
            None,
            with_intercept,
            random_intercept,
            random_slopes,
            random_slopes_per_factor,
            theta_max,
            reml,
            max_iter,
            tol,
        )
    }

    /// Construct a Poisson GLMM.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (add_intercept=None, with_intercept=None, reml=false, max_iter=100, tol=1e-8, random_intercept=true, random_slopes=None, random_slopes_per_factor=None, theta_max=1000.0))]
    fn poisson(
        py: Python<'_>,
        add_intercept: Option<bool>,
        with_intercept: Option<bool>,
        reml: bool,
        max_iter: usize,
        tol: f64,
        random_intercept: bool,
        random_slopes: Option<Vec<usize>>,
        random_slopes_per_factor: Option<Vec<Vec<usize>>>,
        theta_max: f64,
    ) -> PyResult<Self> {
        let with_intercept =
            crate::pymodels::errors::resolve_intercept(py, add_intercept, with_intercept, true)?;
        Self::construct(
            "poisson",
            1.5,
            None,
            with_intercept,
            random_intercept,
            random_slopes,
            random_slopes_per_factor,
            theta_max,
            reml,
            max_iter,
            tol,
        )
    }

    /// Construct a Binomial (logistic) GLMM.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (add_intercept=None, with_intercept=None, reml=false, max_iter=100, tol=1e-8, random_intercept=true, random_slopes=None, random_slopes_per_factor=None, theta_max=1000.0))]
    fn binomial(
        py: Python<'_>,
        add_intercept: Option<bool>,
        with_intercept: Option<bool>,
        reml: bool,
        max_iter: usize,
        tol: f64,
        random_intercept: bool,
        random_slopes: Option<Vec<usize>>,
        random_slopes_per_factor: Option<Vec<Vec<usize>>>,
        theta_max: f64,
    ) -> PyResult<Self> {
        let with_intercept =
            crate::pymodels::errors::resolve_intercept(py, add_intercept, with_intercept, true)?;
        Self::construct(
            "binomial",
            1.5,
            None,
            with_intercept,
            random_intercept,
            random_slopes,
            random_slopes_per_factor,
            theta_max,
            reml,
            max_iter,
            tol,
        )
    }

    /// Construct a negative-binomial (NB2) GLMM with the log link,
    /// `Var(y) = mu + mu^2/theta`.
    ///
    /// By default the size `theta` is estimated by maximising the profiled
    /// Laplace log-likelihood, as `lme4::glmer.nb(nAGQ = 0)`; pass `nb_theta`
    /// to fix it, matching `glmer(family = MASS::negative.binomial(theta), nAGQ = 0)`.
    ///
    /// Parameters
    /// ----------
    /// nb_theta : float, optional
    ///     Fixed negative-binomial size theta. Must be finite and > 0.
    ///     When omitted (default), theta is estimated from the data.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (nb_theta=None, add_intercept=None, with_intercept=None, reml=false, max_iter=100, tol=1e-8, random_intercept=true, random_slopes=None, random_slopes_per_factor=None, theta_max=1000.0))]
    fn negative_binomial(
        py: Python<'_>,
        nb_theta: Option<f64>,
        add_intercept: Option<bool>,
        with_intercept: Option<bool>,
        reml: bool,
        max_iter: usize,
        tol: f64,
        random_intercept: bool,
        random_slopes: Option<Vec<usize>>,
        random_slopes_per_factor: Option<Vec<Vec<usize>>>,
        theta_max: f64,
    ) -> PyResult<Self> {
        if let Some(t) = nb_theta {
            if !t.is_finite() || t <= 0.0 {
                return Err(PyValueError::new_err(format!(
                    "nb_theta must be finite and > 0, got {t}"
                )));
            }
        }
        let with_intercept =
            crate::pymodels::errors::resolve_intercept(py, add_intercept, with_intercept, true)?;
        Self::construct(
            "negative_binomial",
            1.5,
            nb_theta,
            with_intercept,
            random_intercept,
            random_slopes,
            random_slopes_per_factor,
            theta_max,
            reml,
            max_iter,
            tol,
        )
    }

    /// Construct a Gamma GLMM with the log link (`y > 0`), matching
    /// `lme4::glmer(family = Gamma(link = "log"), nAGQ = 0)`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (add_intercept=None, with_intercept=None, reml=false, max_iter=100, tol=1e-8, random_intercept=true, random_slopes=None, random_slopes_per_factor=None, theta_max=1000.0))]
    fn gamma(
        py: Python<'_>,
        add_intercept: Option<bool>,
        with_intercept: Option<bool>,
        reml: bool,
        max_iter: usize,
        tol: f64,
        random_intercept: bool,
        random_slopes: Option<Vec<usize>>,
        random_slopes_per_factor: Option<Vec<Vec<usize>>>,
        theta_max: f64,
    ) -> PyResult<Self> {
        let with_intercept =
            crate::pymodels::errors::resolve_intercept(py, add_intercept, with_intercept, true)?;
        Self::construct(
            "gamma",
            1.5,
            None,
            with_intercept,
            random_intercept,
            random_slopes,
            random_slopes_per_factor,
            theta_max,
            reml,
            max_iter,
            tol,
        )
    }

    /// Construct a Tweedie (compound Poisson-gamma) GLMM with variance power
    /// `1 < power < 2` and the log link (`y >= 0`), matching
    /// `lme4::glmer(family = statmod::tweedie(var.power = power, link.power = 0), nAGQ = 0)`.
    ///
    /// Parameters
    /// ----------
    /// power : float, default 1.5
    ///     Tweedie variance power. Must satisfy `1 < power < 2`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (power=1.5, add_intercept=None, with_intercept=None, reml=false, max_iter=100, tol=1e-8, random_intercept=true, random_slopes=None, random_slopes_per_factor=None, theta_max=1000.0))]
    fn tweedie(
        py: Python<'_>,
        power: f64,
        add_intercept: Option<bool>,
        with_intercept: Option<bool>,
        reml: bool,
        max_iter: usize,
        tol: f64,
        random_intercept: bool,
        random_slopes: Option<Vec<usize>>,
        random_slopes_per_factor: Option<Vec<Vec<usize>>>,
        theta_max: f64,
    ) -> PyResult<Self> {
        if !(power.is_finite() && power > 1.0 && power < 2.0) {
            return Err(PyValueError::new_err(format!(
                "power must satisfy 1 < power < 2, got {power}"
            )));
        }
        let with_intercept =
            crate::pymodels::errors::resolve_intercept(py, add_intercept, with_intercept, true)?;
        Self::construct(
            "tweedie",
            power,
            None,
            with_intercept,
            random_intercept,
            random_slopes,
            random_slopes_per_factor,
            theta_max,
            reml,
            max_iter,
            tol,
        )
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
    /// offset : numpy.ndarray of shape (n_samples,), optional
    ///     Per-observation offset on the linear-predictor scale, e.g.
    ///     ``log(exposure)``; added with a fixed coefficient of 1, as
    ///     ``offset()`` in an lme4 formula. Length must equal n_samples.
    ///
    /// Returns
    /// -------
    /// self
    ///     The fitted model (enables method chaining).
    #[pyo3(signature = (x, y, group, offset=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        group: Vec<u64>,
        offset: Option<PyReadonlyArray1<'py, f64>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        crate::pymodels::errors::validate_xy("GLMM", &x, &y)?;
        let x_mat = x.to_faer();
        let y_col = y.to_faer();
        let n_rows = x_mat.nrows();
        if group.len() != n_rows {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "group length ({}) must equal number of samples ({})",
                group.len(),
                n_rows
            )));
        }
        // Portably convert u64 → usize (safe on all wheel targets)
        let group_usize: Vec<usize> = group.iter().map(|&g| g as usize).collect();

        let offset_col = match offset {
            Some(off) => {
                let off_n = off.shape()[0];
                if off_n != n_rows {
                    return Err(PyValueError::new_err(format!(
                        "offset has {off_n} entries but x/y have {n_rows} rows"
                    )));
                }
                Some(off.to_faer())
            }
            None => None,
        };

        let model = slf.regressor_builder(offset_col).build();

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
    /// offset : numpy.ndarray of shape (n_samples,), optional
    ///     Per-observation offset on the linear-predictor scale, e.g.
    ///     ``log(exposure)``; added with a fixed coefficient of 1, as
    ///     ``offset()`` in an lme4 formula. Length must equal n_samples.
    ///
    /// Returns
    /// -------
    /// self
    ///     The fitted model (enables method chaining).
    #[pyo3(signature = (x, y, groups, offset=None))]
    fn fit_crossed<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        groups: Vec<Vec<u64>>,
        offset: Option<PyReadonlyArray1<'py, f64>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_mat = x.to_faer();
        let y_col = y.to_faer();
        let n_rows = x_mat.nrows();
        for (i, g) in groups.iter().enumerate() {
            if g.len() != n_rows {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "groups[{}] length ({}) must equal number of samples ({})",
                    i,
                    g.len(),
                    n_rows
                )));
            }
        }
        let groups_usize: Vec<Vec<usize>> = groups
            .iter()
            .map(|g| g.iter().map(|&v| v as usize).collect())
            .collect();
        let group_refs: Vec<&[usize]> = groups_usize.iter().map(|g| g.as_slice()).collect();

        let offset_col = match offset {
            Some(off) => {
                let off_n = off.shape()[0];
                if off_n != n_rows {
                    return Err(PyValueError::new_err(format!(
                        "offset has {off_n} entries but x/y have {n_rows} rows"
                    )));
                }
                Some(off.to_faer())
            }
            None => None,
        };

        let model = slf.regressor_builder(offset_col).build();

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
    /// offset : numpy.ndarray of shape (n_samples,), optional
    ///     Per-observation offset added to the linear predictor, e.g.
    ///     ``log(exposure)``; length must equal the rows of `x`.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray of shape (n_samples,)
    ///     Predicted values using fixed effects only (no random effects).
    #[pyo3(signature = (x, offset=None))]
    fn predict_fixed<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        offset: Option<PyReadonlyArray1<'py, f64>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        let x_mat = x.to_faer();
        match offset {
            Some(off) => {
                let n_rows = x_mat.nrows();
                let off_n = off.shape()[0];
                if off_n != n_rows {
                    return Err(PyValueError::new_err(format!(
                        "offset has {off_n} entries but x has {n_rows} rows"
                    )));
                }
                let off_col = off.to_faer();
                Ok(fitted
                    .predict_fixed_with_offset(&x_mat, &off_col)
                    .into_numpy(py))
            }
            None => Ok(fitted.predict_fixed(&x_mat).into_numpy(py)),
        }
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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        Ok(PyArray1::from_slice(py, fitted.fixed_effects()))
    }

    /// Standard errors of the fixed-effect coefficients.
    #[getter]
    fn std_errors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        Ok(PyArray1::from_slice(py, fitted.std_errors()))
    }

    /// Wald z statistics beta/SE of the fixed effects, intercept first —
    /// lme4 `summary()` `z value` column.
    #[getter]
    fn z_values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        Ok(PyArray1::from_vec(py, fitted.z_values()))
    }

    /// Two-sided asymptotic normal p-values `2*(1-Phi(|z|))` of the fixed
    /// effects. For a Gaussian LMM this is the large-sample approximation —
    /// lme4's `lmer` prints none.
    #[getter]
    fn p_values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        Ok(PyArray1::from_vec(py, fitted.p_values()))
    }

    /// Wald confidence intervals for the fixed effects at `level` (default
    /// 0.95), as ``(lower, upper)`` — matches R `confint(method = "Wald")`.
    /// Returns all-NaN bounds when `level` is not in `(0, 1)`.
    #[pyo3(signature = (level=0.95))]
    #[allow(clippy::type_complexity)]
    fn conf_int<'py>(
        &self,
        py: Python<'py>,
        level: f64,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        let (lower, upper) = fitted.conf_int(level);
        Ok((PyArray1::from_vec(py, lower), PyArray1::from_vec(py, upper)))
    }

    /// Fitted intercept, or None when with_intercept=False.
    #[getter]
    fn intercept(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        Ok(fitted.intercept())
    }

    /// Non-intercept fixed effects (slopes only).
    #[getter]
    fn slopes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        Ok(PyArray1::from_slice(py, fitted.slopes()))
    }

    /// Random intercept BLUPs per group (single-factor fits).
    #[getter]
    fn random_effects<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        Ok(PyArray1::from_slice(py, fitted.random_effects()))
    }

    /// Random effect standard deviations: sqrt(diag(Σ)).
    #[getter]
    fn random_sd<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        Ok(PyArray1::from_slice(py, &fitted.random_sd()))
    }

    /// Profiled variance-component ratio σ_b / σ.
    #[getter]
    fn theta(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        Ok(fitted.theta())
    }

    /// Negative-binomial size θ (`Var(y) = μ + μ²/θ`) — estimated, or as fixed
    /// via `negative_binomial(nb_theta=...)`. `None` for every other family.
    /// Distinct from `theta` (the profiled σ_b/σ variance-component ratio).
    #[getter]
    fn nb_theta(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        Ok(fitted.nb_theta())
    }

    /// Residual standard deviation (1.0 for Poisson/Binomial).
    #[getter]
    fn sigma(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        Ok(fitted.sigma())
    }

    /// −2 log-likelihood (or REML criterion for Gaussian).
    #[getter]
    fn deviance(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        Ok(fitted.deviance())
    }

    /// Log-likelihood (= −deviance / 2).
    #[getter]
    fn log_likelihood(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        Ok(fitted.log_likelihood())
    }

    /// Number of distinct groups (for single-factor fits).
    #[getter]
    fn n_groups(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        Ok(fitted.n_groups())
    }

    /// Whether the EM/PIRLS algorithm converged.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        Ok(fitted.converged())
    }

    /// Number of iterations used.
    #[getter]
    fn iterations(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        Ok(fitted.iterations())
    }

    /// Random-effects covariance matrix Sigma (q x q): the intercept first,
    /// then the random slopes in the given order. `None` only when unfitted.
    #[getter]
    fn random_cov<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        PyArray2::from_vec2(py, fitted.random_cov())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Full BLUP matrix (n_groups x q): column 0 is the random intercept,
    /// then the random slopes in the given order.
    #[getter]
    fn random_effects_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        PyArray2::from_vec2(py, fitted.random_effects_matrix())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Random-effects correlation matrix (unit diagonal), derived from `random_cov`.
    #[getter]
    fn random_corr<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        PyArray2::from_vec2(py, &fitted.random_corr())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Number of random-effects columns q (1 + the number of random slopes
    /// for a single-factor fit).
    #[getter]
    fn n_random_effects(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        Ok(fitted.n_random_effects())
    }

    /// Number of grouping factors (1 for `fit`, >= 2 for a crossed `fit_crossed`).
    #[getter]
    fn n_factors(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        Ok(fitted.n_factors())
    }

    /// Whether the Gaussian fit used REML (always False for non-Gaussian families).
    #[getter]
    fn is_reml(&self) -> PyResult<bool> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        Ok(fitted.is_reml())
    }

    /// Random-effects covariance matrix for grouping factor `f` (q_f x q_f).
    /// `None` for a single-factor fit or an out-of-range `f`; component 0 is
    /// the random intercept, then the factor's slopes in the given order.
    fn factor_random_cov<'py>(
        &self,
        py: Python<'py>,
        f: usize,
    ) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        match fitted.factor_random_cov(f) {
            Some(cov) => Ok(Some(
                PyArray2::from_vec2(py, cov).map_err(|e| PyValueError::new_err(e.to_string()))?,
            )),
            None => Ok(None),
        }
    }

    /// Random-effects standard deviations for grouping factor `f`. `None`
    /// for a single-factor fit or an out-of-range `f`; component 0 is the
    /// random intercept, then the factor's slopes in the given order.
    fn factor_random_sd<'py>(
        &self,
        py: Python<'py>,
        f: usize,
    ) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        Ok(fitted
            .factor_random_sd(f)
            .map(|sd| PyArray1::from_vec(py, sd)))
    }

    /// BLUP matrix for grouping factor `f` (n_levels_f x q_f). `None` for a
    /// single-factor fit or an out-of-range `f`; column 0 is the random
    /// intercept, then the factor's slopes in the given order.
    fn factor_random_effects_matrix<'py>(
        &self,
        py: Python<'py>,
        f: usize,
    ) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("GLMM"))?;
        match fitted.factor_random_effects_matrix(f) {
            Some(mat) => Ok(Some(
                PyArray2::from_vec2(py, mat).map_err(|e| PyValueError::new_err(e.to_string()))?,
            )),
            None => Ok(None),
        }
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

impl PyGLMM {
    /// Shared factory-construction path for all six family staticmethods:
    /// validates the random-effects structure options once (`random_slopes`
    /// and `random_slopes_per_factor` are mutually exclusive, a disabled
    /// random intercept needs at least one non-empty slope list, `theta_max`
    /// must be finite and positive) before building `Self`.
    #[allow(clippy::too_many_arguments)]
    fn construct(
        family: &str,
        power: f64,
        nb_theta: Option<f64>,
        with_intercept: bool,
        random_intercept: bool,
        random_slopes: Option<Vec<usize>>,
        random_slopes_per_factor: Option<Vec<Vec<usize>>>,
        theta_max: f64,
        reml: bool,
        max_iter: usize,
        tol: f64,
    ) -> PyResult<Self> {
        if random_slopes.is_some() && random_slopes_per_factor.is_some() {
            return Err(PyValueError::new_err(
                "random_slopes and random_slopes_per_factor are mutually exclusive",
            ));
        }
        let has_slopes = random_slopes.as_ref().is_some_and(|s| !s.is_empty())
            || random_slopes_per_factor
                .as_ref()
                .is_some_and(|per| per.iter().any(|s| !s.is_empty()));
        if !random_intercept && !has_slopes {
            return Err(PyValueError::new_err(
                "random_intercept=False requires at least one random slope \
                 (via random_slopes or random_slopes_per_factor)",
            ));
        }
        if !(theta_max.is_finite() && theta_max > 0.0) {
            return Err(PyValueError::new_err(format!(
                "theta_max must be finite and > 0, got {theta_max}"
            )));
        }
        Ok(Self {
            family: family.into(),
            power,
            nb_theta,
            with_intercept,
            random_intercept,
            random_slopes: random_slopes.unwrap_or_default(),
            random_slopes_per_factor,
            theta_max,
            reml,
            max_iter,
            tol,
            fitted: None,
        })
    }

    /// Build the upstream `GlmmRegressorBuilder` from the current family and
    /// options, optionally with a per-observation offset. Shared by `fit` and
    /// `fit_crossed` so every option is threaded identically on both paths.
    fn regressor_builder(&self, offset: Option<Col<f64>>) -> GlmmRegressorBuilder {
        let mut builder = match self.family.as_str() {
            "poisson" => GlmmRegressor::poisson(),
            "binomial" => GlmmRegressor::binomial(),
            "negative_binomial" => GlmmRegressor::negative_binomial(),
            "gamma" => GlmmRegressor::gamma(),
            "tweedie" => GlmmRegressor::tweedie(self.power),
            _ => GlmmRegressor::gaussian(),
        }
        .with_intercept(self.with_intercept)
        .random_intercept(self.random_intercept)
        .random_slopes(self.random_slopes.clone())
        .reml(self.reml)
        .max_iterations(self.max_iter)
        .tolerance(self.tol)
        .theta_max(self.theta_max);

        if let Some(per) = &self.random_slopes_per_factor {
            builder = builder.random_slopes_per_factor(per.clone());
        }
        if let Some(t) = self.nb_theta {
            builder = builder.nb_theta(t);
        }
        if let Some(o) = offset {
            builder = builder.offset(o);
        }
        builder
    }
}
