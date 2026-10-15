//! PyO3 wrapper for the accelerated failure time (AFT) survival regression
//! engine (`AFT`).

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use anofox_regression::solvers::aft::{AftDistribution, AftRegressor, FittedAft};
use anofox_regression::solvers::penalized_glm::{PriorSpec, VcovType};

use crate::expressions::{parse_aft_distribution, parse_vcov_type};
use crate::pymodels::py_penalized_glm::extract_priors;
use crate::utils::{IntoNumpy, ToFaer};

/// Accelerated failure time (AFT) survival regression with right censoring,
/// for any of the Weibull, log-normal, log-logistic or exponential error
/// distributions.
///
/// Models ``log T = x'beta + sigma * W`` for a fixed error distribution
/// ``W`` — the parametrisation of R's ``survival::survreg``. An observation
/// contributes its density when the event was observed and its survival
/// probability when the row is still censored.
///
/// Parameters
/// ----------
/// distribution : str, default "weibull"
///     Error distribution for ``T``: ``"weibull"``, ``"lognormal"``
///     (or ``"log_normal"``/``"log-normal"``), ``"loglogistic"`` (or
///     ``"log_logistic"``/``"log-logistic"``), or ``"exponential"`` (or
///     ``"exp"``). The exponential distribution fixes ``scale = 1`` (not
///     estimated); every other distribution estimates it.
/// priors : sequence, optional
///     Per-coefficient priors, positionally aligned with the feature
///     columns. Same format as ``PenalizedGLM``'s ``priors``: each entry is
///     ``None``/``"flat"`` (flat), a ``(kind, loc, scale)`` tuple/list (or
///     ``(kind,)``), or a dict ``{"kind", "loc", "scale"}`` — ``kind`` is one
///     of ``"flat"``, ``"normal"``, ``"laplace"``. An optional leading entry
///     applies to the intercept (``p`` or ``p + 1`` entries).
/// vcov_type : str, default "laplace"
///     Covariance at the mode: ``"laplace"``, ``"sandwich"``, or ``"naive"``
///     (see ``PenalizedGLM`` for the exact semantics).
/// add_intercept : bool, default True
///     Whether to include an intercept term.
/// max_iter : int, default 100
///     Maximum Newton iterations.
/// tol : float, default 1e-9
///     Convergence tolerance on the log-likelihood.
/// compute_inference : bool, default True
///     Whether to compute standard errors / z / p values / confidence
///     intervals / the covariance.
/// confidence_level : float, default 0.95
///     Confidence level for confidence intervals.
///
/// Notes
/// -----
/// ``event`` passed to ``fit`` must be 0/1 (bool, int, or float); any other
/// value (e.g. R's 1/2 coding) raises ``ValueError`` rather than being
/// silently misread. Validated against R ``survival::survreg`` on the
/// `anofox-regression` 0.5.22 R reference fixture (all four distributions).
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from polars_statistics import AFT
/// >>>
/// >>> model = AFT(distribution="weibull").fit(x, time, event)
/// >>> model.predict_median(x)
#[pyclass(name = "AFT")]
pub struct PyAFT {
    distribution: AftDistribution,
    priors: Vec<PriorSpec>,
    vcov_type: VcovType,
    with_intercept: bool,
    max_iter: u32,
    tol: f64,
    compute_inference: bool,
    confidence_level: f64,
    fitted: Option<FittedAft>,
}

#[pymethods]
impl PyAFT {
    #[new]
    #[pyo3(signature = (
        distribution="weibull",
        priors=None,
        vcov_type="laplace",
        add_intercept=true,
        max_iter=100,
        tol=1e-9,
        compute_inference=true,
        confidence_level=0.95
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        distribution: &str,
        priors: Option<&Bound<'_, PyAny>>,
        vcov_type: &str,
        add_intercept: bool,
        max_iter: u32,
        tol: f64,
        compute_inference: bool,
        confidence_level: f64,
    ) -> PyResult<Self> {
        if max_iter < 1 {
            return Err(PyValueError::new_err(format!(
                "max_iter must be >= 1, got {max_iter}"
            )));
        }
        if tol <= 0.0 {
            return Err(PyValueError::new_err(format!("tol must be > 0, got {tol}")));
        }
        if !(confidence_level > 0.0 && confidence_level < 1.0) {
            return Err(PyValueError::new_err(format!(
                "confidence_level must be in (0, 1), got {confidence_level}"
            )));
        }

        let parsed_distribution =
            parse_aft_distribution(distribution).map_err(PyValueError::new_err)?;
        let parsed_vcov = parse_vcov_type(vcov_type).map_err(PyValueError::new_err)?;
        let parsed_priors = extract_priors(priors)?;

        Ok(Self {
            distribution: parsed_distribution,
            priors: parsed_priors,
            vcov_type: parsed_vcov,
            with_intercept: add_intercept,
            max_iter,
            tol,
            compute_inference,
            confidence_level,
            fitted: None,
        })
    }

    /// Fit the model. `event` must have the same length as `time` and
    /// contain only 0/1 (bool, int, or float arrays/lists); any other value
    /// (e.g. R's 1/2 coding) raises `ValueError`.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        time: PyReadonlyArray1<'py, f64>,
        event: Vec<f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        crate::pymodels::errors::validate_xy("AFT", &x, &time)?;
        let n = time.shape()[0];
        if event.len() != n {
            return Err(PyValueError::new_err(format!(
                "event has {} entries but x/time have {n} rows",
                event.len()
            )));
        }

        let mut event_bool = Vec::with_capacity(n);
        for (i, &v) in event.iter().enumerate() {
            if v == 0.0 {
                event_bool.push(false);
            } else if v == 1.0 {
                event_bool.push(true);
            } else {
                return Err(PyValueError::new_err(format!(
                    "event[{i}] must be 0 or 1 (got {v}); R's 1/2 coding is not accepted"
                )));
            }
        }

        let x_mat = x.to_faer();
        let time_col = time.to_faer();

        let mut builder = AftRegressor::builder()
            .distribution(slf.distribution)
            .with_intercept(slf.with_intercept)
            .max_iterations(slf.max_iter)
            .tolerance(slf.tol)
            .compute_inference(slf.compute_inference)
            .confidence_level(slf.confidence_level)
            .vcov_type(slf.vcov_type);

        if !slf.priors.is_empty() {
            builder = builder.priors(slf.priors.clone());
        }

        let model = builder.build();
        let fitted = model
            .fit(&x_mat, &time_col, &event_bool)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        slf.fitted = Some(fitted);
        Ok(slf)
    }

    /// Linear predictor `intercept + x * beta` (the location of `log T`).
    fn predict_linear<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted.predict_linear(&x.to_faer()).into_numpy(py))
    }

    /// Predicted `p`-quantile of the survival time (`p=0.5` is the median).
    #[pyo3(signature = (x, p=0.5))]
    fn predict_quantile<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        p: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if !(0.0..=1.0).contains(&p) {
            return Err(PyValueError::new_err(format!(
                "p must be in [0, 1], got {p}"
            )));
        }
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted.predict_quantile(&x.to_faer(), p).into_numpy(py))
    }

    /// Predicted median survival time (`predict_quantile(x, 0.5)`).
    fn predict_median<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted.predict_median(&x.to_faer()).into_numpy(py))
    }

    /// Predicted survival probability `S(t | x) = P(T > t | x)`.
    fn predict_survival<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted.predict_survival(&x.to_faer(), t).into_numpy(py))
    }

    /// Predicted failure probability `F(t | x) = 1 - S(t | x)`.
    fn predict_cdf<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted.predict_cdf(&x.to_faer(), t).into_numpy(py))
    }

    fn is_fitted(&self) -> bool {
        self.fitted.is_some()
    }

    /// Feature coefficients on the log-time scale.
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted.coefficients().into_numpy(py))
    }

    /// Intercept, if fitted (`None` when `add_intercept=False`).
    #[getter]
    fn intercept(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted.intercept())
    }

    /// Scale `sigma` (`1.0`, not estimated, for `distribution="exponential"`).
    #[getter]
    fn scale(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted.scale())
    }

    /// Canonical lower-case distribution name (`"weibull"`, `"lognormal"`,
    /// `"loglogistic"`, or `"exponential"`).
    #[getter]
    fn distribution(&self) -> PyResult<String> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted.distribution().as_str().to_string())
    }

    /// Log-likelihood at the fitted parameters (on the time scale, as
    /// `survreg` reports it).
    #[getter]
    fn log_likelihood(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted.log_likelihood())
    }

    /// Log-likelihood of the intercept(+scale)-only model.
    #[getter]
    fn null_log_likelihood(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted.null_log_likelihood())
    }

    /// Akaike Information Criterion.
    #[getter]
    fn aic(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted.aic())
    }

    /// Bayesian Information Criterion.
    #[getter]
    fn bic(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted.bic())
    }

    /// Whether the Newton iterations converged (the last iterate is
    /// returned either way).
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted.converged())
    }

    /// Newton iterations of the final fit.
    #[getter]
    fn iterations(&self) -> PyResult<u32> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted.fit_result().iterations)
    }

    /// Number of rows used (rows with a non-finite value in `time`, `x`, or
    /// `event`, or a non-positive `time`, are dropped — unless the whole fit
    /// is rejected, e.g. every observation censored).
    #[getter]
    fn n_observations(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted.fit_result().n_observations)
    }

    /// Number of observed events among `n_observations`.
    #[getter]
    fn n_events(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted.fit_result().n_events)
    }

    /// Number of right-censored rows among `n_observations`.
    #[getter]
    fn n_censored(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted.fit_result().n_censored)
    }

    /// Standard errors of the feature coefficients (`None` unless
    /// `compute_inference=True`).
    #[getter]
    fn std_errors<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted
            .inference()
            .map(|inf| PyArray1::from_slice(py, &inf.std_errors)))
    }

    /// Wald z-statistics of the feature coefficients (`None` unless
    /// `compute_inference=True`).
    #[getter]
    fn z_values<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted
            .inference()
            .map(|inf| PyArray1::from_slice(py, &inf.z_values)))
    }

    /// Two-sided p-values of the feature coefficients (`None` unless
    /// `compute_inference=True`).
    #[getter]
    fn p_values<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted
            .inference()
            .map(|inf| PyArray1::from_slice(py, &inf.p_values)))
    }

    /// Lower confidence bounds of the feature coefficients (`None` unless
    /// `compute_inference=True`).
    #[getter]
    fn conf_interval_lower<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted
            .inference()
            .map(|inf| PyArray1::from_slice(py, &inf.ci_lower)))
    }

    /// Upper confidence bounds of the feature coefficients (`None` unless
    /// `compute_inference=True`).
    #[getter]
    fn conf_interval_upper<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted
            .inference()
            .map(|inf| PyArray1::from_slice(py, &inf.ci_upper)))
    }

    /// Standard error of the intercept (`None` without an intercept or
    /// unless `compute_inference=True`). Read directly from upstream's
    /// `AftInference` — never recomputed locally.
    #[getter]
    fn intercept_std_error(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted.inference().and_then(|inf| inf.intercept_std_error))
    }

    /// Wald z-statistic of the intercept (`None` without an intercept or
    /// unless `compute_inference=True`).
    #[getter]
    fn intercept_z_value(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted.inference().and_then(|inf| inf.intercept_z_value))
    }

    /// Two-sided p-value of the intercept (`None` without an intercept or
    /// unless `compute_inference=True`).
    #[getter]
    fn intercept_p_value(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted.inference().and_then(|inf| inf.intercept_p_value))
    }

    /// Confidence interval for the intercept as `(lower, upper)` (`None`
    /// without an intercept or unless `compute_inference=True`).
    #[getter]
    fn intercept_conf_interval(&self) -> PyResult<Option<(f64, f64)>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted.inference().and_then(|inf| {
            match (inf.intercept_ci_lower, inf.intercept_ci_upper) {
                (Some(l), Some(u)) => Some((l, u)),
                _ => None,
            }
        }))
    }

    /// Standard error of `log(scale)` (`None` when the scale is fixed, i.e.
    /// `distribution="exponential"`, or unless `compute_inference=True`).
    #[getter]
    fn log_scale_std_error(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted.inference().and_then(|inf| inf.log_scale_std_error))
    }

    /// Full parameter covariance at the mode (`None` unless
    /// `compute_inference=True`). Parameter order: the intercept first when
    /// fitted, then the feature coefficients, then `log(scale)` when the
    /// scale is estimated — so for a fit with an intercept and `p` features
    /// the matrix is `(1 + p + 1)` square.
    #[getter]
    fn vcov<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted
            .inference()
            .and_then(|inf| inf.vcov.as_ref())
            .map(|v| v.into_numpy(py)))
    }

    /// The penalised observed information at the mode (`None` unless
    /// `compute_inference=True`); same parameter order as `vcov`.
    #[getter]
    fn information<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        Ok(fitted
            .inference()
            .and_then(|inf| inf.information.as_ref())
            .map(|m| m.into_numpy(py)))
    }

    /// Informative repr: class name plus key state; never panics if unfitted.
    /// Model type, error family (R naming; None for estimators without a
    /// likelihood) and link of the fit (anofox-regression ModelInfo).
    #[getter]
    fn model_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        use anofox_regression::HasModelInfo;
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("AFT"))?;
        crate::pymodels::model_info::model_info_dict(py, fitted.model_info())
    }

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
