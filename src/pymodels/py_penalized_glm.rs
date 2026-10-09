//! PyO3 wrapper for the family-generic penalized GLM engine (`PenalizedGLM`).

use faer::Mat;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use anofox_regression::solvers::penalized_glm::{
    FittedPenalizedGlm, PenalizedGlmFamily, PenalizedGlmRegressor, PriorSpec, VcovType,
};

use crate::expressions::{
    parse_penalized_glm_family, parse_vcov_type, penalized_glm_covariance_scale,
    penalized_glm_family_names, prior_spec_from_parts,
};
use crate::utils::{IntoNumpy, ToFaer};

/// Parse a single Python prior entry into a [`PriorSpec`].
///
/// Accepts `None` (flat), a bare string (treated as `kind`, `loc=0.0`,
/// `scale=+inf`), a dict `{"kind", "loc", "scale"}` (`loc`/`scale` optional),
/// or a `(kind,)` / `(kind, loc)` / `(kind, loc, scale)` tuple/list.
fn extract_one_prior(item: &Bound<'_, PyAny>) -> PyResult<PriorSpec> {
    if item.is_none() {
        return prior_spec_from_parts("flat", 0.0, f64::INFINITY).map_err(PyValueError::new_err);
    }
    if let Ok(kind) = item.extract::<String>() {
        return prior_spec_from_parts(&kind, 0.0, f64::INFINITY).map_err(PyValueError::new_err);
    }
    if let Ok(dict) = item.downcast::<PyDict>() {
        let kind: String = dict
            .get_item("kind")?
            .ok_or_else(|| PyValueError::new_err("prior dict entries require a \"kind\" key"))?
            .extract()?;
        let loc: f64 = match dict.get_item("loc")? {
            Some(v) => v.extract()?,
            None => 0.0,
        };
        let scale: f64 = match dict.get_item("scale")? {
            Some(v) => v.extract()?,
            None => f64::INFINITY,
        };
        return prior_spec_from_parts(&kind, loc, scale).map_err(PyValueError::new_err);
    }
    let parts: Vec<Bound<'_, PyAny>> = item.try_iter()?.collect::<PyResult<_>>()?;
    if parts.is_empty() {
        return Err(PyValueError::new_err(
            "prior entry must be None, \"flat\", a (kind, loc, scale) tuple, or a dict",
        ));
    }
    let kind: String = parts[0].extract()?;
    let loc: f64 = if parts.len() > 1 {
        parts[1].extract()?
    } else {
        0.0
    };
    let scale: f64 = if parts.len() > 2 {
        parts[2].extract()?
    } else {
        f64::INFINITY
    };
    prior_spec_from_parts(&kind, loc, scale).map_err(PyValueError::new_err)
}

/// Extract the `priors` constructor argument into `Vec<PriorSpec>`. Shared
/// with the AFT PyModel (Plan 05).
pub(crate) fn extract_priors(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<PriorSpec>> {
    let Some(obj) = obj else {
        return Ok(Vec::new());
    };
    if obj.is_none() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        out.push(extract_one_prior(&item?)?);
    }
    Ok(out)
}

/// Family-generic penalized GLM (IRLS with explicit priors), for any of
/// Poisson, Binomial, Logistic, Negative Binomial, Gamma or Tweedie.
///
/// Without priors and with ``lambda_ = 0`` this is a plain maximum-likelihood
/// GLM matching R's ``glm`` / ``MASS::glm.nb``; with Gaussian priors it
/// returns the posterior mode (as ``arm::bayesglm`` with fixed prior scales).
///
/// Parameters
/// ----------
/// family : str, default "poisson"
///     One of ``"poisson"``, ``"binomial"``, ``"logistic"``,
///     ``"negative_binomial"``, ``"gamma"``, ``"tweedie"``.
/// link : str, optional
///     Link function; defaults to the family's canonical link:
///
///     - ``"poisson"``: ``"log"`` (default), ``"identity"``, ``"sqrt"``.
///     - ``"binomial"``: ``"logit"`` (default), ``"probit"``, ``"cloglog"``.
///     - ``"logistic"``, ``"negative_binomial"``, ``"gamma"``, ``"tweedie"``:
///       fixed (``"logit"`` / ``"log"``).
/// theta : float, optional
///     Fixed Negative Binomial dispersion; ``None`` estimates it by maximum
///     likelihood (``MASS::glm.nb``). Only valid for
///     ``family="negative_binomial"``.
/// power : float, optional
///     Tweedie variance power in ``[1, 2]``; default 1.5. Only valid for
///     ``family="tweedie"``.
/// threshold : float, default 0.5
///     Classification threshold for ``family="logistic"``, in ``(0, 1)``.
/// lambda_ : float, default 0.0
///     Uniform L2 ridge penalty on non-intercept coefficients, composed
///     additively with ``priors``.
/// estimate_dispersion : bool, default False
///     Poisson only: scale the coefficient covariance by the Pearson
///     chi2/df dispersion estimate (R's ``quasipoisson``). Every other
///     family fixes its own dispersion rule and rejects this option.
/// priors : sequence, optional
///     Per-coefficient priors, positionally aligned with the feature
///     columns. Each entry is ``None``/``"flat"`` (flat), a
///     ``(kind, loc, scale)`` tuple/list (or ``(kind,)``), or a dict
///     ``{"kind", "loc", "scale"}`` — ``kind`` is one of ``"flat"``,
///     ``"normal"``, ``"laplace"``. An optional leading entry applies to
///     the intercept (``p`` or ``p + 1`` entries).
/// vcov_type : str, default "laplace"
///     Covariance at the mode: ``"laplace"`` (``(X'WX+P)^-1``, the Laplace
///     approximation and the correct observed information for a MAP
///     estimate), ``"sandwich"`` (targets the frequentist sampling
///     variance of the penalized estimator), or ``"naive"``
///     (``(X'WX)^-1``, ignores the penalty). All three coincide for an
///     unpenalized fit.
/// add_intercept : bool, default True
///     Whether to include an intercept term.
/// max_iter : int, default 100
///     Maximum IRLS iterations.
/// tol : float, default 1e-8
///     Convergence tolerance.
/// compute_inference : bool, default True
///     Whether to compute standard errors / z / p values / confidence
///     intervals / the covariance.
/// confidence_level : float, default 0.95
///     Confidence level for confidence intervals (used only when
///     ``compute_inference=True``).
///
/// Notes
/// -----
/// Validated against R ``glm`` / ``MASS::glm.nb`` / ``negative.binomial(theta)``
/// on the `anofox-regression` 0.5.22 R reference fixture.
#[pyclass(name = "PenalizedGLM")]
pub struct PyPenalizedGLM {
    family: PenalizedGlmFamily,
    lambda_: f64,
    estimate_dispersion: bool,
    priors: Vec<PriorSpec>,
    vcov_type: VcovType,
    with_intercept: bool,
    max_iter: u32,
    tol: f64,
    compute_inference: bool,
    confidence_level: f64,
    fitted: Option<FittedPenalizedGlm>,
}

#[pymethods]
impl PyPenalizedGLM {
    #[new]
    #[pyo3(signature = (
        family="poisson",
        link=None,
        theta=None,
        power=None,
        threshold=0.5,
        lambda_=0.0,
        estimate_dispersion=false,
        priors=None,
        vcov_type="laplace",
        add_intercept=true,
        max_iter=100,
        tol=1e-8,
        compute_inference=true,
        confidence_level=0.95
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        family: &str,
        link: Option<&str>,
        theta: Option<f64>,
        power: Option<f64>,
        threshold: f64,
        lambda_: f64,
        estimate_dispersion: bool,
        priors: Option<&Bound<'_, PyAny>>,
        vcov_type: &str,
        add_intercept: bool,
        max_iter: u32,
        tol: f64,
        compute_inference: bool,
        confidence_level: f64,
    ) -> PyResult<Self> {
        if lambda_ < 0.0 {
            return Err(PyValueError::new_err(format!(
                "lambda_ must be >= 0, got {lambda_}"
            )));
        }
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

        let parsed_family =
            parse_penalized_glm_family(family, link, theta, power, threshold, estimate_dispersion)
                .map_err(PyValueError::new_err)?;
        let parsed_vcov = parse_vcov_type(vcov_type).map_err(PyValueError::new_err)?;
        let parsed_priors = extract_priors(priors)?;

        Ok(Self {
            family: parsed_family,
            lambda_,
            estimate_dispersion,
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

    /// Fit the model. `offset` (if given) must have the same length as `y`.
    #[pyo3(signature = (x, y, offset=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        offset: Option<PyReadonlyArray1<'py, f64>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        crate::pymodels::errors::validate_xy("PenalizedGLM", &x, &y)?;
        let x_mat = x.to_faer();
        let y_col = y.to_faer();

        let offset_col = match offset {
            Some(off) => {
                let n = y.shape()[0];
                let off_n = off.shape()[0];
                if off_n != n {
                    return Err(PyValueError::new_err(format!(
                        "offset has {off_n} entries but x/y have {n} rows"
                    )));
                }
                Some(off.to_faer())
            }
            None => None,
        };

        let mut builder = PenalizedGlmRegressor::builder()
            .family(slf.family)
            .with_intercept(slf.with_intercept)
            .max_iterations(slf.max_iter)
            .tolerance(slf.tol)
            .compute_inference(slf.compute_inference)
            .confidence_level(slf.confidence_level)
            .lambda(slf.lambda_)
            .estimate_dispersion(slf.estimate_dispersion)
            .vcov_type(slf.vcov_type);

        if !slf.priors.is_empty() {
            builder = builder.priors(slf.priors.clone());
        }
        if let Some(off) = offset_col {
            builder = builder.offset(off);
        }

        let model = builder.build();
        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        slf.fitted = Some(fitted);
        Ok(slf)
    }

    /// Predicted means on the response scale (`offset` added to the linear
    /// predictor when given).
    #[pyo3(signature = (x, offset=None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        offset: Option<PyReadonlyArray1<'py, f64>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        let x_mat = x.to_faer();
        let preds = match offset {
            Some(off) => fitted.predict_with_offset(&x_mat, &off.to_faer()),
            None => fitted.predict(&x_mat),
        };
        Ok(preds.into_numpy(py))
    }

    /// Linear predictor `intercept + x * beta` (dropped columns contribute 0).
    fn predict_linear<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        let x_mat = x.to_faer();
        Ok(fitted.predict_linear(&x_mat).into_numpy(py))
    }

    fn is_fitted(&self) -> bool {
        self.fitted.is_some()
    }

    /// Feature coefficients (`NaN` for columns dropped as constant / aliased).
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(fitted.coefficients().into_numpy(py))
    }

    /// Intercept, if fitted (`None` when `add_intercept=False`).
    #[getter]
    fn intercept(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(fitted.intercept())
    }

    /// Lower-case family name (`"poisson"`, `"binomial"`, `"logistic"`,
    /// `"negative_binomial"`, `"gamma"`, or `"tweedie"`).
    #[getter]
    fn family(&self) -> PyResult<String> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(penalized_glm_family_names(&fitted.family()).0.to_string())
    }

    /// Lower-case link name.
    #[getter]
    fn link(&self) -> PyResult<String> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(penalized_glm_family_names(&fitted.family()).1.to_string())
    }

    /// Residual deviance.
    #[getter]
    fn deviance(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(fitted.deviance())
    }

    /// Null (intercept-only) deviance.
    #[getter]
    fn null_deviance(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(fitted.null_deviance())
    }

    /// Deviance pseudo-R-squared, `1 - deviance / null_deviance`.
    #[getter]
    fn pseudo_r_squared(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(fitted.pseudo_r_squared())
    }

    /// Log-likelihood at the fitted means.
    #[getter]
    fn log_likelihood(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(fitted.log_likelihood())
    }

    /// Akaike Information Criterion.
    #[getter]
    fn aic(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(fitted.aic())
    }

    /// Bayesian Information Criterion.
    #[getter]
    fn bic(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(fitted.bic())
    }

    /// Dispersion: 1.0 (Poisson unless `estimate_dispersion=True`, Binomial,
    /// Logistic), the Pearson estimate (Gamma, Tweedie, quasi-Poisson), or
    /// `theta` (Negative Binomial).
    #[getter]
    fn dispersion(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(fitted.dispersion())
    }

    /// Negative Binomial `theta` (fixed or estimated); `None` for every other
    /// family.
    #[getter]
    fn theta(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(fitted.theta())
    }

    /// IRLS iterations of the final fit.
    #[getter]
    fn iterations(&self) -> PyResult<u32> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(fitted.iterations())
    }

    /// Whether IRLS converged (the last iterate is returned either way).
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(fitted.converged())
    }

    /// Number of rows used (rows with a non-finite value in `y`, `x`, or the
    /// offset are dropped).
    #[getter]
    fn n_observations(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(fitted.n_observations())
    }

    /// Training accuracy (`family="logistic"` only); `None` otherwise.
    #[getter]
    fn accuracy(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(fitted.accuracy())
    }

    /// Standard errors of the feature coefficients (`None` unless
    /// `compute_inference=True`).
    #[getter]
    fn std_errors<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
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
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(fitted
            .inference()
            .map(|inf| PyArray1::from_slice(py, &inf.ci_upper)))
    }

    /// Standard error of the intercept (`None` without an intercept or unless
    /// `compute_inference=True`). Read directly from upstream's
    /// `GlmInferenceResult` — never recomputed locally.
    #[getter]
    fn intercept_std_error(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(fitted.inference().and_then(|inf| inf.intercept_std_error))
    }

    /// Wald z-statistic of the intercept (`None` without an intercept or
    /// unless `compute_inference=True`).
    #[getter]
    fn intercept_z_value(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(fitted.inference().and_then(|inf| inf.intercept_z_value))
    }

    /// Two-sided p-value of the intercept (`None` without an intercept or
    /// unless `compute_inference=True`).
    #[getter]
    fn intercept_p_value(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(fitted.inference().and_then(|inf| inf.intercept_p_value))
    }

    /// Confidence interval for the intercept as `(lower, upper)` (`None`
    /// without an intercept or unless `compute_inference=True`).
    #[getter]
    fn intercept_conf_interval(&self) -> PyResult<Option<(f64, f64)>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(fitted.inference().and_then(|inf| {
            match (inf.intercept_ci_lower, inf.intercept_ci_upper) {
                (Some(l), Some(u)) => Some((l, u)),
                _ => None,
            }
        }))
    }

    /// Full parameter covariance, scaled so its diagonal square roots equal
    /// the reported standard errors (like R's `vcov(glm)`); fitted order,
    /// see `matrix_parameters`. `None` unless `compute_inference=True`.
    #[getter]
    fn vcov<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        let scale = penalized_glm_covariance_scale(fitted);
        Ok(fitted
            .inference()
            .and_then(|inf| inf.vcov.as_ref())
            .map(|v| {
                let scaled = Mat::from_fn(v.nrows(), v.ncols(), |i, j| scale * v[(i, j)]);
                (&scaled).into_numpy(py)
            }))
    }

    /// The penalized observed information at the mode (upstream, unscaled);
    /// fitted order, see `matrix_parameters`. `None` unless
    /// `compute_inference=True`.
    #[getter]
    fn information<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(fitted
            .inference()
            .and_then(|inf| inf.information.as_ref())
            .map(|m| m.into_numpy(py)))
    }

    /// What each row/column of `vcov` and `information` refers to: `None`
    /// for the intercept, `Some(j)` for feature `j` of the original design
    /// (dropped columns are absent). `None` unless `compute_inference=True`.
    #[getter]
    fn matrix_parameters(&self) -> PyResult<Option<Vec<Option<usize>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(fitted.inference().map(|inf| inf.matrix_parameters.clone()))
    }

    /// Whether (quasi-)separation was detected (`family="binomial"` /
    /// `"logistic"` only; always `False` for other families).
    #[getter]
    fn has_separation(&self) -> PyResult<bool> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(fitted.separation().has_separation)
    }

    /// Indices (0-based, excludes the intercept) of predictors involved in
    /// separation.
    #[getter]
    fn separated_predictors(&self) -> PyResult<Vec<usize>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(fitted.separation().separated_predictors.clone())
    }

    /// Human-readable separation warning, if any.
    #[getter]
    fn separation_warning(&self) -> PyResult<Option<String>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(fitted.separation().warning_message.clone())
    }

    /// Fitted means over the rows used (see `valid_rows`).
    #[getter]
    fn fitted_values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(PyArray1::from_slice(py, fitted.fitted_values()))
    }

    /// Indices into the original rows that were used (rows with a
    /// non-finite value in `y`, `x`, or the offset are dropped).
    #[getter]
    fn valid_rows(&self) -> PyResult<Vec<usize>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("PenalizedGLM"))?;
        Ok(fitted.result().valid_rows.clone())
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
