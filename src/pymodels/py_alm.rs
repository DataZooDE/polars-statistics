//! PyO3 wrapper for Augmented Linear Model (ALM).

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::solvers::{
    AlmDistribution, AlmLoss, AlmRegressor, FittedAlm, FittedRegressor, LinkFunction, Regressor,
};

use crate::utils::{IntoNumpy, ToFaer};

/// Convert string to AlmDistribution enum.
pub fn parse_distribution(s: &str) -> PyResult<AlmDistribution> {
    match s.to_lowercase().as_str() {
        "normal" | "gaussian" => Ok(AlmDistribution::Normal),
        "laplace" => Ok(AlmDistribution::Laplace),
        "student_t" | "studentt" | "t" => Ok(AlmDistribution::StudentT),
        "logistic" => Ok(AlmDistribution::Logistic),
        "asymmetric_laplace" | "asymmetriclaplace" => Ok(AlmDistribution::AsymmetricLaplace),
        "generalised_normal" | "generalisednormal" | "generalized_normal" => Ok(AlmDistribution::GeneralisedNormal),
        "s" => Ok(AlmDistribution::S),
        "lognormal" | "log_normal" => Ok(AlmDistribution::LogNormal),
        "loglaplace" | "log_laplace" => Ok(AlmDistribution::LogLaplace),
        "logs" | "log_s" => Ok(AlmDistribution::LogS),
        "loggeneralisednormal" | "log_generalised_normal" => Ok(AlmDistribution::LogGeneralisedNormal),
        "gamma" => Ok(AlmDistribution::Gamma),
        "inverse_gaussian" | "inversegaussian" => Ok(AlmDistribution::InverseGaussian),
        "exponential" => Ok(AlmDistribution::Exponential),
        "folded_normal" | "foldednormal" => Ok(AlmDistribution::FoldedNormal),
        "rectified_normal" | "rectifiednormal" => Ok(AlmDistribution::RectifiedNormal),
        "beta" => Ok(AlmDistribution::Beta),
        "logit_normal" | "logitnormal" => Ok(AlmDistribution::LogitNormal),
        "poisson" => Ok(AlmDistribution::Poisson),
        "negative_binomial" | "negativebinomial" | "negbin" => Ok(AlmDistribution::NegativeBinomial),
        "binomial" => Ok(AlmDistribution::Binomial),
        "geometric" => Ok(AlmDistribution::Geometric),
        "cumulative_logistic" | "cumulativelogistic" | "ordinal_logistic" => Ok(AlmDistribution::CumulativeLogistic),
        "cumulative_normal" | "cumulativenormal" | "ordinal_probit" => Ok(AlmDistribution::CumulativeNormal),
        "boxcox_normal" | "boxcoxnormal" | "box_cox" => Ok(AlmDistribution::BoxCoxNormal),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown distribution: '{}'. Available: normal, laplace, student_t, logistic, asymmetric_laplace, generalised_normal, s, lognormal, loglaplace, logs, loggeneralisednormal, gamma, inverse_gaussian, exponential, folded_normal, rectified_normal, beta, logit_normal, poisson, negative_binomial, binomial, geometric, cumulative_logistic, cumulative_normal, boxcox_normal", s)
        )),
    }
}

/// Convert string to LinkFunction enum.
fn parse_link(s: &str) -> PyResult<LinkFunction> {
    match s.to_lowercase().as_str() {
        "identity" => Ok(LinkFunction::Identity),
        "log" => Ok(LinkFunction::Log),
        "logit" => Ok(LinkFunction::Logit),
        "probit" => Ok(LinkFunction::Probit),
        "inverse" => Ok(LinkFunction::Inverse),
        "sqrt" => Ok(LinkFunction::Sqrt),
        "cloglog" | "complementary_log_log" => Ok(LinkFunction::Cloglog),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown link function: '{}'. Available: identity, log, logit, probit, inverse, sqrt, cloglog", s)
        )),
    }
}

/// Convert string to AlmLoss enum.
fn parse_loss(s: &str, role_trim: Option<f64>) -> PyResult<AlmLoss> {
    match s.to_lowercase().as_str() {
        "likelihood" | "mle" => Ok(AlmLoss::Likelihood),
        "mse" => Ok(AlmLoss::MSE),
        "mae" => Ok(AlmLoss::MAE),
        "ham" => Ok(AlmLoss::HAM),
        "role" => Ok(AlmLoss::ROLE {
            trim: role_trim.unwrap_or(0.05),
        }),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown loss function: '{}'. Available: likelihood, mse, mae, ham, role",
            s
        ))),
    }
}

/// Augmented Linear Model (ALM).
///
/// A flexible regression model supporting 24+ distributions with configurable link functions.
/// Based on the greybox R package methodology.
///
/// Parameters
/// ----------
/// distribution : str, default "normal"
///     The distribution family. Options include:
///     - Continuous: "normal", "laplace", "student_t", "logistic", "asymmetric_laplace",
///       "generalised_normal", "s"
///     - Positive: "lognormal", "loglaplace", "logs", "loggeneralisednormal", "gamma",
///       "inverse_gaussian", "exponential", "folded_normal", "rectified_normal"
///     - Bounded (0,1): "beta", "logit_normal"
///     - Count: "poisson", "negative_binomial", "binomial", "geometric"
///     - Ordinal: "cumulative_logistic", "cumulative_normal"
///     - Transformed: "boxcox_normal"
/// link : str, optional
///     Link function. If None, uses canonical link for the distribution.
///     Options: "identity", "log", "logit", "probit", "inverse", "sqrt", "cloglog"
/// loss : str, default "likelihood"
///     Loss function for convergence criterion.
///     Options: "likelihood" (MLE), "mse", "mae", "ham" (Half Absolute Moment), "role" (ROLE)
/// role_trim : float, optional
///     Fraction of observations to trim when using ROLE loss (0.0 to 0.5, default 0.05).
/// with_intercept : bool, default True
///     Whether to include an intercept term.
/// compute_inference : bool, default True
///     Whether to compute statistical inference.
/// confidence_level : float, default 0.95
///     Confidence level for confidence intervals.
/// max_iter : int, default 100
///     Maximum number of iterations.
/// tol : float, default 1e-8
///     Tolerance for convergence.
/// extra_parameter : float, optional
///     Extra parameter for certain distributions (e.g., degrees of freedom for Student-t).
#[pyclass(name = "ALM")]
pub struct PyALM {
    distribution: String,
    link: Option<String>,
    loss: String,
    role_trim: Option<f64>,
    with_intercept: bool,
    compute_inference: bool,
    confidence_level: f64,
    max_iter: usize,
    tol: f64,
    extra_parameter: Option<f64>,
    fitted: Option<FittedAlm>,
}

#[pymethods]
impl PyALM {
    #[new]
    #[pyo3(signature = (distribution="normal", link=None, loss="likelihood", role_trim=None, with_intercept=true, compute_inference=true, confidence_level=0.95, max_iter=100, tol=1e-8, extra_parameter=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        distribution: &str,
        link: Option<&str>,
        loss: &str,
        role_trim: Option<f64>,
        with_intercept: bool,
        compute_inference: bool,
        confidence_level: f64,
        max_iter: usize,
        tol: f64,
        extra_parameter: Option<f64>,
    ) -> Self {
        Self {
            distribution: distribution.to_string(),
            link: link.map(|s| s.to_string()),
            loss: loss.to_string(),
            role_trim,
            with_intercept,
            compute_inference,
            confidence_level,
            max_iter,
            tol,
            extra_parameter,
            fitted: None,
        }
    }

    /// Create a Normal (Gaussian) ALM.
    #[staticmethod]
    #[pyo3(signature = (with_intercept=true, compute_inference=true))]
    fn normal(with_intercept: bool, compute_inference: bool) -> Self {
        Self::new(
            "normal",
            None,
            "likelihood",
            None,
            with_intercept,
            compute_inference,
            0.95,
            100,
            1e-8,
            None,
        )
    }

    /// Create a Laplace ALM.
    #[staticmethod]
    #[pyo3(signature = (with_intercept=true, compute_inference=true))]
    fn laplace(with_intercept: bool, compute_inference: bool) -> Self {
        Self::new(
            "laplace",
            None,
            "likelihood",
            None,
            with_intercept,
            compute_inference,
            0.95,
            100,
            1e-8,
            None,
        )
    }

    /// Create a Student-t ALM.
    #[staticmethod]
    #[pyo3(signature = (df=5.0, with_intercept=true, compute_inference=true))]
    fn student_t(df: f64, with_intercept: bool, compute_inference: bool) -> Self {
        Self::new(
            "student_t",
            None,
            "likelihood",
            None,
            with_intercept,
            compute_inference,
            0.95,
            100,
            1e-8,
            Some(df),
        )
    }

    /// Create a Gamma ALM (for positive continuous data).
    #[staticmethod]
    #[pyo3(signature = (with_intercept=true, compute_inference=true))]
    fn gamma(with_intercept: bool, compute_inference: bool) -> Self {
        Self::new(
            "gamma",
            Some("log"),
            "likelihood",
            None,
            with_intercept,
            compute_inference,
            0.95,
            100,
            1e-8,
            None,
        )
    }

    /// Create a Beta ALM (for data in (0,1)).
    #[staticmethod]
    #[pyo3(signature = (with_intercept=true, compute_inference=true))]
    fn beta(with_intercept: bool, compute_inference: bool) -> Self {
        Self::new(
            "beta",
            Some("logit"),
            "likelihood",
            None,
            with_intercept,
            compute_inference,
            0.95,
            100,
            1e-8,
            None,
        )
    }

    /// Create a Poisson ALM (for count data).
    #[staticmethod]
    #[pyo3(signature = (with_intercept=true, compute_inference=true))]
    fn poisson(with_intercept: bool, compute_inference: bool) -> Self {
        Self::new(
            "poisson",
            Some("log"),
            "likelihood",
            None,
            with_intercept,
            compute_inference,
            0.95,
            100,
            1e-8,
            None,
        )
    }

    /// Fit the ALM model.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data.
    /// y : array-like of shape (n_samples,)
    ///     Target values.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_mat = x.to_faer();
        let y_col = y.to_faer();

        let dist = parse_distribution(&slf.distribution)?;
        let loss = parse_loss(&slf.loss, slf.role_trim)?;

        let mut builder = AlmRegressor::builder()
            .distribution(dist)
            .loss(loss)
            .with_intercept(slf.with_intercept)
            .compute_inference(slf.compute_inference)
            .confidence_level(slf.confidence_level)
            .max_iterations(slf.max_iter)
            .tolerance(slf.tol);

        if let Some(ref link_str) = slf.link {
            let link = parse_link(link_str)?;
            builder = builder.link(link);
        }

        if let Some(extra) = slf.extra_parameter {
            builder = builder.extra_parameter(extra);
        }

        let model = builder.build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        slf.fitted = Some(fitted);
        Ok(slf)
    }

    /// Predict target values.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Samples to predict.
    ///
    /// Returns
    /// -------
    /// array of shape (n_samples,)
    ///     Predicted values.
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

    #[getter]
    fn scale(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok(fitted.scale())
    }

    #[getter]
    fn distribution(&self) -> &str {
        &self.distribution
    }

    #[getter]
    fn link_function(&self) -> Option<&str> {
        self.link.as_deref()
    }

    #[getter]
    fn residuals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok((&fitted.result().residuals).into_numpy(py))
    }

    #[getter]
    fn fitted_values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;

        Ok((&fitted.result().fitted_values).into_numpy(py))
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
