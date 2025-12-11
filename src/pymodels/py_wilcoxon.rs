//! PyO3 wrapper for Wilcoxon signed-rank test.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use anofox_statistics::{wilcoxon_signed_rank, Alternative, WilcoxonResult};

/// Wilcoxon signed-rank test.
///
/// Tests the null hypothesis that two related paired samples come from
/// the same distribution. This is a non-parametric alternative to the
/// paired t-test.
///
/// Parameters
/// ----------
/// alternative : str, default "two-sided"
///     The alternative hypothesis: "two-sided", "less", or "greater".
/// continuity_correction : bool, default True
///     Whether to apply continuity correction for the normal approximation.
/// exact : bool, default False
///     Whether to use exact method (only for small samples).
/// conf_level : float, optional
///     Confidence level for the pseudomedian confidence interval.
/// mu : float, optional
///     Hypothesized location under null hypothesis.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from polars_statistics import WilcoxonSignedRank
/// >>>
/// >>> before = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
/// >>> after = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
/// >>>
/// >>> test = WilcoxonSignedRank()
/// >>> test.fit(before, after)
/// >>> print(test.statistic, test.p_value)
#[pyclass(name = "WilcoxonSignedRank")]
pub struct PyWilcoxonSignedRank {
    alternative: Alternative,
    continuity_correction: bool,
    exact: bool,
    conf_level: Option<f64>,
    mu: Option<f64>,
    fitted: Option<WilcoxonResult>,
}

#[pymethods]
impl PyWilcoxonSignedRank {
    /// Create a new Wilcoxon signed-rank test.
    #[new]
    #[pyo3(signature = (alternative="two-sided", continuity_correction=true, exact=false, conf_level=None, mu=None))]
    fn new(
        alternative: &str,
        continuity_correction: bool,
        exact: bool,
        conf_level: Option<f64>,
        mu: Option<f64>,
    ) -> PyResult<Self> {
        let alt = match alternative.to_lowercase().as_str() {
            "two-sided" | "two_sided" => Alternative::TwoSided,
            "less" => Alternative::Less,
            "greater" => Alternative::Greater,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "alternative must be 'two-sided', 'less', or 'greater'",
                ))
            }
        };

        if let Some(cl) = conf_level {
            if !(0.0..=1.0).contains(&cl) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "conf_level must be between 0 and 1",
                ));
            }
        }

        Ok(Self {
            alternative: alt,
            continuity_correction,
            exact,
            conf_level,
            mu,
            fitted: None,
        })
    }

    /// Perform the Wilcoxon signed-rank test on paired samples.
    ///
    /// Parameters
    /// ----------
    /// x : array-like of shape (n_samples,)
    ///     First sample (e.g., before treatment).
    /// y : array-like of shape (n_samples,)
    ///     Second sample (e.g., after treatment).
    ///
    /// Returns
    /// -------
    /// self
    ///     Fitted test with results.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray1<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_vec: Vec<f64> = x.as_slice()?.to_vec();
        let y_vec: Vec<f64> = y.as_slice()?.to_vec();

        if x_vec.len() != y_vec.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Paired samples must have the same length",
            ));
        }

        let result = wilcoxon_signed_rank(
            &x_vec,
            &y_vec,
            slf.alternative,
            slf.continuity_correction,
            slf.exact,
            slf.conf_level,
            slf.mu,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        slf.fitted = Some(result);

        Ok(slf)
    }

    /// Check if the test has been performed.
    fn is_fitted(&self) -> bool {
        self.fitted.is_some()
    }

    /// Get the test statistic (W statistic).
    #[getter]
    fn statistic(&self) -> PyResult<f64> {
        self.fitted
            .as_ref()
            .map(|r| r.statistic)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Test not fitted"))
    }

    /// Get the p-value.
    #[getter]
    fn p_value(&self) -> PyResult<f64> {
        self.fitted
            .as_ref()
            .map(|r| r.p_value)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Test not fitted"))
    }

    /// Get the pseudomedian estimate.
    #[getter]
    fn estimate(&self) -> PyResult<Option<f64>> {
        self.fitted
            .as_ref()
            .map(|r| r.estimate)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Test not fitted"))
    }

    /// Get the null hypothesis value.
    #[getter]
    fn null_value(&self) -> PyResult<f64> {
        self.fitted
            .as_ref()
            .map(|r| r.null_value)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Test not fitted"))
    }

    /// Get the confidence interval as a numpy array [lower, upper].
    #[getter]
    fn conf_int<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let result = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Test not fitted"))?;

        match &result.conf_int {
            Some(ci) => Ok(Some(PyArray1::from_vec(py, vec![ci.lower, ci.upper]))),
            None => Ok(None),
        }
    }

    /// Get a formatted summary of the test results.
    fn summary(&self) -> PyResult<String> {
        let result = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Test not fitted"))?;

        let alt_str = match self.alternative {
            Alternative::TwoSided => "two-sided",
            Alternative::Less => "less",
            Alternative::Greater => "greater",
        };

        let significance = if result.p_value < 0.05 {
            "Reject H0 at alpha=0.05"
        } else {
            "Fail to reject H0 at alpha=0.05"
        };

        let estimate_str = match result.estimate {
            Some(est) => format!("{:.4}", est),
            None => "N/A".to_string(),
        };

        let conf_int_str = match &result.conf_int {
            Some(ci) => format!("[{:.4}, {:.4}]", ci.lower, ci.upper),
            None => "N/A".to_string(),
        };

        Ok(format!(
            "Wilcoxon Signed-Rank Test\n\
             =========================\n\n\
             W statistic:     {:>12.4}\n\
             P-value:         {:>12.4e}\n\
             Estimate:        {:>12}\n\
             Confidence int:  {:>12}\n\
             Alternative:     {:>12}\n\
             Null value:      {:>12.4}\n\
             Continuity corr: {:>12}\n\
             Exact:           {:>12}\n\n\
             H0: Paired differences have median zero\n\
             Result: {}",
            result.statistic,
            result.p_value,
            estimate_str,
            conf_int_str,
            alt_str,
            result.null_value,
            self.continuity_correction,
            self.exact,
            significance
        ))
    }
}
