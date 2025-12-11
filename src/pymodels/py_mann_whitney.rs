//! PyO3 wrapper for Mann-Whitney U test.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use anofox_statistics::{mann_whitney_u, Alternative, MannWhitneyResult};

/// Mann-Whitney U test (Wilcoxon rank-sum test).
///
/// Tests whether the distribution of values in one sample is stochastically
/// greater than in another sample. This is a non-parametric test that does
/// not assume normality.
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
///     Confidence level for the Hodges-Lehmann estimator confidence interval.
/// mu : float, optional
///     Hypothesized location shift under null hypothesis.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from polars_statistics import MannWhitneyU
/// >>>
/// >>> x = np.random.randn(50)
/// >>> y = np.random.randn(50) + 0.5
/// >>>
/// >>> test = MannWhitneyU()
/// >>> test.fit(x, y)
/// >>> print(test.statistic, test.p_value)
#[pyclass(name = "MannWhitneyU")]
pub struct PyMannWhitneyU {
    alternative: Alternative,
    continuity_correction: bool,
    exact: bool,
    conf_level: Option<f64>,
    mu: Option<f64>,
    fitted: Option<MannWhitneyResult>,
}

#[pymethods]
impl PyMannWhitneyU {
    /// Create a new Mann-Whitney U test.
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

    /// Perform the Mann-Whitney U test on two samples.
    ///
    /// Parameters
    /// ----------
    /// x : array-like of shape (n_samples,)
    ///     First sample.
    /// y : array-like of shape (n_samples,)
    ///     Second sample.
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

        let result = mann_whitney_u(
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

    /// Get the test statistic (U statistic).
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

    /// Get the Hodges-Lehmann estimator (location shift estimate).
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
            "Mann-Whitney U Test\n\
             ===================\n\n\
             U statistic:     {:>12.4}\n\
             P-value:         {:>12.4e}\n\
             Estimate:        {:>12}\n\
             Confidence int:  {:>12}\n\
             Alternative:     {:>12}\n\
             Null value:      {:>12.4}\n\
             Continuity corr: {:>12}\n\
             Exact:           {:>12}\n\n\
             H0: The distributions are equal\n\
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
