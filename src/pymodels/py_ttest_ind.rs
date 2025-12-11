//! PyO3 wrapper for independent samples t-test.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use anofox_statistics::{t_test, Alternative, TTestKind, TTestResult};

/// Result of a statistical test (basic version for backward compat).
pub(crate) struct TestResult {
    pub statistic: f64,
    pub p_value: f64,
    pub n1: usize,
    pub n2: Option<usize>,
}

/// Independent samples t-test.
///
/// Calculates the T-test for the means of two independent samples.
///
/// Parameters
/// ----------
/// alternative : str, default "two-sided"
///     The alternative hypothesis: "two-sided", "less", or "greater".
/// equal_var : bool, default False
///     If True, perform a standard independent t-test assuming equal
///     population variances. If False (default), perform Welch's t-test.
/// mu : float, default 0.0
///     The hypothesized difference in means under the null hypothesis.
/// conf_level : float, default 0.95
///     Confidence level for the confidence interval.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from polars_statistics import TTestInd
/// >>>
/// >>> x = np.random.randn(50)
/// >>> y = np.random.randn(50) + 0.5
/// >>>
/// >>> test = TTestInd(alternative="two-sided")
/// >>> test.fit(x, y)
/// >>> print(test.statistic, test.p_value)
#[pyclass(name = "TTestInd")]
pub struct PyTTestInd {
    alternative: Alternative,
    equal_var: bool,
    mu: f64,
    conf_level: f64,
    fitted: Option<TTestResult>,
}

#[pymethods]
impl PyTTestInd {
    /// Create a new independent samples t-test.
    #[new]
    #[pyo3(signature = (alternative="two-sided", equal_var=false, mu=0.0, conf_level=0.95))]
    fn new(alternative: &str, equal_var: bool, mu: f64, conf_level: f64) -> PyResult<Self> {
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

        if !(0.0..=1.0).contains(&conf_level) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "conf_level must be between 0 and 1",
            ));
        }

        Ok(Self {
            alternative: alt,
            equal_var,
            mu,
            conf_level,
            fitted: None,
        })
    }

    /// Perform the t-test on two samples.
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

        let kind = if slf.equal_var {
            TTestKind::Student
        } else {
            TTestKind::Welch
        };

        let result = t_test(
            &x_vec,
            &y_vec,
            kind,
            slf.alternative,
            slf.mu,
            Some(slf.conf_level),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        slf.fitted = Some(result);

        Ok(slf)
    }

    /// Check if the test has been performed.
    fn is_fitted(&self) -> bool {
        self.fitted.is_some()
    }

    /// Get the test statistic.
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

    /// Get the degrees of freedom.
    #[getter]
    fn df(&self) -> PyResult<f64> {
        self.fitted
            .as_ref()
            .map(|r| r.df)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Test not fitted"))
    }

    /// Get the mean of the first sample.
    #[getter]
    fn mean_x(&self) -> PyResult<f64> {
        self.fitted
            .as_ref()
            .map(|r| r.mean_x)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Test not fitted"))
    }

    /// Get the mean of the second sample.
    #[getter]
    fn mean_y(&self) -> PyResult<Option<f64>> {
        self.fitted
            .as_ref()
            .map(|r| r.mean_y)
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

        let var_str = if self.equal_var {
            "True (Student's t)"
        } else {
            "False (Welch's t)"
        };

        let significance = if result.p_value < 0.05 {
            "Reject H0 at alpha=0.05"
        } else {
            "Fail to reject H0 at alpha=0.05"
        };

        let conf_int_str = match &result.conf_int {
            Some(ci) => format!("[{:.4}, {:.4}]", ci.lower, ci.upper),
            None => "N/A".to_string(),
        };

        let mean_y_str = match result.mean_y {
            Some(my) => format!("{:.4}", my),
            None => "N/A".to_string(),
        };

        Ok(format!(
            "Independent Samples T-Test\n\
             ==========================\n\n\
             Test statistic:  {:>12.4}\n\
             P-value:         {:>12.4e}\n\
             Degrees of freedom: {:>9.2}\n\
             Mean x:          {:>12.4}\n\
             Mean y:          {:>12}\n\
             Confidence int:  {:>12}\n\
             Alternative:     {:>12}\n\
             Equal variance:  {:>12}\n\
             Null value (mu): {:>12.4}\n\n\
             Result: {}",
            result.statistic,
            result.p_value,
            result.df,
            result.mean_x,
            mean_y_str,
            conf_int_str,
            alt_str,
            var_str,
            result.null_value,
            significance
        ))
    }
}
