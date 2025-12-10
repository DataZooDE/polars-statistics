//! PyO3 wrapper for independent samples t-test.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use anofox_statistics::{t_test, Alternative, TTestKind};

/// Result of a statistical test.
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
    fitted: Option<TestResult>,
}

#[pymethods]
impl PyTTestInd {
    /// Create a new independent samples t-test.
    #[new]
    #[pyo3(signature = (alternative="two-sided", equal_var=false))]
    fn new(alternative: &str, equal_var: bool) -> PyResult<Self> {
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

        Ok(Self {
            alternative: alt,
            equal_var,
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

        let result = t_test(&x_vec, &y_vec, kind, slf.alternative)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        slf.fitted = Some(TestResult {
            statistic: result.statistic,
            p_value: result.p_value,
            n1: x_vec.len(),
            n2: Some(y_vec.len()),
        });

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

        Ok(format!(
            "Independent Samples T-Test\n\
             ==========================\n\n\
             Test statistic:  {:>12.4}\n\
             P-value:         {:>12.4e}\n\
             Alternative:     {:>12}\n\
             Equal variance:  {:>12}\n\
             Sample sizes:    n1={}, n2={}\n\n\
             Result: {}",
            result.statistic,
            result.p_value,
            alt_str,
            var_str,
            result.n1,
            result.n2.unwrap_or(0),
            significance
        ))
    }
}
