//! PyO3 wrapper for paired samples t-test.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use anofox_statistics::{t_test, Alternative, TTestKind};

use super::py_ttest_ind::TestResult;

/// Paired samples t-test.
///
/// Calculates the T-test for the means of two paired samples.
///
/// Parameters
/// ----------
/// alternative : str, default "two-sided"
///     The alternative hypothesis: "two-sided", "less", or "greater".
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from polars_statistics import TTestPaired
/// >>>
/// >>> before = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
/// >>> after = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
/// >>>
/// >>> test = TTestPaired()
/// >>> test.fit(before, after)
/// >>> print(test.statistic, test.p_value)
#[pyclass(name = "TTestPaired")]
pub struct PyTTestPaired {
    alternative: Alternative,
    fitted: Option<TestResult>,
}

#[pymethods]
impl PyTTestPaired {
    /// Create a new paired samples t-test.
    #[new]
    #[pyo3(signature = (alternative="two-sided"))]
    fn new(alternative: &str) -> PyResult<Self> {
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
            fitted: None,
        })
    }

    /// Perform the paired t-test on two samples.
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

        let result = t_test(&x_vec, &y_vec, TTestKind::Paired, slf.alternative)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        slf.fitted = Some(TestResult {
            statistic: result.statistic,
            p_value: result.p_value,
            n1: x_vec.len(),
            n2: None,
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

        let significance = if result.p_value < 0.05 {
            "Reject H0 at alpha=0.05"
        } else {
            "Fail to reject H0 at alpha=0.05"
        };

        Ok(format!(
            "Paired Samples T-Test\n\
             =====================\n\n\
             Test statistic:  {:>12.4}\n\
             P-value:         {:>12.4e}\n\
             Alternative:     {:>12}\n\
             Sample size:     n={}\n\n\
             Result: {}",
            result.statistic, result.p_value, alt_str, result.n1, significance
        ))
    }
}
