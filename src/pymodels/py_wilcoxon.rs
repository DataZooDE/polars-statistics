//! PyO3 wrapper for Wilcoxon signed-rank test.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use anofox_statistics::wilcoxon_signed_rank;

use super::py_ttest_ind::TestResult;

/// Wilcoxon signed-rank test.
///
/// Tests the null hypothesis that two related paired samples come from
/// the same distribution. This is a non-parametric alternative to the
/// paired t-test.
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
    fitted: Option<TestResult>,
}

#[pymethods]
impl PyWilcoxonSignedRank {
    /// Create a new Wilcoxon signed-rank test.
    #[new]
    fn new() -> Self {
        Self { fitted: None }
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

        let result = wilcoxon_signed_rank(&x_vec, &y_vec)
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

    /// Get a formatted summary of the test results.
    fn summary(&self) -> PyResult<String> {
        let result = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Test not fitted"))?;

        let significance = if result.p_value < 0.05 {
            "Reject H0 at alpha=0.05"
        } else {
            "Fail to reject H0 at alpha=0.05"
        };

        Ok(format!(
            "Wilcoxon Signed-Rank Test\n\
             =========================\n\n\
             W statistic:     {:>12.4}\n\
             P-value:         {:>12.4e}\n\
             Sample size:     n={}\n\n\
             H0: Paired differences have median zero\n\
             Result: {}",
            result.statistic, result.p_value, result.n1, significance
        ))
    }
}
