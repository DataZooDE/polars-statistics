//! PyO3 wrapper for Brown-Forsythe test.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use anofox_statistics::brown_forsythe;

use super::py_ttest_ind::TestResult;

/// Brown-Forsythe test for equality of variances.
///
/// Tests the null hypothesis that all input samples are from populations
/// with equal variances. It is more robust to departures from normality
/// than Levene's test.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from polars_statistics import BrownForsythe
/// >>>
/// >>> x = np.random.randn(50)
/// >>> y = np.random.randn(50) * 2  # Different variance
/// >>>
/// >>> test = BrownForsythe()
/// >>> test.fit(x, y)
/// >>> print(test.statistic, test.p_value)
#[pyclass(name = "BrownForsythe")]
pub struct PyBrownForsythe {
    fitted: Option<TestResult>,
}

#[pymethods]
impl PyBrownForsythe {
    /// Create a new Brown-Forsythe test.
    #[new]
    fn new() -> Self {
        Self { fitted: None }
    }

    /// Perform the Brown-Forsythe test on two samples.
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

        let groups: [&[f64]; 2] = [&x_vec, &y_vec];
        let result = brown_forsythe(&groups)
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

        let significance = if result.p_value < 0.05 {
            "Reject H0 (variances are unequal) at alpha=0.05"
        } else {
            "Fail to reject H0 (variances may be equal) at alpha=0.05"
        };

        Ok(format!(
            "Brown-Forsythe Test\n\
             ===================\n\n\
             Test statistic:  {:>12.4}\n\
             P-value:         {:>12.4e}\n\
             Sample sizes:    n1={}, n2={}\n\n\
             H0: All groups have equal variances\n\
             Result: {}",
            result.statistic,
            result.p_value,
            result.n1,
            result.n2.unwrap_or(0),
            significance
        ))
    }
}
