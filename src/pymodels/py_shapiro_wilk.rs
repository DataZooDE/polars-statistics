//! PyO3 wrapper for Shapiro-Wilk test.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use anofox_statistics::shapiro_wilk;

use super::py_ttest_ind::TestResult;

/// Shapiro-Wilk test for normality.
///
/// Tests the null hypothesis that the data was drawn from a normal
/// distribution. This is one of the most powerful tests for normality.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from polars_statistics import ShapiroWilk
/// >>>
/// >>> x = np.random.randn(50)
/// >>>
/// >>> test = ShapiroWilk()
/// >>> test.fit(x)
/// >>> print(test.statistic, test.p_value)
#[pyclass(name = "ShapiroWilk")]
pub struct PyShapiroWilk {
    fitted: Option<TestResult>,
}

#[pymethods]
impl PyShapiroWilk {
    /// Create a new Shapiro-Wilk test.
    #[new]
    fn new() -> Self {
        Self { fitted: None }
    }

    /// Perform the Shapiro-Wilk test on a sample.
    ///
    /// Parameters
    /// ----------
    /// x : array-like of shape (n_samples,)
    ///     Sample data (should have 3 to 5000 observations).
    ///
    /// Returns
    /// -------
    /// self
    ///     Fitted test with results.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_vec: Vec<f64> = x.as_slice()?.to_vec();

        if x_vec.len() < 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Sample must have at least 3 observations",
            ));
        }

        let result = shapiro_wilk(&x_vec)
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
            "Reject H0 (data is NOT normally distributed) at alpha=0.05"
        } else {
            "Fail to reject H0 (data may be normally distributed) at alpha=0.05"
        };

        Ok(format!(
            "Shapiro-Wilk Normality Test\n\
             ===========================\n\n\
             W statistic:     {:>12.6}\n\
             P-value:         {:>12.4e}\n\
             Sample size:     n={}\n\n\
             H0: Data is normally distributed\n\
             Result: {}",
            result.statistic, result.p_value, result.n1, significance
        ))
    }
}
