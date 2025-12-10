//! PyO3 wrapper for Yuen's trimmed mean test.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use anofox_statistics::yuen_test;

use super::py_ttest_ind::TestResult;

/// Yuen's test for trimmed means.
///
/// Compares the trimmed means of two samples. This test is robust to
/// outliers and non-normality.
///
/// Parameters
/// ----------
/// trim : float, default 0.2
///     Proportion of observations to trim from each end of the
///     distribution. Must be between 0 and 0.5.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from polars_statistics import YuenTest
/// >>>
/// >>> x = np.random.randn(50)
/// >>> y = np.random.randn(50) + 0.5
/// >>>
/// >>> test = YuenTest(trim=0.2)
/// >>> test.fit(x, y)
/// >>> print(test.statistic, test.p_value)
#[pyclass(name = "YuenTest")]
pub struct PyYuenTest {
    trim: f64,
    fitted: Option<TestResult>,
}

#[pymethods]
impl PyYuenTest {
    /// Create a new Yuen test.
    #[new]
    #[pyo3(signature = (trim=0.2))]
    fn new(trim: f64) -> PyResult<Self> {
        if !(0.0..=0.5).contains(&trim) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "trim must be between 0 and 0.5",
            ));
        }

        Ok(Self { trim, fitted: None })
    }

    /// Perform Yuen's test on two samples.
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

        let result = yuen_test(&x_vec, &y_vec, slf.trim)
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
            "Reject H0 at alpha=0.05"
        } else {
            "Fail to reject H0 at alpha=0.05"
        };

        Ok(format!(
            "Yuen's Trimmed Mean Test\n\
             ========================\n\n\
             Test statistic:  {:>12.4}\n\
             P-value:         {:>12.4e}\n\
             Trim proportion: {:>12.2}\n\
             Sample sizes:    n1={}, n2={}\n\n\
             H0: Trimmed means are equal\n\
             Result: {}",
            result.statistic,
            result.p_value,
            self.trim,
            result.n1,
            result.n2.unwrap_or(0),
            significance
        ))
    }
}
