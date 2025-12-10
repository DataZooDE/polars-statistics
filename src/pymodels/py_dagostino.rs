//! PyO3 wrapper for D'Agostino-Pearson test.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use anofox_statistics::dagostino_k_squared;

use super::py_ttest_ind::TestResult;

/// D'Agostino-Pearson K-squared test for normality.
///
/// Tests the null hypothesis that the data was drawn from a normal
/// distribution. This test is based on the skewness and kurtosis
/// of the sample.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from polars_statistics import DAgostino
/// >>>
/// >>> x = np.random.randn(50)
/// >>>
/// >>> test = DAgostino()
/// >>> test.fit(x)
/// >>> print(test.statistic, test.p_value)
#[pyclass(name = "DAgostino")]
pub struct PyDAgostino {
    fitted: Option<TestResult>,
}

#[pymethods]
impl PyDAgostino {
    /// Create a new D'Agostino-Pearson test.
    #[new]
    fn new() -> Self {
        Self { fitted: None }
    }

    /// Perform the D'Agostino-Pearson test on a sample.
    ///
    /// Parameters
    /// ----------
    /// x : array-like of shape (n_samples,)
    ///     Sample data (should have at least 20 observations).
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

        if x_vec.len() < 20 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Sample must have at least 20 observations",
            ));
        }

        let result = dagostino_k_squared(&x_vec)
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

    /// Get the test statistic (K-squared).
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
            "D'Agostino-Pearson Normality Test\n\
             =================================\n\n\
             K-squared:       {:>12.4}\n\
             P-value:         {:>12.4e}\n\
             Sample size:     n={}\n\n\
             H0: Data is normally distributed\n\
             Result: {}",
            result.statistic, result.p_value, result.n1, significance
        ))
    }
}
