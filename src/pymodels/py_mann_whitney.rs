//! PyO3 wrapper for Mann-Whitney U test.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use anofox_statistics::mann_whitney_u;

use super::py_ttest_ind::TestResult;

/// Mann-Whitney U test (Wilcoxon rank-sum test).
///
/// Tests whether the distribution of values in one sample is stochastically
/// greater than in another sample. This is a non-parametric test that does
/// not assume normality.
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
    fitted: Option<TestResult>,
}

#[pymethods]
impl PyMannWhitneyU {
    /// Create a new Mann-Whitney U test.
    #[new]
    fn new() -> Self {
        Self { fitted: None }
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

        let result = mann_whitney_u(&x_vec, &y_vec)
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
            "Mann-Whitney U Test\n\
             ===================\n\n\
             U statistic:     {:>12.4}\n\
             P-value:         {:>12.4e}\n\
             Sample sizes:    n1={}, n2={}\n\n\
             H0: The distributions are equal\n\
             Result: {}",
            result.statistic,
            result.p_value,
            result.n1,
            result.n2.unwrap_or(0),
            significance
        ))
    }
}
