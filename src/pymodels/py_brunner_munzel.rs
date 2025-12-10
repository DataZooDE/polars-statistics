//! PyO3 wrapper for Brunner-Munzel test.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use anofox_statistics::{brunner_munzel, Alternative};

use super::py_ttest_ind::TestResult;

/// Brunner-Munzel test for stochastic equality.
///
/// Tests the null hypothesis that when values are taken from each
/// group, the probabilities of getting larger values are equal.
/// This is a robust alternative to the Mann-Whitney U test.
///
/// Parameters
/// ----------
/// alternative : str, default "two-sided"
///     The alternative hypothesis: "two-sided", "less", or "greater".
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from polars_statistics import BrunnerMunzel
/// >>>
/// >>> x = np.random.randn(50)
/// >>> y = np.random.randn(50) + 0.5
/// >>>
/// >>> test = BrunnerMunzel()
/// >>> test.fit(x, y)
/// >>> print(test.statistic, test.p_value)
#[pyclass(name = "BrunnerMunzel")]
pub struct PyBrunnerMunzel {
    alternative: Alternative,
    fitted: Option<TestResult>,
}

#[pymethods]
impl PyBrunnerMunzel {
    /// Create a new Brunner-Munzel test.
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

    /// Perform the Brunner-Munzel test on two samples.
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

        let result = brunner_munzel(&x_vec, &y_vec, slf.alternative)
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

        let significance = if result.p_value < 0.05 {
            "Reject H0 at alpha=0.05"
        } else {
            "Fail to reject H0 at alpha=0.05"
        };

        Ok(format!(
            "Brunner-Munzel Test\n\
             ===================\n\n\
             Test statistic:  {:>12.4}\n\
             P-value:         {:>12.4e}\n\
             Alternative:     {:>12}\n\
             Sample sizes:    n1={}, n2={}\n\n\
             H0: P(X < Y) = P(Y < X) (stochastic equality)\n\
             Result: {}",
            result.statistic,
            result.p_value,
            alt_str,
            result.n1,
            result.n2.unwrap_or(0),
            significance
        ))
    }
}
