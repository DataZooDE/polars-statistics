//! PyO3 wrapper for Brunner-Munzel test.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use anofox_statistics::{brunner_munzel, Alternative, BrunnerMunzelResult};

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
/// alpha : float, optional
///     Significance level for the confidence interval.
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
    alpha: Option<f64>,
    fitted: Option<BrunnerMunzelResult>,
}

#[pymethods]
impl PyBrunnerMunzel {
    /// Create a new Brunner-Munzel test.
    #[new]
    #[pyo3(signature = (alternative="two-sided", alpha=None))]
    fn new(alternative: &str, alpha: Option<f64>) -> PyResult<Self> {
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

        if let Some(a) = alpha {
            if !(0.0..=1.0).contains(&a) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "alpha must be between 0 and 1",
                ));
            }
        }

        Ok(Self {
            alternative: alt,
            alpha,
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

        let result = brunner_munzel(&x_vec, &y_vec, slf.alternative, slf.alpha)
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

    /// Get the stochastic superiority estimate P(X < Y).
    #[getter]
    fn estimate(&self) -> PyResult<f64> {
        self.fitted
            .as_ref()
            .map(|r| r.estimate)
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

        let significance = if result.p_value < 0.05 {
            "Reject H0 at alpha=0.05"
        } else {
            "Fail to reject H0 at alpha=0.05"
        };

        let conf_int_str = match &result.conf_int {
            Some(ci) => format!("[{:.4}, {:.4}]", ci.lower, ci.upper),
            None => "N/A".to_string(),
        };

        Ok(format!(
            "Brunner-Munzel Test\n\
             ===================\n\n\
             Test statistic:  {:>12.4}\n\
             P-value:         {:>12.4e}\n\
             Degrees of freedom: {:>9.2}\n\
             Estimate P(X<Y): {:>12.4}\n\
             Confidence int:  {:>12}\n\
             Alternative:     {:>12}\n\n\
             H0: P(X < Y) = P(Y < X) (stochastic equality)\n\
             Result: {}",
            result.statistic,
            result.p_value,
            result.df,
            result.estimate,
            conf_int_str,
            alt_str,
            significance
        ))
    }
}
