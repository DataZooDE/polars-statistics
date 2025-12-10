//! PyO3 wrapper for Kruskal-Wallis H test.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use anofox_statistics::kruskal_wallis;

/// Result of Kruskal-Wallis test (with multiple groups).
struct KruskalResult {
    statistic: f64,
    p_value: f64,
    n_groups: usize,
    group_sizes: Vec<usize>,
}

/// Kruskal-Wallis H test.
///
/// Tests the null hypothesis that the population medians of all groups
/// are equal. This is a non-parametric alternative to one-way ANOVA.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from polars_statistics import KruskalWallis
/// >>>
/// >>> g1 = np.random.randn(30)
/// >>> g2 = np.random.randn(30) + 0.5
/// >>> g3 = np.random.randn(30) + 1.0
/// >>>
/// >>> test = KruskalWallis()
/// >>> test.fit(g1, g2, g3)
/// >>> print(test.statistic, test.p_value)
#[pyclass(name = "KruskalWallis")]
pub struct PyKruskalWallis {
    fitted: Option<KruskalResult>,
}

#[pymethods]
impl PyKruskalWallis {
    /// Create a new Kruskal-Wallis test.
    #[new]
    fn new() -> Self {
        Self { fitted: None }
    }

    /// Perform the Kruskal-Wallis test on multiple groups.
    ///
    /// Parameters
    /// ----------
    /// *groups : array-like
    ///     Two or more arrays of sample observations.
    ///
    /// Returns
    /// -------
    /// self
    ///     Fitted test with results.
    #[pyo3(signature = (*groups))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        groups: Vec<PyReadonlyArray1<'py, f64>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if groups.len() < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "At least 2 groups are required",
            ));
        }

        let group_vecs: Vec<Vec<f64>> = groups
            .iter()
            .map(|g| g.as_slice().map(|s| s.to_vec()))
            .collect::<Result<Vec<_>, _>>()?;

        let group_sizes: Vec<usize> = group_vecs.iter().map(|g| g.len()).collect();
        let group_refs: Vec<&[f64]> = group_vecs.iter().map(|g| g.as_slice()).collect();

        let result = kruskal_wallis(&group_refs)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        slf.fitted = Some(KruskalResult {
            statistic: result.statistic,
            p_value: result.p_value,
            n_groups: groups.len(),
            group_sizes,
        });

        Ok(slf)
    }

    /// Check if the test has been performed.
    fn is_fitted(&self) -> bool {
        self.fitted.is_some()
    }

    /// Get the test statistic (H statistic).
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

        let sizes_str = result
            .group_sizes
            .iter()
            .enumerate()
            .map(|(i, n)| format!("n{}={}", i + 1, n))
            .collect::<Vec<_>>()
            .join(", ");

        Ok(format!(
            "Kruskal-Wallis H Test\n\
             =====================\n\n\
             H statistic:     {:>12.4}\n\
             P-value:         {:>12.4e}\n\
             Number of groups: {}\n\
             Sample sizes:    {}\n\n\
             H0: All groups have equal population medians\n\
             Result: {}",
            result.statistic, result.p_value, result.n_groups, sizes_str, significance
        ))
    }
}
