//! PyO3 wrapper for MomentAccumulator — streaming sufficient-statistics utility.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

use anofox_regression::solvers::MomentAccumulator;

use crate::utils::IntoNumpy;

/// Streaming sufficient-statistics accumulator for online OLS / Ridge fitting.
///
/// Accumulates the cross-product moments ``XᵀX``, ``Xᵀy``, ``Σx``, ``Σy``,
/// and ``n`` one row at a time. Once populated, pass the accumulator to
/// :meth:`OLS.fit_from_accumulator` or :meth:`Ridge.fit_from_accumulator` to
/// obtain fitted coefficients without ever materialising the full N×p design
/// matrix.
///
/// Accumulators can also be merged (e.g., after parallel accumulation on
/// worker threads) via :meth:`merge`.
///
/// Parameters
/// ----------
/// n_features : int
///     Number of predictors (columns of the design matrix, excluding
///     any intercept column the solver adds internally).
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from polars_statistics import OLS, MomentAccumulator
/// >>>
/// >>> X = np.random.randn(100, 2)
/// >>> y = X @ [1.0, 2.0] + 0.1 * np.random.randn(100)
/// >>>
/// >>> acc = MomentAccumulator(n_features=2)
/// >>> for i in range(len(y)):
/// ...     acc.push_row(X[i], y[i])
/// >>> model = OLS().fit_from_accumulator(acc)
/// >>> model.coefficients
#[pyclass(name = "MomentAccumulator")]
pub struct PyMomentAccumulator {
    pub(crate) inner: MomentAccumulator,
}

#[pymethods]
impl PyMomentAccumulator {
    /// Create a new accumulator for ``n_features`` predictors.
    #[new]
    fn new(n_features: usize) -> Self {
        Self {
            inner: MomentAccumulator::new(n_features),
        }
    }

    /// Incorporate one observation into the running moments.
    ///
    /// Parameters
    /// ----------
    /// x_row : 1-D array of shape (n_features,)
    ///     Predictor values for the observation.  Must have length equal to
    ///     ``n_features`` specified at construction time.
    /// y : float
    ///     Response value for the observation.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``x_row`` has the wrong length.
    fn push_row(&mut self, x_row: PyReadonlyArray1<'_, f64>, y: f64) -> PyResult<()> {
        // Threat T-04-12: explicit length check before calling push_row (Pitfall 4).
        let slice = x_row.as_slice().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "x_row must be a contiguous (C-order) 1-D float64 array; \
                 try passing np.ascontiguousarray(x_row) if it is a slice",
            )
        })?;
        let n = slice.len();
        let expected = self.inner.n_features();
        if n != expected {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "x_row has {n} elements but accumulator expects {expected}"
            )));
        }
        // CRITICAL: push_row takes &[f64] — use as_slice(), NOT to_faer() (anti-pattern).
        self.inner
            .push_row(slice, y)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Merge another accumulator into this one (in-place).
    ///
    /// Enables parallel accumulation: run separate :class:`MomentAccumulator`
    /// instances on worker threads and then merge the results.
    fn merge(&mut self, other: &PyMomentAccumulator) -> PyResult<()> {
        self.inner
            .merge(&other.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Reset all moments to zero (keeps ``n_features`` unchanged).
    fn clear(&mut self) {
        self.inner.clear();
    }

    /// Number of observations pushed so far.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// Number of predictors this accumulator was created for.
    #[getter]
    fn n_features(&self) -> usize {
        self.inner.n_features()
    }

    /// Sum of response values Σyᵢ.
    #[getter]
    fn sum_y(&self) -> f64 {
        self.inner.sum_y()
    }

    /// Sum of predictor vectors Σxᵢ, shape (n_features,).
    #[getter]
    fn sum_x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.sum_x().into_numpy(py)
    }

    /// Cross-product matrix XᵀX, shape (n_features, n_features).
    #[getter]
    fn xtx<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.xtx().into_numpy(py)
    }

    /// Cross-product vector Xᵀy, shape (n_features,).
    #[getter]
    fn xty<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.xty().into_numpy(py)
    }
}
