//! PyO3 wrappers for bootstrap methods.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

use anofox_statistics::{CircularBlockBootstrap, StationaryBootstrap};

/// Stationary Bootstrap for time series data.
///
/// Generates bootstrap samples using random blocks of varying length.
/// Suitable for weakly stationary time series.
///
/// Parameters
/// ----------
/// expected_block_length : float
///     Expected length of blocks (typical: n^(1/3) where n is data length).
/// seed : int, optional
///     Random seed for reproducibility.
#[pyclass(name = "StationaryBootstrap")]
pub struct PyStationaryBootstrap {
    inner: StationaryBootstrap,
}

#[pymethods]
impl PyStationaryBootstrap {
    #[new]
    #[pyo3(signature = (expected_block_length, seed=None))]
    fn new(expected_block_length: f64, seed: Option<u64>) -> Self {
        Self {
            inner: StationaryBootstrap::new(expected_block_length, seed),
        }
    }

    /// Generate a single bootstrap sample.
    ///
    /// Parameters
    /// ----------
    /// data : array-like
    ///     The original time series data.
    /// length : int, optional
    ///     Length of bootstrap sample (default: same as data).
    ///
    /// Returns
    /// -------
    /// array
    ///     Bootstrap sample.
    #[pyo3(signature = (data, length=None))]
    fn sample<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray1<'py, f64>,
        length: Option<usize>,
    ) -> Bound<'py, PyArray1<f64>> {
        let data_vec: Vec<f64> = data.as_array().to_vec();
        let n = length.unwrap_or(data_vec.len());
        let sample = self.inner.sample(&data_vec, n);
        PyArray1::from_vec(py, sample)
    }

    /// Generate multiple bootstrap samples.
    ///
    /// Parameters
    /// ----------
    /// data : array-like
    ///     The original time series data.
    /// n_samples : int
    ///     Number of bootstrap samples to generate.
    /// length : int, optional
    ///     Length of each bootstrap sample (default: same as data).
    ///
    /// Returns
    /// -------
    /// array of shape (n_samples, length)
    ///     Matrix of bootstrap samples.
    #[pyo3(signature = (data, n_samples, length=None))]
    fn samples<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray1<'py, f64>,
        n_samples: usize,
        length: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let data_vec: Vec<f64> = data.as_array().to_vec();
        let n = length.unwrap_or(data_vec.len());
        let samples = self.inner.samples(&data_vec, n, n_samples);

        PyArray2::from_vec2(py, &samples)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
}

/// Circular Block Bootstrap for time series data.
///
/// Generates bootstrap samples using fixed-length blocks with wrap-around.
///
/// Parameters
/// ----------
/// block_length : int
///     Length of each block.
/// seed : int, optional
///     Random seed for reproducibility.
#[pyclass(name = "CircularBlockBootstrap")]
pub struct PyCircularBlockBootstrap {
    inner: CircularBlockBootstrap,
}

#[pymethods]
impl PyCircularBlockBootstrap {
    #[new]
    #[pyo3(signature = (block_length, seed=None))]
    fn new(block_length: usize, seed: Option<u64>) -> Self {
        Self {
            inner: CircularBlockBootstrap::new(block_length, seed),
        }
    }

    /// Generate a single bootstrap sample.
    ///
    /// Parameters
    /// ----------
    /// data : array-like
    ///     The original time series data.
    /// length : int, optional
    ///     Length of bootstrap sample (default: same as data).
    ///
    /// Returns
    /// -------
    /// array
    ///     Bootstrap sample.
    #[pyo3(signature = (data, length=None))]
    fn sample<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray1<'py, f64>,
        length: Option<usize>,
    ) -> Bound<'py, PyArray1<f64>> {
        let data_vec: Vec<f64> = data.as_array().to_vec();
        let n = length.unwrap_or(data_vec.len());
        let sample = self.inner.sample(&data_vec, n);
        PyArray1::from_vec(py, sample)
    }

    /// Generate multiple bootstrap samples.
    ///
    /// Parameters
    /// ----------
    /// data : array-like
    ///     The original time series data.
    /// n_samples : int
    ///     Number of bootstrap samples to generate.
    /// length : int, optional
    ///     Length of each bootstrap sample (default: same as data).
    ///
    /// Returns
    /// -------
    /// array of shape (n_samples, length)
    ///     Matrix of bootstrap samples.
    #[pyo3(signature = (data, n_samples, length=None))]
    fn samples<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray1<'py, f64>,
        n_samples: usize,
        length: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let data_vec: Vec<f64> = data.as_array().to_vec();
        let n = length.unwrap_or(data_vec.len());
        let samples = self.inner.samples(&data_vec, n, n_samples);

        PyArray2::from_vec2(py, &samples)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
}
