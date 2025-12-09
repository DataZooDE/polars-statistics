//! Array conversion utilities between numpy and faer types.

use faer::{Col, Mat};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// Trait for converting numpy arrays to faer matrices.
pub trait IntoFaer<T> {
    /// Convert to the target faer type.
    fn into_faer(&self) -> T;
}

impl IntoFaer<Mat<f64>> for PyReadonlyArray2<'_, f64> {
    fn into_faer(&self) -> Mat<f64> {
        let shape = self.shape();
        let nrows = shape[0];
        let ncols = shape[1];

        // Convert to standard layout if needed and get slice
        let array = self.as_array();
        Mat::from_fn(nrows, ncols, |i, j| array[[i, j]])
    }
}

impl IntoFaer<Col<f64>> for PyReadonlyArray1<'_, f64> {
    fn into_faer(&self) -> Col<f64> {
        let array = self.as_array();
        let len = array.len();
        Col::from_fn(len, |i| array[i])
    }
}

/// Trait for converting faer types to numpy arrays.
pub trait IntoNumpy<'py> {
    type Output;
    /// Convert to numpy array.
    fn into_numpy(self, py: Python<'py>) -> Self::Output;
}

impl<'py> IntoNumpy<'py> for &Col<f64> {
    type Output = Bound<'py, PyArray1<f64>>;

    fn into_numpy(self, py: Python<'py>) -> Self::Output {
        let len = self.nrows();
        let data: Vec<f64> = (0..len).map(|i| self[i]).collect();
        PyArray1::from_vec(py, data)
    }
}

impl<'py> IntoNumpy<'py> for Col<f64> {
    type Output = Bound<'py, PyArray1<f64>>;

    fn into_numpy(self, py: Python<'py>) -> Self::Output {
        (&self).into_numpy(py)
    }
}

impl<'py> IntoNumpy<'py> for &Mat<f64> {
    type Output = Bound<'py, PyArray2<f64>>;

    fn into_numpy(self, py: Python<'py>) -> Self::Output {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let data: Vec<Vec<f64>> = (0..nrows)
            .map(|i| (0..ncols).map(move |j| self[(i, j)]).collect())
            .collect();
        PyArray2::from_vec2(py, &data).expect("Failed to create 2D array")
    }
}

/// Convert a Vec<f64> to a numpy array.
pub fn vec_to_numpy<'py>(py: Python<'py>, data: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, data)
}

/// Convert a Vec<f64> to a faer Col.
pub fn vec_to_col(data: &[f64]) -> Col<f64> {
    Col::from_fn(data.len(), |i| data[i])
}

/// Convert a faer Col to a Vec<f64>.
pub fn col_to_vec(col: &Col<f64>) -> Vec<f64> {
    (0..col.nrows()).map(|i| col[i]).collect()
}
