//! Shared helper computing R `broom::augment`-style per-observation
//! diagnostics for any PyModel wrapping an `Augmentable` fitted model.
//!
//! Used by `OLS.augment`/`WLS.augment` (16.3-05) and the eight GLM PyModel
//! `.augment()` methods added in 16.3-06.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use anofox_regression::diagnostics::{augment, augment_with, Augmentable};

use crate::utils::{IntoNumpy, ToFaer};

/// Per-observation diagnostics (R `broom::augment`) for any `Augmentable`
/// fitted model.
///
/// `x`/`y` must be the training data (without the intercept column) and
/// agree on row count; `weights` are prior weights (`None` uses the fit's own
/// weights, e.g. WLS). `residual_type` is `"response"`/`"pearson"`/
/// `"deviance"` (`None` uses the upstream default: response for linear
/// models, deviance for GLMs).
///
/// Returns a dict with keys `fitted`, `residual`, `resid_type`,
/// `std_residual`, `stud_residual`, `leverage`, `cooks_d`, `dffits`.
pub(crate) fn augment_dict<'py, M: Augmentable + ?Sized>(
    py: Python<'py>,
    fitted: &M,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    residual_type: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let n_x = x.shape()[0];
    let n_y = y.shape()[0];
    if n_x != n_y {
        return Err(PyValueError::new_err(format!(
            "augment: x has {n_x} rows but y has {n_y} — they must match."
        )));
    }

    let x_mat = x.to_faer();
    let y_col = y.to_faer();
    let w_col = weights.as_ref().map(|w| w.to_faer());

    let rt = match residual_type {
        Some(s) => Some(crate::expressions::parse_residual_type(s).map_err(PyValueError::new_err)?),
        None => None,
    };

    let result = match rt {
        Some(rt) => augment_with(fitted, &x_mat, &y_col, w_col.as_ref(), rt),
        None => augment(fitted, &x_mat, &y_col, w_col.as_ref()),
    }
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("fitted", (&result.fitted).into_numpy(py))?;
    dict.set_item("residual", (&result.residual).into_numpy(py))?;
    dict.set_item("resid_type", result.resid_type.as_str())?;
    dict.set_item("std_residual", (&result.std_residual).into_numpy(py))?;
    dict.set_item("stud_residual", (&result.stud_residual).into_numpy(py))?;
    dict.set_item("leverage", (&result.leverage).into_numpy(py))?;
    dict.set_item("cooks_d", (&result.cooks_d).into_numpy(py))?;
    dict.set_item("dffits", (&result.dffits).into_numpy(py))?;
    Ok(dict)
}
