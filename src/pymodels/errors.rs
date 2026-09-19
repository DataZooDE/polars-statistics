//! Shared, contextual error constructors for the PyO3 model / test classes
//! (ERR-01 / ERR-02).
//!
//! Historically every getter and method guarded its unfitted state with a
//! generic `PyErr::new::<PyRuntimeError, _>("Model not fitted")`.  That message
//! never told the user *which* model failed nor *how* to fix it.  These helpers
//! centralise the wording so the error names the class and points at the
//! required `.fit(...)` call, while keeping the exception *type* stable
//! (`RuntimeError` for unfitted state, `ValueError` for bad input) so existing
//! `pytest.raises(RuntimeError)` / `pytest.raises(ValueError)` assertions keep
//! holding.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyFutureWarning, PyRuntimeError, PyValueError};
use pyo3::{PyErr, PyResult, Python};

/// Error raised when a *regression* model method/getter is used before `.fit`.
///
/// Example message: ``OLS is not fitted — call `.fit(X, y)` first.``
pub fn not_fitted_err(model: &str) -> PyErr {
    PyRuntimeError::new_err(format!("{model} is not fitted — call `.fit(X, y)` first."))
}

/// Error raised when a *statistical test* class result is accessed before
/// `.fit` (tests are fitted with the sample data, not an ``X, y`` pair).
///
/// Example message: ``TTestInd is not fitted — call `.fit(...)` first.``
pub fn not_fitted_test_err(model: &str) -> PyErr {
    PyRuntimeError::new_err(format!("{model} is not fitted — call `.fit(...)` first."))
}

/// Error raised when the feature matrix ``X`` and target ``y`` disagree on the
/// number of rows.  Uses ``ValueError`` (sklearn convention for bad input).
pub fn x_y_row_mismatch_err(model: &str, n_x: usize, n_y: usize) -> PyErr {
    PyValueError::new_err(format!(
        "{model}.fit: X has {n_x} rows but y has {n_y} — they must match \
         (X is (n_samples, n_features), y is (n_samples,))."
    ))
}

/// Error raised when there are fewer samples than free parameters, so the
/// system is under-determined (rank deficiency by construction).
pub fn too_few_samples_err(model: &str, n_samples: usize, n_params: usize) -> PyErr {
    PyValueError::new_err(format!(
        "{model}.fit: {n_samples} samples is not enough to estimate {n_params} \
         parameters — provide at least as many rows as columns (plus one for the \
         intercept), or reduce the number of features."
    ))
}

/// Error raised when an input array is empty.
pub fn empty_input_err(model: &str) -> PyErr {
    PyValueError::new_err(format!(
        "{model}.fit: received empty input — X and y must contain at least one row."
    ))
}

/// Resolve the intercept flag for a model *constructor* from the preferred
/// `add_intercept` kwarg and the deprecated `with_intercept` kwarg (API-01).
///
/// Semantics (mirroring the Python `_resolve_intercept` used by the
/// expression builders):
///
/// * both given -> `ValueError` (they are mutually exclusive),
/// * only `with_intercept` given -> a `FutureWarning` is emitted and its value
///   is used (kept fully back-compatible; `with_intercept` is *not* removed),
/// * only `add_intercept` given -> used as-is,
/// * neither -> `default`.
pub fn resolve_intercept(
    py: Python<'_>,
    add_intercept: Option<bool>,
    with_intercept: Option<bool>,
    default: bool,
) -> PyResult<bool> {
    match (add_intercept, with_intercept) {
        (Some(_), Some(_)) => Err(PyValueError::new_err(
            "Cannot specify both 'add_intercept' and 'with_intercept'. \
             Use 'add_intercept' (with_intercept is deprecated).",
        )),
        (None, Some(w)) => {
            PyErr::warn(
                py,
                &py.get_type::<PyFutureWarning>(),
                std::ffi::CString::new(
                    "Parameter 'with_intercept' is deprecated and will be removed in a \
                     future release. Use 'add_intercept' instead.",
                )
                .unwrap()
                .as_c_str(),
                1,
            )?;
            Ok(w)
        }
        (Some(a), None) => Ok(a),
        (None, None) => Ok(default),
    }
}

/// Validate a `(X, y)` pair for a regressor's `fit` before handing it to the
/// solver.  Catches the two highest-signal, most confusing failure modes:
///
/// * empty input (0 rows), and
/// * an `X`/`y` row-count mismatch,
///
/// both of which otherwise surface as an opaque solver error or a panic deep in
/// faer.  Returns `Ok(())` when the shapes are compatible so the caller can
/// proceed to fitting.
pub fn validate_xy(
    model: &str,
    x: &PyReadonlyArray2<'_, f64>,
    y: &PyReadonlyArray1<'_, f64>,
) -> PyResult<()> {
    let x_shape = x.shape();
    let n_x = x_shape[0];
    let n_y = y.shape()[0];
    if n_x == 0 || n_y == 0 {
        return Err(empty_input_err(model));
    }
    if n_x != n_y {
        return Err(x_y_row_mismatch_err(model, n_x, n_y));
    }
    // Fewer rows than columns is rank-deficient by construction — the solver
    // would either error opaquely or emit NaN coefficients. Flag it early with
    // an actionable message. (We check against n_features, not n_features + 1,
    // so this never false-positives on the exactly-determined case.)
    let n_features = x_shape[1];
    if n_x < n_features {
        return Err(too_few_samples_err(model, n_x, n_features));
    }
    Ok(())
}
