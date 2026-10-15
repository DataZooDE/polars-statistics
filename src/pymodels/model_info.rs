//! Shared helper turning an upstream `anofox_regression::ModelInfo` into a
//! Python dict (UPST-09, #66).
//!
//! `ModelInfo` (anofox-regression 0.5.20, `HasModelInfo` trait) is a static,
//! `Copy` description of a fit: the estimator name (`model_type`), its error
//! family in R naming (`family`, `None` for non-likelihood estimators), and
//! the link function (`link`). Every `model_info` getter across the 30
//! `HasModelInfo`-backed `PyModel` classes (16.3-08) delegates here so the
//! dict shape is defined exactly once.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use anofox_regression::ModelInfo;

/// Build the `{"model_type": str, "family": str | None, "link": str}` dict
/// for a fitted model's [`ModelInfo`].
pub(crate) fn model_info_dict(py: Python<'_>, info: ModelInfo) -> PyResult<Bound<'_, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("model_type", info.model_type)?;
    dict.set_item("family", info.family)?;
    dict.set_item("link", info.link)?;
    Ok(dict)
}
