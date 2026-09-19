//! Shared result-ergonomics helpers (ERGO-01/02/03).
//!
//! Every PyO3 model / test class exposes its results through `#[getter]`
//! properties.  Rather than hand-writing a `to_dict` / `__repr__` / `summary`
//! for all ~40 classes (which would be error-prone copy-paste), this module
//! provides generic implementations that *introspect the class's own
//! properties* at runtime and materialise them.  The [`impl_ergonomics!`]
//! macro stamps the three methods onto a class in one line.
//!
//! Design decisions
//! ----------------
//! * **Unfitted state never panics.** `__repr__` reports `fitted=False`; a
//!   getter that raises `RuntimeError` (the "… not fitted" guard) is treated as
//!   "unavailable" and simply omitted from `to_dict` / skipped in `__repr__`.
//! * `to_dict()` on an unfitted model returns `{"fitted": false}` (a plain,
//!   non-raising dict) so callers can branch without a `try/except`.
//! * numpy arrays and other rich getter values are passed through verbatim so
//!   the dict is directly usable (e.g. `d["coefficients"]` is an `ndarray`).

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Names of introspection / control methods that are never treated as result
/// fields for `to_dict` / `__repr__`.
const SKIP_PROPS: &[&str] = &["is_fitted"];

/// Return `true` if the class (or an ancestor) reports itself fitted.
///
/// Classes without an `is_fitted` method (e.g. accumulators, bootstrap
/// samplers, immutable result objects) are always considered "fitted" so their
/// state is materialised.
fn is_fitted(slf: &Bound<'_, PyAny>) -> bool {
    match slf.call_method0("is_fitted") {
        Ok(v) => v.extract::<bool>().unwrap_or(true),
        Err(_) => true,
    }
}

/// Collect the names of the `property` (getter) descriptors declared on the
/// class, in declaration order where possible.
fn property_names<'py>(slf: &Bound<'py, PyAny>) -> PyResult<Vec<String>> {
    let py = slf.py();
    let cls = slf.get_type();
    let builtins = py.import("builtins")?;
    let property_ty = builtins.getattr("property")?;

    let mut names: Vec<String> = Vec::new();
    // Walk the MRO so inherited properties are captured too.
    let mro = cls.getattr("__mro__")?;
    for base in mro.try_iter()? {
        let base = base?;
        let dict = match base.getattr("__dict__") {
            Ok(d) => d,
            Err(_) => continue,
        };
        let keys = match dict.call_method0("keys") {
            Ok(k) => k,
            Err(_) => continue,
        };
        for key in keys.try_iter()? {
            let key = key?;
            let name: String = key.extract()?;
            if name.starts_with("__") || SKIP_PROPS.contains(&name.as_str()) {
                continue;
            }
            let attr = dict.get_item(&name)?;
            // Getters appear either as builtin `property` objects (pure-Python
            // subclasses) or, for PyO3 `#[getter]`, as `getset_descriptor`.
            let is_property = attr.is_instance(&property_ty)?;
            let is_getset = attr
                .get_type()
                .name()
                .ok()
                .and_then(|n| n.extract::<String>().ok())
                .map(|n| n == "getset_descriptor")
                .unwrap_or(false);
            if (is_property || is_getset) && !names.contains(&name) {
                names.push(name);
            }
        }
    }
    Ok(names)
}

/// Build a dict of `{property_name: value}` for every getter that resolves
/// without raising.  Getters that raise (typically the unfitted guard) are
/// skipped.
pub fn to_dict<'py>(slf: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    let py = slf.py();
    let dict = PyDict::new(py);

    if !is_fitted(slf) {
        dict.set_item("fitted", false)?;
        return Ok(dict);
    }

    for name in property_names(slf)? {
        match slf.getattr(name.as_str()) {
            Ok(value) => dict.set_item(name, value)?,
            Err(_) => continue,
        }
    }
    Ok(dict)
}

/// Format a scalar getter value for a compact `__repr__`, or return `None` if
/// the value is not a simple scalar worth showing inline.
fn repr_scalar(value: &Bound<'_, PyAny>) -> Option<String> {
    if let Ok(b) = value.extract::<bool>() {
        return Some(b.to_string());
    }
    if let Ok(i) = value.extract::<i64>() {
        return Some(i.to_string());
    }
    if let Ok(f) = value.extract::<f64>() {
        if f.is_finite() {
            return Some(format!("{:.4}", f));
        }
        return Some(f.to_string());
    }
    None
}

/// Build a concise, informative `__repr__` such as
/// `OLS(fitted=True, r_squared=0.9820, n_observations=4)` or `OLS(fitted=False)`.
///
/// Picks a small, stable set of "headline" scalar getters when present, else
/// falls back to the first couple of scalar getters available.
pub fn repr(slf: &Bound<'_, PyAny>) -> PyResult<String> {
    let cls_name: String = slf.get_type().name()?.extract()?;

    let fitted = is_fitted(slf);
    let has_is_fitted = slf.call_method0("is_fitted").is_ok();

    if has_is_fitted && !fitted {
        return Ok(format!("{}(fitted=False)", cls_name));
    }

    // Preferred headline fields, in priority order. Only those that exist and
    // resolve to a scalar are shown, capped to keep the repr short.
    const PREFERRED: &[&str] = &[
        "statistic",
        "p_value",
        "r_squared",
        "adj_r_squared",
        "score",
        "coefficients",
        "intercept",
        "n_observations",
        "n_features",
        "n_samples",
        "df",
        "aic",
    ];

    let mut parts: Vec<String> = Vec::new();
    if has_is_fitted {
        parts.push("fitted=True".to_string());
    }

    let available = property_names(slf)?;
    let mut shown = 0usize;
    for cand in PREFERRED {
        if shown >= 4 {
            break;
        }
        if !available.iter().any(|n| n == cand) {
            continue;
        }
        if let Ok(value) = slf.getattr(*cand) {
            if let Some(s) = repr_scalar(&value) {
                parts.push(format!("{}={}", cand, s));
                shown += 1;
            }
        }
    }

    // Fallback: if nothing headline matched, show the first scalar getters.
    if shown == 0 {
        for name in &available {
            if shown >= 3 {
                break;
            }
            if let Ok(value) = slf.getattr(name.as_str()) {
                if let Some(s) = repr_scalar(&value) {
                    parts.push(format!("{}={}", name, s));
                    shown += 1;
                }
            }
        }
    }

    Ok(format!("{}({})", cls_name, parts.join(", ")))
}

/// Generic, readable multi-line summary used by classes that do not supply a
/// bespoke domain-specific summary.  Lists every resolvable scalar / value
/// getter as `name: value`.
pub fn generic_summary(slf: &Bound<'_, PyAny>) -> PyResult<String> {
    let cls_name: String = slf.get_type().name()?.extract()?;

    if !is_fitted(slf) {
        return Ok(format!(
            "{name} (not fitted)\n{underline}\nCall `.fit(...)` to populate results.",
            name = cls_name,
            underline = "=".repeat(cls_name.len()),
        ));
    }

    let header = format!("{} Results", cls_name);
    let mut out = String::new();
    out.push_str(&header);
    out.push('\n');
    out.push_str(&"=".repeat(header.len()));
    out.push_str("\n\n");

    for name in property_names(slf)? {
        let value = match slf.getattr(name.as_str()) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let rendered = if let Some(s) = repr_scalar(&value) {
            s
        } else if value.is_none() {
            "None".to_string()
        } else {
            // numpy arrays / lists: use their own repr, collapsed to one line.
            match value.repr() {
                Ok(r) => {
                    let s: String = r.extract().unwrap_or_default();
                    s.replace('\n', " ")
                }
                Err(_) => continue,
            }
        };
        out.push_str(&format!("{:<24} {}\n", format!("{}:", name), rendered));
    }
    Ok(out)
}
