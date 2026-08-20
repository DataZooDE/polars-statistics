# Phase 4: Regression API Parity - Pattern Map

**Mapped:** 2026-08-12
**Files analyzed:** 20 new/modified files (12 new PyModel files, 1 modified expressions file,
1 modified Python exprs builder, 2 modified existing PyModels for HC extension,
2 modified existing PyModels for fit_from_accumulator, 3 shared registration files)
**Analogs found:** 20 / 20 (some are role-match only, no exact analogs for GLMM/PSpline/MomentAccumulator)

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `src/pymodels/py_gamma.rs` | PyModel | request-response | `src/pymodels/py_tweedie.rs` | exact |
| `src/pymodels/py_glmm.rs` | PyModel | request-response | `src/pymodels/py_poisson.rs` + `py_tweedie.rs` | partial |
| `src/pymodels/py_pspline.rs` | PyModel | request-response | `src/pymodels/py_ridge.rs` | role-match |
| `src/pymodels/py_theil_sen.rs` | PyModel | request-response | `src/pymodels/py_huber.rs` | exact |
| `src/pymodels/py_ransac.rs` | PyModel | request-response | `src/pymodels/py_huber.rs` | role-match |
| `src/pymodels/py_bayesian_ridge.rs` | PyModel | request-response | `src/pymodels/py_ridge.rs` | role-match |
| `src/pymodels/py_ard.rs` | PyModel | request-response | `src/pymodels/py_ridge.rs` | role-match |
| `src/pymodels/py_lars.rs` | PyModel | request-response | `src/pymodels/py_elastic_net.rs` | role-match |
| `src/pymodels/py_passive_aggressive.rs` | PyModel | event-driven (streaming) | `src/pymodels/py_huber.rs` | partial |
| `src/pymodels/py_moment_accumulator.rs` | PyModel (utility) | batch/streaming | none | no analog |
| `src/pymodels/py_ridge.rs` (extend HC) | PyModel | request-response | `src/pymodels/py_ols.rs` lines 355-412 | exact |
| `src/pymodels/py_wls.rs` (extend HC) | PyModel | request-response | `src/pymodels/py_ols.rs` lines 355-412 | exact |
| `src/pymodels/py_ols.rs` (extend accumulator) | PyModel | batch | `src/pymodels/py_ols.rs` itself | self |
| `src/pymodels/py_ridge.rs` (extend accumulator) | PyModel | batch | `src/pymodels/py_ols.rs` | role-match |
| `src/expressions/regression.rs` (5 new exprs) | expression | request-response | `regression.rs` lines 2189-2278 | exact |
| `python/polars_statistics/exprs/regression.py` (builders) | expression builder | request-response | `exprs/regression.py` lines 1484-1535 | exact |
| `src/pymodels/mod.rs` (registration) | config | — | `src/pymodels/mod.rs` lines 1-72 | exact |
| `src/lib.rs` (#[pymodule] registration) | config | — | `src/lib.rs` lines 30-85 | exact |
| `python/polars_statistics/__init__.py` (export) | config | — | `__init__.py` lines 6-50 | exact |
| `tests/test_*.py` (10 new smoke test files) | test | — | `tests/test_models.py` lines 1-60 | exact |

---

## Pattern Assignments

### `src/pymodels/py_gamma.rs` (PyModel, GLM-family)

**Analog:** `src/pymodels/py_tweedie.rs` (entire file, 237 lines)
**Rationale:** GammaRegressor is a thin re-skin of TweedieRegressor. PyTweedie is the exact structural template.

**Imports pattern** (`py_tweedie.rs` lines 1-8):
```rust
//! PyO3 wrapper for Tweedie regression.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::solvers::{FittedRegressor, FittedTweedie, Regressor, TweedieRegressor};

use crate::utils::{IntoNumpy, ToFaer};
```
For Gamma, replace the import line with:
```rust
use anofox_regression::solvers::{FittedGamma, FittedRegressor, GammaRegressor, Regressor};
```

**Struct + #[new] pattern** (`py_tweedie.rs` lines 36-68):
```rust
#[pyclass(name = "Gamma")]
pub struct PyGamma {
    with_intercept: bool,
    max_iter: usize,
    tol: f64,
    lambda_: f64,
    fitted: Option<FittedGamma>,
}

#[pymethods]
impl PyGamma {
    #[new]
    #[pyo3(signature = (with_intercept=true, max_iter=25, tol=1e-8, lambda_=0.0))]
    fn new(with_intercept: bool, max_iter: usize, tol: f64, lambda_: f64) -> Self {
        Self { with_intercept, max_iter, tol, lambda_, fitted: None }
    }
```

**fit pattern** (`py_tweedie.rs` lines 115-142):
```rust
fn fit<'py>(
    mut slf: PyRefMut<'py, Self>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<PyRefMut<'py, Self>> {
    let x_mat = x.to_faer();
    let y_col = y.to_faer();

    let model = GammaRegressor::builder()
        .with_intercept(slf.with_intercept)
        .max_iterations(slf.max_iter)
        .tolerance(slf.tol)
        .lambda(slf.lambda_)
        .build();

    let fitted = model
        .fit(&x_mat, &y_col)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    slf.fitted = Some(fitted);
    Ok(slf)
}
```

**predict pattern** (`py_tweedie.rs` lines 144-158):
```rust
fn predict<'py>(
    &self,
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let fitted = self
        .fitted
        .as_ref()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
    let x_mat = x.to_faer();
    let predictions = fitted.predict(&x_mat);
    Ok(predictions.into_numpy(py))
}
```

**Getter pattern** (`py_tweedie.rs` lines 164-230):
```rust
#[getter]
fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let fitted = self.fitted.as_ref()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
    Ok(fitted.coefficients().into_numpy(py))
}

#[getter]
fn intercept(&self) -> PyResult<Option<f64>> {
    let fitted = self.fitted.as_ref()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
    Ok(fitted.intercept())
}

#[getter]
fn std_errors<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
    let fitted = self.fitted.as_ref()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
    Ok(fitted.result().std_errors.as_ref().map(|se| se.into_numpy(py)))
}

#[getter]
fn aic(&self) -> PyResult<Option<f64>> {
    let fitted = self.fitted.as_ref()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
    Ok(Some(fitted.result().aic))
}
```

**Extra Gamma-specific getter:** Add `converged` and `predict_eta`:
```rust
#[getter]
fn converged(&self) -> PyResult<bool> {
    let fitted = self.fitted.as_ref()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
    Ok(fitted.converged())
}

fn predict_eta<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'py, f64>)
    -> PyResult<Bound<'py, PyArray1<f64>>> {
    let fitted = self.fitted.as_ref()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
    Ok(fitted.predict_eta(&x.to_faer()).into_numpy(py))
}
```

---

### `src/pymodels/py_glmm.rs` (PyModel, GLMM — unique fit signature)

**Analog:** `src/pymodels/py_tweedie.rs` (factory staticmethod pattern, lines 70-113) +
RESEARCH.md Pattern 2 (GLMM group fit, lines 934-958)

**Key deviation from standard pattern:** `fit(x, y, group)` — not `fit(x, y)`.

**Struct pattern** (based on py_tweedie.rs lines 36-45, adapted):
```rust
#[pyclass(name = "GLMM")]
pub struct PyGLMM {
    family: String,          // "gaussian" | "poisson" | "binomial"
    with_intercept: bool,
    random_intercept: bool,
    random_slopes: Vec<usize>,
    reml: bool,
    max_iter: usize,
    tol: f64,
    fitted: Option<FittedGlmm>,
}
```

**Factory staticmethod pattern** (`py_tweedie.rs` lines 70-83):
```rust
#[staticmethod]
#[pyo3(signature = (with_intercept=true, reml=true, max_iter=100, tol=1e-8))]
fn gaussian(with_intercept: bool, reml: bool, max_iter: usize, tol: f64) -> Self {
    Self { family: "gaussian".into(), with_intercept, reml, max_iter, tol,
           random_intercept: true, random_slopes: vec![], fitted: None }
}
```

**fit with group array** (RESEARCH.md pattern 2, lines 934-953):
```rust
fn fit<'py>(
    mut slf: PyRefMut<'py, Self>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    group: Vec<u64>,   // accept u64; convert to usize portably
) -> PyResult<PyRefMut<'py, Self>> {
    let x_mat = x.to_faer();
    let y_col = y.to_faer();
    let group_usize: Vec<usize> = group.iter().map(|&g| g as usize).collect();
    let model = match slf.family.as_str() {
        "poisson" => GlmmRegressor::poisson(),
        "binomial" => GlmmRegressor::binomial(),
        _ => GlmmRegressor::gaussian(),
    }
    .with_intercept(slf.with_intercept)
    .random_intercept(slf.random_intercept)
    .random_slopes(slf.random_slopes.clone())
    .reml(slf.reml)
    .max_iterations(slf.max_iter)
    .tolerance(slf.tol)
    .build();
    let fitted = model.fit(&x_mat, &y_col, &group_usize)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    slf.fitted = Some(fitted);
    Ok(slf)
}
```

**fit_crossed with groups-of-groups** (RESEARCH.md pattern 2, lines 955-958):
```rust
fn fit_crossed<'py>(
    mut slf: PyRefMut<'py, Self>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    groups: Vec<Vec<u64>>,
) -> PyResult<PyRefMut<'py, Self>> {
    let groups_usize: Vec<Vec<usize>> = groups.iter()
        .map(|g| g.iter().map(|&v| v as usize).collect()).collect();
    let group_refs: Vec<&[usize]> = groups_usize.iter().map(|g| g.as_slice()).collect();
    // ...model.fit_crossed(&x_mat, &y_col, &group_refs)
}
```

**FactorSummary as list-of-dicts** (no existing analog — use PyDict pattern from `hc_inference`):
```rust
fn factors<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, pyo3::types::PyDict>>> {
    let fitted = self.fitted.as_ref()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
    fitted.factors().iter().map(|f| {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("n_levels", f.n_levels)?;
        d.set_item("sd", f.sd)?;
        d.set_item("blups", PyArray1::from_slice(py, &f.blups))?;
        Ok(d)
    }).collect()
}
```

---

### `src/pymodels/py_pspline.rs` (PyModel, smoother)

**Analog:** `src/pymodels/py_huber.rs` (builder pattern, fit/predict/is_fitted/getters).
Extra getters: `edf`, `sigma2` from `FittedPSpline`.

**Key deviation:** Validate `x.ncols() == 1` before calling `.to_faer()`:
```rust
fn fit<'py>(
    mut slf: PyRefMut<'py, Self>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<PyRefMut<'py, Self>> {
    if x.shape()[1] != 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "PSpline requires a single-column predictor matrix (n, 1)."
        ));
    }
    // ...standard pattern follows
}
```

**Builder construction pattern** (PSplineRegressor uses method-chaining, not `.builder()`):
```rust
let model = PSplineRegressor::new()
    .with_n_basis(slf.n_basis)
    .with_penalty_order(slf.penalty_order);
let model = if let Some(lam) = slf.lambda { model.with_lambda(lam) } else { model };
let model = model.build();
```

**edf / sigma2 getters** (follow `py_huber.rs` `scale` getter pattern, lines 120-127):
```rust
#[getter]
fn edf(&self) -> PyResult<f64> {
    let fitted = self.fitted.as_ref()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
    Ok(fitted.edf())
}
```

---

### `src/pymodels/py_theil_sen.rs` (PyModel, robust)

**Analog:** `src/pymodels/py_huber.rs` (entire file, 201 lines) — exact structural match.
TheilSen has no inference table (NaN for f_statistic/aic/bic) — expose them via `result()` as-is.

**Key difference from Huber:** builder params differ; no `scale`/`epsilon`/`outliers`; has `r_squared`:
```rust
#[pyclass(name = "TheilSen")]
pub struct PyTheilSen {
    with_intercept: bool,
    max_subpopulation: usize,
    n_subsamples: Option<usize>,
    max_iter: usize,
    tol: f64,
    random_state: u64,
    fitted: Option<FittedTheilSen>,
}
```
Builder call:
```rust
let mut b = TheilSenRegressor::builder()
    .with_intercept(slf.with_intercept)
    .max_subpopulation(slf.max_subpopulation)
    .max_iter(slf.max_iter)
    .tolerance(slf.tol)
    .random_state(slf.random_state);
if let Some(ns) = slf.n_subsamples { b = b.n_subsamples(ns); }
let model = b.build();
```

**r_squared, mse, rmse getters** (`py_huber.rs` lines 163-200):
```rust
#[getter]
fn r_squared(&self) -> PyResult<f64> {
    let fitted = self.fitted.as_ref()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
    Ok(fitted.result().r_squared)
}
```

---

### `src/pymodels/py_ransac.rs` (PyModel, robust + inlier_mask)

**Analog:** `src/pymodels/py_huber.rs` for overall structure.
**Extra:** `inlier_mask` getter returns `PyArray1<bool>` — use `PyArray1::from_slice` pattern from `py_huber.rs` `outliers` getter (lines 142-149):

```rust
#[getter]
fn inlier_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let fitted = self.fitted.as_ref()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
    Ok(PyArray1::from_slice(py, fitted.inlier_mask()))
}
```

Builder call (note optional params):
```rust
let mut b = RansacRegressor::builder()
    .with_intercept(slf.with_intercept)
    .max_trials(slf.max_trials)
    .stop_probability(slf.stop_probability)
    .random_state(slf.random_state);
if let Some(ms) = slf.min_samples { b = b.min_samples(ms); }
if let Some(rt) = slf.residual_threshold { b = b.residual_threshold(rt); }
if let Some(si) = slf.stop_n_inliers { b = b.stop_n_inliers(si); }
let model = b.build();
```

---

### `src/pymodels/py_bayesian_ridge.rs` + `src/pymodels/py_ard.rs` (PyModel, regularized)

**Analog:** `src/pymodels/py_huber.rs` overall structure; note `fit_intercept` not `with_intercept`.

**Struct pattern** (BayesianRidge):
```rust
#[pyclass(name = "BayesianRidge")]
pub struct PyBayesianRidge {
    fit_intercept: bool,
    max_iter: usize,
    tol: f64,
    alpha_1: f64, alpha_2: f64,
    lambda_1: f64, lambda_2: f64,
    alpha_init: Option<f64>,
    lambda_init: Option<f64>,
    fitted: Option<FittedBayesianRidge>,
}
```

**#[new] signature** (using `fit_intercept`, not `with_intercept`):
```rust
#[pyo3(signature = (fit_intercept=true, max_iter=300, tol=1e-3,
                    alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6,
                    alpha_init=None, lambda_init=None))]
```

**Extra getters for BayesianRidge:**
```rust
#[getter]
fn alpha_(&self) -> PyResult<f64> { /* fitted.alpha() */ }
#[getter]
fn lambda_(&self) -> PyResult<f64> { /* fitted.lambda() */ }
#[getter]
fn sigma_diag<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // Col<f64> analog: use IntoNumpy on a Col created from sigma_diag slice
}
```

**For PyARD:** same structure, add `threshold_lambda: f64` field; `lambdas` getter returns `PyArray1<f64>`:
```rust
#[getter]
fn lambdas<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let fitted = self.fitted.as_ref()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
    Ok(PyArray1::from_slice(py, fitted.lambdas()))
}
```

---

### `src/pymodels/py_lars.rs` (PyModel, path-based regularized)

**Analog:** `src/pymodels/py_elastic_net.rs` (lines 1-60) for alpha/method string param pattern.

**String-dispatch for LarsMethod** (same pattern as HcType dispatch in `py_ols.rs` lines 382-387):
```rust
let method = match slf.method.as_str() {
    "lasso" => LarsMethod::Lasso,
    _ => LarsMethod::Lar,
};
```

**alphas getter** (returns path, use `PyArray1::from_slice`):
```rust
#[getter]
fn alphas<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let fitted = self.fitted.as_ref()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
    Ok(PyArray1::from_slice(py, fitted.alphas()))
}
```

Note: `fit_intercept` (not `with_intercept`) in builder.

---

### `src/pymodels/py_passive_aggressive.rs` (PyModel, streaming)

**Analog:** `src/pymodels/py_huber.rs` for the batch `fit` path.
**No direct analog** for `partial_fit` + `PaState` pattern.

**Struct with optional PaState:**
```rust
#[pyclass(name = "PassiveAggressive")]
pub struct PyPassiveAggressive {
    c: f64, epsilon: f64, with_intercept: bool,
    max_iter: usize, tol: f64, shuffle: bool,
    loss: String,   // "epsilon_insensitive" | "squared_epsilon_insensitive"
    random_state: u64,
    fitted: Option<FittedPassiveAggressive>,
    state: Option<PaState>,  // for partial_fit
}
```

**PaLoss string dispatch** (same pattern as LarsMethod):
```rust
let loss = match slf.loss.as_str() {
    "squared_epsilon_insensitive" => PaLoss::SquaredEpsilonInsensitive,
    _ => PaLoss::EpsilonInsensitive,
};
```

**partial_fit method:**
```rust
fn partial_fit(&mut self, x_row: PyReadonlyArray1<f64>, y_value: f64) -> PyResult<()> {
    let n_features = x_row.len();
    let state = self.state.get_or_insert_with(|| PaState::new(n_features));
    let model = /* build from self fields */;
    model.partial_fit(state, x_row.as_slice().unwrap(), y_value)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}
```

---

### `src/pymodels/py_moment_accumulator.rs` (PyModel, utility/streaming)

**Analog:** None. Unique utility class. No existing PyModel wraps a non-regressor.

**Struct pattern** (closest: private field + getter convention from any PyModel):
```rust
#[pyclass(name = "MomentAccumulator")]
pub struct PyMomentAccumulator {
    inner: MomentAccumulator,
}

#[pymethods]
impl PyMomentAccumulator {
    #[new]
    fn new(n_features: usize) -> Self {
        Self { inner: MomentAccumulator::new(n_features) }
    }

    fn push_row(&mut self, x_row: PyReadonlyArray1<f64>, y: f64) -> PyResult<()> {
        // Note: push_row takes &[f64], NOT Col<f64> — use as_slice(), not to_faer()
        self.inner.push_row(x_row.as_slice().unwrap(), y)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    fn merge(&mut self, other: &PyMomentAccumulator) -> PyResult<()> {
        self.inner.merge(&other.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    fn clear(&mut self) { self.inner.clear(); }

    #[getter] fn n(&self) -> usize { self.inner.n() }
    #[getter] fn n_features(&self) -> usize { self.inner.n_features() }
    #[getter] fn sum_y(&self) -> f64 { self.inner.sum_y() }

    #[getter]
    fn xtx<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        // Mat<f64> → 2D numpy: use IntoNumpy2 if available, else manual ndarray
        self.inner.xtx().into_numpy(py)
    }
}
```

**`fit_from_accumulator` addition to `PyOLS` and `PyRidge`** — add method to their existing `#[pymethods]` block:
```rust
fn fit_from_accumulator<'py>(
    mut slf: PyRefMut<'py, Self>,
    acc: &PyMomentAccumulator,
) -> PyResult<PyRefMut<'py, Self>> {
    // OlsRegressor::fit_from_accumulator(&acc.inner) — verify method name at implementation time
    let model = OlsRegressor::builder()
        .with_intercept(slf.with_intercept)
        .build();
    let fitted = model.fit_from_accumulator(&acc.inner)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    slf.fitted = Some(fitted);
    Ok(slf)
}
```

---

### `src/pymodels/py_ridge.rs` + `src/pymodels/py_wls.rs` — HC extension

**Analog:** `src/pymodels/py_ols.rs` lines 355-412 (copy verbatim, adjust types)

```rust
// Add to PyRidge and PyWLS #[pymethods] blocks
#[pyo3(signature = (x, hc_type="hc1"))]
fn hc_inference<'py>(
    &self,
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    hc_type: &str,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let fitted = self.fitted.as_ref()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
    let hc = match hc_type {
        "hc0" => HcType::HC0,
        "hc2" => HcType::HC2,
        "hc3" => HcType::HC3,
        _ => HcType::HC1,
    };
    let x_mat = x.to_faer();
    let residuals = fitted.result().residuals.clone();  // Col<f64> from RegressionResult
    let coef = fitted.result().coefficients.clone();
    let intercept = fitted.intercept();
    let n = fitted.result().n_observations;
    let p = fitted.result().n_parameters;
    let aliased = vec![false; x_mat.ncols()];
    let result = compute_hc_inference(
        &x_mat, &residuals, &aliased, /* with_intercept= */ true, hc,
        &coef, intercept, 0.95, n - p,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("std_errors", result.std_errors.into_numpy(py))?;
    dict.set_item("t_statistics", result.t_statistics.into_numpy(py))?;
    dict.set_item("p_values", result.p_values.into_numpy(py))?;
    dict.set_item("conf_interval_lower", result.conf_interval_lower.into_numpy(py))?;
    dict.set_item("conf_interval_upper", result.conf_interval_upper.into_numpy(py))?;
    if let Some(ref int_inf) = result.intercept {
        dict.set_item("intercept_std_error", int_inf.std_error)?;
        dict.set_item("intercept_t_statistic", int_inf.t_statistic)?;
        dict.set_item("intercept_p_value", int_inf.p_value)?;
    }
    Ok(dict)
}
```

**Required new import in py_ridge.rs / py_wls.rs:**
```rust
use anofox_regression::{HcType};
use anofox_regression::inference::compute_hc_inference;
```

---

### `src/expressions/regression.rs` — 5 new GLM diagnostic expressions

**Analog:** `src/expressions/regression.rs` lines 2189-2278 (`logistic_pearson_residuals_fit` + `poisson_pearson_residuals_fit` pattern)

**Pattern to copy** (lines 2189-2207):
```rust
pub fn logistic_pearson_residuals_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let lambda = inputs[1].f64()?.get(0).unwrap_or(0.0);
    let with_intercept = inputs[2].bool()?.get(0).unwrap_or(true);
    let (x, y) = match build_xy_data(inputs, 0, 3) {
        Ok(d) => d,
        Err(_) => return residual_diag_nan_output(),
    };
    let n = y.nrows();
    let model = build_binomial_logit(lambda, with_intercept);
    match model.fit(&x, &y) {
        Ok(f) => residual_diag_output(col_into_vec(&f.pearson_residuals()), n),
        Err(_) => residual_diag_nan_output(),
    }
}

#[polars_expr(output_type_func=residual_diag_output_dtype)]
fn pl_logistic_pearson_residuals(inputs: &[Series]) -> PolarsResult<Series> {
    logistic_pearson_residuals_fit(inputs)
}
```

**Naming convention for new expressions** (decision: per-family separate functions per RESEARCH.md open question resolution):
- `gamma_standardized_pearson_residuals_fit` / `pl_gamma_standardized_pearson_residuals`
- `gamma_standardized_deviance_residuals_fit` / `pl_gamma_standardized_deviance_residuals`
- `gamma_dispersion_deviance_fit` / `pl_gamma_dispersion_deviance` (scalar output)
- `gamma_dispersion_pearson_fit` / `pl_gamma_dispersion_pearson` (scalar output)
- `gamma_pearson_chi_squared_fit` / `pl_gamma_pearson_chi_squared` (scalar output)

**Output dtype for scalar outputs:** `chi_squared_output_dtype` already exists (line 2038 in regression.rs) — reuse for single-float diagnostics.

**Output dtype for per-row residuals:** `residual_diag_output_dtype` already exists (line 1589) — reuse for standardized residuals.

---

### Python expression builders — `python/polars_statistics/exprs/regression.py`

**Analog:** `exprs/regression.py` lines 1484-1535 (`logistic_pearson_residuals` + `poisson_pearson_residuals`)

```python
def gamma_standardized_pearson_residuals(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Standardized Pearson residuals from an internal Gamma GLM fit."""
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_gamma_standardized_pearson_residuals",
        args=_glm_residual_args(y, x, lambda_, add_intercept),
        returns_scalar=True,
    )
```

---

## Shared Patterns

### Standard PyModel skeleton
**Source:** `src/pymodels/py_tweedie.rs` (entire file) and `src/pymodels/py_huber.rs` (entire file)
**Apply to:** All 10 new PyModel files

Every PyModel must contain:
1. `#[pyclass(name = "ClassName")]` struct with `fitted: Option<FittedXxx>` private field
2. `#[new]` with `#[pyo3(signature = (...))]` listing all defaults
3. `fit<'py>(mut slf: PyRefMut<'py, Self>, x: PyReadonlyArray2<'py, f64>, y: PyReadonlyArray1<'py, f64>) -> PyResult<PyRefMut<'py, Self>>`
4. `predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray1<f64>>>`
5. `fn is_fitted(&self) -> bool { self.fitted.is_some() }`
6. All `#[getter]` methods guard with the same `.ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?` pattern

### Error handling
**Source:** `src/pymodels/py_tweedie.rs` line 138, `py_huber.rs` line 73
**Apply to:** All new PyModel files — all `model.fit()` errors:
```rust
.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
```
All "not fitted" guard:
```rust
.ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?
```

### numpy↔faer bridge
**Source:** `src/pymodels/py_tweedie.rs` lines 120-121, 154-155
**Apply to:** All new PyModel files
```rust
use crate::utils::{IntoNumpy, ToFaer};
// Array2 → Mat<f64>:
let x_mat = x.to_faer();
// Col<f64> → Array1:
Ok(col.into_numpy(py))
// &[bool] → PyArray1<bool>:
Ok(PyArray1::from_slice(py, slice))
// NOTE: MomentAccumulator push_row takes &[f64] — use x_row.as_slice().unwrap(), NOT to_faer()
```

### Builder + optional param pattern
**Source:** `src/pymodels/py_tweedie.rs` lines 129-133, `py_ols.rs` lines 91-103
**Apply to:** All models with optional builder params (PSpline lambda, TheilSen n_subsamples, RANSAC optional thresholds):
```rust
let mut builder = XxxRegressor::builder().param1(v1).param2(v2);
if let Some(opt) = slf.optional_field { builder = builder.optional_setter(opt); }
let model = builder.build();
```

### String enum dispatch
**Source:** `src/pymodels/py_ols.rs` lines 95-101 (SolverType), lines 382-387 (HcType)
**Apply to:** LarsMethod, PaLoss, GLMM family string — never register enums as `#[pyclass]`
```rust
let variant = match slf.method_str.as_str() {
    "variant_a" => Enum::VariantA,
    _ => Enum::Default,
};
```

---

## Shared Registration Files (must-serialize — single task per wave)

These three files are touched by every new PyModel added. Each wave must dedicate **one single
registration task** that edits all three AFTER the PyModel Rust files are written. Never split
registration across multiple tasks in the same wave.

### `src/pymodels/mod.rs`
**Source:** Lines 1-72 (full file)
**Pattern:** Add two lines per new model — one `mod` declaration in the regression models section,
one `pub use` export in the regression model exports section:
```rust
// In "Regression models" mod declarations section (lines 3-24):
mod py_gamma;

// In "Regression model exports" section (lines 38-59):
pub use py_gamma::PyGamma;
```

### `src/lib.rs` `#[pymodule]` block
**Source:** Lines 29-85
**Pattern:** Add one `m.add_class::<pymodels::PyXxx>()?;` under the appropriate comment section.
New GLM-family models go under the `// GLM Models` comment (line 45):
```rust
// GLM Models
m.add_class::<pymodels::PyGamma>()?;
m.add_class::<pymodels::PyGLMM>()?;
```
New robust/sklearn solvers go under a new `// Robust & Sklearn-Style Solvers` comment after line 44.

### `python/polars_statistics/__init__.py`
**Source:** Lines 6-50 (import block), lines 242-470 (`__all__` list)
**Pattern:** Add to BOTH the import block and the `__all__` list:
```python
# In import block (after existing GLM imports):
from polars_statistics._polars_statistics import (
    ...
    Gamma,
    GLMM,
    PSpline,
    TheilSen,
    RANSAC,
    BayesianRidge,
    ARD,
    LARS,
    PassiveAggressive,
    MomentAccumulator,
)

# In __all__ list (same names):
"Gamma",
"GLMM",
# etc.
```

---

## Test File Pattern

### All new `tests/test_*.py` smoke tests
**Analog:** `tests/test_models.py` lines 1-60

**Template for each new test file:**
```python
"""Smoke tests for XxxRegressor PyModel."""
import numpy as np
import pytest
from polars_statistics import Xxx

class TestXxx:
    def test_fit_basic(self):
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X @ np.array([1.0, 2.0]) + np.random.randn(50) * 0.1
        model = Xxx().fit(X, y)
        assert model.is_fitted()
        assert len(model.coefficients) == 2
        assert np.all(np.isfinite(model.coefficients))

    def test_predict_shape(self):
        np.random.seed(0)
        X = np.random.randn(30, 2)
        y = X @ np.array([1.0, -1.0]) + 0.05 * np.random.randn(30)
        model = Xxx().fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (30,)
        assert np.all(np.isfinite(preds))

    def test_not_fitted_raises(self):
        model = Xxx()
        with pytest.raises(Exception):
            _ = model.coefficients
```

**GLMM deviations:**
```python
from polars_statistics import GLMM
group = np.array([0, 0, 1, 1, 2, 2, ...], dtype=np.int64)  # one per obs
model = GLMM.gaussian().fit(X, y, group.tolist())
```

**MomentAccumulator deviations:**
```python
from polars_statistics import OLS, MomentAccumulator
acc = MomentAccumulator(n_features=2)
for i in range(50):
    acc.push_row(X[i], y[i])
model = OLS().fit_from_accumulator(acc)
assert model.is_fitted()
```

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `src/pymodels/py_moment_accumulator.rs` | PyModel utility | batch/streaming | No existing utility-class PyModel (all existing ones are regressors/test models) |
| `src/pymodels/py_glmm.rs` — `fit_crossed` method | PyModel | request-response | No multi-group fit signature exists in any current PyModel |
| `src/pymodels/py_passive_aggressive.rs` — `partial_fit` | PyModel | streaming | No online/partial-fit pattern exists in any current PyModel |

For these, the executor should follow the PyO3 PyRefMut pattern for `fit` (from py_tweedie.rs) and use `&mut self` for mutating methods like `partial_fit` and `push_row`.

---

## Metadata

**Analog search scope:** `src/pymodels/`, `src/expressions/regression.rs`,
`python/polars_statistics/exprs/regression.py`, `tests/`, `src/lib.rs`
**Files scanned:** 8 source files read in full; 2 read in targeted sections
**Pattern extraction date:** 2026-08-12
