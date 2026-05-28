//! polars-statistics: Statistical testing and regression for Polars DataFrames.
//!
//! This crate provides Polars expressions wrapping `anofox-regression` and
//! `anofox-statistics`. It is consumable two ways:
//!
//! - As an **rlib** by other Rust crates: `polars-statistics = "0.4"` and
//!   `use polars_statistics::expressions::*;` — no pyo3 linkage, no `python` feature.
//! - As a **cdylib** Python extension: built by `maturin` with the default
//!   `python` feature, producing the `_polars_statistics` module.

pub mod expressions;

#[cfg(feature = "python")]
mod pymodels;
#[cfg(feature = "python")]
mod utils;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3_polars::PolarsAllocator;

#[cfg(feature = "python")]
#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

/// Python module for polars-statistics.
#[cfg(feature = "python")]
#[pymodule]
fn _polars_statistics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Linear Models
    m.add_class::<pymodels::PyOLS>()?;
    m.add_class::<pymodels::PyRidge>()?;
    m.add_class::<pymodels::PyElasticNet>()?;
    m.add_class::<pymodels::PyWLS>()?;
    m.add_class::<pymodels::PyRLS>()?;
    m.add_class::<pymodels::PyBLS>()?;
    m.add_class::<pymodels::PyPls>()?;

    // Robust Regression
    m.add_class::<pymodels::PyQuantile>()?;
    m.add_class::<pymodels::PyIsotonic>()?;
    m.add_class::<pymodels::PyHuber>()?;

    // GLM Models
    m.add_class::<pymodels::PyLogistic>()?;
    m.add_class::<pymodels::PyLogisticRegression>()?;
    m.add_class::<pymodels::PyPoisson>()?;
    m.add_class::<pymodels::PyNegativeBinomial>()?;
    m.add_class::<pymodels::PyTweedie>()?;
    m.add_class::<pymodels::PyProbit>()?;
    m.add_class::<pymodels::PyCloglog>()?;

    // Augmented Linear Model
    m.add_class::<pymodels::PyALM>()?;

    // Dynamic Linear Model
    m.add_class::<pymodels::PyLmDynamic>()?;

    // Demand Classification
    m.add_class::<pymodels::PyAid>()?;
    m.add_class::<pymodels::PyAidResult>()?;

    // Bootstrap
    m.add_class::<pymodels::PyStationaryBootstrap>()?;
    m.add_class::<pymodels::PyCircularBlockBootstrap>()?;

    // Parametric Tests
    m.add_class::<pymodels::PyTTestInd>()?;
    m.add_class::<pymodels::PyTTestPaired>()?;
    m.add_class::<pymodels::PyBrownForsythe>()?;
    m.add_class::<pymodels::PyYuenTest>()?;

    // Non-Parametric Tests
    m.add_class::<pymodels::PyMannWhitneyU>()?;
    m.add_class::<pymodels::PyWilcoxonSignedRank>()?;
    m.add_class::<pymodels::PyKruskalWallis>()?;
    m.add_class::<pymodels::PyBrunnerMunzel>()?;

    // Distributional Tests
    m.add_class::<pymodels::PyShapiroWilk>()?;
    m.add_class::<pymodels::PyDAgostino>()?;

    Ok(())
}
