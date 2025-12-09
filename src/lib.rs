//! polars-statistics: Statistical testing and regression for Polars DataFrames.
//!
//! This crate provides Python bindings for statistical analysis with Polars,
//! wrapping the `regress-rs` and `anofox-statistics` Rust libraries.

use pyo3::prelude::*;
use pyo3_polars::PolarsAllocator;

mod expressions;
mod pymodels;
mod utils;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

/// Python module for polars-statistics.
#[pymodule]
fn _polars_statistics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Linear Models
    m.add_class::<pymodels::PyOLS>()?;
    m.add_class::<pymodels::PyRidge>()?;
    m.add_class::<pymodels::PyElasticNet>()?;
    m.add_class::<pymodels::PyWLS>()?;
    m.add_class::<pymodels::PyRLS>()?;
    m.add_class::<pymodels::PyBLS>()?;

    // GLM Models
    m.add_class::<pymodels::PyLogistic>()?;
    m.add_class::<pymodels::PyPoisson>()?;
    m.add_class::<pymodels::PyNegativeBinomial>()?;
    m.add_class::<pymodels::PyTweedie>()?;
    m.add_class::<pymodels::PyProbit>()?;
    m.add_class::<pymodels::PyCloglog>()?;

    // Bootstrap
    m.add_class::<pymodels::PyStationaryBootstrap>()?;
    m.add_class::<pymodels::PyCircularBlockBootstrap>()?;

    Ok(())
}
