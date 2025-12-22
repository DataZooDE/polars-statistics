//! Polars expression functions for statistical tests and regression.
//!
//! These functions use the #[polars_expr] macro to create expressions
//! that work with group_by and over operations.

mod categorical;
mod correlation;
mod distributional;
mod forecast;
mod modern;
mod nonparametric;
mod output_types;
mod parametric;
mod regression;
mod tost;

// Re-exports for polars plugin FFI - these are used via Python
#[allow(unused_imports)]
pub use categorical::*;
#[allow(unused_imports)]
pub use correlation::*;
#[allow(unused_imports)]
pub use distributional::*;
#[allow(unused_imports)]
pub use forecast::*;
#[allow(unused_imports)]
pub use modern::*;
#[allow(unused_imports)]
pub use nonparametric::*;
#[allow(unused_imports)]
pub use output_types::*;
#[allow(unused_imports)]
pub use parametric::*;
#[allow(unused_imports)]
pub use regression::*;
#[allow(unused_imports)]
pub use tost::*;
