//! Polars expression functions for statistical tests and regression.
//!
//! These functions use the #[polars_expr] macro to create expressions
//! that work with group_by and over operations.

mod output_types;
mod parametric;
mod nonparametric;
mod distributional;
mod forecast;
mod modern;
mod regression;

pub use output_types::*;
pub use parametric::*;
pub use nonparametric::*;
pub use distributional::*;
pub use forecast::*;
pub use modern::*;
pub use regression::*;
