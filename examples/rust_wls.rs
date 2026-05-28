//! Per-group WLS via the rlib path — mirrors the per-(site, part) fit loop
//! used by `kaercher-forecast` and the `anofox-orchestration` metalearner.
//!
//! Run with: `cargo run --example rust_wls --no-default-features`

use polars::prelude::*;
use polars_statistics::expressions::wls_fit;

fn main() -> PolarsResult<()> {
    // Synthetic panel: two sites, three observations each.
    // y_i ≈ intercept + slope * x_i, with per-site parameters.
    let df = df!(
        "site"   => &["A", "A", "A", "B", "B", "B"],
        "y"      => &[1.0_f64, 3.0, 5.0, 2.0, 5.0, 8.0],
        "weight" => &[1.0_f64, 1.0, 1.0, 1.0, 1.0, 1.0],
        "x1"     => &[0.0_f64, 1.0, 2.0, 0.0, 1.0, 2.0],
    )?;

    // Per-(site) WLS fit. Each partition is a contiguous DataFrame.
    let groups = df.partition_by(["site"], /* stable */ true)?;

    for group in groups {
        let site = group.column("site")?.str()?.get(0).unwrap_or("?");

        // wls_fit input contract:
        //   [0] y, [1] weights, [2] with_intercept (bool scalar),
        //   [3] solve_method (str scalar, nullable), [4..] x columns
        let y = group.column("y")?.as_materialized_series().clone();
        let w = group.column("weight")?.as_materialized_series().clone();
        let x1 = group.column("x1")?.as_materialized_series().clone();
        let with_intercept = Series::new("with_intercept".into(), &[true]);
        let solver = Series::new("solve_method".into(), &[None::<&str>]);

        let result = wls_fit(&[y, w, with_intercept, solver, x1])?;
        let fitted = result.struct_()?;

        let intercept = fitted
            .field_by_name("intercept")?
            .f64()?
            .get(0)
            .unwrap_or(f64::NAN);
        let coefs = fitted.field_by_name("coefficients")?;
        let r2 = fitted
            .field_by_name("r_squared")?
            .f64()?
            .get(0)
            .unwrap_or(f64::NAN);

        println!("site={site} intercept={intercept:.4} coefs={coefs:?} r2={r2:.4}");
    }

    Ok(())
}
