# Testing Patterns

**Analysis Date:** 2026-08-11

## Test Framework

**Runner:**
- Cargo built-in test runner (Rust standard)
- No explicit test framework dependency (uses `#[test]` attribute)
- Config: No `Cargo.toml` `[profile.test]` section detected

**Assertion Library:**
- Standard Rust `assert!`, `assert_eq!`, `assert_ne!`
- Custom helpers for struct field extraction (e.g., `assert_n_obs_nonzero`, `assert_f64_finite_or_zero`)

**Run Commands:**
```bash
cargo test --no-default-features --test rust_api       # Test Rust API (no Python feature)
cargo test                                              # All tests with default features
cargo test --lib                                        # Library tests only
cargo test --test rust_api -- --nocapture             # Run with output
```

## Test File Organization

**Location:**
- Primary: `tests/rust_api.rs` - integration test file for Rust API
- No unit tests in source files (no `#[cfg(test)]` modules detected)
- Pattern: Separate integration tests directory per Rust convention

**Naming:**
- Test functions: snake_case prefixed by test category
  - `linear_regression_fits` - Tests linear regression models
  - `glm_fits` - Tests generalized linear model variants
  - `diagnostic_fits` - Tests regression diagnostics
  - `parametric_tests` - Tests parametric hypothesis tests
  - `nonparametric_tests` - Tests nonparametric hypothesis tests
  - `distributional_tests` - Tests for normality/distribution
  - `correlation_fits` - Tests correlation methods
  - `categorical_fits` - Tests categorical/contingency table tests
  - `forecast_comparison_tests` - Tests forecast comparison metrics

**File structure:**
```
tests/
└── rust_api.rs          # 1851 lines, single comprehensive test file
```

## Test Structure

**Suite Organization:**
```rust
// From tests/rust_api.rs (lines 1-20)
//! Integration tests for the public Rust API exposed by `polars_statistics::expressions`.
//!
//! These tests exercise every `*_fit` function reachable from a downstream Rust crate
//! via the rlib path. They must compile and pass with:
//!
//! ```text
//! cargo test --no-default-features --test rust_api
//! ```
//!
//! Test depth:
//! - The four headline regressors (OLS / WLS / Ridge / Quantile) are validated against
//!   known coefficients on synthetic linear data.
//! - All other `*_fit` functions are smoke-tested for reachability and no-panic
//!   behaviour on legitimate inputs.
```

**Patterns:**

1. **Helper Functions** (lines 23-71):
   - Data generators: `linear_xy()`, `two_samples()` - return `(Vec<f64>, Vec<f64>)`
   - Series constructors: `series_f64(name, vals)`, `scalar_f64(name, v)`, `scalar_bool(name, v)`
   - Assertion helpers: `assert_n_obs_nonzero(s, field)`, `assert_f64_finite_or_zero(s, field)`

2. **Test Methods** - Each `#[test]` function:
   - Generates test data via helpers
   - Constructs input `Vec<Series>` slice with named parameters
   - Calls public `*_fit()` function directly
   - Asserts output struct fields or length

3. **Assertion Patterns**:
   ```rust
   // Coefficient validation (lines 138-143)
   assert!(
       (intercept - 1.0).abs() < 1e-6,
       "OLS intercept ~ 1.0, got {intercept}"
   );
   
   // Field presence and non-zero (line 144)
   assert_n_obs_nonzero(&out, "n_observations");
   
   // Output length match (line 342)
   assert_eq!(out.len(), y.len());
   ```

## Mocking

**Framework:** No mocking framework detected (mockall, mocktopus, etc.)

**Patterns:**
- No mocking used; tests work directly with public API
- Synthetic test data generated inline (linear, count, binary, continuous positive)
- No external dependencies mocked

**What to Mock:**
- Not applicable; this codebase does not use mocks

**What NOT to Mock:**
- Core anofox-regression and anofox-statistics dependencies are direct runtime deps
- Test data synthesized to avoid external I/O or service calls

## Fixtures and Factories

**Test Data:**
```rust
// Linear data: y = 1.0 + 2.0 * x (lines 23-28)
fn linear_xy() -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi).collect();
    (x, y)
}

// Two correlated samples (lines 31-35)
fn two_samples() -> (Vec<f64>, Vec<f64>) {
    let a: Vec<f64> = (0..25).map(|i| (i as f64) * 0.5 + 1.0).collect();
    let b: Vec<f64> = (0..25).map(|i| (i as f64) * 0.5 + 1.3).collect();
    (a, b)
}

// Binary outcome data (lines 586-590)
let x_bin: Vec<f64> = (0..20).map(|i| i as f64).collect();
let y_bin: Vec<f64> = x_bin
    .iter()
    .map(|&xi| if xi >= 10.0 { 1.0 } else { 0.0 })
    .collect();
```

**Location:**
- Inline in test functions
- No separate fixtures or factories module
- Hardcoded data for reproducibility and simplicity

## Coverage

**Requirements:** No enforced coverage targets

**View Coverage:**
- Use `cargo tarpaulin` (not configured in repo)
- Run via: `cargo tarpaulin --out Html`
- Coverage tracking not part of standard test workflow

## Test Types

**Unit Tests:**
- Scope: Individual `*_fit` functions
- Approach: Direct function calls with synthetic data
- Coverage:
  - Core linear models (OLS, Ridge, Quantile, WLS) tested with known coefficients
  - GLM models (Logistic, Poisson, NegBin, Tweedie) smoke-tested
  - Diagnostics (VIF, leverage, cooks distance, residuals) smoke-tested
  - Parametric tests (t-test, Brown-Forsythe) smoke-tested
  - Non-parametric tests (Mann-Whitney, Kruskal-Wallis) smoke-tested
  - Categorical tests (binomial, chi-square, Fisher exact) smoke-tested
  - Correlation tests (Pearson, Spearman, Kendall, partial/semi-partial) smoke-tested

**Integration Tests:**
- Scope: Cross-feature workflows
- Approach: Test multiple `*_fit` variants (e.g., `ols_fit`, `ols_summary_fit`, `ols_predict_fit` in one test)
- Coverage: Regression pipeline (fit → summary → predict), GLM pipeline, diagnostic chains
- Enabled when: Default features (includes Python) or `--no-default-features`

**E2E Tests:**
- Framework: Not used
- Reason: No end-to-end application workflow; crate is library-only
- Coverage: Python bindings tested via separate test suite (not in this repo)

## Common Patterns

**Smoke Testing:**
```rust
// Example: elastic_net_fit (lines 264-274)
{
    let inputs = vec![
        series_f64("y", &y),
        scalar_f64("lambda", 0.01),
        scalar_f64("alpha", 0.5),
        scalar_bool("with_intercept", true),
        series_f64("x1", &x),
    ];
    let out = elastic_net_fit(&inputs).expect("elastic_net_fit failed");
    assert_n_obs_nonzero(&out, "n_observations");
}
```
- Pattern: Call function, assert it doesn't panic, check basic field presence
- Used for: Functions where exact output values are unpredictable or expensive to validate

**Coefficient Validation:**
```rust
// Example: OLS exact fit (lines 138-143)
assert!(
    (intercept - 1.0).abs() < 1e-6,
    "OLS intercept ~ 1.0, got {intercept}"
);
assert!((slope - 2.0).abs() < 1e-6, "OLS slope ~ 2.0, got {slope}");
assert!((r2 - 1.0).abs() < 1e-6, "OLS R^2 ~ 1.0, got {r2}");
```
- Pattern: Use tolerance ranges (e.g., 1e-6 for exact fits, 1e-3 for regularized, 0.5 for robust fits)
- Used for: Core regression models validated against synthetic data with known solutions

**Error Testing:**
```rust
// Example: Function call with error expectation (line 119)
let out = ols_fit(&inputs).expect("ols_fit failed");
```
- Pattern: Use `.expect()` with descriptive message; panic on error during test
- No explicit error case testing; assumes valid inputs

**Parametric Testing:**
- Not detected (no parametrize/quickcheck used)
- Test data hardcoded per function
- Same input contract validated across fit/summary/predict variants

**Null Policy Testing:**
```rust
// Example: Predict with null_policy parameter (lines 338-343)
let inputs = vec![
    // ... data setup ...
    scalar_str("null_policy", "drop"),
    // ... features ...
];
let out = quantile_predict_fit(&inputs, pk).expect("quantile_predict_fit failed");
assert_eq!(out.len(), y.len());
```
- Pattern: Test both null-handling modes (typically "drop" vs others)
- Used for: Predict variants that handle missing data

---

*Testing analysis: 2026-08-11*
