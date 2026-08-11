# Codebase Concerns

**Analysis Date:** 2026-08-11

## Tech Debt

### Monolithic Regression Module

**Issue:** `src/expressions/regression.rs` has grown to 5,496 lines, making it difficult to navigate, test, and maintain specific regression types.

**Files:** `src/expressions/regression.rs`

**Impact:**
- Long file makes code review and bug fixes slower
- Related functions spread across 5000+ lines reduce code cohesion
- IDE navigation and refactoring become problematic
- Single point of failure — any corruption affects all regression models

**Fix approach:**
- Split into modules by model type: `regression/ols.rs`, `regression/glm.rs`, `regression/robust.rs`, `regression/diagnostics.rs`
- Create internal `build_xy_with_null_policy()` in shared module for reuse
- Extract output type definitions to separate module
- Maintain backwards compatibility via re-exports in `mod.rs`

### Repeated Output Structure Builders

**Issue:** Similar struct output construction patterns repeated across multiple expression modules.

**Files:**
- `src/expressions/categorical.rs` (lines 26–56, 59–71, 74–90)
- `src/expressions/correlation.rs` (lines 14–42, 45–64)
- `src/expressions/regression.rs` (many locations)

**Impact:**
- Inconsistency in field ordering or data type handling
- Harder to update all tests when output schema changes
- Increases maintenance burden

**Fix approach:**
- Create `src/utils/output_builders.rs` with builder structs for common patterns
- Example: `CorrelationOutputBuilder::new(estimate).with_statistic(stat).with_pvalue(p).build()`
- Use trait implementations to reduce repetition

### Array Conversion `.expect()` Calls

**Issue:** `src/utils/array_conversion.rs:67` uses `.expect("Failed to create 2D array")` in PyArray2 conversion without alternative handling.

**Files:** `src/utils/array_conversion.rs:67`

**Impact:**
- Python-side crashes not caught gracefully
- No recovery mechanism if array conversion fails (e.g., invalid dimensions)
- Exception message may not help users debug

**Fix approach:**
- Return `PyResult<Bound<'py, PyArray2<f64>>>` to propagate Python exceptions
- Add validation before conversion: check shape consistency
- Convert expectation to explicit error with context: `PyArray2::from_vec2(py, &data).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid array shape: {}", e)))`

## Known Bugs

### Unsafe `.unwrap()` on Index Access in Regression

**Issue:** Multiple `.unwrap()` calls on index access assume validity without verification.

**Symptoms:**
- Panic at runtime if `valid_indices` or `y_series.get()` return `None`
- Occurs during regression fitting with certain null handling policies

**Files:**
- `src/expressions/regression.rs:3619` — `y_series.get(valid_indices[i]).unwrap()`
- `src/expressions/regression.rs:3677` — same pattern
- `src/expressions/regression.rs:3683` — `.unwrap()` after `.and_then(|ca| ca.get(valid_indices[row]))`

**Trigger:**
1. Apply "drop" null policy with regression
2. If `valid_mask` construction misses edge case (e.g., empty column chunks)
3. Index into `y_series` at position that doesn't exist

**Workaround:** None — will panic.

**Why it's safe (for now):**
- `valid_indices` is constructed from `valid_mask.iter().enumerate().filter_map(...)`
- Each index in `valid_indices` only included if corresponding `valid_mask[i] == true`
- `y_series.get()` checked before building `valid_indices`
- However, implicit contract fragile — not documented

**Fix approach:**
- Convert to `?` operator with descriptive error: 
  ```rust
  let y_val = y_series.get(valid_indices[i])
    .ok_or_else(|| polars_err!(ComputeError: "Lost valid row at index {}", valid_indices[i]))?;
  ```
- Add integration test with edge case: empty groups, single-row groups, all nulls in subset

### ICC Placeholder Always Returns NaN

**Issue:** Intraclass correlation coefficient (ICC) function is a stub that returns NaN for all outputs.

**Symptoms:** ICC always shows `NaN` regardless of input, making results unusable.

**Files:** `src/expressions/correlation.rs:270–304`

**Current behavior:**
```rust
let estimate = Series::new("estimate".into(), &[f64::NAN]);
let statistic = Series::new("statistic".into(), &[f64::NAN]);
let p_value = Series::new("p_value".into(), &[f64::NAN]);
```

**Impact:** Users calling `ps.icc()` get no useful output. Breaking because documented in API but non-functional.

**Fix approach:**
1. Implement proper ICC using matrix input (subjects × raters)
2. Add variant support: ICC(1,1), ICC(2,1), ICC(3,1), ICC(2,k), ICC(3,k)
3. Use existing `anofox-statistics` for calculation if available, else implement from Shrout & Fleiss (1979)
4. Add tests validating against R `irr::icc()` output

## Error Handling Deficiencies

### Implicit Unwrap on Series Type Conversion

**Issue:** Many `.f64()?.get(0).unwrap_or(default)` patterns assume type is correct after `?` check but provide no validation.

**Files:** `src/expressions/regression.rs` (lines 534, 580, 581, 627–629, 675, 724–725, 769, 819–823, 872–877, 927–930)

**Impact:**
- Silent failures if parameter extraction returns `None` before `.get(0)`
- Type errors absorbed by `?`, only error message from `.f64()` visible
- Example: `inputs[1].f64()?.get(0).unwrap_or(1.0)` fails silently if input not float

**Fix approach:**
- Extract to validation function:
  ```rust
  fn get_param_f64(inputs: &[Series], idx: usize, name: &str, default: f64) -> PolarsResult<f64> {
    inputs[idx]
      .f64()
      .map_err(|_| polars_err!(ComputeError: "Parameter {} must be float", name))?
      .get(0)
      .ok_or_else(|| polars_err!(ComputeError: "Parameter {} is empty", name))
      .or(Ok(default))
  }
  ```
- Use consistently across all parameter extraction

### No Bounds Checking on Matrix Operations

**Issue:** `.from_fn()` matrix construction relies on external validity; no bounds checking.

**Files:** `src/expressions/regression.rs:3622–3637`

**Impact:**
- If `n_features` or `n_rows` computed incorrectly, matrix created with wrong shape
- Fitted model later fails or produces invalid predictions
- No error message pinpoints the root cause

**Fix approach:**
- Add explicit validation after building matrices:
  ```rust
  if x_fit.nrows() != y.nrows() {
    return Err(polars_err!(ComputeError: 
      "Feature matrix {} rows, target {} rows", x_fit.nrows(), y.nrows()));
  }
  ```

## Performance Bottlenecks

### Data Cloning in Array Conversion

**Issue:** `src/utils/array_conversion.rs` clones data twice for f64 conversion.

**Files:** `src/utils/array_conversion.rs:44–46` (Col), `src/utils/array_conversion.rs:64–66` (Mat)

**Current pattern:**
```rust
let data: Vec<f64> = (0..len).map(|i| self[i]).collect();  // Clone #1
PyArray1::from_vec(py, data)  // Move into PyArray
```

**Impact:**
- For large matrices (1000 × 1000), allocates 2MB twice before deallocation
- Reduces throughput in repeated model fitting
- Noticeable with `over()` on wide data or large `group_by()` groups

**Fix approach:**
- Use `PyArray1::from_slice()` with temporary storage when possible
- Benchmark vs. current approach — may be acceptable for typical dataset sizes
- Document decision

### Redundant Valid Mask Iteration

**Issue:** `build_xy_with_null_policy()` iterates valid_mask twice in "drop" path.

**Files:** `src/expressions/regression.rs:3664–3668`

**Pattern:**
```rust
let valid_indices: Vec<usize> = valid_mask
  .iter()
  .enumerate()
  .filter_map(|(i, &v)| if v { Some(i) } else { None })
  .collect();
```

Used again later for every access to y_series/X columns.

**Impact:**
- For dataset with 1M rows and 50K valid rows, creates 1M allocation for mask + 50K for indices
- Regression per group with many groups multiplies this overhead

**Fix approach:**
- Reuse `valid_indices` throughout function
- Consider `SmallVec<[usize; 256]>` if most groups are small

## Security Considerations

### No Input Validation on Regression Inputs

**Risk:** User can pass arbitrary data shapes without size validation.

**Files:** `src/expressions/regression.rs` (all fitting functions)

**Current mitigation:** None explicit — relies on downstream `anofox-regression` crate.

**Recommendations:**
1. Add checks in `build_xy_with_null_policy()`:
   - Minimum n_observations > n_features (or even 3× for safe inference)
   - Warn if n_observations < 30 (small sample)
   - Max matrix size guard (e.g., 1B elements) to prevent OOM
2. Document assumptions in docstrings
3. Return `Err` with helpful message if constraints violated

### Python -> Rust Deserialization Not Validated

**Issue:** Strings passed from Python (e.g., solver type, null policy) validated only by string matching.

**Files:**
- `src/expressions/regression.rs:35–50` — `parse_solver_type()`, `parse_hc_type()`
- Multiple `match` statements with `_` fallback to defaults

**Risk:**
- Silent fallback to default if user typo (e.g., "cholesky" → "qr")
- No error message explains why

**Recommendations:**
- Return `Err` instead of `None` on unknown value:
  ```rust
  fn parse_solver_type(s: Option<&str>) -> PolarsResult<SolverType> {
    match s {
      Some("svd") => Ok(SolverType::Svd),
      Some("cholesky") => Ok(SolverType::Cholesky),
      Some(other) => Err(polars_err!(ComputeError: "Unknown solver: {}", other)),
      None => Ok(SolverType::Qr),
    }
  }
  ```
- Migrate callers to `?` operator
- Add tests for invalid string inputs

## Fragile Areas

### Null Policy Logic Complexity

**Files:** `src/expressions/regression.rs:3583–3699` (build_xy_with_null_policy)

**Why fragile:**
- Two code paths ("zero_fill" and "drop") with nearly identical setup but different suffix logic
- 117 lines of conditional index management
- Implicit assumption that valid_indices stay in sync with valid_mask
- No assertion that indices are actually used

**Safe modification:**
1. Extract common parts into helper:
   ```rust
   fn build_valid_mask(inputs: &[Series], y_idx: usize, x_start: usize, n_features: usize) -> PolarsResult<Vec<bool>>
   ```
2. Extract "drop" path logic into separate function
3. Add integration tests for edge cases:
   - All nulls in one column
   - Empty groups
   - Single row groups
   - Alternating null patterns

**Test coverage:** See test gaps below.

### GLM Link Function String Parsing

**Files:** `src/expressions/regression.rs` (all GLM expressions use string-based link functions)

**Why fragile:**
- No centralized validation of link function names
- Each GLM implementation redoes the string matching
- Case-sensitive string comparison vulnerable to typos

**Safe modification:**
- Create enum `LinkFunctionSpec` with validated string parser
- Use throughout GLMs
- Add validation tests

## Test Coverage Gaps

### Null Handling Edge Cases

**What's not tested:** 
- All nulls in X for "zero_fill" (should work, fill with 0)
- All nulls in y for "drop" (should error with "No valid rows")
- Groups with only 1 valid row (underdetermined system)
- Alternating null patterns in wide data

**Files to test:**
- `tests/test_group_by_over.py` (expand with null scenarios)
- Add Rust test in `tests/rust_api.rs` for edge cases

**Priority:** High — null handling is common data quality issue

### Regression Diagnostics with Small Samples

**What's not tested:**
- n_observations = n_features (perfect fit)
- n_observations < n_features (underdetermined)
- Single-row groups from groupby

**Files:** `tests/test_diagnostics_toolkit.py`, `tests/test_residual_diagnostics.py`

**Priority:** Medium — advanced users may hit these

### ICC Function (Placeholder)

**What's not tested:** Anything — function is non-functional.

**Files:** `tests/test_correlation.py` (expand to test ICC variants)

**Priority:** Critical — feature is documented but broken

### Python Model Class Error Handling

**What's not tested:**
- Calling methods on unfitted models (should error gracefully)
- Invalid parameter combinations
- Extreme value inputs (very large/small numbers, infinities)

**Files:** Create `tests/test_model_errors.py`

**Priority:** Medium — robustness

## Missing Critical Features

### ICC Implementation

**Problem:** ICC stub always returns NaN. Users relying on this get unusable results.

**Blocks:** Any analysis requiring inter-rater reliability assessment.

**Implementation status:** TODO (see "Known Bugs" section for details)

### Matrix Input Support for Correlation Functions

**Problem:** Some correlations (ICC, multitrait-multimethod) require 2D matrix input; current API only supports column pairs.

**Blocks:** Advanced correlation studies.

**Approach:** Add matrix input variant to correlation expressions.

## Scaling Limits

### Memory Usage in Large Group-By Operations

**Current capacity:** Group-by with 100K+ groups on 1M-row dataset with 50+ features works but may hit memory limits.

**Limit:** 
- Each group allocates matrices: X_fit (n × p), X_pred (n × p), y (n × 1), valid_mask (n bits)
- For 1M rows, 50 features: ~800MB per group alone
- With 100K groups, OOM likely

**Scaling path:**
1. Add lazy evaluation for group-by (only compute groups requested in output)
2. Implement streaming regression (online updates) for very large groups
3. Add memory usage warnings in documentation

## Dependencies at Risk

### anofox-regression 0.5.4

**Risk:** External crate not widely used; single-maintainer dependency.

**Impact:** If maintainer stops supporting or introduces breaking changes, this crate blocked.

**Mitigation:**
- Monitor upstream issues: https://github.com/sipemu/anofox-regression-rs
- Keep changelog up-to-date for transitions
- Add vendoring option if critical

### anofox-statistics 0.4.1

**Risk:** Same as anofox-regression — tightly coupled.

**Impact:** Cross-dependency failures.

**Mitigation:** Same as above.

### pyo3 0.26 + numpy 0.26

**Risk:** PyO3 frequently releases breaking changes; numpy ABI shifts can cause wheel incompatibility.

**Impact:**
- Wheel breaks on new Python version
- maturin build fails silently on version mismatches

**Recommendations:**
1. Pin PyO3 to MSRV (minimum supported release)
2. Test wheel against multiple Python versions in CI
3. Add pre-built wheels for Python 3.9–3.13 to CI artifacts

## Documentation Gaps

### Null Handling Policy Not Clearly Documented

**Issue:** `null_policy` parameter in regression functions not explained in user-facing docs.

**Files:** `README.md` (silent), `docs/api/` (no mention)

**Impact:** Users unaware behavior differs between "drop" and "zero_fill" modes; silent behavior changes cause debugging headaches.

**Recommendations:**
1. Add section to README under "Advanced Usage"
2. Add docstring to each regression function explaining policy
3. Show examples with `null_policy="drop"` and `null_policy="zero_fill"`

### Regression Diagnostics Not Linked from Main Docs

**Issue:** Diagnostics (VIF, leverage, Cook's distance) exist but not highlighted as primary features.

**Impact:** Users may not discover them.

**Recommendations:**
1. Add "Diagnostics" section to README with examples
2. Link from regression examples to diagnostic functions

---

*Concerns audit: 2026-08-11*
