# Phase 3: Statistics API Parity - Pattern Map

**Mapped:** 2026-08-11
**Files analyzed:** 8 (5 modified, 3 modified-with-new-additions)
**Analogs found:** 8 / 8

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/expressions/parametric.rs` | expression (3 new fns) | request-response | `src/expressions/parametric.rs` (existing ttest_ind_fit) | exact — same file, same pattern |
| `src/expressions/modern.rs` | expression (1 new fn) | request-response | `src/expressions/modern.rs` (energy_distance_fit) | exact — same file, same pattern |
| `src/expressions/correlation.rs` | expression (1 fn replaced) | request-response | `src/expressions/correlation.rs` (partial_cor_fit) | exact — variable-arity pattern in same file |
| `src/expressions/output_types.rs` | config / schema (4 new fns) | — | `src/expressions/output_types.rs` (tost_output_dtype) | exact — richest existing output_dtype in same file |
| `python/polars_statistics/exprs/parametric.py` | Python builder (3 new fns) | request-response | `python/polars_statistics/exprs/parametric.py` (ttest_ind) | exact — same file |
| `python/polars_statistics/exprs/modern.py` | Python builder (1 new fn) | request-response | `python/polars_statistics/exprs/modern.py` (energy_distance) | exact — same file, seed-nullable pattern |
| `python/polars_statistics/exprs/correlation.py` | Python builder (1 fn rewritten) | request-response | `python/polars_statistics/exprs/correlation.py` (partial_cor) | role-match — multi-expr varargs builder |
| `python/polars_statistics/exprs/__init__.py` | config / registration | — | itself (existing import blocks) | exact |
| `python/polars_statistics/__init__.py` | config / registration | — | itself (existing import + __all__ blocks) | exact |
| `tests/test_statistics_parity.py` | test (new file) | — | existing tests in `tests/` | role-match |

---

## Pattern Assignments

### `src/expressions/output_types.rs` — 4 new `*_output_dtype` functions

**Analog:** `src/expressions/output_types.rs` — `tost_output_dtype` (lines 28-41, the richest
existing schema) and `correlation_output_dtype` (lines 44-54).

**Existing output_dtype signature pattern** (lines 6-12 and 28-41):
```rust
pub fn stats_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("statistic".into(), DataType::Float64),
        Field::new("p_value".into(), DataType::Float64),
    ];
    Ok(Field::new("stats".into(), DataType::Struct(fields)))
}

pub fn tost_output_dtype(_input_fields: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("estimate".into(), DataType::Float64),
        Field::new("ci_lower".into(), DataType::Float64),
        Field::new("ci_upper".into(), DataType::Float64),
        Field::new("bound_lower".into(), DataType::Float64),
        Field::new("bound_upper".into(), DataType::Float64),
        Field::new("tost_p_value".into(), DataType::Float64),
        Field::new("equivalent".into(), DataType::Boolean),
        Field::new("alpha".into(), DataType::Float64),
        Field::new("n".into(), DataType::UInt32),
    ];
    Ok(Field::new("tost".into(), DataType::Struct(fields)))
}
```

**Copy this pattern for each new output_dtype.** The `UInt32` type (not Float64) is used for
count fields (n_groups, n_subjects, n_raters). The outer `Field::new` name matches the expression
result name (e.g., `"one_way_anova"`, `"two_way_anova"`, `"icc"`).

**Four new functions to add** (exact field lists from RESEARCH.md, verified against crate structs):

`one_way_anova_output_dtype` — fields: statistic (F64), df_between (F64), df_within (F64),
p_value (F64), ss_between (F64, NaN for Welch), ss_within (F64, NaN for Welch), ms_between (F64,
NaN for Welch), ms_within (F64, NaN for Welch), eta_squared (F64, computed, NaN for Welch),
n_groups (U32). Outer name: `"one_way_anova"`.

`two_way_anova_output_dtype` — fields: a_ss, a_df, a_ms, a_f, a_p_value, b_ss, b_df, b_ms, b_f,
b_p_value, ab_ss, ab_df, ab_ms, ab_f, ab_p_value (all F64); residual_ss, residual_df, residual_ms
(F64); grand_mean (F64); n (U32). Outer name: `"two_way_anova"`.

`repeated_measures_anova_output_dtype` — fields: ws_f, ws_df, ws_ss, ws_ms, ws_p_value, error_df,
error_ss, error_ms, mauchly_w, mauchly_p_value, gg_epsilon, gg_p_value, hf_epsilon, hf_p_value,
grand_mean (all F64). Outer name: `"repeated_measures_anova"`.

`icc_output_dtype` — fields: icc (F64), f_value (F64), df1 (F64), df2 (F64), p_value (F64),
ci_lower (F64), ci_upper (F64), n_subjects (U32), n_raters (U32). Outer name: `"icc"`.

Note: `energy_distance_nd` reuses the existing `stats_output_dtype` (statistic + p_value) — no
new output_dtype function needed.

---

### `src/expressions/parametric.rs` — 3 new fit functions + `#[polars_expr]` shims

**Analog:** `src/expressions/parametric.rs` `ttest_ind_fit` / `pl_ttest_ind` (lines 20-48) for
the overall function structure, plus `brown_forsythe_fit` (lines 83-97) for the multi-group slice
pattern.

**Imports to add** (copy existing import block at lines 1-8, add anofox-statistics ANOVA items):
```rust
use anofox_statistics::{t_test, yuen_test, Alternative, TTestKind};
// Add:
use anofox_statistics::parametric::{one_way_anova, two_way_anova, repeated_measures_anova, AnovaKind};
use crate::expressions::output_types::{
    generic_stats_output, stats_output_dtype,
    one_way_anova_output_dtype, two_way_anova_output_dtype, repeated_measures_anova_output_dtype,
};
```
(Assumption A2 in RESEARCH.md: if `AnovaKind` is at crate root, use `anofox_statistics::AnovaKind`
instead. Verify with `grep pub_use` on crate lib.rs before writing.)

**Core fit function pattern** (lines 20-48 — ttest_ind_fit):
```rust
pub fn ttest_ind_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;
    let y = inputs[1].f64()?;
    let alt_str = inputs[2].str()?.get(0).unwrap_or("two-sided");
    let equal_var = inputs[3].bool()?.get(0).unwrap_or(false);
    let mu = inputs[4].f64()?.get(0).unwrap_or(0.0);
    let conf_level = inputs[5].f64()?.get(0).unwrap_or(0.95);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();

    let alternative = parse_alternative(alt_str);
    let kind = if equal_var { TTestKind::Student } else { TTestKind::Welch };

    match t_test(&x_vec, &y_vec, kind, alternative, mu, Some(conf_level)) {
        Ok(result) => generic_stats_output(result.statistic, result.p_value, "ttest_ind"),
        Err(_) => generic_stats_output(f64::NAN, f64::NAN, "ttest_ind"),
    }
}

#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_ttest_ind(inputs: &[Series]) -> PolarsResult<Series> {
    ttest_ind_fit(inputs)
}
```

**Error-output helper pattern** (from `correlation.rs` lines 44-59 — `correlation_error_output`):
Each new ANOVA fit function needs its own error-output helper that returns the same struct shape
with all-NaN / 0 values. Copy the `correlation_error_output` structure:
```rust
fn one_way_anova_error_output() -> PolarsResult<Series> {
    let statistic = Series::new("statistic".into(), &[f64::NAN]);
    // ... all fields as NaN or 0u32 ...
    let df = StructChunked::from_series(
        "one_way_anova".into(), 1,
        [&statistic, /* ... */].into_iter(),
    )?;
    Ok(df.into_series())
}
```

**StructChunked construction pattern** (from `correlation.rs` lines 36-41):
```rust
let df = StructChunked::from_series(
    name.into(),
    1,
    [&field1, &field2, &field3].into_iter(),
)?;
Ok(df.into_series())
```
All ANOVA output builders use this exact `StructChunked::from_series` call. Length is always `1`
(scalar output per group).

**Parse-helper pattern** (lines 11-17, `parse_alternative`):
```rust
fn parse_alternative(s: &str) -> Alternative {
    match s.to_lowercase().as_str() {
        "less" => Alternative::Less,
        "greater" => Alternative::Greater,
        _ => Alternative::TwoSided,
    }
}
```
Add `parse_anova_kind` and (in correlation.rs) `parse_icc_type` following the identical
`match s.to_lowercase().as_str()` form.

**`#[polars_expr]` shim pattern** — always a two-line thin wrapper:
```rust
#[polars_expr(output_type_func=one_way_anova_output_dtype)]
fn pl_one_way_anova(inputs: &[Series]) -> PolarsResult<Series> {
    one_way_anova_fit(inputs)
}
```
The public function is `one_way_anova_fit`; the FFI entry point is `pl_one_way_anova` (private,
`fn` not `pub fn`).

---

### `src/expressions/modern.rs` — 1 new fit function + shim

**Analog:** `src/expressions/modern.rs` `energy_distance_fit` / `pl_energy_distance` (lines 11-30)
and `mmd_test_fit` (lines 33-53). These are the exact template.

**Full existing function** (lines 11-30):
```rust
pub fn energy_distance_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;
    let y = inputs[1].f64()?;
    let n_perm = inputs[2].u32()?.get(0).unwrap_or(999) as usize;
    let seed = inputs[3].u64()?.get(0);

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();

    match energy_distance_test_1d(&x_vec, &y_vec, n_perm, seed) {
        Ok(result) => generic_stats_output(result.statistic, result.p_value, "energy_distance"),
        Err(_) => generic_stats_output(f64::NAN, f64::NAN, "energy_distance"),
    }
}

#[polars_expr(output_type_func=stats_output_dtype)]
fn pl_energy_distance(inputs: &[Series]) -> PolarsResult<Series> {
    energy_distance_fit(inputs)
}
```

**Import to add** (line 6):
```rust
use anofox_statistics::{energy_distance_test_1d, mmd_test_1d};
// Add:
use anofox_statistics::energy_distance_test;  // the nD overload
```
(Verify exact import path; may be `anofox_statistics::modern::energy_distance_test` if not
re-exported at root.)

**nD energy input contract** — variable-arity, follows `partial_cor_fit` pattern (see below):
```
inputs[0] = UInt32 literal: d (number of dimensions / features per observation)
inputs[1] = UInt32 literal: n_permutations
inputs[2] = UInt64 literal: seed (nullable)
inputs[3..3+d] = f64 Series (X sample, one Series per dimension)
inputs[3+d..3+2d] = f64 Series (Y sample, one Series per dimension)
```

The nD wrapper collects inputs[3..] into `Vec<Vec<f64>>` for x and y, then transposes from
column-per-dimension into row-per-observation:
```rust
// Each observation is a Vec<f64> of length d
let mut x_data: Vec<Vec<f64>> = Vec::new(); // x_data[obs_i] = Vec of d features
// inputs[3+j] is dimension j of x; collect each into a column, then transpose
```

The output reuses `generic_stats_output(result.statistic, result.p_value, "energy_distance_nd")`
and `stats_output_dtype` — no new output_dtype.

---

### `src/expressions/correlation.rs` — `icc_fit` replacement + `parse_icc_type` helper

**Analog:** `src/expressions/correlation.rs` `partial_cor_fit` (lines 167-211) — this is the
exact variable-arity pattern to copy for the rater-column loop.

**Full partial_cor_fit** (lines 167-211):
```rust
pub fn partial_cor_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;
    let y = inputs[1].f64()?;
    let n_covariates = inputs[2].u32()?.get(0).unwrap_or(1) as usize;

    let x_vec: Vec<f64> = x.into_no_null_iter().collect();
    let y_vec: Vec<f64> = y.into_no_null_iter().collect();

    let mut covariates: Vec<Vec<f64>> = Vec::new();
    for i in 0..n_covariates {
        if let Some(cov_series) = inputs.get(3 + i) {
            if let Ok(cov) = cov_series.f64() {
                covariates.push(cov.into_no_null_iter().collect());
            }
        }
    }

    if covariates.is_empty() {
        return correlation_error_output("partial_cor");
    }

    let cov_refs: Vec<&[f64]> = covariates.iter().map(|v| v.as_slice()).collect();

    match partial_cor(&x_vec, &y_vec, &cov_refs) {
        Ok(result) => {
            // ... build StructChunked ...
        }
        Err(_) => correlation_error_output("partial_cor"),
    }
}

#[polars_expr(output_type_func=correlation_output_dtype)]
fn pl_partial_cor(inputs: &[Series]) -> PolarsResult<Series> {
    partial_cor_fit(inputs)
}
```

**New icc_fit input contract** (replacing current stub at lines 270-303):
```
inputs[0] = UInt32 literal: n_raters
inputs[1] = String literal: icc_type ("icc1", "icc2", "icc3", "icc1k", "icc2k", "icc3k")
inputs[2..2+n_raters] = f64 Series, one per rater column (rows = subjects)
```

**Current stub to DELETE** (lines 270-303) — the entire body of `icc_fit` and the `pl_icc` shim:
```rust
pub fn icc_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let values = inputs[0].f64()?;
    let _icc_type_str = inputs[1].str()?.get(0).unwrap_or("icc1");
    let _conf_level = inputs[2].f64()?.get(0).unwrap_or(0.95);
    // ICC requires a 2D matrix structure - this is a simplified placeholder
    // TODO: Implement proper ICC with matrix input
    let estimate = Series::new("estimate".into(), &[f64::NAN]);
    // ... all-NaN fields ...
}

#[polars_expr(output_type_func=correlation_output_dtype)]   // <-- also changes to icc_output_dtype
fn pl_icc(inputs: &[Series]) -> PolarsResult<Series> {
    icc_fit(inputs)
}
```

**Imports to add** to correlation.rs (current line 6-9):
```rust
use anofox_statistics::{
    distance_cor_test, kendall, partial_cor, pearson, semi_partial_cor, spearman,
    CorrelationResult, KendallVariant,
};
// Add:
use anofox_statistics::correlation::{icc, ICCResult, ICCType};
use crate::expressions::output_types::{correlation_output_dtype, icc_output_dtype};
```

**Matrix transpose pattern** — raters-indexed to subjects-indexed:
```rust
// After collecting: matrix[rater_index] = Vec<f64> of subject scores
// icc() expects: data[subject_index] = Vec<f64> of rater scores
let n_subjects = matrix[0].len();
let data: Vec<Vec<f64>> = (0..n_subjects)
    .map(|s| (0..n_raters).map(|r| matrix[r][s]).collect())
    .collect();
```

**icc_error_output helper** — same pattern as `correlation_error_output` (lines 44-59) but
matching the new `icc_output_dtype` schema (icc, f_value, df1, df2, p_value, ci_lower, ci_upper,
n_subjects, n_raters — all NaN / 0u32):
```rust
fn icc_error_output() -> PolarsResult<Series> {
    let icc_val = Series::new("icc".into(), &[f64::NAN]);
    let f_value = Series::new("f_value".into(), &[f64::NAN]);
    // ...
    let n_subjects = Series::new("n_subjects".into(), &[0u32]);
    let n_raters = Series::new("n_raters".into(), &[0u32]);
    let df = StructChunked::from_series(
        "icc".into(), 1,
        [&icc_val, &f_value, /* ... */ &n_subjects, &n_raters].into_iter(),
    )?;
    Ok(df.into_series())
}
```

---

### `python/polars_statistics/exprs/parametric.py` — 3 new builder functions

**Analog:** `python/polars_statistics/exprs/parametric.py` `ttest_ind` (lines 14-87) — exact
docstring format, `isinstance` str→expr conversion, `is_finite()` filter, `register_plugin_function`
call with `returns_scalar=True`.

**Full ttest_ind pattern** (lines 66-87):
```python
if isinstance(x, str):
    x = pl.col(x)
if isinstance(y, str):
    y = pl.col(y)

x_clean = x.filter(x.is_finite())
y_clean = y.filter(y.is_finite())

return register_plugin_function(
    plugin_path=LIB,
    function_name="pl_ttest_ind",
    args=[
        x_clean,
        y_clean,
        pl.lit(alternative, dtype=pl.String),
        pl.lit(equal_var, dtype=pl.Boolean),
        pl.lit(mu, dtype=pl.Float64),
        pl.lit(conf_level, dtype=pl.Float64),
    ],
    returns_scalar=True,
)
```

**one_way_anova builder** — two column args (value + group), both filtered on `value.is_finite()`:
```python
value_clean = value_expr.filter(value_expr.is_finite())
group_clean = group_expr.filter(value_expr.is_finite())  # same mask — preserve alignment
```

**two_way_anova builder** — three column args (value, factor_a, factor_b). Factor columns must be
encoded as UInt32 by the Python builder before passing:
```python
# In the builder, after converting to expr:
factor_a_encoded = factor_a_expr.cast(pl.Categorical).to_physical().cast(pl.UInt32)
factor_b_encoded = factor_b_expr.cast(pl.Categorical).to_physical().cast(pl.UInt32)
# Filter: apply value.is_finite() mask to all three columns
```

**repeated_measures_anova builder** — four column args (value, subject_id, condition, bool
literal) plus `compute_sphericity` as `pl.lit(..., dtype=pl.Boolean)`:
```python
return register_plugin_function(
    plugin_path=LIB,
    function_name="pl_repeated_measures_anova",
    args=[
        value_clean,
        subject_clean,
        condition_clean,
        pl.lit(compute_sphericity, dtype=pl.Boolean),
    ],
    returns_scalar=True,
)
```

---

### `python/polars_statistics/exprs/modern.py` — 1 new builder function

**Analog:** `python/polars_statistics/exprs/modern.py` `energy_distance` (lines 21-69) — the
nullable seed pattern and `_to_expr` helper.

**Nullable seed pattern** (lines 57-58):
```python
seed_expr = pl.lit(seed, dtype=pl.UInt64) if seed is not None else pl.lit(None, dtype=pl.UInt64)
```

**Multi-column args pattern for energy_distance_nd:**
```python
def energy_distance_nd(
    x_cols: list[Union[pl.Expr, str]],
    y_cols: list[Union[pl.Expr, str]],
    n_permutations: int = 999,
    seed: int | None = None,
) -> pl.Expr:
    d = len(x_cols)
    # validate len(x_cols) == len(y_cols)
    x_exprs = [_to_expr(c).cast(pl.Float64) for c in x_cols]
    y_exprs = [_to_expr(c).cast(pl.Float64) for c in y_cols]
    seed_expr = pl.lit(seed, dtype=pl.UInt64) if seed is not None else pl.lit(None, dtype=pl.UInt64)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_energy_distance_nd",
        args=[
            pl.lit(d, dtype=pl.UInt32),
            pl.lit(n_permutations, dtype=pl.UInt32),
            seed_expr,
            *x_exprs,
            *y_exprs,
        ],
        returns_scalar=True,
    )
```

---

### `python/polars_statistics/exprs/correlation.py` — `icc` function rewrite

**Analog:** `python/polars_statistics/exprs/correlation.py` `partial_cor` function (search for
`def partial_cor` — uses variable-arity `*covariates` args with `n_covariates` literal).

**Current icc signature to REPLACE** (lines 377-422):
```python
def icc(
    values: Union[pl.Expr, str],
    icc_type: Literal["icc1", "icc2", "icc3", "icc2k", "icc3k"] = "icc1",
    conf_level: float = 0.95,
) -> pl.Expr:
```

**New signature pattern** — mirroring the partial_cor multi-arg approach with n_raters literal:
```python
def icc(
    *rater_cols: Union[pl.Expr, str],
    icc_type: Literal["icc1", "icc2", "icc3", "icc1k", "icc2k", "icc3k"] = "icc2",
) -> pl.Expr:
    # Multi-column null handling: filter rows where ANY column is non-finite
    rater_exprs = [pl.col(c) if isinstance(c, str) else c for c in rater_cols]
    mask = pl.all_horizontal([col.is_finite() for col in rater_exprs])
    clean_cols = [col.filter(mask) for col in rater_exprs]

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_icc",
        args=[
            pl.lit(len(rater_cols), dtype=pl.UInt32),
            pl.lit(icc_type, dtype=pl.String),
            *clean_cols,
        ],
        returns_scalar=True,
    )
```

Note: `icc` is already imported in `exprs/__init__.py` (line 50) and `__init__.py` (line 96/325)
— no registration change needed for icc itself.

---

### `python/polars_statistics/exprs/__init__.py` — add 4 new imports

**Analog:** existing import blocks in the same file.

**Pattern to copy** (lines 3-8 — parametric block):
```python
from polars_statistics.exprs.parametric import (
    ttest_ind,
    ttest_paired,
    brown_forsythe,
    yuen_test,
)
```

Add `one_way_anova`, `two_way_anova`, `repeated_measures_anova` to the parametric block.
Add `energy_distance_nd` to the modern block (lines 27-30).
Add all four to `__all__` list (lines 189-366).

---

### `python/polars_statistics/__init__.py` — add 4 new re-exports

**Pattern to copy** (lines 54-59 — parametric block):
```python
from polars_statistics.exprs import (
    ttest_ind,
    ttest_paired,
    brown_forsythe,
    yuen_test,
    ...
    energy_distance,
    mmd_test,
    ...
)
```

Add `one_way_anova`, `two_way_anova`, `repeated_measures_anova`, `energy_distance_nd` to the
existing `from polars_statistics.exprs import (...)` block and to `__all__` (lines 238-462).

---

### `tests/test_statistics_parity.py` — new test file

**Analog:** Look at any existing test in `tests/` for import style, but the structure is prescribed
by RESEARCH.md. The smoke-test pattern is:

```python
import polars as pl
import polars_statistics as ps

class TestOneWayAnova:
    def test_returns_correct_schema(self):
        df = pl.DataFrame({
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "group": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        })
        result = df.select(ps.one_way_anova("value", "group"))
        assert result.shape == (1, 1)
        struct_val = result[0, 0]
        assert "statistic" in struct_val
        assert "p_value" in struct_val
        assert struct_val["statistic"] >= 0
        assert 0.0 <= struct_val["p_value"] <= 1.0
```

Each test class covers one STAT-XX requirement. Five classes total:
`TestOneWayAnova`, `TestTwoWayAnova`, `TestRmAnova`, `TestEnergyDistanceNd`, `TestIcc`.

---

## Shared Patterns

### `#[polars_expr]` + `pub fn *_fit` Two-Level Structure

**Source:** `src/expressions/parametric.rs` lines 20-48 (`ttest_ind_fit` / `pl_ttest_ind`)
**Apply to:** All five new Rust expressions

Every expression has:
1. `pub fn *_fit(inputs: &[Series]) -> PolarsResult<Series>` — public, testable, contains all logic
2. `#[polars_expr(output_type_func=*_output_dtype)] fn pl_*(inputs: &[Series]) -> PolarsResult<Series>`
   — private FFI entry point, delegates to `*_fit`

### NaN-on-error pattern

**Source:** `src/expressions/modern.rs` line 21-23; `src/expressions/correlation.rs` lines 44-59
**Apply to:** All five new Rust expressions

On `Err(_)` from any crate call, return the same Struct schema with all-NaN / 0u32 values, never
propagate the error as a Polars error. This keeps the expression compatible with group_by (a
failing group returns NaN, not an exception).

Pattern: define a dedicated `*_error_output()` helper that constructs the NaN struct, call it
in the `Err(_)` match arm.

### `into_no_null_iter().collect()` for Series→Vec

**Source:** `src/expressions/parametric.rs` lines 28-29; `src/expressions/correlation.rs` line 181
**Apply to:** All five new Rust expressions

```rust
let x_vec: Vec<f64> = x.into_no_null_iter().collect();
```
Nulls are silently dropped. For multi-column inputs, alignment is preserved by applying the same
finite mask in Python before passing (see multi-column null handling pitfall in RESEARCH.md).

### `register_plugin_function` with `returns_scalar=True`

**Source:** `python/polars_statistics/exprs/parametric.py` lines 75-87
**Apply to:** All five new Python builders

```python
return register_plugin_function(
    plugin_path=LIB,
    function_name="pl_<name>",
    args=[...],
    returns_scalar=True,
)
```
`LIB = Path(__file__).parent.parent` is defined at module level in every builder file.

### Glob re-export via `mod.rs` — no manual registration

**Source:** `src/expressions/mod.rs` lines 18-37
**Apply to:** All five new Rust expressions

Any `pub fn` added to `parametric.rs`, `modern.rs`, or `correlation.rs` is automatically
exported via the existing `pub use parametric::*` etc. glob. No edits to `mod.rs` are needed.

---

## No Analog Found

None. All new/modified files have exact or role-match analogs in the codebase.

---

## Metadata

**Analog search scope:** `src/expressions/`, `python/polars_statistics/exprs/`,
`python/polars_statistics/__init__.py`
**Files read:** 9 source files
**Pattern extraction date:** 2026-08-11
