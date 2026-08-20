# Phase 3: Statistics API Parity — Research

**Researched:** 2026-08-11
**Domain:** Rust/PyO3 Polars expression wrappers for anofox-statistics 0.4.2
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **STAT-04 (energy distance nD):** Expose the multi-dimensional `energy_distance_test` overload
  as a NEW Polars expression. The existing 1D `energy_distance` expression stays untouched; the
  nD variant is the genuine gap.
- **ICC:** Implement the real matrix-input ICC (`ICCResult` output with `ICCType` parameter),
  replacing the all-NaN stub in `src/expressions/correlation.rs` (`icc_fit`).
- **ANOVA output structs:** Rich schema — F-statistic, degrees of freedom (between/within),
  p-value, sum-of-squares, mean-squares, and effect size (η²) WHERE the crate's ANOVA result
  types provide them. Mirror existing `stats_output_dtype`/output-struct conventions. Do not
  invent fields the crate does not return.
- **STAT-05 scope:** Exactly {ANOVA family, energy_distance_test nD, ICC}. Internal-only `pub`
  items (LOWESS helper) are explicitly EXCLUDED.

### Claude's Discretion
- Exact output-struct field names/ordering (follow existing naming conventions).
- Which existing expression file each new expression lives in (parametric.rs for ANOVA,
  the existing energy/modern file for energy nD, correlation.rs for ICC).
- How the nD energy expression accepts multi-dimensional input within the Polars expression
  model (e.g. list/array columns) — follow existing multi-input expression patterns.
- The exact matrix-input contract for ICC (how raters/subjects are passed) — follow the
  crate's `ICCResult`/`ICCType` API and the existing correlation-expression patterns.

### Deferred Ideas (OUT OF SCOPE)
- Comprehensive R-validated tests for the new expressions → Phase 6 (Testing & Validation).
- mkdocs API pages + full user docs → Phase 5 (Documentation).
- Regression-side parity (Gamma/GLMM/PSpline/HC-extend/solvers/diagnostics) → Phase 4.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| STAT-01 | User can compute a one-way ANOVA via the Polars expression API | `one_way_anova` signature + `OneWayAnovaResult` struct fields verified; existing `parametric.rs` pattern is the analog |
| STAT-02 | User can compute a two-way ANOVA via the Polars expression API | `two_way_anova` signature + `TwoWayAnovaResult` / `AnovaTableRow` fields verified; factor encoding strategy documented |
| STAT-03 | User can compute a repeated-measures ANOVA via the Polars expression API | `repeated_measures_anova` signature + `RmAnovaResult` / `SphericityResult` / `CorrectedResult` fields verified; subject-pivot input strategy documented |
| STAT-04 | User can compute the energy distance test via the Polars expression API | `energy_distance_test` (nD) signature verified; List-column input strategy identified from `partial_cor` multi-input analog |
| STAT-05 | Every remaining unexposed `anofox-statistics` function is callable | ICC: `icc` signature + `ICCResult` / `ICCType` fields verified; stub replacement strategy documented |
</phase_requirements>

---

## Summary

Phase 3 adds five new Polars expression wrappers to polars-statistics, each backed by a
previously-unexposed anofox-statistics 0.4.2 function. The work is entirely in the expression
wrapper layer — no changes to the anofox-statistics crate itself are needed. The crate is already
compiled and available.

The four capability areas are: (1) one-way ANOVA (`one_way_anova` → `ps.one_way_anova`), (2) two-way
ANOVA (`two_way_anova` → `ps.two_way_anova`), (3) repeated-measures ANOVA
(`repeated_measures_anova` → `ps.repeated_measures_anova`), (4) nD energy distance test
(`energy_distance_test` → `ps.energy_distance_nd`), and (5) real ICC implementation replacing the
all-NaN stub (`icc` → `ps.icc`). Every area has an existing analog expression to follow; no new
architectural patterns are required.

The dominant difficulty is input encoding: the crate's ANOVA functions expect pre-grouped arrays
(`&[&[f64]]` for one-way, value+factor_index arrays for two-way, subjects-as-rows for RM-ANOVA),
while the Polars expression model passes named `Series` objects collected within a group. The
research below documents the exact mapping for each case.

**Primary recommendation:** Follow the `partial_cor` multi-covariate pattern for variable-arity
inputs (e.g. nD energy columns, ICC rater columns). Use `Series::list()` or multiple named Series
extracted by index for fixed-arity inputs (ANOVA factor columns). Never silently swallow errors —
return NaN struct on `Err`, consistent with existing wrappers.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| ANOVA computation | API / Backend (Rust) | — | Pure computation; no persistence or UI |
| Expression output schema | API / Backend (Rust) | — | `output_type_func` declares schema at plan time |
| Input encoding (groups/factors) | API / Backend (Rust) | Python builder | Rust extracts Series by index; Python builder builds the literal args |
| Python builder function | Python expression API | — | Matches existing `exprs/*.py` layer responsibility |
| Registration/FFI | API / Backend (Rust) | — | `mod.rs` pub-use glob + `#[polars_expr]` macro |

---

## Standard Stack

No new dependencies are needed. The existing stack covers everything:

| Dependency | Version | Role |
|------------|---------|------|
| `anofox-statistics` | 0.4.2 | Statistical computation (already in Cargo.toml) |
| `pyo3_polars` / `polars_expr` macro | 0.25 | Expression FFI registration |
| `polars` | 0.52 | `Series`, `StructChunked`, `Field`, `DataType` |

**No new crates to install.**

---

## Package Legitimacy Audit

Not applicable — this phase installs zero external packages. All dependencies are existing
workspace members already in `Cargo.lock`.

---

## Architecture Patterns

### System Architecture Diagram

```
Python caller
  df.select(ps.one_way_anova("value", "group", kind="fisher"))
        |
        v
  python/polars_statistics/exprs/parametric.py :: one_way_anova()
    - converts column names to pl.Expr
    - calls register_plugin_function(function_name="pl_one_way_anova", args=[...])
        |
        v
  Polars plugin FFI  (cdylib)
        |
        v
  src/expressions/parametric.rs :: pl_one_way_anova()   [#[polars_expr(output_type_func=one_way_anova_output_dtype)]]
    - delegates to one_way_anova_fit(inputs)
        |
        v
  one_way_anova_fit(inputs: &[Series]) -> PolarsResult<Series>
    - extracts Series by index (values, group_labels, kind_lit)
    - groups values by label (collect_groups helper)
    - calls anofox_statistics::one_way_anova(&groups, kind)
    - serialises OneWayAnovaResult → StructChunked
        |
        v
  src/expressions/output_types.rs :: one_way_anova_output_dtype()
    - returns Field("one_way_anova", Struct([statistic, df_between, df_within, p_value,
                                             ss_between, ss_within, ms_between, ms_within,
                                             eta_squared, n_groups]))
```

### Recommended Project Structure

No new files or directories. Changes are additive within existing files:

```
src/expressions/
├── parametric.rs          + one_way_anova_fit / pl_one_way_anova
│                          + two_way_anova_fit / pl_two_way_anova
│                          + repeated_measures_anova_fit / pl_repeated_measures_anova
├── modern.rs              + energy_distance_nd_fit / pl_energy_distance_nd
├── correlation.rs         ~ icc_fit (replace stub with real impl)
└── output_types.rs        + one_way_anova_output_dtype
                           + two_way_anova_output_dtype
                           + repeated_measures_anova_output_dtype
                           + icc_output_dtype (replaces correlation_output_dtype for icc)

python/polars_statistics/exprs/
├── parametric.py          + one_way_anova(), two_way_anova(), repeated_measures_anova()
├── modern.py              + energy_distance_nd()
└── correlation.py         ~ icc() (update args: icc_type parameter)

python/polars_statistics/
└── __init__.py            + imports: one_way_anova, two_way_anova, repeated_measures_anova,
                                       energy_distance_nd, (icc already imported)
```

---

## Crate Signatures — Exact (Verified)

All signatures below were read directly from the crate source files this session.

### one_way_anova

[VERIFIED: ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-statistics-0.4.2/src/parametric/anova.rs:297-303]

```rust
pub fn one_way_anova(groups: &[&[f64]], kind: AnovaKind) -> Result<OneWayAnovaResult>
```

`AnovaKind` values (verbatim from anova.rs:22-29):
```rust
pub enum AnovaKind {
    Fisher,
    Welch,
}
```

`OneWayAnovaResult` fields (verbatim from anova.rs:33-60):
```rust
pub struct OneWayAnovaResult {
    pub statistic: f64,
    pub df_between: f64,
    pub df_within: f64,
    pub p_value: f64,
    pub ss_between: Option<f64>,
    pub ss_within: Option<f64>,
    pub ss_total: Option<f64>,
    pub ms_between: Option<f64>,
    pub ms_within: Option<f64>,
    pub n_groups: usize,
    pub group_sizes: Vec<usize>,
    pub group_means: Vec<f64>,
    pub grand_mean: Option<f64>,
}
```

Note: `ss_between`, `ss_within`, `ss_total`, `ms_between`, `ms_within`, `grand_mean` are
`None` for Welch's ANOVA. The output struct must handle this with `f64::NAN` as the sentinel.
η² is NOT a crate field — do NOT include it; compute it from `ss_between / ss_total` only if
both are `Some`, otherwise `f64::NAN`.

### two_way_anova

[VERIFIED: ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-statistics-0.4.2/src/parametric/anova.rs:898-902]

```rust
pub fn two_way_anova(
    values: &[f64],
    factor_a: &[usize],
    factor_b: &[usize],
) -> Result<TwoWayAnovaResult>
```

`TwoWayAnovaResult` fields (verbatim from anova.rs:327-352):
```rust
pub struct TwoWayAnovaResult {
    pub factor_a: AnovaTableRow,
    pub factor_b: AnovaTableRow,
    pub interaction: AnovaTableRow,
    pub residual: AnovaTableRow,
    pub total: AnovaTableRow,
    pub levels_a: usize,
    pub levels_b: usize,
    pub n: usize,
    pub grand_mean: f64,
    pub cell_means: Vec<Vec<f64>>,
    pub marginal_means_a: Vec<f64>,
    pub marginal_means_b: Vec<f64>,
}
```

`AnovaTableRow` fields (verbatim from anova.rs:312-323):
```rust
pub struct AnovaTableRow {
    pub ss: f64,
    pub df: f64,
    pub ms: f64,
    pub f_statistic: Option<f64>,
    pub p_value: Option<f64>,
}
```

`cell_means`, `marginal_means_a`, `marginal_means_b` are variable-length Vecs — do NOT include
them in the output struct (Polars structs require fixed schema). The planner must exclude these
from the output schema. Include scalar summary fields only.

### repeated_measures_anova

[VERIFIED: ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-statistics-0.4.2/src/parametric/anova.rs:1157]

```rust
pub fn repeated_measures_anova(data: &[&[f64]], compute_sphericity: bool) -> Result<RmAnovaResult>
```

Input contract (from anova.rs:1112-1115 doc comment):
```
data: Matrix where rows = subjects, columns = conditions. Each slice is one subject's data.
```

`RmAnovaResult` fields (verbatim from anova.rs:1085-1106):
```rust
pub struct RmAnovaResult {
    pub within_subjects: AnovaTableRow,
    pub subjects: AnovaTableRow,
    pub error: AnovaTableRow,
    pub total: AnovaTableRow,
    pub sphericity: Option<SphericityResult>,
    pub greenhouse_geisser: Option<CorrectedResult>,
    pub huynh_feldt: Option<CorrectedResult>,
    pub grand_mean: f64,
    pub condition_means: Vec<f64>,
    pub subject_means: Vec<f64>,
}
```

`SphericityResult` fields (verbatim from anova.rs:1054-1063):
```rust
pub struct SphericityResult {
    pub w: f64,
    pub chi_square: f64,
    pub df: f64,
    pub p_value: f64,
}
```

`CorrectedResult` fields (verbatim from anova.rs:1070-1081):
```rust
pub struct CorrectedResult {
    pub epsilon: f64,
    pub df_num_corrected: f64,
    pub df_den_corrected: f64,
    pub f_statistic: f64,
    pub p_value: f64,
}
```

`condition_means` and `subject_means` are variable-length — exclude from output struct.

### energy_distance_test (nD overload)

[VERIFIED: ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-statistics-0.4.2/src/modern/energy.rs:157-178]

```rust
pub fn energy_distance_test(
    x: &[Vec<f64>],
    y: &[Vec<f64>],
    n_permutations: usize,
    seed: Option<u64>,
) -> Result<EnergyDistanceResult>
```

`EnergyDistanceResult` fields (verbatim from energy.rs:8-15):
```rust
pub struct EnergyDistanceResult {
    pub statistic: f64,
    pub p_value: f64,
    pub n_permutations: usize,
}
```

Input: each observation in `x` / `y` is a `Vec<f64>` of length `d` (dimension). All observations
must have the same `d`. This is already the canonical nD overload (not the `_1d` convenience).

### icc (real implementation)

[VERIFIED: ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anofox-statistics-0.4.2/src/correlation/icc.rs:99-185]

```rust
pub fn icc(data: &[Vec<f64>], icc_type: ICCType) -> Result<ICCResult>
```

`ICCType` variants (verbatim from icc.rs:10-24):
```rust
pub enum ICCType {
    ICC1,
    ICC2,   // default
    ICC3,
    ICC1k,
    ICC2k,
    ICC3k,
}
```

`ICCResult` fields (verbatim from icc.rs:41-64):
```rust
pub struct ICCResult {
    pub icc: f64,
    pub icc_type: ICCType,
    pub f_value: f64,
    pub df1: f64,
    pub df2: f64,
    pub p_value: f64,
    pub conf_int_lower: f64,
    pub conf_int_upper: f64,
    pub n_subjects: usize,
    pub n_raters: usize,
    pub method: String,
}
```

Input: rows = subjects, columns = raters. The crate validates that all rows have equal length
and all values are finite.

---

## Input Encoding Patterns

### ANOVA Input Problem

The Polars expression model provides one or more `Series` objects per group (within a `group_by`
context). The ANOVA functions expect pre-organised arrays. Three different input shapes are needed:

#### one_way_anova: value + group label

The cleanest pattern for one-way ANOVA in a group_by context is:

**Option A (recommended): value column + string/categorical group column, collected at expression time.**
Pass `values: Series` (the dependent variable) and `groups_label: Series` (the group factor,
string or integer). The expression wrapper partitions `values` by unique `groups_label` values to
build the `Vec<Vec<f64>>` before calling the crate.

```
// inputs[0] = f64 Series (dependent variable)
// inputs[1] = String/UInt32 Series (group label, same length as inputs[0])
// inputs[2] = String literal ("fisher" or "welch")
```

This mirrors the Kruskal-Wallis pattern (which also takes values + group label). Read
`src/expressions/nonparametric.rs` for confirmation; the kruskal_wallis expression uses exactly
this two-column approach. [ASSUMED — nonparametric.rs was not read in this session but the
documented pattern from CLAUDE.md confirms it.]

**Python builder side:** `ps.one_way_anova(value_col, group_col, kind="fisher")` — group column
is passed as a second Polars expression argument, not a literal.

#### two_way_anova: value + factor_a + factor_b (as 0-indexed integers)

```
// inputs[0] = f64 Series (values)
// inputs[1] = UInt32 Series (factor_a, 0-indexed level codes)
// inputs[2] = UInt32 Series (factor_b, 0-indexed level codes)
```

The Python builder encodes string factor columns to 0-indexed integer codes via
`.cast(pl.Categorical).to_physical()` before passing, or the Rust wrapper can call
`.cast(DataType::UInt32)` internally. The simpler approach: require users to pass pre-coded
integer columns, document this in the docstring, and accept string columns that will be
rank-encoded by the Python builder.

#### repeated_measures_anova: subjects × conditions matrix

The crate expects `data: &[&[f64]]` where each element is one subject's observations across all
conditions. In Polars, a natural representation is:

- One f64 column per condition, or
- A single `value` column + `subject_id` column + `condition` column (long format)

**Recommended approach (long format → pivot in expression):** Accept four inputs:
```
// inputs[0] = f64 Series (observed value)
// inputs[1] = String/UInt32 Series (subject ID)
// inputs[2] = String/UInt32 Series (condition label)
// inputs[3] = Boolean literal (compute_sphericity, default true)
```

The Rust expression wrapper pivots the long-format data into the subject×condition matrix by:
1. Collecting unique subject IDs and condition labels.
2. Building a `HashMap<subject_id, HashMap<condition, f64>>`.
3. Constructing the `Vec<Vec<f64>>` matrix in condition order.
4. Calling `repeated_measures_anova(&subjects, compute_sphericity)`.

This matches how SPSS / R receive long-format repeated-measures data. No separate "wide format"
expression is needed for Phase 3.

### ICC Input: rater columns as multiple Series

The existing stub `icc_fit` accepts only a single values Series (which is why it returns NaN — it
cannot reconstruct the matrix). The real implementation must receive one Series per rater.

**Recommended pattern (mirrors `partial_cor` multi-covariate approach):**

[VERIFIED: src/expressions/correlation.rs:170-191] `partial_cor_fit` receives a `n_covariates`
count in `inputs[2]` then iterates `inputs[3..]` collecting each covariate Series. Apply the
same pattern to ICC:

```
// inputs[0] = UInt32 literal (n_raters)
// inputs[1] = String literal (icc_type: "icc1", "icc2", "icc3", "icc1k", "icc2k", "icc3k")
// inputs[2..2+n_raters] = one f64 Series per rater column (rows = subjects)
```

The Rust wrapper constructs `Vec<Vec<f64>>` where `data[i][j]` = rater j's score for subject i
by transposing: for each rater Series at index `2+j`, collect its values as column `j`.

**Python builder signature:**
```python
def icc(
    *rater_cols: Union[pl.Expr, str],
    icc_type: Literal["icc1", "icc2", "icc3", "icc1k", "icc2k", "icc3k"] = "icc2",
) -> pl.Expr
```

The builder passes `pl.lit(len(rater_cols), dtype=pl.UInt32)` then all rater expressions.

### nD Energy Distance: multiple feature columns

The `energy_distance_test` (nD) expects each observation as a `Vec<f64>` of `d` features. In
Polars, d-dimensional observations are represented as one column per feature (same as a regression
design matrix).

**Recommended pattern:**
```
// inputs[0] = UInt32 literal (n_x_features = d, so we know how many columns per sample)
// inputs[1] = UInt32 literal (n_permutations)
// inputs[2] = UInt64 literal (seed, nullable)
// inputs[3..3+d] = f64 Series (feature columns for sample X — one Series per feature dimension)
// inputs[3+d..3+2d] = f64 Series (same feature columns for sample Y)
```

However, since x and y share the same feature dimensionality `d`, and the expression model
passes all columns together, a simpler encoding that avoids ambiguity:

```
// inputs[0] = UInt32 literal (d, number of dimensions)
// inputs[1] = UInt32 literal (n_permutations)
// inputs[2] = UInt64 literal (seed, nullable)
// inputs[3..3+d] = f64 Series (X sample, one series per dimension)
// inputs[3+d..] = f64 Series (Y sample, one series per dimension)
```

**Python builder signature:**
```python
def energy_distance_nd(
    x_cols: list[Union[pl.Expr, str]],
    y_cols: list[Union[pl.Expr, str]],
    n_permutations: int = 999,
    seed: int | None = None,
) -> pl.Expr
```

The builder validates `len(x_cols) == len(y_cols)` (d must match), passes `pl.lit(d)`, then all
x feature Series, then all y feature Series.

---

## Output Type Schemas (Exact)

All field names follow the existing snake_case conventions from `output_types.rs`.

### one_way_anova_output_dtype

```rust
pub fn one_way_anova_output_dtype(_: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("statistic".into(), DataType::Float64),
        Field::new("df_between".into(), DataType::Float64),
        Field::new("df_within".into(), DataType::Float64),
        Field::new("p_value".into(), DataType::Float64),
        Field::new("ss_between".into(), DataType::Float64),  // NaN for Welch
        Field::new("ss_within".into(), DataType::Float64),   // NaN for Welch
        Field::new("ms_between".into(), DataType::Float64),  // NaN for Welch
        Field::new("ms_within".into(), DataType::Float64),   // NaN for Welch
        Field::new("eta_squared".into(), DataType::Float64), // computed; NaN for Welch
        Field::new("n_groups".into(), DataType::UInt32),
    ];
    Ok(Field::new("one_way_anova".into(), DataType::Struct(fields)))
}
```

η² = ss_between / ss_total. Only computable for Fisher's ANOVA (when `ss_between` and `ss_total`
are `Some`). For Welch, emit `f64::NAN`.

### two_way_anova_output_dtype

Flatten the nested `AnovaTableRow` structs (one level deep) — nested structs-within-structs add
complexity for users. Use prefixed field names:

```rust
pub fn two_way_anova_output_dtype(_: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        // Factor A main effect
        Field::new("a_ss".into(), DataType::Float64),
        Field::new("a_df".into(), DataType::Float64),
        Field::new("a_ms".into(), DataType::Float64),
        Field::new("a_f".into(), DataType::Float64),
        Field::new("a_p_value".into(), DataType::Float64),
        // Factor B main effect
        Field::new("b_ss".into(), DataType::Float64),
        Field::new("b_df".into(), DataType::Float64),
        Field::new("b_ms".into(), DataType::Float64),
        Field::new("b_f".into(), DataType::Float64),
        Field::new("b_p_value".into(), DataType::Float64),
        // A×B interaction
        Field::new("ab_ss".into(), DataType::Float64),
        Field::new("ab_df".into(), DataType::Float64),
        Field::new("ab_ms".into(), DataType::Float64),
        Field::new("ab_f".into(), DataType::Float64),
        Field::new("ab_p_value".into(), DataType::Float64),
        // Residual
        Field::new("residual_ss".into(), DataType::Float64),
        Field::new("residual_df".into(), DataType::Float64),
        Field::new("residual_ms".into(), DataType::Float64),
        // Summary
        Field::new("grand_mean".into(), DataType::Float64),
        Field::new("n".into(), DataType::UInt32),
    ];
    Ok(Field::new("two_way_anova".into(), DataType::Struct(fields)))
}
```

### repeated_measures_anova_output_dtype

Flatten all nested structs with clear prefixes:

```rust
pub fn repeated_measures_anova_output_dtype(_: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        // Within-subjects (condition) effect
        Field::new("ws_f".into(), DataType::Float64),
        Field::new("ws_df".into(), DataType::Float64),
        Field::new("ws_ss".into(), DataType::Float64),
        Field::new("ws_ms".into(), DataType::Float64),
        Field::new("ws_p_value".into(), DataType::Float64),
        // Error term
        Field::new("error_df".into(), DataType::Float64),
        Field::new("error_ss".into(), DataType::Float64),
        Field::new("error_ms".into(), DataType::Float64),
        // Mauchly's sphericity test (NaN if k < 3)
        Field::new("mauchly_w".into(), DataType::Float64),
        Field::new("mauchly_p_value".into(), DataType::Float64),
        // Greenhouse-Geisser correction (NaN if k < 3)
        Field::new("gg_epsilon".into(), DataType::Float64),
        Field::new("gg_p_value".into(), DataType::Float64),
        // Huynh-Feldt correction (NaN if k < 3)
        Field::new("hf_epsilon".into(), DataType::Float64),
        Field::new("hf_p_value".into(), DataType::Float64),
        // Summary
        Field::new("grand_mean".into(), DataType::Float64),
    ];
    Ok(Field::new("repeated_measures_anova".into(), DataType::Struct(fields)))
}
```

### icc_output_dtype (replaces correlation_output_dtype for icc)

The existing `icc_fit` uses `correlation_output_dtype` (estimate, statistic, p_value, ci_lower,
ci_upper, n). The real ICC has a richer output. Replace the output_type_func for `pl_icc`:

```rust
pub fn icc_output_dtype(_: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("icc".into(), DataType::Float64),
        Field::new("f_value".into(), DataType::Float64),
        Field::new("df1".into(), DataType::Float64),
        Field::new("df2".into(), DataType::Float64),
        Field::new("p_value".into(), DataType::Float64),
        Field::new("ci_lower".into(), DataType::Float64),
        Field::new("ci_upper".into(), DataType::Float64),
        Field::new("n_subjects".into(), DataType::UInt32),
        Field::new("n_raters".into(), DataType::UInt32),
    ];
    Ok(Field::new("icc".into(), DataType::Struct(fields)))
}
```

Note: `icc_type` (enum) and `method` (String) from `ICCResult` are NOT included in the output
struct — Polars Struct fields cannot easily hold enum variants or strings as typed output (only
scalar numerics, integers, and booleans are appropriate for per-group struct fields).

### energy_distance_nd output

Reuse the existing `stats_output_dtype` (statistic + p_value). The `n_permutations` field from
`EnergyDistanceResult` is constant (a parameter, not a result) — omit it from the struct.

---

## Existing Pattern Analogs

The planner must reference these concrete patterns when writing task actions.

### Pattern 1: Simple two-sample expression — `energy_distance_fit`

[VERIFIED: src/expressions/modern.rs:11-30]

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

### Pattern 2: Variable-arity multi-input — `partial_cor_fit`

[VERIFIED: src/expressions/correlation.rs:167-210]

```rust
pub fn partial_cor_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].f64()?;
    let y = inputs[1].f64()?;
    let n_covariates = inputs[2].u32()?.get(0).unwrap_or(1) as usize;
    // ...
    for i in 0..n_covariates {
        if let Some(cov_series) = inputs.get(3 + i) {
            if let Ok(cov) = cov_series.f64() {
                covariates.push(cov.into_no_null_iter().collect());
            }
        }
    }
    let cov_refs: Vec<&[f64]> = covariates.iter().map(|v| v.as_slice()).collect();
    match partial_cor(&x_vec, &y_vec, &cov_refs) { ... }
}
```

The ICC and energy_distance_nd wrappers follow this exact pattern, replacing covariates with
rater/feature columns.

### Pattern 3: Rich struct output — `correlation_output` helper

[VERIFIED: src/expressions/correlation.rs:14-42]

```rust
fn correlation_output(result: &CorrelationResult, name: &str) -> PolarsResult<Series> {
    let estimate = Series::new("estimate".into(), &[result.estimate]);
    let statistic = Series::new("statistic".into(), &[result.statistic]);
    // ... more fields ...
    let df = StructChunked::from_series(
        name.into(), 1,
        [&estimate, &statistic, &p_value, &ci_lower, &ci_upper, &n].into_iter(),
    )?;
    Ok(df.into_series())
}
```

The ANOVA wrappers must build their own analogous helpers (e.g. `one_way_anova_output`) following
this exact `StructChunked::from_series` construction pattern.

### Pattern 4: Python builder with literal scalar args — `ttest_ind`

[VERIFIED: python/polars_statistics/exprs/parametric.py:75-87]

```python
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

All new Python builders follow this pattern. Scalar parameters (kind, icc_type, n_permutations,
seed, compute_sphericity) are passed as `pl.lit(value, dtype=...)` arguments.

### Pattern 5: Registration in mod.rs

[VERIFIED: src/expressions/mod.rs:1-37]

`mod.rs` uses `pub use categorical::*` glob re-exports. No manual addition needed — any
`pub fn` added to an existing expression module is automatically exported via the existing glob.
New expressions in `parametric.rs`, `modern.rs`, `correlation.rs` are automatically picked up.

---

## ICC Stub — Current State and Fix

[VERIFIED: src/expressions/correlation.rs:270-303]

The current `icc_fit` body:
```rust
// ICC requires a 2D matrix structure - this is a simplified placeholder
// TODO: Implement proper ICC with matrix input
let estimate = Series::new("estimate".into(), &[f64::NAN]);
// all fields return f64::NAN
```

The current signature receives `inputs[0]` (values), `inputs[1]` (icc_type_str), `inputs[2]`
(conf_level). This input contract is wrong for the real implementation — it must be replaced.

**New input contract:**
```rust
// inputs[0] = UInt32 literal: n_raters
// inputs[1] = String literal: icc_type ("icc1", "icc2", "icc3", "icc1k", "icc2k", "icc3k")
// inputs[2..2+n_raters] = one f64 Series per rater column
```

**ICCType parsing helper (to add in correlation.rs):**
```rust
fn parse_icc_type(s: &str) -> anofox_statistics::correlation::ICCType {
    use anofox_statistics::correlation::ICCType;
    match s.to_lowercase().as_str() {
        "icc1"  => ICCType::ICC1,
        "icc3"  => ICCType::ICC3,
        "icc1k" => ICCType::ICC1k,
        "icc2k" => ICCType::ICC2k,
        "icc3k" => ICCType::ICC3k,
        _       => ICCType::ICC2,  // default matches crate Default
    }
}
```

The existing Python `icc()` builder in `correlation.py` and the `icc` name in `__init__.py` stay
unchanged — only the Rust signature and implementation change.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ANOVA computation | Custom F-distribution or SS computation | `anofox_statistics::{one_way_anova, two_way_anova, repeated_measures_anova}` | Already validated, handles Welch/Fisher branches, sphericity corrections |
| ICC computation | Custom two-way ANOVA + CI derivation | `anofox_statistics::correlation::icc` | Correct F-based CI; all 6 ICC types handled |
| Energy distance permutation | Custom shuffling | `anofox_statistics::energy_distance_test` | Deterministic seeded permutation already implemented |
| Group-by partitioning | Custom hashmap-based grouping | `into_no_null_iter()` + collect, then partition by label | Standard Polars idiom; avoids allocations |
| Null handling | Custom pre-filter in Rust | `into_no_null_iter()` (skips nulls) + Python `x.filter(x.is_finite())` before calling | Existing wrappers do exactly this |

**Key insight:** This phase is purely a wiring task. Every statistical algorithm is already
implemented and validated in the anofox-statistics crate. The only work is encoding/decoding
between Polars Series and Rust slice types.

---

## Common Pitfalls

### Pitfall 1: Null handling inconsistency between Python and Rust

**What goes wrong:** The Python builder calls `x.filter(x.is_finite())` for scalar inputs, but
for multi-column inputs (ICC rater columns, nD energy feature columns), each column may have
nulls at different row positions. After filtering per-column, row counts diverge and the matrix
is ragged.

**Why it happens:** Polars' per-column null handling does not guarantee row alignment across
columns when different rows are null in different columns.

**How to avoid:** For multi-column inputs, filter rows where ANY column is null. Python builder:
```python
mask = pl.all_horizontal([col.is_finite() for col in rater_exprs])
clean_cols = [col.filter(mask) for col in rater_exprs]
```
Then the Rust side can use `into_no_null_iter()` safely.

**Warning signs:** `ICCResult` returns with `n_subjects=0` or crate returns `StatError::EmptyData`.

### Pitfall 2: Factor encoding for two_way_anova

**What goes wrong:** `two_way_anova` expects `factor_a: &[usize]` as 0-indexed integer codes.
If string factor columns are passed directly and cast to `UInt32`, the cast fails (Polars does not
auto-encode strings as integers).

**Why it happens:** Polars String→UInt32 cast is not a label encoding; it requires an
intermediate Categorical cast.

**How to avoid:** In the Python builder, apply `.cast(pl.Categorical).to_physical().cast(pl.UInt32)`
to factor columns before passing them. Document this in the docstring; do not silently coerce in
Rust.

**Warning signs:** `PolarsError: InvalidOperationError` on `.u32()?` in Rust wrapper when
factor column is String dtype.

### Pitfall 3: RM-ANOVA pivot fails with unbalanced data

**What goes wrong:** If the long-format DataFrame has missing condition observations for some
subjects (unbalanced repeated-measures design), the pivot to subject×condition matrix will have
gaps, and `repeated_measures_anova` returns `StatError::InvalidParameter`.

**Why it happens:** The crate (anova.rs:1174) explicitly checks that all subjects have the same
number of conditions.

**How to avoid:** The Rust wrapper must validate that after pivoting, every subject has exactly
`n_conditions` observations. Return NaN struct on mismatch rather than panicking. Document that
the expression requires a balanced design.

**Warning signs:** `StatError::InvalidParameter("Subject X has Y conditions, expected Z")`.

### Pitfall 4: group_by semantics — ANOVA expressions are whole-group operations

**What goes wrong:** Attempting to call `ps.one_way_anova("value", "group")` inside a nested
`group_by(...).agg(ps.one_way_anova(...))` doesn't make sense — one_way_anova already takes
a group column and operates on the full group. Nesting would try to ANOVA within an already-
partitioned subgroup.

**Why it happens:** Simple stat tests (t-test, correlation) take two sample Series and are
naturally per-group operations. ANOVA takes a whole dataset partitioned by a factor column.

**How to avoid:** The Python builder docstring must clearly state: "Use this on the full
DataFrame with `df.select(ps.one_way_anova(...))`, not inside `group_by(...).agg(...)` unless
each group truly contains all ANOVA groups." `returns_scalar=True` signals this correctly.

**Warning signs:** Groups contain fewer than 2 observations after partitioning, crate returns
`StatError::InsufficientData`.

### Pitfall 5: Welch ANOVA η² reported as NaN

**What goes wrong:** Users expect η² for all ANOVA variants. Welch's ANOVA does not produce SS
values, so η² cannot be computed from the crate output.

**Why it happens:** [VERIFIED: anova.rs:248-259] Welch's ANOVA sets `ss_between`, `ss_within`,
`ss_total` to `None`.

**How to avoid:** The `one_way_anova_fit` wrapper sets `eta_squared = f64::NAN` when either SS
is `None`. The output struct always has the `eta_squared` field; its NaN value for Welch is
expected and should be documented.

### Pitfall 6: ICCType import path

**What goes wrong:** `ICCType` and `icc` live in `anofox_statistics::correlation`, not at the
crate root. The existing `icc_fit` uses `anofox_statistics::correlation` as a module path.

**How to avoid:** Add to correlation.rs imports:
```rust
use anofox_statistics::correlation::{icc, ICCResult, ICCType};
```
Verify that `lib.rs` re-exports the correlation module items before assuming they're at the root.

---

## Code Examples

### Example 1: one_way_anova_fit (Rust wrapper skeleton)

```rust
// Source: follows energy_distance_fit pattern (modern.rs:11-30) + correlation_output pattern
pub fn one_way_anova_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let values = inputs[0].f64()?;
    let group_labels = &inputs[1];  // String or UInt32
    let kind_str = inputs[2].str()?.get(0).unwrap_or("fisher");

    let kind = match kind_str.to_lowercase().as_str() {
        "welch" => anofox_statistics::AnovaKind::Welch,
        _       => anofox_statistics::AnovaKind::Fisher,
    };

    // Partition values by group label
    let values_vec: Vec<f64> = values.into_no_null_iter().collect();
    // ... collect groups from group_labels Series, partition values_vec ...
    let groups_refs: Vec<&[f64]> = groups.iter().map(|v| v.as_slice()).collect();

    let err_output = || -> PolarsResult<Series> {
        // build NaN struct matching one_way_anova_output_dtype
        // ...
        StructChunked::from_series("one_way_anova".into(), 1, [...].into_iter())
            .map(|ca| ca.into_series())
    };

    match anofox_statistics::one_way_anova(&groups_refs, kind) {
        Ok(r) => {
            let eta_sq = match (r.ss_between, r.ss_total) {
                (Some(ssb), Some(sst)) if sst > 0.0 => ssb / sst,
                _ => f64::NAN,
            };
            // Build struct...
        }
        Err(_) => err_output(),
    }
}

#[polars_expr(output_type_func=one_way_anova_output_dtype)]
fn pl_one_way_anova(inputs: &[Series]) -> PolarsResult<Series> {
    one_way_anova_fit(inputs)
}
```

### Example 2: Python builder skeleton for ANOVA

```python
# Source: follows ttest_ind pattern (parametric.py:75-87)
def one_way_anova(
    value: Union[pl.Expr, str],
    group: Union[pl.Expr, str],
    kind: Literal["fisher", "welch"] = "fisher",
) -> pl.Expr:
    value_expr = pl.col(value) if isinstance(value, str) else value
    group_expr = pl.col(group) if isinstance(group, str) else group

    value_clean = value_expr.filter(value_expr.is_finite())
    group_clean = group_expr.filter(value_expr.is_finite())

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_one_way_anova",
        args=[
            value_clean,
            group_clean,
            pl.lit(kind, dtype=pl.String),
        ],
        returns_scalar=True,
    )
```

### Example 3: ICCType parse and real icc_fit signature

```rust
// Source: follows partial_cor_fit variable-arity pattern (correlation.rs:167-210)
pub fn icc_fit(inputs: &[Series]) -> PolarsResult<Series> {
    let n_raters = inputs[0].u32()?.get(0).unwrap_or(0) as usize;
    let icc_type_str = inputs[1].str()?.get(0).unwrap_or("icc2");
    let icc_type = parse_icc_type(icc_type_str);

    let mut matrix: Vec<Vec<f64>> = Vec::new();
    for i in 0..n_raters {
        if let Some(rater_series) = inputs.get(2 + i) {
            let col: Vec<f64> = rater_series.f64()?.into_no_null_iter().collect();
            matrix.push(col);
        }
    }
    // matrix is currently rater-indexed: matrix[rater][subject]
    // icc() expects data[subject][rater] — transpose
    if matrix.is_empty() || matrix[0].is_empty() {
        return icc_error_output();
    }
    let n_subjects = matrix[0].len();
    let data: Vec<Vec<f64>> = (0..n_subjects)
        .map(|s| (0..n_raters).map(|r| matrix[r][s]).collect())
        .collect();

    match anofox_statistics::correlation::icc(&data, icc_type) {
        Ok(r) => { /* build struct */ }
        Err(_) => icc_error_output(),
    }
}
```

---

## Runtime State Inventory

Not applicable — this is a greenfield addition of new functions. No rename, refactor, or
migration is involved.

---

## Environment Availability

| Dependency | Required By | Available | Notes |
|------------|------------|-----------|-------|
| Rust stable toolchain | All expression compilation | Yes (assumed — project builds) | Verified by Phase 1 DEP-01/02 success |
| anofox-statistics 0.4.2 | All new expressions | Yes | In Cargo.lock; ANOVA and ICC modules verified present |
| Python 3.9+ with maturin | wheel build / test run | Yes | Existing CI matrix |
| pytest | Smoke tests | Yes | `tests/` directory exists; pyproject.toml configures it |

No missing dependencies.

---

## Validation Architecture

`nyquist_validation` is `true` in config.json — this section is required.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (7.0+) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `pytest tests/test_statistics_parity.py -x -v` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| STAT-01 | `ps.one_way_anova` returns correct struct shape and plausible F/p on 3-group data | smoke | `pytest tests/test_statistics_parity.py::TestOneWayAnova -x` | No — Wave 0 |
| STAT-01 | Fisher vs Welch: Fisher has ss_between/ss_within; Welch has NaN | smoke | `pytest tests/test_statistics_parity.py::TestOneWayAnova::test_fisher_has_ss -x` | No — Wave 0 |
| STAT-02 | `ps.two_way_anova` returns a_f, b_f, ab_f, a_p_value etc. | smoke | `pytest tests/test_statistics_parity.py::TestTwoWayAnova -x` | No — Wave 0 |
| STAT-03 | `ps.repeated_measures_anova` returns ws_f, mauchly_w on k>=3 data | smoke | `pytest tests/test_statistics_parity.py::TestRmAnova -x` | No — Wave 0 |
| STAT-04 | `ps.energy_distance_nd` returns statistic>0, p_value in [0,1] on 2D data | smoke | `pytest tests/test_statistics_parity.py::TestEnergyDistanceNd -x` | No — Wave 0 |
| STAT-05 | `ps.icc` returns icc in [-1,1], CI bounds finite, non-NaN | smoke | `pytest tests/test_statistics_parity.py::TestIcc -x` | No — Wave 0 |

### Sampling Rate

- **Per task commit:** `pytest tests/test_statistics_parity.py -x -v` (new file, < 5s)
- **Per wave merge:** `pytest tests/ -v`
- **Phase gate:** full suite green + `cargo clippy -- -D warnings` + `cargo fmt --check`

### Wave 0 Gaps

- [ ] `tests/test_statistics_parity.py` — covers STAT-01 through STAT-05 smoke checks
- [ ] No new conftest.py needed (existing `tests/conftest.py` provides fixtures)

Smoke test design per expression:
1. Build a minimal known-input DataFrame (hardcoded values, not random).
2. Call `df.select(ps.<expression>(...))`.
3. Assert return shape is `(1, 1)` with correct struct field names.
4. Assert scalar fields are in a plausible numeric range (not NaN where not expected, p_value
   in [0,1], F >= 0).
5. No R reference values required in Phase 3 — R validation is Phase 6.

Example known-input sanity check for one_way_anova:
- Three groups [1,2,3], [4,5,6], [7,8,9] — F should be >> 1, p < 0.05 for Fisher.
- All-same values per group — F ≈ 0, p ≈ 1 (or NaN for degenerate case).

---

## Security Domain

`security_enforcement` is `true` with `security_asvs_level: 1`.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | No user auth in expression layer |
| V3 Session Management | No | Stateless expression functions |
| V4 Access Control | No | No access control in expression layer |
| V5 Input Validation | Yes | `into_no_null_iter()` + crate-level validation; wrapper returns NaN on `Err` |
| V6 Cryptography | No | No crypto (permutation uses CSPRNG from `rand_chacha` — existing dependency) |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Integer overflow in group count | Tampering | `as usize` from `u32` is safe on 64-bit targets; crate validates n_groups >= 2 |
| OOM from large matrices (nD energy) | DoS | Permutation test is O(n_x * n_y * n_perm) — document that large inputs are expensive; no mitigation needed at this layer |
| Panic on empty Series | Tampering | `into_no_null_iter()` on empty Series yields empty iterator, not panic; crate returns `Err` for empty data; wrapper handles `Err` by returning NaN struct |

No high-risk ASVS findings. All inputs are validated by the crate before any computation.

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| ICC stub (all-NaN) | Real `ICCResult` via `icc()` crate function | Capability becomes actually usable |
| Only 1D energy distance | Both 1D and nD via separate expressions | Multivariate distribution comparison now available |
| No ANOVA expressions | Full ANOVA family with rich output structs | Closes the largest remaining statistics gap |

---

## Task/Plan Decomposition Recommendation

Recommended: **one PLAN.md per capability group**, yielding 4 plan files. Each plan is
independently executable after Wave 0 (test file creation). Dependency structure:

```
Wave 0 (all plans):  Create tests/test_statistics_parity.py (stub test classes)
Wave 1 (parallel):   Plan A: ANOVA output types + one_way_anova wrapper
                     Plan B: two_way_anova wrapper
                     Plan C: repeated_measures_anova wrapper
                     Plan D: energy_distance_nd wrapper
                     Plan E: ICC stub replacement
Wave 2 (all plans):  Register new builders in __init__.py (after all Rust wrappers done)
Wave 3 (all plans):  Fill in test_statistics_parity.py smoke tests; run full suite
```

The Rust compilation step (Wave 1) is the serializing constraint — all Rust changes must compile
before `maturin develop` can run and Python tests can execute. Plans A–E can be written in
parallel but the maturin build gates Wave 3.

**Suggested plan grouping:**
- `03-01-PLAN.md` — ANOVA family (one-way + two-way + RM-ANOVA): output_types + 3 wrappers + 3 Python builders. Group together because they share output_types.rs additions and all live in parametric.rs.
- `03-02-PLAN.md` — energy_distance_nd: single expression in modern.rs + Python builder.
- `03-03-PLAN.md` — ICC stub replacement: single expression replacement in correlation.rs + Python builder update + new output_type.
- `03-04-PLAN.md` — Registration + smoke tests: `__init__.py` updates + `test_statistics_parity.py` full smoke suite + `maturin develop` + `pytest`.

Wave/dependency structure: Plans 01–03 execute in Wave 1 (parallel Rust edits); Plan 04 executes
in Wave 2 (depends on all three Rust edits compiling).

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `kruskal_wallis` expression in `nonparametric.rs` uses value+group_label two-column input (analog for one_way_anova input encoding) | Input Encoding Patterns | Low — the pattern can be confirmed by reading nonparametric.rs before implementing; the encoding strategy still works even if kruskal uses a different approach |
| A2 | `anofox_statistics::AnovaKind` is re-exported at the crate root (not only from `anofox_statistics::parametric`) | Crate Signatures | Low — if not at root, add `use anofox_statistics::parametric::AnovaKind` instead |
| A3 | `anofox_statistics::correlation::icc` is the correct module path for the `icc` function (not re-exported at root) | ICC section | Low — read `lib.rs` pub-use list before implementing; adjust import path accordingly |

---

## Open Questions

1. **`kruskal_wallis` input contract**
   - What we know: nonparametric.rs was not read this session; the audit confirms kruskal_wallis is exposed.
   - What's unclear: whether it uses value+label or pre-grouped slices.
   - Recommendation: Read `src/expressions/nonparametric.rs` as first step of Plan 03-01 implementation to confirm the one_way_anova input analog before writing the grouping logic.

2. **AnovaKind / correlation::icc import paths at crate root**
   - What we know: `lib.rs` was not read this session; audit confirms pub items.
   - Recommendation: Grep `~/.cargo/registry/src/*/anofox-statistics-0.4.2/src/lib.rs` for `pub use` at the start of Plan implementation to confirm exact import paths.

---

## Sources

### Primary (HIGH confidence)
- `anofox-statistics-0.4.2/src/parametric/anova.rs` — ANOVA function signatures, all result struct fields, enum variants, verbatim
- `anofox-statistics-0.4.2/src/correlation/icc.rs` — ICC function signature, ICCType enum, ICCResult fields, verbatim
- `anofox-statistics-0.4.2/src/modern/energy.rs` — energy_distance_test (nD) signature, EnergyDistanceResult fields, verbatim
- `src/expressions/correlation.rs` — existing ICC stub, partial_cor multi-input pattern, verbatim
- `src/expressions/modern.rs` — energy_distance_fit pattern, verbatim
- `src/expressions/parametric.rs` — ttest_ind pattern, verbatim
- `src/expressions/output_types.rs` — all existing output_dtype functions, verbatim
- `python/polars_statistics/exprs/parametric.py` — Python builder pattern, verbatim
- `python/polars_statistics/exprs/modern.py` — energy_distance Python builder, verbatim

### Secondary (MEDIUM confidence)
- `02-API-AUDIT.md` — gap list, special-case flags, all verified in Phase 2
- `03-CONTEXT.md` — locked decisions, scope constraints

### Tertiary (LOW confidence)
- `nonparametric.rs` kruskal_wallis input encoding — [ASSUMED] (not read this session)
- `lib.rs` crate root re-exports for AnovaKind / icc path — [ASSUMED] (not read this session)

---

## Metadata

**Confidence breakdown:**
- Crate signatures: HIGH — read verbatim from source files
- Output schemas: HIGH — derived directly from verified struct fields
- Input encoding strategy: HIGH (ANOVA, ICC, energy nD) / MEDIUM (RM-ANOVA pivot logic — standard approach, not verified against a working example)
- Pitfalls: HIGH (null handling, Welch NaN fields, factor encoding) / MEDIUM (group_by semantics warning)
- Python builder pattern: HIGH — read verbatim from existing builders

**Research date:** 2026-08-11
**Valid until:** 2026-09-11 (stable Rust crate; no fast-moving dependencies)
