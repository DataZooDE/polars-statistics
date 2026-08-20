---
phase: 03-statistics-api-parity
reviewed: 2026-08-12T10:00:00Z
depth: standard
files_reviewed: 10
files_reviewed_list:
  - src/expressions/parametric.rs
  - src/expressions/modern.rs
  - src/expressions/correlation.rs
  - src/expressions/output_types.rs
  - python/polars_statistics/exprs/parametric.py
  - python/polars_statistics/exprs/modern.py
  - python/polars_statistics/exprs/correlation.py
  - python/polars_statistics/__init__.py
  - python/polars_statistics/exprs/__init__.py
  - tests/test_statistics_parity.py
findings:
  critical: 3
  warning: 3
  info: 1
  total: 7
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-08-12T10:00:00Z
**Depth:** standard
**Files Reviewed:** 10
**Status:** issues_found

## Summary

Phase 3 adds the ANOVA family (one-way Fisher/Welch, two-way, repeated-measures), `energy_distance_nd`, and a real matrix-input ICC replacing the all-NaN stub. The struct field order is consistent between every `*_fit` StructChunked and its matching `*_output_dtype`. The variable-arity input parsing index math (kind literal at `inputs[inputs.len()-1]`, rater columns at `inputs[2..]`, dimension columns at `inputs[3..3+d]`) is correct. The ICC transpose (`matrix[rater][subject]` → `data[subject][rater]`) is correctly implemented. The repeated-measures ANOVA balanced-design check is logically sound.

Three correctness blockers were found, all in the Python builders, related to how the `is_finite()` mask interacts with non-value columns. Three warnings cover a missing matrix-bounds guard in ICC, a potential eta-squared miscalculation, and an inconsistency in encoding strategy between `two_way_anova` and `repeated_measures_anova`.

---

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: `two_way_anova` factor encoding computed before `is_finite()` filter — codes become non-zero-based when entire factor levels are eliminated

**File:** `python/polars_statistics/exprs/parametric.py:122-133`

**Issue:** `rank(method="dense")` is applied to the full (unfiltered) factor column, then `filter(value.is_finite())` is applied afterwards. If all rows belonging to a particular factor level have a non-finite value, that level is eliminated from the filtered column — but the remaining dense-rank codes retain their pre-filter values and no longer start at 0.

Concrete example: `factor_a = ["A", "B", "B", "C"]`, `values = [NaN, NaN, 3.0, 4.0]`. Dense ranks (over full column) = `[1, 2, 2, 3]`. After `filter(value.is_finite())` → `[2, 3]`. After `- 1` → `[1, 2]`. The Rust contract requires 0-indexed codes; `two_way_anova` receives `[1, 2]` with no 0, producing incorrect factor A degrees-of-freedom and sum-of-squares because the number of detected levels (max code + 1 = 3) is wrong.

**Fix:** Apply `rank("dense")` on the filtered column, not on the pre-filtered one. Rearrange to filter first, then encode:

```python
mask = value.is_finite() & factor_a.is_not_null() & factor_b.is_not_null()
value_clean = value.filter(mask)
# Encode AFTER filtering so codes are always 0..n_levels-1
factor_a_clean = (
    factor_a.filter(mask).rank(method="dense").cast(pl.UInt32) - pl.lit(1, dtype=pl.UInt32)
)
factor_b_clean = (
    factor_b.filter(mask).rank(method="dense").cast(pl.UInt32) - pl.lit(1, dtype=pl.UInt32)
)
```

---

### CR-02: `two_way_anova` and `repeated_measures_anova` masks do not exclude null factor/subject/condition values — `into_no_null_iter()` silently creates length misalignment

**File:** `python/polars_statistics/exprs/parametric.py:130` (two-way), `python/polars_statistics/exprs/parametric.py:206` (repeated-measures)

**Issue:** The finite mask is `value.is_finite()` only. It does not require factor_a, factor_b, subject, or condition to be non-null. When those columns contain nulls at rows where the value is finite, those rows pass the mask. After `filter`, the factor/subject/condition series retains those null entries. On the Rust side, `inputs[1].u32()?.into_no_null_iter().collect()` silently drops null rows, producing a `factor_a` vector shorter than the `values` vector. The crate then processes misaligned vectors, yielding silently incorrect results (not an error).

For `repeated_measures_anova` with `Categorical` encoding: `subject.cast(pl.Categorical).to_physical()` produces null (not an integer) for null subjects. The null rows pass `value.is_finite()` and enter the Rust function, where they are silently dropped by `into_no_null_iter()`, breaking the subject-index alignment assumed by the pivot step.

**Fix (two-way ANOVA):** Extend the mask (also fixes CR-01 if combined with filter-first encoding):
```python
mask = value.is_finite() & factor_a.is_not_null() & factor_b.is_not_null()
```

**Fix (repeated-measures ANOVA):** Extend the mask:
```python
mask = value.is_finite() & subject.is_not_null() & condition.is_not_null()
```

---

### CR-03: `repeated_measures_anova` uses global Polars Categorical catalog for subject/condition encoding — codes are non-deterministic across sessions and group contexts

**File:** `python/polars_statistics/exprs/parametric.py:202-203`

**Issue:** `subject.cast(pl.Categorical).to_physical()` depends on the Polars global string cache. In a fresh session the codes start at 0. But when the expression runs inside a `group_by(...).agg(...)`, each group is a sub-DataFrame slice; the Categorical catalog is global so earlier groups' values occupy code slots 0…k, and the first value in the current group may receive code k+1 instead of 0. The Rust function treats the codes as 0-indexed dense identifiers for the pivot step; a non-zero starting code causes `cell_map.get(&(subj, cond))` lookups to succeed but the `unique_subjects` vector to contain those large codes, which never appear in the decoded matrix, causing the balance check to incorrectly report an unbalanced design and return the all-NaN struct.

`two_way_anova` explicitly avoids this pattern with `rank(method="dense")` and documents the reason. `repeated_measures_anova` should use the same approach.

**Fix:**
```python
# Replace Categorical encoding with rank-based encoding (same as two_way_anova):
subject_enc = (
    subject.rank(method="dense").cast(pl.UInt32) - pl.lit(1, dtype=pl.UInt32)
)
condition_enc = (
    condition.rank(method="dense").cast(pl.UInt32) - pl.lit(1, dtype=pl.UInt32)
)
```

---

## Warnings

### WR-01: ICC Rust transpose uses `n_raters` from the scalar literal to index `matrix[]` without verifying `matrix.len() == n_raters` — theoretical out-of-bounds panic

**File:** `src/expressions/correlation.rs:327-343`

**Issue:** `n_raters` is read from `inputs[0]` (a scalar literal passed by Python). The loop `for i in 0..n_raters` collects columns via `inputs.get(2 + i)` and pushes only those that are `Some`. If the Polars plugin framework delivers fewer columns than `n_raters` says (edge case: expression rewrite, custom caller), `matrix.len() < n_raters`. The subsequent transpose at line 343:

```rust
.map(|s| (0..n_raters).map(|r| matrix[r][s]).collect())
```

will panic with an index-out-of-bounds when `r >= matrix.len()`. In normal Python-builder use this cannot occur, but the contract is undocumented and unguarded defensively.

**Fix:** Add a guard after the collection loop:
```rust
if matrix.len() != n_raters {
    return icc_error_output();
}
```

---

### WR-02: `one_way_anova` eta-squared silently returns NaN for Fisher ANOVA if `r.ss_total` is `None` — fallback to `ss_between + ss_within` is missing

**File:** `src/expressions/parametric.rs:205-207`

**Issue:**
```rust
let eta_sq = match (r.ss_between, r.ss_total) {
    (Some(ssb), Some(sst)) if sst > 0.0 => ssb / sst,
    _ => f64::NAN,
};
```

If the `anofox_statistics` crate's `OneWayAnovaResult` does not populate `ss_total` for the Fisher variant (returning `None`), eta-squared will be NaN even though both `r.ss_between` and `r.ss_within` are `Some` with finite values. The correct fallback is `ssb / (ssb + ssw)`, which is algebraically equivalent. There is no test that asserts `eta_squared` is finite for the Fisher case via the expression path (as opposed to an expected numeric value), so this silent NaN would go undetected until R-validation in Phase 6.

**Fix:**
```rust
let eta_sq = match (r.ss_between, r.ss_total, r.ss_within) {
    (Some(ssb), Some(sst), _) if sst > 0.0 => ssb / sst,
    (Some(ssb), None, Some(ssw)) if (ssb + ssw) > 0.0 => ssb / (ssb + ssw),
    _ => f64::NAN,
};
```

---

### WR-03: `energy_distance_nd` accepts `d=0` (two empty column lists) without raising `ValueError` — silently returns NaN output

**File:** `python/polars_statistics/exprs/modern.py:119-148`

**Issue:** The Python builder validates `len(x_cols) != len(y_cols)` but not `len(x_cols) == 0`. Calling `energy_distance_nd([], [], seed=42)` produces `d=0`, sends no data columns to Rust, and returns `{statistic: NaN, p_value: NaN}` silently. For a test that compares two multivariate distributions, receiving 0 dimensions with no error is misleading — the caller likely made a programming mistake. The test suite does not cover this degenerate input.

**Fix:**
```python
if len(x_cols) == 0:
    raise ValueError("x_cols and y_cols must be non-empty (d >= 1)")
```

---

## Info

### IN-01: Test suite has no null/NaN-in-factor coverage for `two_way_anova` or `repeated_measures_anova`

**File:** `tests/test_statistics_parity.py`

**Issue:** No test exercises a `two_way_anova` or `repeated_measures_anova` call where factor/subject/condition columns contain null or NaN values. The three critical bugs above (CR-01, CR-02, CR-03) would pass the current test suite undetected. The unbalanced-design test for repeated-measures ANOVA covers one edge case, but the null-alignment bug (CR-02) requires a different fixture (finite values with null identifiers).

**Fix:** Add fixtures with null factor entries:
```python
def test_two_way_anova_null_factor_row_excluded():
    # Row where factor_a is null but value is finite should be silently dropped,
    # not cause misalignment between values and factor_a vectors.
    df = pl.DataFrame({
        "v": [1.0, 2.0, 3.0, 4.0],
        "fa": ["A", None, "B", "A"],   # null in factor_a
        "fb": ["P", "P", "P", "Q"],
    })
    result = df.select(ps.two_way_anova("v", "fa", "fb"))
    v = result[0, 0]
    # Should not panic; n should reflect only the 3 non-null rows
    assert v["n"] == 3
```

---

_Reviewed: 2026-08-12T10:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
