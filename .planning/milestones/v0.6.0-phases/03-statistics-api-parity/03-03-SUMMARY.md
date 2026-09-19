---
phase: 03-statistics-api-parity
plan: "03"
subsystem: correlation-expressions
tags: [icc, rust, pyo3, polars-expr, statistics, STAT-05]
status: complete

requires:
  - "03-01-SUMMARY.md"
depends_on: ["03-01"]
provides:
  - "ps.icc(*rater_cols, icc_type=) — real matrix-input ICC (STAT-05)"
  - "parse_icc_type, icc_error_output helpers in correlation.rs"
affects:
  - "src/expressions/correlation.rs"
  - "python/polars_statistics/exprs/correlation.py"
  - "src/expressions/output_types.rs (used read-only — icc_output_dtype already present)"

tech_stack:
  added: []
  patterns:
    - "variable-arity rater-column input via partial_cor_fit pattern"
    - "subjects×raters matrix transpose before calling anofox_statistics::icc()"
    - "pl.all_horizontal finite mask for multi-column null alignment"

key_files:
  created: []
  modified:
    - "src/expressions/correlation.rs — icc_fit replaced, parse_icc_type + icc_error_output added, pl_icc output_type_func changed"
    - "python/polars_statistics/exprs/correlation.py — icc() builder rewritten to *rater_cols + icc_type"
    - "tests/test_correlation.py — TestICC.test_icc_basic updated to new multi-rater API"
    - "tests/test_statistics_parity.py — TestIcc class (5 smoke tests) added for STAT-05"

decisions:
  - "Use anofox_statistics::icc (crate-root re-export) not correlation::icc — confirmed via lib.rs grep"
  - "Zero-rater guard in Python builder (pl.all_horizontal([]) throws) — send n_raters=0 literals, Rust guard handles empty matrix"
  - "Pre-existing ruff UP violations in correlation.py left out of scope (not in CI, pre-existing before this plan)"

metrics:
  duration: "~7 minutes"
  completed: "2026-08-12T05:40:49Z"
  tasks_completed: 2
  tasks_total: 2
  commits: 1
  files_changed: 4

actuals:
  tokens: 18000
  tasks: 2
  commits: 1
---

# Phase 03 Plan 03: ICC Stub Replacement Summary

Real matrix-input ICC via `anofox_statistics::icc()` + `ICCType` replacing the all-NaN stub — STAT-05 fully implemented.

## What Was Built

Replaced the all-NaN `icc_fit` stub in `src/expressions/correlation.rs` with a real matrix-input ICC using the `anofox_statistics::icc()` function. The old stub (which returned NaN for every field regardless of input) is now a fully-functional expression that:

- Accepts `n_raters` (UInt32 literal), `icc_type` (String literal), then one f64 Series per rater column
- Transposes the rater-indexed matrix to subjects-indexed before calling the crate
- Returns a 9-field struct: `icc`, `f_value`, `df1`, `df2`, `p_value`, `ci_lower`, `ci_upper`, `n_subjects`, `n_raters`
- Uses the existing `icc_output_dtype` from `output_types.rs` (pre-added by plan 03-01)
- Returns the all-NaN error struct on empty/degenerate input without panicking

The Python `icc()` builder was rewritten from a single-column signature to `icc(*rater_cols, icc_type="icc2")`, applying a shared `pl.all_horizontal` finite mask across all rater columns to prevent row-alignment divergence.

## Commits

| Hash | Message |
|------|---------|
| 113c2a6 | feat(03-03): replace all-NaN icc_fit stub with real matrix-input ICC (STAT-05) |

## Smoke Test Results

All 5 new TestIcc tests pass, plus 40 existing correlation tests remain green (45 total):

- `test_returns_correct_schema` — struct has all 9 expected fields
- `test_icc_value_finite_and_in_range` — icc in [-1,1], n_subjects=4, n_raters=3
- `test_ci_bounds_finite` — ci_lower <= ci_upper, both finite
- `test_icc_type_parameter_honored` — icc2 vs icc3 produce different values on same data
- `test_degenerate_no_raters_returns_nan_struct` — zero-rater call returns NaN struct, no panic

Sample output on 4 subjects × 3 raters with strong agreement:
```
icc=0.992, f_value=397.4, df1=3.0, df2=6.0, p_value=2.7e-07,
ci_lower=0.952, ci_upper=0.999, n_subjects=4, n_raters=3
```

## CI Gate Results

| Gate | Result |
|------|--------|
| `cargo build --features python` | PASS |
| `cargo clippy --all-targets --features python -- -D warnings` | PASS |
| `cargo fmt --check` | PASS |
| `pytest tests/test_statistics_parity.py tests/test_correlation.py` | 45/45 PASS |
| ruff (not in CI) | Pre-existing 21 violations in file (all in pre-03-03 code); ruff not run in `.github/workflows/ci.yml` |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Zero-rater Python guard**
- **Found during:** Task 1 verification
- **Issue:** `pl.all_horizontal([])` with an empty list throws `ComputeError: cannot return empty fold` when the Python builder is called with zero rater columns
- **Fix:** Added early-return path in Python builder when `len(rater_exprs) == 0` — sends `n_raters=0` and only the two literals to Rust; the Rust `matrix.is_empty()` guard returns `icc_error_output()`
- **Files modified:** `python/polars_statistics/exprs/correlation.py`
- **Commit:** 113c2a6

### Pre-existing Out-of-Scope Issues

**ruff UP violations in correlation.py** — 21 violations (UP035, UP007, UP006, UP045) in pre-existing functions using `Union[...]`, `List[...]`, `Optional[...]` type annotations. Confirmed pre-existing via `git stash` test: identical violations existed before any plan-03-03 change. Not caused by this plan. Logged below per broken-windows protocol.

## Known Stubs

None. The all-NaN ICC stub is fully replaced. The `// TODO: Implement proper ICC` comment is gone.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes at trust boundaries. All changes are within the existing expression layer:
- Input: user-provided f64 DataFrame columns (already inside Polars session)
- Output: 9-field scalar struct (numeric fields only)
- Error path: NaN struct on `Err`, no panic, no leakage

All STRIDE threats from the plan's threat model are mitigated:
- T-03-07: empty/single-rater guard via `matrix.is_empty()` check before calling crate
- T-03-08: shared `pl.all_horizontal` finite mask in Python builder
- T-03-09: `parse_icc_type` falls back to ICC2 on unknown strings

## Self-Check

**Checking created/modified files exist:**
- `src/expressions/correlation.rs` — FOUND, contains `pub fn icc_fit`, `parse_icc_type`, `icc_error_output`
- `python/polars_statistics/exprs/correlation.py` — FOUND, contains rewritten `icc(*rater_cols)`
- `tests/test_statistics_parity.py` — FOUND, contains `class TestIcc`
- `tests/test_correlation.py` — FOUND, updated `TestICC.test_icc_basic`

**Checking commits exist:**
- 113c2a6 — FOUND

**Checking TODO gone:**
- `grep "TODO: Implement proper ICC" src/expressions/correlation.rs` — NOT FOUND (stub removed)

**Checking output_type_func:**
- `grep "output_type_func=icc_output_dtype" src/expressions/correlation.rs` — FOUND

## Self-Check: PASSED
