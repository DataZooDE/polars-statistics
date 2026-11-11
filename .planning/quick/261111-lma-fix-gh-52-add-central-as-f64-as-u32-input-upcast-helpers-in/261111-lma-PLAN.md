---
phase: quick-261111-lma
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/expressions/inputs.rs
  - src/expressions/mod.rs
  - src/expressions/aft.rs
  - src/expressions/categorical.rs
  - src/expressions/correlation.rs
  - src/expressions/distributional.rs
  - src/expressions/forecast.rs
  - src/expressions/modern.rs
  - src/expressions/nonparametric.rs
  - src/expressions/parametric.rs
  - src/expressions/penalized_glm.rs
  - src/expressions/posthoc.rs
  - src/expressions/regression.rs
  - src/expressions/resampling.rs
  - src/expressions/shrinkage.rs
  - src/expressions/tost.rs
  - tests/rust_api.rs
  - tests/test_dtype_upcast.py
  - CHANGELOG.md
autonomous: true
requirements:
  - GH-52
must_haves:
  truths:
    - "ps.spearman on two UInt32 columns (the GH #52 repro) returns the same struct as on the same data cast to Float64, instead of raising 'expected `Float64`, got `u32`'."
    - "Every *_fit input that is read as Float64 accepts any primitive numeric dtype (Int8..Int64, UInt8..UInt64, Float32), Boolean (0.0/1.0) or an all-null column, through one helper: as_f64 in src/expressions/inputs.rs."
    - "Every *_fit input that is read as UInt32 accepts any integer dtype through a strict cast (as_u32). A negative or out-of-range value raises an error; it is never silently nulled, wrapped or truncated, and float dtypes are not accepted."
    - "Non-numeric inputs (e.g. String) still fail with the original dtype error that names `Float64`."
    - "Regression design-matrix builders and partial/semi-partial correlation covariates upcast integer columns. They no longer NaN-fill, zero-fill, panic on, or silently drop them."
    - "Results for Int32/UInt32/Int64/Float32 inputs equal the Float64 results in both the Rust integration test and the Python test suite."
  artifacts:
    - path: src/expressions/inputs.rs
      provides: "pub(crate) as_f64 / as_u32 input upcast helpers + #[cfg(test)] unit tests"
    - path: tests/rust_api.rs
      provides: "#[test] fn input_dtype_upcast (rlib-level dtype parity + strict-cast + error-preservation checks)"
    - path: tests/test_dtype_upcast.py
      provides: "dtype-parametrized Python parity tests (Int32/UInt32/Int64/Float32 vs Float64)"
    - path: CHANGELOG.md
      provides: "[Unreleased] ### Fixed entry referencing #52"
  key_links:
    - from: src/expressions/mod.rs
      to: src/expressions/inputs.rs
      via: "private `mod inputs;` with no `pub use` re-export (crate-internal API only)"
    - from: "*_fit bodies in src/expressions/*.rs"
      to: src/expressions/inputs.rs
      via: "fully-qualified super::inputs::as_f64(..)? / super::inputs::as_u32(..)? call sites (no new use statements)"
---

<objective>
Fix GitHub issue #52. `ps.spearman('char_tokens', 'char_types')` on a `u32` column fails with
"invalid series dtype: expected `Float64`, got `u32`". The cause is that every Rust `*_fit`
function reads its inputs through the strict typed Polars accessors. Some Python wrappers in
`python/polars_statistics/exprs/*.py` cast first and others don't, so behaviour is inconsistent.
The fix goes in the Rust layer so that Python users and Rust rlib `*_fit` callers both benefit.

Purpose: one crate-internal pair of upcast helpers (`as_f64`, `as_u32`) becomes the single input
coercion point for all of `src/expressions`. Every input accessor routes through them. This also
fixes some silent wrong results that come from the same root cause:
- regression X-matrix builders turned non-Float64 columns into NaN/0.0 cells, and in one place panicked;
- partial/semi-partial correlation silently dropped non-Float64 covariates.

Output: `src/expressions/inputs.rs` (helpers + unit tests), call-site conversions in 14 expression
files, a Rust integration test, a dtype-parametrized Python test module, and a CHANGELOG entry.
No version bump.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.claude/CLAUDE.md
@src/expressions/mod.rs
@src/expressions/correlation.rs
@tests/rust_api.rs

<interfaces>
New crate-internal API (src/expressions/inputs.rs). Not re-exported from `crate::expressions`:

```rust
pub(crate) fn as_f64(s: &Series) -> PolarsResult<Float64Chunked>;
pub(crate) fn as_u32(s: &Series) -> PolarsResult<UInt32Chunked>;
```

Both return OWNED ChunkedArrays. The old accessors returned `&Float64Chunked` / `&UInt32Chunked`.
Cloning a ChunkedArray is an Arc bump and does not copy data. Call sites name them by full path,
e.g. `super::inputs::as_f64(&inputs[0])?` (from any `crate::expressions::<module>` top-level fn,
`super` is `crate::expressions`).

A coordinator prototype of the helper bodies exists at
/tmp/claude-1000/-home-simonm-projects-rust-polars-statistics/51dd6dcf-e04c-4b63-82a1-5eaa287d343d/scratchpad/proto/src/expressions/inputs.rs
and compiles against polars 0.52. It may be gone by execution time; Task 1 fully specifies the behaviour.

Existing input contracts used by the new tests (see tests/rust_api.rs):
- spearman_fit / pearson_fit: [x, y, conf_level(f64 scalar)] -> struct{estimate, statistic, p_value, ci_lower, ci_upper, n}
- chisq_test_fit: [table(u32 flattened), n_rows(u32), n_cols(u32), correction(bool)] -> struct{statistic, p_value, df, n}
- ols_fit: [y, with_intercept(bool), solve_method(str|null), x1, ...] (see linear_regression_fits)
- partial_cor_fit: [x, y, n_covariates(u32), cov1, ...]
- kruskal_wallis_fit: [g1, g2, g3, ...]
- two_way_anova_fit: [values(f64), factor_a(u32 codes), factor_b(u32 codes)] (see anova_and_energy_fits ~L2503)
- condition_number_fit: [with_intercept, x1, ...] (see diagnostic_fits ~L1116)
- ols_predict_fit(&inputs, kwargs()) (see ~L490): goes through build_xy_with_null_policy
</interfaces>

Sibling-item coupling (261111-lm9, same files, applied serially in either order):
- lm9 inserts `#[cfg(feature = "python")]` lines above every `use pyo3_polars::derive::polars_expr;`
  and every `#[polars_expr` line.
- lm9 splits/cfg-gates some `use crate::expressions::...` statements.
- lm9's verify counts `allow(dead_code)` / `allow(unused` attributes in src/ and expects that baseline unchanged.

Therefore this plan edits ONLY call sites inside fn bodies, plus one new `mod inputs;` line in
mod.rs. It adds NO `use` statements to the expression files and NO `#[allow(...)]` attributes
anywhere.

Measured baselines (HEAD e026384, unaffected by lm9's attribute-only edits):
- single-index typed accessors on `inputs` elements in src/expressions: 332 (256 Float64 + 76 UInt32), 0 of them in comment lines
- non-comment `.f64()`/`.u32()` accessor lines outside inputs.rs and aft.rs: 363
- `.f64()`/`.u32()` accessor lines in aft.rs: 5
- `^mod inputs;` in mod.rs: 0; `#52` in CHANGELOG.md: 0; tests/test_dtype_upcast.py: absent
- no existing test asserts the old "expected Float64" error (grep of tests/ is clean)
</context>

<tasks>

<task type="tracer">
  <name>Task 1: Tracer — as_f64/as_u32 helpers wired end-to-end through spearman_fit and chisq_test_fit</name>
  <files>src/expressions/inputs.rs, src/expressions/mod.rs, src/expressions/correlation.rs, src/expressions/categorical.rs, tests/rust_api.rs</files>
  <action>
Wire the GH #52 repro path (spearman) and one UInt32 path (chisq) through the new helpers
end-to-end at the rlib public-API level. This proves the helper contract before the bulk conversion.

1. Create src/expressions/inputs.rs.
   - Start with a `//!` module doc: these are the shared input-coercion helpers for every expression
     `*_fit` function, and they exist to fix GH #52. Each fn gets `///` docs (project convention).
   - Both fns are `pub(crate)`. Do not use `pub`: the public Rust API must not grow.
   - `as_f64(s: &Series) -> PolarsResult<Float64Chunked>`, matching on `s.dtype()`:
     - `DataType::Float64`: return a clone of the existing chunked array.
     - any dtype where `is_primitive_numeric()`, `is_bool()` or `is_null()` is true: `s.cast(&DataType::Float64)?`, then take its Float64 chunked array and clone it.
     - every other dtype: fall through to the plain Float64 accessor, so the caller still gets Polars' original "expected `Float64`" error. This covers String, Date, Datetime, Duration, Decimal, Categorical, List and Struct.
   - `as_u32(s: &Series) -> PolarsResult<UInt32Chunked>`, matching on `s.dtype()`:
     - `DataType::UInt32`: clone.
     - any dtype where `is_integer()` or `is_null()` is true: `s.strict_cast(&DataType::UInt32)?`, then clone its UInt32 chunked array. Strict cast is required: a negative or out-of-range value must ERROR. Never use a non-strict cast, which would silently null it.
     - every other dtype, including floats and Boolean: fall through to the plain UInt32 accessor's original error. No silent float truncation.
   - The doc comments must state:
     - nulls are preserved;
     - Boolean maps to 0.0/1.0;
     - integers with magnitude above 2^53 round to the nearest f64 (the same as the Python wrappers' existing `.cast(pl.Float64)`);
     - as_u32 rejects floats on purpose.
2. In the same file, add a `#[cfg(test)] mod tests` covering:
   - as_f64 on Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32 and Boolean series returns the expected f64 values. Include one series with a null and check it stays null.
   - as_f64 on a Float64 series returns equal values.
   - as_f64 on a Null-dtype series (`Series::new_null`) returns an all-null Float64Chunked of the same length.
   - as_f64 on a String series is Err, and the error text contains "Float64".
   - as_u32 on Int64 [0, 5, 7] returns [0, 5, 7], and on UInt64 it is Ok.
   - as_u32 is Err for Int64 [-1], for Int64 [u32::MAX as i64 + 1], for Float64 and for Boolean.
   - The dtype-i8/i16/u8/u16 polars features are already enabled in Cargo.toml.
3. In src/expressions/mod.rs, add the line `mod inputs;` between `mod forecast;` and `mod modern;`.
   - Do NOT add a glob re-export line for this module, unlike the other modules in mod.rs. The helpers stay crate-private.
   - Do NOT add any `#[allow(...)]` attribute (sibling 261111-lm9 counts them).
4. In src/expressions/correlation.rs `spearman_fit`, change the three reads of `inputs[0]`, `inputs[1]`
   and `inputs[2]` to `super::inputs::as_f64(&inputs[N])?`. Keep the `.get(0)` on the conf_level read.
   In src/expressions/categorical.rs `chisq_test_fit`, change the reads of the table (`inputs[0]`),
   n_rows (`inputs[1]`) and n_cols (`inputs[2]`) to `super::inputs::as_u32(&inputs[N])?`. Leave the
   `.bool()` read alone.
   - Use the fully-qualified `super::inputs::` path, per the coordinator's "call sites only" rule. Do
     NOT add `use` statements: sibling 261111-lm9 rewrites the `use` blocks of these files, and
     call-site-only edits keep the two diffs merge-disjoint.
   - If the compiler reports a borrow mismatch, add `&` at the use site. The helpers return owned arrays.
5. In tests/rust_api.rs, append a new `#[test] fn input_dtype_upcast()` at the end of the file.
   - Use small non-negative integral data so every dtype represents it exactly.
   - Add a local field-reader helper if no existing one fits.
   - Cases:
     - (a) spearman_fit with x as UInt32 and y as Int64 is Ok. Its `estimate` and `p_value` equal (abs diff < 1e-12) those of spearman_fit on the same data as Float64, and `n` is nonzero (assert_n_obs_nonzero).
     - (b) spearman_fit with a String x is Err, and `err.to_string()` contains "Float64".
     - (c) chisq_test_fit with an Int64 table [10, 20, 30, 40] and Int32 n_rows/n_cols scalars gives the same `statistic` and `p_value` as the UInt32 version.
     - (d) chisq_test_fit with an Int64 table [10, -1, 30, 40] is Err. This is the strict cast at work: no silent null.
   - Add one line to the file's top `//!` test-depth doc noting dtype-upcast coverage.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/polars-statistics && test -f src/expressions/inputs.rs && grep -qE '^pub\(crate\) fn as_f64\(' src/expressions/inputs.rs && grep -qE '^pub\(crate\) fn as_u32\(' src/expressions/inputs.rs && ! grep -qE '^pub fn as_' src/expressions/inputs.rs && grep -q '^mod inputs;' src/expressions/mod.rs && ! grep -q 'pub use inputs' src/expressions/mod.rs && cargo test --no-default-features --lib expressions::inputs 2>&1 | tee /dev/stderr | grep -qE 'test result: ok\. [1-9][0-9]* passed' && cargo test --no-default-features --test rust_api input_dtype_upcast 2>&1 | tee /dev/stderr | grep -qE 'test result: ok\. 1 passed' && cargo fmt --all -- --check && cargo clippy --all-targets -- -D warnings</automated>
  </verify>
  <done>
- inputs.rs exists with exactly two `pub(crate)` helpers and a passing unit-test module.
- mod.rs registers `mod inputs;` privately.
- spearman_fit and chisq_test_fit read through the helpers.
- `input_dtype_upcast` passes: UInt32/Int64 spearman equals Float64; String still errors naming Float64; Int64 chisq table equals UInt32; a negative count errors.
- fmt and clippy (-D warnings, all targets) are clean.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Route every remaining src/expressions input read through the helpers (mechanical + manual + hoisted design-matrix builders)</name>
  <files>src/expressions/aft.rs, src/expressions/categorical.rs, src/expressions/correlation.rs, src/expressions/distributional.rs, src/expressions/forecast.rs, src/expressions/modern.rs, src/expressions/nonparametric.rs, src/expressions/parametric.rs, src/expressions/penalized_glm.rs, src/expressions/posthoc.rs, src/expressions/regression.rs, src/expressions/resampling.rs, src/expressions/shrinkage.rs, src/expressions/tost.rs, tests/rust_api.rs</files>
  <behavior>
    - ols_fit with Int64 x1 (the linear_xy data) gives intercept ~1.0 and slope ~2.0, equal (abs diff < 1e-9) to the Float64 path. Before this change the design matrix was NaN-filled.
    - ols_predict_fit with an Int64 x column (build_xy_with_null_policy, default drop policy) is Ok and its predictions equal the Float64 path. Before this change it could panic on an unwrap.
    - condition_number_fit with Int64 x columns equals the Float64 result (a hoisted diagnostic builder).
    - partial_cor_fit with an Int64 covariate equals the Float64-covariate result, and its estimate differs from pearson_fit(x, y) by more than 1e-6. Before this change the covariate was silently skipped.
    - kruskal_wallis_fit with Int32 groups equals the Float64 groups result (the map-closure path).
    - two_way_anova_fit with Int64 factor codes equals the UInt32-codes result (the multi-line UInt32 path).
  </behavior>
  <action>
This touches 14 source files, but most of the edit is one scripted substitution, so context cost is
low. The judgment work is the manual sites listed in Step C and Step D. Keep every edit inside fn
bodies. Add no `use` statements and no `#[allow(...)]` attributes. Both rules exist for the
sibling-coupling reasons given in Task 1, step 4.

Step A (mechanical). From the repo root run
`perl -pi -e 's/\binputs\[(\w+)\]\.f64\(\)\?/super::inputs::as_f64(&inputs[$1])?/g; s/\binputs\[(\w+)\]\.u32\(\)\?/super::inputs::as_u32(&inputs[$1])?/g' src/expressions/*.rs`.
It rewrites every single-index typed read of an `inputs` element (about 326 remaining after Task 1).
It is idempotent.

Step B (compile fallout). Run `cargo check --all-targets` and fix each error at its use site.
- The known case is posthoc.rs `elementwise4`. `cols` now holds owned arrays, so the four reads of
  the form `get(cols[N], i)` become `get(&cols[N], i)`.
- Fix any other borrow mismatch by adding `&` at the use. Never revert to the typed accessor.

Step C (manual sites the regex cannot reach). Convert each one to the helper. Line numbers are
approximate at HEAD and shift by a line or so if lm9's cfg lines are present.
- forecast.rs: the three `.map(|s| ...)` closures over model-loss inputs (~L133, ~L210, ~L268) become `.map(|s| super::inputs::as_f64(s).map(|ca| ca.into_no_null_iter().collect()))`.
- nonparametric.rs kruskal_wallis_fit (~L120) and parametric.rs one_way_anova group collection (~L286): the same closure rewrite.
- parametric.rs two_way_anova_fit (~L407, ~L412): the multi-line UInt32 reads of `inputs[1]` and `inputs[2]` become `super::inputs::as_u32(&inputs[1])?` and the same for `inputs[2]`.
- correlation.rs icc_fit rater loop (~L329, `rater_series`).
- modern.rs energy-distance ND column loops (~L74, ~L83, `col_series`).
- categorical.rs trailing_conf_level (~L25, `s`).
- posthoc.rs long_format (~L318, `values`).
- penalized_glm.rs series_to_vec_nan (~L394, `s`) and resolve_offset (~L416, `offset`).
- aft.rs aft_inputs: the time column (~L146) and the feature columns (~L207). Keep their `.map_err(|e| format!(...))` String-error wrappers, and reword "must be Float64" to "must be numeric".
- correlation.rs partial_cor_fit (~L180) and semi_partial_cor_fit (~L231): replace the `if let Ok(cov) = ...` skip with `let cov = super::inputs::as_f64(cov_series)?;` followed by the push. A covariate that cannot be converted must now raise. It must not be silently dropped.

Step D (hoist design-matrix reads in regression.rs out of `Mat::from_fn` closures).
- Each site below currently does a typed read plus `.ok()` per CELL. Replace it with ONE conversion
  per column, computed immediately before the `Mat::from_fn`, after the function's existing
  early-return guards so those keep their current behaviour. The conversion is a
  `Vec<Float64Chunked>` collected from `super::inputs::as_f64(&inputs[start + c])` over the feature
  range with `.collect::<PolarsResult<Vec<_>>>()?`.
- The closure then reads `x_cols[col].get(row)` (or `get(valid_indices[row])`) with that site's
  EXISTING null fill (NaN or 0.0, unchanged).
- Sites:
  - build_xy_data (~L308).
  - condition_number_fit (~L1247), check_binary_separation_fit (~L1354), check_count_sparsity_fit (~L1406), vif_fit (~L1475, index `inputs[col]`), leverage_fit (~L1539), high_leverage_points_fit (~L1925, fill 0.0), high_vif_predictors_fit (~L2023), generalized_vif_fit (~L2087).
  - augment_fit: hoist once BEFORE the validity scan (~L3004), over the x columns from index 2. Iterate the hoisted columns in that scan, and reuse them in the ~L3043 closure (keep its NaN fallback). Update the invariant comment above it so it no longer names the removed accessor. This is a comment-only change.
  - build_xy_with_null_policy (~L4326–4440): hoist once at the top for both policies. Replace the drop-policy null scan's `if let Ok(col) = ...` with iteration over the hoisted columns, and use them in all four `Mat::from_fn` closures. The drop-policy x_fit keeps its `.unwrap()`, which is now sound because the scan covers every column.
- Choice made here: non-numeric X columns now return an Err from these functions. Before, they were silently NaN- or zero-filled, or panicked. This matches every other input in the crate.
- Fallback if needed: if the full suites in Task 3 show an existing test depended on the old silent
  fallback for a non-numeric column, preserve that one function's old behaviour. Hoist
  `super::inputs::as_f64(..).ok()` into a `Vec<Option<Float64Chunked>>` for it, and record the
  deviation in the SUMMARY.

Step E (leave alone):
- aft.rs's two reads of the already-cast event column (`casted`, ~L171 and ~L579);
- every `.bool()`, `.str()` and `.u64()` parameter read;
- any read of an output or intermediate Series.

Step F (tests). Extend `input_dtype_upcast` in tests/rust_api.rs with the cases in `<behavior>`.
Mirror each fn's existing call shape in rust_api.rs, e.g. ols_predict_fit's `kwargs()` near L490 and
condition_number_fit near L1116.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/polars-statistics && test "$(grep -rnP 'inputs\[\w+\]\.(f64|u32)\(\)\?' src/expressions | wc -l)" -eq 0 && test "$(grep -rnP '\.(f64|u32)\(\)' src/expressions | grep -v '^src/expressions/inputs.rs:' | grep -v '^src/expressions/aft.rs:' | grep -vP '^[^:]+:\d+:\s*//' | wc -l)" -eq 0 && test "$(grep -nP '\.(f64|u32)\(\)' src/expressions/aft.rs | grep -vP '^\d+:\s*//' | wc -l)" -eq 2 && test "$(grep -rnoP 'super::inputs::as_(f64|u32)\(' src/expressions | wc -l)" -ge 340 && ! grep -q 'pub use inputs' src/expressions/mod.rs && cargo fmt --all -- --check && cargo clippy --all-targets -- -D warnings && { ! grep -q 'cargo clippy --no-default-features --lib' .github/workflows/ci.yml || cargo clippy --no-default-features --lib -- -D warnings; } && cargo test --no-default-features --lib && cargo test --no-default-features --test rust_api 2>&1 | tee /dev/stderr | grep -qE 'test result: ok\. [1-9][0-9]* passed; 0 failed'</automated>
  </verify>
  <done>
- No single-index typed accessor on an `inputs` element remains in src/expressions (baseline 332, now 0).
- Outside inputs.rs, the only remaining `.f64()`/`.u32()` reads are aft.rs's two already-cast event-column reads (aft.rs: baseline 5, now 2; other files: baseline 363, now 0).
- At least 340 helper call sites exist.
- Regression design-matrix reads are hoisted one cast per column.
- partial/semi-partial covariates raise instead of being skipped.
- fmt and clippy are clean, plus no-default clippy whenever the sibling's CI gate is present.
- All lib unit tests and the full rust_api suite pass, including the extended input_dtype_upcast.
  </done>
</task>

<task type="auto">
  <name>Task 3: Python dtype-parity tests + CHANGELOG entry + full gates</name>
  <files>tests/test_dtype_upcast.py, CHANGELOG.md</files>
  <precondition>`uv --version` succeeds and the Rust toolchain from Tasks 1-2 builds (`cargo check` passed).</precondition>
  <action>
1. Environment setup:
   - Run `uv sync --extra dev`. It installs the lockfile-pinned dev extras (maturin, pytest, scipy, statsmodels). A fresh .venv has none of them.
   - Then run `uv run maturin develop --release`.
   - uv's editable build does NOT pick up Rust source changes on `uv run`, so always rebuild explicitly with this command after Rust edits.
2. Create tests/test_dtype_upcast.py. Follow ruff rules: line length 100, py39 target. Add
   `from __future__ import annotations` if you use PEP 604 hints.
   - Module docstring: integer/Float32 inputs must give the same results as Float64 (GH #52).
   - `DTYPES = [pl.Int32, pl.UInt32, pl.Int64, pl.Float32]`, with readable pytest ids.
   - Data: small non-negative integers, so every dtype represents them exactly, with enough rows
     (about 20) for every test. Use distinct-ish sequences for x, y, a covariate correlated with
     both, three groups/raters, a group key `g`, and a 2x2 count column.
   - A recursive `_assert_same(a, b)` helper:
     - dicts: same keys, recurse into values;
     - lists: same length, recurse elementwise;
     - floats: both NaN, or `math.isclose(rel_tol=1e-12, abs_tol=1e-15)`;
     - anything else: `==`.
     Compare the struct dict from `df.select(expr.alias("r")).item()` for the cast columns against the
     same expression on Float64-cast columns.
   - Tests:
     - (a) `test_issue_52_spearman_uint32_repro`: columns literally named `char_tokens` and `char_types`, both UInt32. `ps.spearman("char_tokens", "char_types")` does not raise and matches the Float64 result.
     - (b) One test parametrized over DTYPES × named cases: pearson, spearman, kendall, ttest_ind, mann_whitney_u, kruskal_wallis(a, b, c), one_way_anova(a, b, c), shapiro_wilk, icc(r1, r2, r3), partial_cor(x, y, [cov]) and ols(y, x1, x2). Cast every data column the case uses to the dtype. Read each function's real signature in python/polars_statistics/exprs/*.py. Do not invent kwargs.
     - (c) `test_partial_cor_integer_covariate_is_used`: with an Int64 covariate, the estimate equals the Float64-covariate estimate and differs from `ps.pearson(x, y)` by more than 1e-6. The Python wrapper passes covariates uncast, and the Rust side used to skip them.
     - (d) chisq_test(counts, n_rows=2, n_cols=2) parametrized over the integer dtypes only (Int32, UInt32, Int64).
     - (e) A group_by test parametrized over DTYPES: `df.group_by("g", maintain_order=True).agg(ps.spearman("x", "y").alias("r"))` equals the Float64 version.
   - Boolean stays out of the Python matrix. Most wrappers call `is_finite()` in Python before the plugin runs, which is outside this fix. The Rust unit tests from Task 1 cover the Boolean upcast.
3. CHANGELOG.md, under `## [Unreleased]`:
   - Add a `### Fixed` heading. If sibling 261111-lm9 already created `### Fixed` there, append to it instead. Never create a second heading.
   - Add bullets ending `(#52)` that say:
     - every expression and Rust `*_fit` function now upcasts Int8–Int64, UInt8–UInt64, Float32, Boolean and all-null inputs to Float64 wherever it reads floating-point data. `ps.spearman` on a `u32` column no longer fails with "expected `Float64`, got `u32`".
     - integer dtypes are strict-cast to UInt32 wherever counts or codes are read, so negative or out-of-range values raise instead of being nulled.
     - integer covariates passed to `ps.partial_cor` / `ps.semi_partial_cor` were silently dropped and are now used.
     - Rust `*_fit` callers passing non-Float64 numeric X columns to OLS and the regression diagnostics no longer get NaN- or zero-filled design matrices, and null-dropping predict no longer panics on them.
     - non-numeric columns still raise a dtype error. This now also applies to regression X columns and correlation covariates, which previously degraded silently.
   - Do not bump any version.
4. Run `uv run pytest tests/test_dtype_upcast.py -q`, then the full `uv run pytest tests/ -q`, then
   `uvx ruff@0.16.8 check .` (the version CI pins in .github/workflows/ci.yml). All must be green.
   If an existing test fails because it relied on the old silent fallback, apply Task 2's documented
   per-function fallback rather than editing that test.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/polars-statistics && uv sync --extra dev && uv run maturin develop --release && uv run pytest tests/test_dtype_upcast.py -q 2>&1 | tee /dev/stderr | tail -1 | grep -qE '([4-9][0-9]|[1-9][0-9]{2,}) passed' && uv run pytest tests/ -q && uvx ruff@0.16.8 check . && test "$(awk '/^## \[Unreleased\]/{f=1;next} /^## \[/{f=0} f && /#52/' CHANGELOG.md | wc -l)" -ge 1 && test "$(awk '/^## \[Unreleased\]/{f=1;next} /^## \[/{f=0} f && /^### Fixed/' CHANGELOG.md | wc -l)" -eq 1 && test "$(grep -m1 '^version' Cargo.toml)" = 'version = "0.8.0"' && test "$(grep -m1 '^version' pyproject.toml)" = 'version = "0.8.0"'</automated>
  </verify>
  <done>
- tests/test_dtype_upcast.py has at least 40 passing tests: the GH #52 repro, the DTYPES × 11-case parity matrix, the covariate-used check, integer-dtype chisq and the group_by parity.
- The full pytest suite and ruff (0.16.8) are green.
- CHANGELOG [Unreleased] has exactly one `### Fixed` heading with the #52 entry.
- No version string changed.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| caller DataFrame column → `*_fit` input Series | User-controlled dtype and values enter the statistical kernels. The new coercion step decides which dtypes are accepted and how values are converted. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-261111-lma-01 | Tampering (data integrity) | `as_u32` (counts / factor codes) | high | mitigate | `strict_cast` to UInt32, so negative or out-of-range integers ERROR instead of becoming null (non-strict cast) or wrapping. Unit tests cover -1 and u32::MAX+1; rust_api checks that chisq_test_fit with a -1 cell errors. |
| T-261111-lma-02 | Tampering (data integrity) | `as_u32` on float input | medium | mitigate | Float dtypes are not accepted. They fall through to the original dtype error, so 2.7 is never silently truncated to 2. A unit test asserts Float64 → Err. |
| T-261111-lma-03 | Tampering (data integrity) | `as_f64` on non-numeric dtypes | medium | mitigate | Only primitive-numeric, Boolean and Null dtypes are cast. String, Date, Datetime, Duration, Decimal, Categorical and nested types keep the original "expected Float64" error (unit test + rust_api String case). |
| T-261111-lma-04 | Tampering (precision) | `as_f64` on Int64/UInt64 values beyond 2^53 | low | accept | Standard IEEE-754 rounding, the same as the Python wrappers' existing `.cast(pl.Float64)`. Documented in the helper doc comment. |
| T-261111-lma-05 | Denial of Service | regression.rs `build_xy_with_null_policy` drop path | medium | mitigate | `.unwrap()` over a failed per-cell typed read could panic on non-Float64 X columns. The hoisted `as_f64(..)?` returns a PolarsError instead, and the rust_api ols_predict_fit Int64 case covers it. |
| T-261111-lma-06 | Denial of Service | per-cell casting cost in `Mat::from_fn` | low | mitigate | Casts are hoisted out of the per-cell closures: one cast per column, never per cell. The Float64 fast path is an Arc clone with no data copy. |
| T-261111-lma-07 | Tampering (silent wrong result) | correlation.rs partial/semi-partial covariates; regression NaN/0.0-filled X | medium | mitigate | The `if let Ok` skip and the `.ok()` fallbacks are replaced by `?`, so a column is either used correctly or the call errors. The Python test asserts an Int64 covariate changes the estimate, and rust_api asserts the Int64 X path equals Float64. |
| T-261111-lma-SC | Tampering | npm/pip/cargo installs | low | accept | No new dependencies. `uv sync --extra dev` installs only the existing lockfile-pinned dev extras, and `uvx ruff@0.16.8` matches the CI pin. |
</threat_model>

<verification>
- `grep -rnP 'inputs\[\w+\]\.(f64|u32)\(\)\?' src/expressions | wc -l` gives 0 (baseline 332).
- Outside inputs.rs, the only `.f64()`/`.u32()` reads left in src/expressions are aft.rs's 2 already-cast event-column reads.
- `cargo fmt --all -- --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test --no-default-features --lib` and `cargo test --no-default-features --test rust_api` are green.
- `uv run maturin develop --release`, then `uv run pytest tests/ -q` and `uvx ruff@0.16.8 check .`, are green.
- src/expressions/mod.rs has a private `mod inputs;`, no re-export and no new `#[allow]`. This item's own commits add no `use` lines to the expression files: `git diff -U0 <base>..HEAD -- src/expressions | grep -c '^+use '` is 0, where <base> is the commit this item started from. A plain working-tree `git diff` is not valid here, because the sibling item may have uncommitted edits in the shared tree.
</verification>

<success_criteria>
- The GH #52 repro works: `ps.spearman` on UInt32 columns returns the Float64-equivalent struct.
- Every input read in src/expressions goes through `as_f64` / `as_u32`.
- Integer→UInt32 conversion is strict. Non-numeric inputs keep their dtype error.
- Integer X columns and covariates are no longer silently corrupted or dropped.
- Rust unit tests, the rust_api integration test and the Python dtype-parity tests pass alongside the full existing suites.
- The CHANGELOG [Unreleased] ### Fixed entry references #52. No version bump.
</success_criteria>

<output>
Create `.planning/quick/261111-lma-fix-gh-52-add-central-as-f64-as-u32-input-upcast-helpers-in/261111-lma-SUMMARY.md` when done
</output>
