---
name: polars-statistics-anova
description: Use polars-statistics ANOVA — run one-way ANOVA (Fisher or Welch), two-way ANOVA, repeated-measures ANOVA, or intraclass correlation coefficient (ICC) as Polars expressions.
version: "0.8.0"
user-invocable: false
---

# polars-statistics-anova

ANOVA and ICC as native Polars expressions. All four functions return a Struct column
containing the complete test output. Use `pl.col(...).struct.field("p_value")` or
`ps.unnest(result, col)` to extract individual fields.

---

## Overview

`polars-statistics` exposes four ANOVA/ICC functions through the **expression API**
(`df.select(ps.one_way_anova(...))`, `df.group_by(...).agg(ps.two_way_anova(...))`).
The results are Polars Struct columns — access fields via native `.struct.field()` or
use the `ps.unnest()` helper from `polars-statistics-results`.

There are no PyModel-class equivalents for the ANOVA family — these are expression-only.

---

## When to use this skill

Route here when the user asks about any of the following:

- One-way ANOVA, F-test across groups, Welch ANOVA, `one_way_anova`
- Two-way factorial ANOVA, interaction effect, `two_way_anova`
- Repeated-measures ANOVA, within-subjects design, `repeated_measures_anova`
- Sphericity, Mauchly's W, Greenhouse-Geisser, Huynh-Feldt epsilon
- Intraclass correlation, inter-rater reliability, ICC, `icc`
- Which ANOVA function to use (one column per group vs. long format, etc.)
- ANOVA struct output fields, how to extract `p_value`, `eta_squared`, `ss_between`

Also route here for questions about **format gotchas**: wide vs. long input for
`one_way_anova`, raw vs. pre-encoded factors for `two_way_anova` and
`repeated_measures_anova`, and the separate-rater-column contract for `icc`.

---

## API Reference

### one_way_anova

```python
ps.one_way_anova(*groups, kind="fisher")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `*groups` | column names or expressions | required | Two or more group columns (one column per group, **wide format**) |
| `kind` | `"fisher"` or `"welch"` | `"fisher"` | Variance assumption: Fisher assumes equal variances; Welch does not |

**Output struct fields:**

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | f64 | F-statistic |
| `df_between` | f64 | Between-groups degrees of freedom |
| `df_within` | f64 | Within-groups degrees of freedom |
| `p_value` | f64 | p-value |
| `ss_between` | f64 | Between-groups sum of squares (**NaN for Welch**) |
| `ss_within` | f64 | Within-groups sum of squares (**NaN for Welch**) |
| `ms_between` | f64 | Between-groups mean square (**NaN for Welch**) |
| `ms_within` | f64 | Within-groups mean square (**NaN for Welch**) |
| `eta_squared` | f64 | Effect size (**NaN for Welch**) |
| `n_groups` | u32 | Number of groups |

> **Welch note:** `kind="welch"` only populates `statistic`, `df_between`, `df_within`,
> `p_value`, and `n_groups`. The SS/MS/eta_squared fields are `NaN` by design — Welch's
> test does not compute the pooled SS decomposition.

---

### two_way_anova

```python
ps.two_way_anova(value, factor_a, factor_b)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | column name or expression | Continuous outcome |
| `factor_a` | column name or expression | First factor — raw string or integer column, **not pre-encoded** |
| `factor_b` | column name or expression | Second factor — raw string or integer column, **not pre-encoded** |

**Output struct fields** (flattened two-way ANOVA table):

| Field | Description |
|-------|-------------|
| `a_ss`, `a_df`, `a_ms`, `a_f`, `a_p_value` | Factor A main effect: SS, df, MS, F, p |
| `b_ss`, `b_df`, `b_ms`, `b_f`, `b_p_value` | Factor B main effect: SS, df, MS, F, p |
| `ab_ss`, `ab_df`, `ab_ms`, `ab_f`, `ab_p_value` | A×B interaction: SS, df, MS, F, p |
| `residual_ss`, `residual_df`, `residual_ms` | Residual/error: SS, df, MS |
| `grand_mean` | Grand mean of the outcome |
| `n` | Total number of observations |

---

### repeated_measures_anova

```python
ps.repeated_measures_anova(value, subject, condition, compute_sphericity=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `value` | column name or expression | required | Observed values (long format) |
| `subject` | column name or expression | required | Subject identifier — raw string or integer column, **not pre-encoded** |
| `condition` | column name or expression | required | Condition/time-point — raw string or integer column, **not pre-encoded** |
| `compute_sphericity` | bool | `True` | Whether to compute Mauchly's W and GG/HF corrections (requires k >= 3 conditions) |

**Output struct fields:**

| Field | Description |
|-------|-------------|
| `ws_f` | Within-subjects F-statistic |
| `ws_df` | Within-subjects degrees of freedom |
| `ws_ss` | Within-subjects sum of squares |
| `ws_ms` | Within-subjects mean square |
| `ws_p_value` | Within-subjects p-value |
| `error_df` | Error degrees of freedom |
| `error_ss` | Error sum of squares |
| `error_ms` | Error mean square |
| `mauchly_w` | Mauchly's W statistic (NaN if `compute_sphericity=False`) |
| `mauchly_p_value` | Mauchly's W p-value |
| `gg_epsilon` | Greenhouse-Geisser epsilon |
| `gg_p_value` | GG-corrected p-value |
| `hf_epsilon` | Huynh-Feldt epsilon |
| `hf_p_value` | HF-corrected p-value |
| `grand_mean` | Grand mean of `value` |

> **Balance requirement:** Every subject must appear in every condition exactly once
> (balanced design). Unbalanced input returns an all-NaN struct without raising an error.
> Verify balance before calling:
> `df.group_by("subject").agg(pl.col("condition").n_unique())` — all counts must equal
> the number of conditions.

---

### icc

```python
ps.icc(*rater_cols, icc_type="icc2")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `*rater_cols` | column names or expressions | required (2+) | One column per rater; **each rater is a separate positional argument** — rows are subjects |
| `icc_type` | str | `"icc2"` | ICC model variant (see table below) |

**ICC type lookup:**

| `icc_type` | Model | Description |
|------------|-------|-------------|
| `"icc1"` | One-way random, single rater | Raters are a random sample; generalize to single-rater use |
| `"icc2"` | Two-way random, single rater | Both subjects and raters are random samples — **most common default** |
| `"icc3"` | Two-way mixed, single rater | Raters are fixed; generalize to these specific raters |
| `"icc1k"` | One-way random, average of k raters | ICC1 model, reliability of the mean of k raters |
| `"icc2k"` | Two-way random, average of k raters | ICC2 model, reliability of the mean of k raters |
| `"icc3k"` | Two-way mixed, average of k raters | ICC3 model, reliability of the mean of k raters |

**Output struct fields:**

| Field | Type | Description |
|-------|------|-------------|
| `icc` | f64 | Intraclass correlation coefficient |
| `f_value` | f64 | F-statistic |
| `df1` | f64 | Between-subjects degrees of freedom |
| `df2` | f64 | Within-raters (error) degrees of freedom |
| `p_value` | f64 | p-value for H0: ICC = 0 |
| `ci_lower` | f64 | Lower 95% confidence bound |
| `ci_upper` | f64 | Upper 95% confidence bound |
| `n_subjects` | f64 | Number of subjects (rows) |
| `n_raters` | f64 | Number of raters (columns passed) |

---

## Worked Examples

### Example 1: One-way ANOVA — wide format (Fisher and Welch)

Three treatment groups stored as separate columns (wide format).

```python
import polars as pl
import polars_statistics as ps
import numpy as np

rng = np.random.default_rng(42)

df = pl.DataFrame({
    "control": rng.normal(50, 5, 30).tolist(),
    "drug_a":  rng.normal(54, 5, 30).tolist(),
    "drug_b":  rng.normal(58, 5, 30).tolist(),
})

# Fisher's one-way ANOVA (equal-variance assumption)
result = df.select(ps.one_way_anova("control", "drug_a", "drug_b").alias("anova"))
flat = result["anova"].struct.unnest()
print(flat)
# shape: (1, 10)
# Columns: statistic, df_between, df_within, p_value,
#          ss_between, ss_within, ms_between, ms_within, eta_squared, n_groups

# Extract p-value directly:
p_val = result["anova"].struct.field("p_value")[0]
print(f"p = {p_val:.4f}")

# Welch variant (no equal-variance assumption):
result_w = df.select(
    ps.one_way_anova("control", "drug_a", "drug_b", kind="welch").alias("anova")
)
# ss_between, ms_between, eta_squared are NaN for Welch -- expected behavior
print(result_w["anova"].struct.field("statistic")[0])   # F-statistic is populated
print(result_w["anova"].struct.field("eta_squared")[0])  # NaN -- by design
```

### Example 2: Two-way ANOVA — raw factor columns

Pass raw string or integer factor columns. The expression label-encodes them internally.

```python
import polars as pl
import polars_statistics as ps

# 2 x 2 factorial design, 3 obs per cell (balanced)
df = pl.DataFrame({
    "score":  [68., 72., 70.,  74., 78., 76.,  80., 84., 82.,  86., 90., 88.],
    "method": ["A", "A", "A",  "A", "A", "A",  "B", "B", "B",  "B", "B", "B"],
    "group":  ["X", "X", "X",  "Y", "Y", "Y",  "X", "X", "X",  "Y", "Y", "Y"],
})

result = df.select(ps.two_way_anova("score", "method", "group").alias("anova"))
flat = result["anova"].struct.unnest()
print(flat)
# Columns: a_ss, a_df, a_ms, a_f, a_p_value,
#          b_ss, b_df, b_ms, b_f, b_p_value,
#          ab_ss, ab_df, ab_ms, ab_f, ab_p_value,
#          residual_ss, residual_df, residual_ms, grand_mean, n

# Extract interaction p-value:
ab_p = result["anova"].struct.field("ab_p_value")[0]
print(f"A x B interaction p-value: {ab_p:.4f}")
```

### Example 3: Repeated-measures ANOVA — long format with sphericity corrections

```python
import polars as pl
import polars_statistics as ps

# 5 subjects, 3 time points (balanced long format)
df = pl.DataFrame({
    "score":   [10., 12., 11., 14., 13.,   # time1
                14., 15., 13., 17., 16.,   # time2
                18., 20., 17., 21., 19.],  # time3
    "subject": ["s1", "s2", "s3", "s4", "s5"] * 3,
    "time":    ["t1"] * 5 + ["t2"] * 5 + ["t3"] * 5,
})

result = df.select(
    ps.repeated_measures_anova("score", "subject", "time", compute_sphericity=True).alias("rm")
)
rm_val = result["rm"][0]

# Use GG-corrected p-value when Mauchly's test is significant (p < 0.05):
if rm_val["mauchly_p_value"] < 0.05:
    print(f"Sphericity violated -- GG corrected p = {rm_val['gg_p_value']:.4f}")
else:
    print(f"Sphericity OK -- uncorrected p = {rm_val['ws_p_value']:.4f}")

print(f"GG epsilon = {rm_val['gg_epsilon']:.4f}  HF epsilon = {rm_val['hf_epsilon']:.4f}")
```

### Example 4: ICC — separate rater columns

Three raters score five subjects. Pass each rater as a separate positional column argument.

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "rater_a": [5.0, 6.0, 7.0, 8.0, 6.0],
    "rater_b": [5.5, 6.5, 6.5, 8.5, 6.5],
    "rater_c": [5.0, 6.0, 7.0, 8.0, 7.0],
})

# ICC2 (two-way random, single rater) -- the default
result = df.select(ps.icc("rater_a", "rater_b", "rater_c", icc_type="icc2").alias("res"))
val = result["res"][0]
print(f"ICC = {val['icc']:.4f}  95% CI [{val['ci_lower']:.4f}, {val['ci_upper']:.4f}]")
print(f"n_subjects={int(val['n_subjects'])}  n_raters={int(val['n_raters'])}")

# ICC2k -- reliability of the AVERAGE of all three raters:
result_k = df.select(ps.icc("rater_a", "rater_b", "rater_c", icc_type="icc2k").alias("res"))
print(f"ICC2k = {result_k['res'][0]['icc']:.4f}")
```

### Example 5: Extracting results with ps.unnest

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "g1": [10., 11., 12., 10., 11.],
    "g2": [14., 15., 14., 16., 15.],
    "g3": [20., 21., 22., 20., 21.],
})

result = df.select(ps.one_way_anova("g1", "g2", "g3").alias("anova"))

# Flatten all struct fields into columns using the helper:
flat = ps.unnest(result, "anova")
print(flat.columns)
# ['statistic', 'df_between', 'df_within', 'p_value',
#  'ss_between', 'ss_within', 'ms_between', 'ms_within', 'eta_squared', 'n_groups']

# Or access a single field without unnesting:
p_val = result["anova"].struct.field("p_value")[0]
print(f"p = {p_val:.4f}")
```

---

## Gotchas / Notes

### Gotcha 1: one_way_anova uses WIDE format — one column per group

`one_way_anova` expects the data in **wide format**: each group occupies its own column.
It does NOT take a value column plus a group label column.

**Wrong approach — trying to use long format inside group_by:**

The pattern `df_long.group_by("treatment").agg(ps.one_way_anova("score"))` does not work
as intended because each Polars group only contains one treatment group's values.
`one_way_anova` needs all ANOVA groups present simultaneously in wide columns.

**Correct approach — wide format, direct call:**

```python
# Pivot long to wide first, then call:
df_wide = df_long.pivot(on="treatment", values="score", index=None, aggregate_function="first")
df_wide.select(ps.one_way_anova("control", "drug_a", "drug_b"))
```

The same wide-format rule applies to `ps.kruskal_wallis` (the non-parametric alternative
when normality is not assumed).

### Gotcha 2: icc takes SEPARATE RATER COLUMNS — not a matrix column

The current API contract requires each rater to be a **separate positional column
argument**. Rows are subjects. The number of raters equals the number of columns passed.

**Wrong — single stacked or nested column:**

Passing a single matrix column or a list column to `ps.icc()` does not raise an error
but produces incorrect results silently. Do not use `pl.concat_list("r1", "r2")` or
similar constructions as the sole argument.

**Correct — one column per rater:**

```python
df.select(ps.icc("rater1", "rater2", "rater3"))          # 3 raters, n rows = n subjects
df.select(ps.icc("r1", "r2", icc_type="icc1"))           # 2 raters, one-way random model
```

This is a breaking change from an older matrix-input contract. Code written before v0.7.0
that passed a matrix column must be updated to the positional-column form.

### Gotcha 3: two_way_anova and repeated_measures_anova require RAW (unencoded) factor columns

Both expressions perform their own internal label-encoding via `rank("dense") - 1` on the
raw string or integer factor columns after null-filtering. If you pre-encode the factors
before calling the expression, the Polars Categorical catalog assigns codes globally across
the full column (not necessarily 0-based per group), which can produce incorrect ANOVA
tables.

**Wrong — pre-encoding factors:**

Casting a factor column with `.cast(pl.Categorical).to_physical()` before passing it to
`two_way_anova` or `repeated_measures_anova` may produce wrong results because catalog
codes are not guaranteed to be contiguous 0-based integers per group.

**Correct — raw string or integer columns:**

```python
# Pass the original string column directly -- no cast, no encoding:
df.select(ps.two_way_anova("outcome", "factor_a", "factor_b"))

# Subject and condition in repeated measures also stay raw:
df.select(ps.repeated_measures_anova("score", "subject_id", "condition"))
```

### Gotcha 4: Welch ANOVA does not compute SS or eta_squared

When `kind="welch"` is specified, `one_way_anova` returns `NaN` for `ss_between`,
`ss_within`, `ms_between`, `ms_within`, and `eta_squared`. Only `statistic`,
`df_between`, `df_within`, `p_value`, and `n_groups` are populated.

This is expected behavior. Welch's test uses adjusted (non-integer) degrees of freedom
and does not perform the standard sum-of-squares decomposition. Do not treat these NaN
values as a computation error.

### Gotcha 5: repeated_measures_anova silently returns all-NaN for unbalanced designs

If any subject is missing any condition (unbalanced design), the expression returns an
all-NaN struct without raising an exception. Always verify balance before calling:

```python
counts = df.group_by("subject").agg(pl.col("condition").n_unique().alias("n_cond"))
assert counts["n_cond"].n_unique() == 1, "Unbalanced design -- fill or drop missing cells"
```

---

## See also

- polars-statistics-hypothesis-tests — t-tests, Mann-Whitney, Kruskal-Wallis, normality tests, correlation tests
- polars-statistics-regression — OLS, Ridge, Elastic Net, GLM, and all regression families
- polars-statistics-results — extract struct fields, unnest result columns, to_dict(), summary()
- polars-statistics-getting-started — install, model-selection decision matrix, sklearn migration
