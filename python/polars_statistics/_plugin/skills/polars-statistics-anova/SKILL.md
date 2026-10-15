---
name: polars-statistics-anova
description: Use polars-statistics ANOVA — run one-way ANOVA (Fisher or Welch), two-way ANOVA, repeated-measures ANOVA, or intraclass correlation coefficient (ICC); follow up with post-hoc pairwise comparisons (Tukey HSD, pairwise t-tests, Dunn's test) and p-value adjustment (p_adjust) as Polars expressions.
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

This skill also covers the **post-hoc / multiple-testing family** that follows a
significant omnibus test: `tukey_hsd`, `pairwise_t_test`, `dunn_test` (pairwise group
comparisons), and the `p_adjust`/`ptukey`/`qtukey` building blocks they use internally.

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
- Post-hoc comparisons after a significant omnibus test — which groups differ:
  `tukey_hsd`, `pairwise_t_test`, `dunn_test`
- Multiple-testing / p-value adjustment: `p_adjust`, Holm/Hochberg/Hommel/Bonferroni/BH
  (fdr)/BY
- The studentized range distribution: `ptukey`, `qtukey`

Also route here for questions about **format gotchas**: wide vs. long input for
`one_way_anova`, raw vs. pre-encoded factors for `two_way_anova` and
`repeated_measures_anova`, the separate-rater-column contract for `icc`, and the
long-format convention (`values`, `groups`) used by the post-hoc family below.

---

## API Reference

### one_way_anova

```python
# verify: skip -- function signature reference (not runnable Python; no DataFrame context)
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
# verify: skip -- function signature reference (not runnable Python; no DataFrame context)
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
# verify: skip -- function signature reference (not runnable Python; no DataFrame context)
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
# verify: skip -- function signature reference (not runnable Python; no DataFrame context)
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

### Post-hoc comparisons & multiple testing

Run the omnibus test first (`one_way_anova` for parametric data, `kruskal_wallis` for
non-parametric data — see `polars-statistics-hypothesis-tests`). If it is significant,
follow up with one of these to find which pairs of groups actually differ. Unlike
`one_way_anova`'s wide format, these six functions take **long format**: one flat
`values` column plus one `groups` label column (see Gotcha 6 below).

```python
# verify: skip -- function signature reference (not runnable Python; no DataFrame context)
ps.tukey_hsd(values, groups, conf_level=0.95)
ps.pairwise_t_test(values, groups, pooled_sd=True, p_adjust_method="holm", alternative="two-sided")
ps.dunn_test(values, groups, p_adjust_method="holm")
ps.p_adjust(p, method="holm")
ps.ptukey(q, nmeans, df, nranges=1, lower_tail=True)
ps.qtukey(p, nmeans, df, nranges=1, lower_tail=True)
```

| Function | Parameter | Default | Description |
|----------|-----------|---------|-------------|
| `tukey_hsd` | `values`, `groups` | required | Long-format observations / group labels |
| `tukey_hsd` | `conf_level` | `0.95` | Family-wise CI level for `ci_lower`/`ci_upper`; must be in `(0, 1)`, checked at build time |
| `pairwise_t_test` | `pooled_sd` | `True` | `True` pools the SD over all groups (R default); `False` runs a per-pair Welch t-test |
| `pairwise_t_test` | `p_adjust_method` | `"holm"` | One of `"holm"`, `"hochberg"`, `"hommel"`, `"bonferroni"`, `"BH"` (alias `"fdr"`), `"BY"`, `"none"` (case-insensitive) |
| `pairwise_t_test` | `alternative` | `"two-sided"` | `"two-sided"`/`"two_sided"`/`"two.sided"`, `"less"`, or `"greater"`, for `group2 - group1` |
| `dunn_test` | `p_adjust_method` | `"holm"` | Same method set as `pairwise_t_test` |
| `p_adjust` | `p`, `method` | `"holm"` | Column of raw p-values; same method set |
| `ptukey`/`qtukey` | `q`/`p`, `nmeans`, `df`, `nranges`, `lower_tail` | `nranges=1`, `lower_tail=True` | Studentized-range CDF / quantile; each arg may be a scalar or a column |

**`tukey_hsd`/`pairwise_t_test`/`dunn_test` return** `List[Struct{group1, group2, n1, n2, estimate, statistic, df, p_value, p_adj, ci_lower, ci_upper}]` —
one row per group pair (`k(k-1)/2` rows for `k` groups), ordered like R's `TukeyHSD`
(B-A, C-A, C-B, ...). `pairwise_t_test`'s `ci_lower`/`ci_upper` and `dunn_test`'s
`df`/`ci_lower`/`ci_upper` are `NaN` (R's `pairwise.t.test`/`dunn.test` compute none of
those). `p_adjust`/`ptukey`/`qtukey` return a bare `Float64` (`returns_scalar=False`, no
Struct wrapper), same length as the input.

**R equivalents:** `TukeyHSD(aov(values ~ factor(groups)), conf.level = conf_level)`;
`pairwise.t.test(values, groups, p.adjust.method, pool.sd, alternative)`;
`dunn.test::dunn.test(values, groups, method, altp = TRUE)` / `FSA::dunnTest`;
`p.adjust(p, method)`; `ptukey(q, nmeans, df, nranges, lower.tail)`;
`qtukey(p, nmeans, df, nranges, lower.tail)`.

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

### Example 6: Post-hoc comparisons after ANOVA / Kruskal-Wallis, and p-value adjustment

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    "group": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
})

# 1. Omnibus test first (wide format -- pivot the long data, see Gotcha 1):
df_wide = (
    df.with_row_index("row_id")
    .pivot(on="group", values="value", index="row_id", aggregate_function="first")
    .drop("row_id")
)
omnibus = df_wide.select(ps.one_way_anova("a", "b", "c").alias("anova"))
print(omnibus["anova"].struct.field("p_value")[0])

# 2. If significant, follow up with Tukey HSD on the original long-format data:
tukey = (
    df.select(ps.tukey_hsd("value", "group").alias("ph"))
    .select(pl.col("ph").explode().struct.unnest())
)
print(tukey.filter(pl.col("p_adj") < 0.05))

# 3. Kruskal-Wallis (wide) -> Dunn's test (long) for the non-parametric case:
kw = df_wide.select(ps.kruskal_wallis("a", "b", "c").alias("kw"))
print(kw["kw"].struct.field("p_value")[0])

dunn = (
    df.select(ps.dunn_test("value", "group").alias("ph"))
    .select(pl.col("ph").explode().struct.unnest())
)
print(dunn)

# 4. Adjust p-values from several independent families with .over():
p_df = pl.DataFrame({
    "family": ["x", "x", "x", "y", "y"],
    "p": [0.001, 0.01, 0.03, 0.04, 0.2],
})
adjusted = p_df.with_columns(
    ps.p_adjust("p", method="BH").over("family").alias("p_adj")
)
print(adjusted)
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
import polars as pl
import polars_statistics as ps

# Pivot long to wide first (add row index as key — required by Polars pivot), then call:
df_long = pl.DataFrame({
    "treatment": ["control", "drug_a", "drug_b"] * 10,
    "score": [50.0, 55.0, 60.0] * 10,
})
df_wide = (
    df_long.with_row_index("row_id")
    .pivot(on="treatment", values="score", index="row_id", aggregate_function="first")
    .drop("row_id")
)
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
import polars as pl
import polars_statistics as ps

rater_df = pl.DataFrame({
    "rater1": [5.0, 6.0, 7.0, 8.0, 6.0],
    "rater2": [5.5, 6.5, 6.5, 8.5, 6.5],
    "r1":     [5.0, 6.0, 7.0, 8.0, 6.0],
    "r2":     [5.5, 6.5, 6.5, 8.5, 6.5],
    "rater3": [5.0, 6.0, 7.0, 8.0, 7.0],
})
rater_df.select(ps.icc("rater1", "rater2", "rater3"))          # 3 raters, n rows = n subjects
rater_df.select(ps.icc("r1", "r2", icc_type="icc1"))           # 2 raters, one-way random model
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
import polars as pl
import polars_statistics as ps

# Pass the original string column directly -- no cast, no encoding:
df_twoway = pl.DataFrame({
    "outcome":  [68., 72., 70., 74., 78., 76., 80., 84., 82., 86., 90., 88.],
    "factor_a": ["A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B"],
    "factor_b": ["X", "X", "X", "Y", "Y", "Y", "X", "X", "X", "Y", "Y", "Y"],
})
df_twoway.select(ps.two_way_anova("outcome", "factor_a", "factor_b"))

# Subject and condition in repeated measures also stay raw:
df_rm = pl.DataFrame({
    "score":      [10., 12., 14., 15., 18., 20.],
    "subject_id": ["s1", "s2", "s1", "s2", "s1", "s2"],
    "condition":  ["t1", "t1", "t2", "t2", "t3", "t3"],
})
df_rm.select(ps.repeated_measures_anova("score", "subject_id", "condition"))
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
import polars as pl

df = pl.DataFrame({
    "subject":   ["s1", "s2", "s3", "s1", "s2", "s3", "s1", "s2", "s3"],
    "condition": ["t1", "t1", "t1", "t2", "t2", "t2", "t3", "t3", "t3"],
    "score":     [10., 12., 11., 14., 15., 13., 18., 20., 17.],
})
counts = df.group_by("subject").agg(pl.col("condition").n_unique().alias("n_cond"))
assert counts["n_cond"].n_unique() == 1, "Unbalanced design -- fill or drop missing cells"
```

### Gotcha 6: Post-hoc functions use LONG format — values + groups columns

`tukey_hsd`, `pairwise_t_test`, and `dunn_test` take one flat `values` column plus one
`groups` label column — the opposite convention from `one_way_anova`'s wide format
(Gotcha 1). `groups` is cast to `pl.String`; numeric group labels are compared as
strings (zero-pad them, e.g. `"01"`/`"02"`/`"10"`, if numeric sort order matters).

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "group": ["a", "a", "b", "b", "c", "c"],
})
df.select(ps.tukey_hsd("value", "group"))   # long format -- correct
```

### Gotcha 7: Post-hoc rows are dropped (NA) first; fewer than 2 usable groups gives an empty list

Rows with a null/NaN `value` or a null `group` are dropped before the comparison runs
(R's `NA` row-drop semantics). If fewer than 2 usable groups remain, the result is an
**empty list**, never an exception. Inside `group_by().agg()`, each group is analysed
independently — the same caveat as `one_way_anova` (Gotcha 1): a post-hoc call nested in
a `group_by` only ever sees that one group's rows, so it is typically called at the
whole-frame level, not per-group.

### Gotcha 8: `estimate` sign and Dunn's `z` sign convention

For `tukey_hsd` and `pairwise_t_test`, `estimate = mean(group2) - mean(group1)`. For
`dunn_test`, `estimate` is the difference in mean ranks (`group2 - group1`) — but
`dunn.test`/`FSA::dunnTest` report the **opposite sign** for their own `z` statistic;
negate when porting an R-authored analysis. For `tukey_hsd`, `p_value == p_adj` always —
Tukey's test is already simultaneous, so there is no separate unadjusted p-value.

### Gotcha 9: `p_adjust` is a WHOLE-COLUMN operation — use `.over()` for independent families

`p_adjust` adjusts an entire family of p-values together, not one row at a time. Passing
several independent families (e.g. one per experiment) in a single column without
`.over(family_col)` adjusts them as **one** combined family, which is usually wrong.
Null/NaN entries are kept (null stays null, NaN stays NaN) and excluded from the
comparison count `n`. Inside `group_by().agg()`, `p_adjust` returns one adjusted list
per group — equivalent to `.over()` for this purpose.

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({"family": ["x", "x", "y", "y"], "p": [0.01, 0.04, 0.02, 0.03]})
df.with_columns(ps.p_adjust("p", method="BH").over("family").alias("p_adj"))  # correct
# df.with_columns(ps.p_adjust("p", method="BH").alias("p_adj"))  # wrong: treats x+y as one family
```

### Gotcha 10: `ptukey`/`qtukey` broadcast scalars and return NaN for invalid parameters

Every argument (`q`/`p`, `nmeans`, `df`, `nranges`) may be a scalar or a column; a scalar
broadcasts against any column arguments, but all column arguments must share the same
length. Both functions return `NaN` (never raise) for `NaN` input or invalid parameters
(`df < 2`, `nranges < 1`, `nmeans < 2`), and `qtukey` additionally returns `NaN` for `p`
outside `[0, 1]`. `qtukey`'s Newton-Raphson solver converges to its own `1e-4` criterion,
so `ptukey(qtukey(p, ...), ...)` round-trips only to `1e-6`, not full machine precision —
this is expected, not a bug.

---

## See also

- polars-statistics-hypothesis-tests — t-tests, Mann-Whitney, Kruskal-Wallis, normality tests, correlation tests
- polars-statistics-regression — OLS, Ridge, Elastic Net, GLM, and all regression families
- polars-statistics-results — extract struct fields, unnest result columns, to_dict(), summary()
- polars-statistics-getting-started — install, model-selection decision matrix, sklearn migration
- docs/api/tests/posthoc.md — full post-hoc / multiple-testing reference and R validation notes
