---
name: polars-statistics-results
description: Consume polars-statistics expression results — extract struct fields, unnest result columns, call to_dict() and summary() on PyModel classes, handle PyRuntimeError for unfitted models, or work with the typed .pyi API.
version: "0.8.0"
user-invocable: false
---

## Overview

Expression functions such as `df.select(ps.ttest_ind("a", "b"))` and
`df.group_by(...).agg(ps.ols("y", "x"))` return **Polars Struct columns**, not Python
objects. The `polars_statistics.results` module provides two helpers — `unnest` and
`struct_to_dict` — that give Struct columns the same one-call ergonomics that fitted PyModel
classes have. PyModel classes (e.g. `OLS`, `TTestInd`) additionally expose `.to_dict()`,
`.summary()`, and `.is_fitted()` as universal introspection methods.

**Two orthogonal result surfaces:**

- **Expression results** — Polars Struct columns from `df.select(...)` /
  `df.group_by(...).agg(...)`. Unpack with `ps.unnest()`, `ps.struct_to_dict()`, or native
  Polars `.struct.field()` / `.unnest()`.
- **PyModel results** — fitted objects with named property access (`.coefficients`,
  `.r_squared`, etc.) plus `.to_dict()`, `.summary()`, and `.is_fitted()`.

---

## When to use this skill

Route here when the user asks about:
- How to extract fields from a Struct result column (e.g. `p_value`, `r_squared`)
- Using `ps.unnest()` or `ps.struct_to_dict()` to unpack expression results
- What `.to_dict()`, `.summary()`, or `.is_fitted()` do on a fitted model
- The `PyRuntimeError("Model not fitted")` error and how to guard against it
- Which fields a specific expression returns (output-struct field reference)
- The typed `.pyi` stub file for property return types
- How to tidy up per-group regression coefficients from a `group_by` aggregation

For the full expression API (signatures, parameters) see the family-specific skills listed
in **See also** at the bottom.

---

## API Reference

### `ps.unnest` — flatten a result Struct into flat columns

```python
# verify: skip -- function signature reference (not runnable Python; type annotations in call context)
ps.unnest(
    df: pl.DataFrame,
    column: str | None = None,
    *,
    prefix: str | None = None,
    keep_others: bool = True,
) -> pl.DataFrame
```

[Source: python/polars_statistics/results.py]

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pl.DataFrame` | required | DataFrame containing the result Struct column |
| `column` | `str \| None` | `None` | Name of the Struct column to flatten. Auto-detected when `None` and exactly one Struct column exists |
| `prefix` | `str \| None` | `None` | If given, every produced field is renamed `f"{prefix}{field}"` to disambiguate two struct results unnested into the same frame |
| `keep_others` | `bool` | `True` | If `True`, non-struct columns are preserved alongside the flattened fields. If `False`, only the flattened struct fields are returned |

**Returns:** `pl.DataFrame` with the struct fields promoted to top-level columns.

**Auto-detection rules:**
- `column=None` + exactly one Struct column → auto-selects it
- `column=None` + zero Struct columns → raises `ValueError: No Struct column found to unnest`
- `column=None` + two or more Struct columns → raises `ValueError: Multiple Struct columns found (...); pass column= to pick one`

---

### `ps.struct_to_dict` — convert a result Struct into a Python dict

```python
# verify: skip -- function signature reference (not runnable Python; type annotations in call context)
ps.struct_to_dict(
    df: pl.DataFrame,
    column: str | None = None,
) -> dict[str, Any] | list[dict[str, Any]]
```

[Source: python/polars_statistics/results.py]

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pl.DataFrame` | required | DataFrame containing the result Struct column |
| `column` | `str \| None` | `None` | Name of the Struct column. Auto-detected when `None` and exactly one Struct column exists |

**Returns:**
- `dict[str, Any]` — when the Struct column has exactly one row (the common case for a
  whole-frame test such as `df.select(ps.ttest_ind(...))`)
- `list[dict[str, Any]]` — when the Struct column has multiple rows (e.g. a per-group
  aggregation); one dict per row

**Auto-detection rules:** same as `unnest` — raises `ValueError` on 0 or 2+ Struct columns
when `column=None`.

---

### Native Polars struct access (without helpers)

When you need only a single field, or when working in a lazy pipeline, the native Polars
struct API is more efficient than the helper functions:

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "b": [2.0, 2.5, 3.5, 4.5, 5.5, 7.0],
})

# Access one field from a whole-frame result:
p_val = df.select(ps.ttest_ind("a", "b").alias("t"))["t"].struct.field("p_value")

# Unnest all fields using native Polars (no alias needed if single struct column):
flat = df.select(ps.ttest_ind("a", "b").alias("t")).unnest("t")
assert "statistic" in flat.columns and "p_value" in flat.columns

# Inline field extraction inside a select/with_columns:
r = df.select(ps.ttest_ind("a", "b").alias("t")).with_columns(
    pl.col("t").struct.field("statistic").alias("t_stat"),
    pl.col("t").struct.field("p_value").alias("p"),
)
assert "t_stat" in r.columns and "p" in r.columns
```

**Per-group aggregation — explode + unnest pattern:**

When `group_by().agg()` produces a `List[Struct]` column (e.g. from `ols_summary`), the
pattern is `.explode(col).unnest(col)`:

```python
import polars as pl
import polars_statistics as ps
import numpy as np

rng = np.random.default_rng(1)
df = pl.DataFrame({
    "group": ["A"] * 30 + ["B"] * 30,
    "y":     rng.normal(5, 1, 60).tolist(),
    "x":     rng.normal(2, 0.5, 60).tolist(),
})

# ols_summary returns List[Struct{term, estimate, std_error, statistic, p_value}]
coef_df = (
    df.group_by("group")
    .agg(ps.ols_summary("y", "x").alias("coef"))
    .explode("coef")
    .unnest("coef")
)
# Result columns: group, term, estimate, std_error, statistic, p_value
assert set(["group", "term", "estimate", "std_error", "statistic", "p_value"]).issubset(coef_df.columns)
```

---

### Universal PyModel methods

Every PyModel class (fitted or unfitted) exposes three universal introspection methods:

| Method | Return type | Description |
|--------|-------------|-------------|
| `.summary()` | `str` | Human-readable coefficient / statistics table |
| `.to_dict()` | `dict` | Machine-readable mapping of all result fields |
| `.is_fitted()` | `bool` | Returns `True` if `.fit()` has been called; `False` otherwise |

These are available on all 44 PyModel classes including `OLS`, `Ridge`, `ElasticNet`,
`TTestInd`, `MannWhitneyU`, `Logistic`, `ALM`, `Aid`, and every other class in the library.

**`model_info` (0.8.0):** every PyModel class whose fitted type implements `HasModelInfo`
(OLS, WLS, Ridge, ElasticNet, LARS, every GLM class, `GLMM`, `PenalizedGLM`, `AFT`, and more)
exposes `model_info -> dict` — `{"model_type": str, "family": str | None, "link": str}`.
`family` is `None` for non-likelihood estimators (`OLS`, `Ridge`, `Quantile`, ...). Since
0.8.0, `.to_dict()` also includes `model_info` as one of its keys.

**`.augment(x, y, residual_type=None)` (0.8.0):** `OLS`, `WLS` (`weights=None` additionally),
`Logistic`, `Probit`, `Cloglog`, `LogisticRegression`, `Poisson`, `NegativeBinomial`,
`Tweedie`, and `Gamma` expose per-observation diagnostics as a `dict` of numpy arrays —
`{"fitted", "residual", "std_residual", "stud_residual", "leverage", "cooks_d", "dffits",
"resid_type"}` — the same seven fields as the `ps.augment` expression's Struct, plus a
trailing `resid_type` key reporting the resolved residual type. It always describes exactly
what that instance was fit with. See `polars-statistics-regression` for the full
model-by-model semantics.

**Guarding fitted-property access:**

Accessing a fitted property (`.coefficients`, `.r_squared`, `.p_value`, etc.) on a model
that has not yet been fitted raises a `PyRuntimeError` with the message
"Model not fitted". Always guard with `is_fitted()` or wrap in `try/except` when access
is conditional:

```python
from polars_statistics import OLS
import numpy as np

rng = np.random.default_rng(42)
X = rng.standard_normal((50, 2))
y = X[:, 0] * 2.0 + X[:, 1] * -1.0 + rng.standard_normal(50) * 0.3

m = OLS(add_intercept=True)
# m.r_squared  # would raise PyRuntimeError — not yet fitted

m.fit(X, y)
if m.is_fitted():                 # safe guard
    print(m.r_squared)
    print(m.summary())
    d = m.to_dict()               # {'coefficients': [...], 'r_squared': ..., ...}
```

---

### Output struct field reference

| Expression | Key output fields |
|------------|-------------------|
| `ttest_ind`, `ttest_paired` | `statistic`, `p_value` |
| `brown_forsythe`, `yuen_test` | `statistic`, `p_value` |
| `mann_whitney_u`, `wilcoxon_signed_rank` | `statistic`, `p_value`, `estimate`, `ci_lower`, `ci_upper` (`estimate`/`ci_lower`/`ci_upper` are `NaN` unless `conf_level` is given) |
| `kruskal_wallis`, `brunner_munzel` | `statistic`, `p_value` |
| `shapiro_wilk`, `dagostino` | `statistic`, `p_value` |
| `pearson`, `spearman`, `kendall` | `estimate`, `statistic`, `p_value`, `ci_lower`, `ci_upper`, `n` |
| `distance_cor` | `estimate`, `statistic`, `p_value`, `ci_lower`, `ci_upper`, `n` |
| `icc` | `icc`, `f_value`, `df1`, `df2`, `p_value`, `ci_lower`, `ci_upper`, `n_subjects`, `n_raters` |
| `binom_test`, `prop_test_one`, `prop_test_two` | `estimate`, `statistic`, `p_value`, `ci_lower`, `ci_upper`, `n` |
| `chisq_test`, `g_test`, `mcnemar_test` | `statistic`, `p_value`, `df`, `n` |
| `fisher_exact` | `statistic` (sample odds ratio `ad/bc`), `p_value`, `ci_lower`, `ci_upper` (Woolf interval) |
| `fisher_exact_conditional` | `statistic` (conditional MLE odds ratio), `p_value`, `ci_lower`, `ci_upper` (exact conditional interval) |
| `cohens_d`, `cohens_d_one_sample`, `rank_biserial`, `matched_pairs_rank_biserial`, `rank_biserial_from_u` | `estimate` |
| `jarque_bera` | `statistic`, `p_value`, `skewness`, `kurtosis`, `n` |
| `bootstrap_mean_ci`, `bootstrap_ci` | `estimate`, `se`, `ci_lower`, `ci_upper`, `conf_level`, `n_bootstrap` |
| `mmd_test`, `energy_distance` | `statistic`, `p_value` |
| TOST functions | `estimate`, `ci_lower`, `ci_upper`, `bound_lower`, `bound_upper`, `tost_p_value`, `equivalent`, `alpha`, `n` |
| `one_way_anova` | `statistic`, `df_between`, `df_within`, `p_value`, `ss_between`, `ss_within`, `ms_between`, `ms_within`, `eta_squared`, `n_groups` |
| `two_way_anova` | `a_ss`, `a_df`, `a_ms`, `a_f`, `a_p_value`, `b_ss`, `b_df`, `b_ms`, `b_f`, `b_p_value`, `ab_ss`, `ab_df`, `ab_ms`, `ab_f`, `ab_p_value`, `residual_ss`, `residual_df`, `residual_ms`, `grand_mean`, `n` |
| `repeated_measures_anova` | `ws_f`, `ws_df`, `ws_ss`, `ws_ms`, `ws_p_value`, `error_df`, `error_ss`, `error_ms`, `mauchly_w`, `mauchly_p_value`, `gg_epsilon`, `gg_p_value`, `hf_epsilon`, `hf_p_value`, `grand_mean` |
| `ols` | `intercept`, `coefficients`, `r_squared`, `adj_r_squared`, `mse`, `rmse`, `f_statistic`, `f_pvalue`, `aic`, `bic`, `n_observations` |
| `ols_predict` | `ols_prediction`, `ols_lower`, `ols_upper` (prefix customizable with `name=`) |
| `ols_summary` | `List[Struct{term, estimate, std_error, statistic, p_value}]` — use `.explode().unnest()` |
| `penalized_glm` | `intercept`, `coefficients`, `deviance`, `null_deviance`, `pseudo_r_squared`, `log_likelihood`, `aic`, `bic`, `dispersion`, `theta`, `accuracy`, `has_separation`, `iterations`, `converged`, `n_observations` (15 fields) |
| `penalized_glm_summary`, `aft_summary` | `List[Struct{term, estimate, std_error, statistic, p_value, ci_lower, ci_upper}]` (widened 7-field tidy summary; `aft_summary` rows are `intercept`/`x1..xp`/`log_scale`, the last omitted for `distribution="exponential"`) — use `.explode().unnest()` |
| `aft` | `intercept`, `coefficients`, `scale`, `log_likelihood`, `null_log_likelihood`, `aic`, `bic`, `iterations`, `converged`, `n_observations`, `n_events`, `n_censored` (12 fields) |
| `penalized_glm_predict` | `penalized_glm_prediction`, `penalized_glm_lower`, `penalized_glm_upper` (prefix customizable with `name=`; `lower`/`upper` always `NaN` — no interval-prediction method upstream) |
| `aft_predict_quantile`, `aft_predict_median`, `aft_predict_survival`, `aft_predict_cdf` | `{name}_prediction`, `{name}_lower`, `{name}_upper` (default `name` is the function name, e.g. `aft_predict_survival_prediction`; `lower`/`upper` always `NaN`) |
| `tukey_hsd`, `pairwise_t_test`, `dunn_test` | `List[Struct{group1, group2, n1, n2, estimate, statistic, df, p_value, p_adj, ci_lower, ci_upper}]` — one row per group pair; use `.explode().unnest()`; `pairwise_t_test`'s `ci_lower`/`ci_upper` and `dunn_test`'s `df`/`ci_lower`/`ci_upper` are `NaN` |
| `p_adjust`, `ptukey`, `qtukey` | Bare `Float64` (no Struct wrapper, `returns_scalar=False`) — same length as the input column |
| `augment` | `Struct{fitted, residual, std_residual, stud_residual, leverage, cooks_d, dffits}` — one row per input row (`returns_scalar=False`, like the `*_predict` expressions) |
| `eb_shrink` | `Struct{shrunken, shrunken_se, weight, mu, mu_se, tau_squared, i_squared, q, n_groups}` — one row per input row; `mu`/`mu_se`/`tau_squared`/`i_squared`/`q`/`n_groups` are whole-sample scalars repeated on every row |

**Note on `ols_predict` naming:** the prefix of the three prediction fields matches the
`name=` kwarg you pass: `ps.ols_predict("y", "x", name="fitted")` produces
`fitted_prediction`, `fitted_lower`, `fitted_upper`.

---

### Typed `.pyi` API

The file `python/polars_statistics/_polars_statistics.pyi` contains complete type stubs
for all 45 PyModel classes. It is the authoritative source for:

- Property return types (e.g. `.coefficients -> numpy.ndarray`, `.r_squared -> float`)
- Method signatures including optional vs. required parameters
- `fit()` overloads — e.g. `WLS.fit(x, y, weights)` (three args) vs. the standard two-arg form

When the installed `.pyi` and this cheat-sheet differ, the `.pyi` takes precedence — it is
generated from the shipped Rust extension and reflects the exact runtime surface.

---

## Worked Examples

### Example 1: Whole-frame test — auto-detect and struct_to_dict

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "control":   [4.2, 3.8, 5.1, 4.6, 3.9, 4.1],
    "treatment": [5.5, 6.1, 5.8, 6.3, 5.9, 6.0],
})

# One struct column -> auto-detected; 1 row -> returns dict
result = df.select(ps.ttest_ind("control", "treatment").alias("t"))
d = ps.struct_to_dict(result, "t")
print(d)
# {'statistic': <float>, 'p_value': <float>}

print(f"p = {d['p_value']:.4f}")
```

### Example 2: unnest with auto-detect (single Struct column)

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 9.0]})

result = df.select(ps.pearson("a", "b").alias("cor"))

# Auto-detect: one struct column, no column= needed
flat = ps.unnest(result)
print(flat.columns)
# ['estimate', 'statistic', 'p_value', 'ci_lower', 'ci_upper', 'n']
print(flat["estimate"][0])
```

### Example 3: Multiple struct columns — pass column= explicitly

After a `group_by().agg()` that produces two struct columns, `ps.unnest(result)` would
raise `ValueError: Multiple Struct columns found`. Pass `column=` to specify which one:

```python
import polars as pl
import polars_statistics as ps
import numpy as np

rng = np.random.default_rng(42)
df = pl.DataFrame({
    "group":   ["A"] * 30 + ["B"] * 30,
    "y":       rng.normal(5, 1, 60).tolist(),
    "x":       rng.normal(2, 0.5, 60).tolist(),
    "control": rng.normal(3, 1, 60).tolist(),
})

# Two struct result columns per group
agg = df.group_by("group").agg(
    ps.ttest_ind("y", "control").alias("t"),
    ps.ols("y", "x").alias("reg"),
)

# Must specify column= because two Struct columns exist
t_flat = ps.unnest(agg, "t")
reg_flat = ps.unnest(agg, "reg")

# Or use prefix= to merge both into one frame:
merged = ps.unnest(agg, "t", prefix="t_")
merged = ps.unnest(merged, "reg", prefix="reg_")
print(merged.columns)
# ['group', 't_statistic', 't_p_value', 'reg_intercept', 'reg_coefficients', ...]
```

### Example 4: Per-group coefficient table — explode + unnest

`ols_summary` returns `List[Struct]` (one row per coefficient). Use `.explode().unnest()`:

```python
import polars as pl
import polars_statistics as ps
import numpy as np

rng = np.random.default_rng(7)
df = pl.DataFrame({
    "region": ["N"] * 40 + ["S"] * 40,
    "sales":  rng.normal(100, 10, 80).tolist(),
    "price":  rng.normal(50, 5, 80).tolist(),
    "promo":  rng.normal(0, 1, 80).tolist(),
})

coef_df = (
    df.group_by("region")
    .agg(ps.ols_summary("sales", "price", "promo").alias("coef"))
    .explode("coef")
    .unnest("coef")
)
print(coef_df.columns)
# ['region', 'term', 'estimate', 'std_error', 'statistic', 'p_value']
print(coef_df.filter(pl.col("term") == "price"))
```

### Example 5: PyModel to_dict and summary

```python
from polars_statistics import OLS
import numpy as np

rng = np.random.default_rng(99)
X = rng.standard_normal((80, 3))
y = X[:, 0] * 1.5 - X[:, 1] * 0.8 + rng.standard_normal(80) * 0.3

m = OLS(add_intercept=True).fit(X, y)

# Human-readable table
print(m.summary())

# Machine-readable dict
d = m.to_dict()
print(list(d.keys()))
# ['coefficients', 'intercept', 'r_squared', 'adj_r_squared', 'mse', 'rmse',
#  'f_statistic', 'f_pvalue', 'aic', 'bic', 'n_observations', ...]

# Unfitted guard
m2 = OLS()
if m2.is_fitted():             # False - safe
    print(m2.r_squared)        # never reached
```

### Example 6: Native field access and inline struct extraction

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "g": ["X"] * 20 + ["Y"] * 20,
    "a": list(range(20)) + list(range(5, 25)),
    "b": list(range(1, 21)) + list(range(3, 23)),
})
df = df.with_columns(pl.col("a", "b").cast(pl.Float64))

# Access a single field inline (no unnest needed):
p_values = (
    df.group_by("g")
    .agg(ps.ttest_ind("a", "b").alias("t"))
    .with_columns(pl.col("t").struct.field("p_value").alias("p"))
    .drop("t")
)
print(p_values)
```

---

## Gotchas / Notes

**Pass `column=` explicitly when a frame has two or more Struct columns**
After `group_by(...).agg(ps.ttest_ind(...).alias("t"), ps.ols(...).alias("reg"))` the result
has two Struct columns. Calling `ps.unnest(result)` raises
`ValueError: Multiple Struct columns found`. Pass the column name explicitly:
`ps.unnest(result, "t")`.

**Add `.alias("name")` before calling `ps.unnest()`**
Expression results are unnamed by default. If you aggregate multiple expressions in one
`group_by().agg()`, each needs an alias so `unnest` knows which struct column to expand.
`ps.unnest(result)` auto-detects the struct column only when exactly one struct column
exists in the DataFrame.

**Unfitted model access raises `PyRuntimeError`**
Calling `.coefficients`, `.r_squared`, or any other fitted property on a model before
`.fit()` raises a `PyRuntimeError`. The error message is "Model not fitted". Guard with
`.is_fitted()` before accessing any fitted property, especially when a model object is
passed around and the calling site cannot assume it has been fitted.

**`struct_to_dict` returns a list for multi-row results**
For a whole-frame test (`df.select(ps.ttest_ind(...))`), the Struct column has one row and
`struct_to_dict` returns a `dict`. For a per-group aggregation it returns `list[dict]`
(one dict per group row). Check the return type or use `unnest` when operating in a
DataFrame context.

**`ols_summary` produces `List[Struct]`, not a plain Struct**
The `ols_summary` and other `*_summary` expressions return a list of coefficient structs
(one element per term). Access them with `.explode(col).unnest(col)`, not with a plain
`.unnest(col)`. Using plain `.unnest()` will raise a type error because the column is
`List[Struct]`, not `Struct`.

**`with_intercept=` is deprecated — use `add_intercept=`**
The `with_intercept` parameter still works but emits a `FutureWarning`. All new code should
use `add_intercept=True/False`.

**The `.pyi` stub is authoritative for return types**
Property return types for PyModel classes (e.g. whether `.coefficients` is `list[float]` or
`numpy.ndarray`) are defined in `python/polars_statistics/_polars_statistics.pyi`. When in
doubt, consult the stub file rather than inferring from `summary()` output.

---

## See also

- polars-statistics-hypothesis-tests
- polars-statistics-anova
- polars-statistics-regression
- polars-statistics-getting-started
