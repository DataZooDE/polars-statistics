# Post-hoc Tests & Multiple Comparisons

Pairwise multiple-comparison tests that follow an omnibus test (`one_way_anova` or
`kruskal_wallis`), plus the generic p-value adjustment and studentized-range distribution
functions they build on.

> **Validation:** All six functions are validated against R `TukeyHSD(aov(...))`,
> `pairwise.t.test`, `dunn.test(altp = TRUE)` / `FSA::dunnTest`, `p.adjust`, `ptukey`, and
> `qtukey` on the upstream `anofox-statistics` 0.4.7 reference fixtures.

## Long Format vs `one_way_anova`'s Wide Format

`one_way_anova`/`kruskal_wallis` take one column per group (wide format):

```text
df.select(ps.one_way_anova("control", "treatment_a", "treatment_b"))
```

`tukey_hsd`, `pairwise_t_test`, and `dunn_test` instead take **long format**: one flat
`values` column plus one `groups` label column, mirroring how real DataFrames store
grouped observations.

```text
df.select(ps.tukey_hsd("value", "group"))
```

**When to use which:** run the omnibus test (`one_way_anova` for parametric data,
`kruskal_wallis` for non-parametric data) first. If it is significant, follow up with
`tukey_hsd` (parametric, equal variances), `pairwise_t_test` (parametric, pooled or Welch),
or `dunn_test` (rank-based, matches `kruskal_wallis`'s assumptions) to find which pairs of
groups actually differ.

---

## `tukey_hsd`

Tukey's honest significant difference: all pairwise mean differences with simultaneous
(family-wise) confidence intervals, matching R's
`TukeyHSD(aov(values ~ factor(groups)), conf.level = conf_level)`.

```python title="Signature"
ps.tukey_hsd(
    values: pl.Expr | str,
    groups: pl.Expr | str,
    conf_level: float = 0.95,
) -> pl.Expr
```

**Parameters:**
- `values` — observations, or a column name.
- `groups` — group labels, or a column name. Cast to `pl.String`; numeric group columns are
  compared as strings (zero-pad if numeric sort order matters).
- `conf_level` (default `0.95`) — family-wise confidence level for `ci_lower`/`ci_upper`.
  Must be in `(0, 1)`, checked when the expression is **built** (before `.collect()`).

**Returns:** `List[Struct{group1, group2, n1, n2, estimate, statistic, df, p_value, p_adj, ci_lower, ci_upper}]`
— see [Pairwise Comparison Output](../outputs.md#pairwise-comparison-output) for the full field
table. `estimate` is `mean(group2) - mean(group1)`. `p_value == p_adj` (Tukey's test is already
simultaneous — there is no separate unadjusted p-value). For `k` groups there are `k(k-1)/2`
rows, ordered like R's `TukeyHSD` (B-A, C-A, C-B, ...).

**R equivalent:** `TukeyHSD(aov(values ~ factor(groups)), conf.level = conf_level)`.

**Notes:** Rows with a null/NaN value or a null group are dropped first (R's `NA` row-drop
semantics). Fewer than 2 usable groups give an empty list, never an exception. Inside
`group_by().agg()`, each group is analysed independently (same caveat as `one_way_anova`).

**Example:**
```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    "group": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
})

result = df.select(
    ps.tukey_hsd("value", "group").alias("ph")
).select(pl.col("ph").explode().struct.unnest())
print(result)
```

---

## `pairwise_t_test`

Pairwise t-tests between every pair of groups, with p-value adjustment for multiple
comparisons, matching R's
`pairwise.t.test(values, groups, p.adjust.method, pool.sd, alternative)`.

```python title="Signature"
ps.pairwise_t_test(
    values: pl.Expr | str,
    groups: pl.Expr | str,
    pooled_sd: bool = True,
    p_adjust_method: str = "holm",
    alternative: str = "two-sided",
) -> pl.Expr
```

**Parameters:**
- `values`/`groups` — same long-format convention as `tukey_hsd`.
- `pooled_sd` (default `True`) — if `True`, every comparison uses the standard deviation
  pooled over **all** groups (R's default). If `False`, each pair runs a Welch two-sample
  t-test using only that pair's data.
- `p_adjust_method` (default `"holm"`) — one of `"holm"`, `"hochberg"`, `"hommel"`,
  `"bonferroni"`, `"BH"` (alias `"fdr"`), `"BY"`, `"none"` (case-insensitive); see
  [`p_adjust`](#p_adjust).
- `alternative` (default `"two-sided"`) — `"two-sided"`/`"two_sided"`/`"two.sided"`,
  `"less"`, or `"greater"`, for `group2 - group1`.

**Returns:** `List[Struct{group1, group2, n1, n2, estimate, statistic, df, p_value, p_adj, ci_lower, ci_upper}]`
(`ci_lower`/`ci_upper` are `NaN` — R's `pairwise.t.test` computes no confidence intervals). See
[Pairwise Comparison Output](../outputs.md#pairwise-comparison-output).

**R equivalent:** `pairwise.t.test(values, groups, p.adjust.method = p_adjust_method, pool.sd = pooled_sd, alternative = alternative)`.

**Notes:** Same NA-row-drop, degenerate-input, and `group_by` semantics as `tukey_hsd`.
`ValueError` is raised at build time for an unknown `p_adjust_method` or `alternative`.

**Example:**
```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    "group": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
})

result = df.select(
    ps.pairwise_t_test("value", "group", p_adjust_method="bonferroni").alias("ph")
).select(pl.col("ph").explode().struct.unnest())
print(result)
```

---

## `dunn_test`

Dunn's (1964) pairwise rank-mean comparison, the standard post-hoc test after a significant
`kruskal_wallis`, matching `dunn.test::dunn.test(values, groups, method, altp = TRUE)` /
`FSA::dunnTest`.

```python title="Signature"
ps.dunn_test(
    values: pl.Expr | str,
    groups: pl.Expr | str,
    p_adjust_method: str = "holm",
) -> pl.Expr
```

**Parameters:**
- `values`/`groups` — same long-format convention as `tukey_hsd`.
- `p_adjust_method` (default `"holm"`) — same set of methods as `pairwise_t_test`.

**Returns:** `List[Struct{group1, group2, n1, n2, estimate, statistic, df, p_value, p_adj, ci_lower, ci_upper}]`
(`df`/`ci_lower`/`ci_upper` are `NaN`). `estimate` is the difference in mean ranks
(`group2 - group1`). **Sign-convention note:** `dunn.test`/`FSA` report the opposite sign for
their own `z` statistic. See [Pairwise Comparison Output](../outputs.md#pairwise-comparison-output).

**R equivalent:** `dunn.test::dunn.test(values, groups, method = p_adjust_method, altp = TRUE)` /
`FSA::dunnTest(values ~ groups, method = p_adjust_method)`.

**Notes:** Same NA-row-drop, degenerate-input, and `group_by` semantics as `tukey_hsd`.

**Example:**
```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    "group": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
})

result = df.select(
    ps.dunn_test("value", "group").alias("ph")
).select(pl.col("ph").explode().struct.unnest())
print(result)
```

---

## `p_adjust`

Adjust a family of p-values for multiple comparisons, matching R's `p.adjust(p, method)`.

```python title="Signature"
ps.p_adjust(
    p: pl.Expr | str,
    method: str = "holm",
) -> pl.Expr
```

**Parameters:**
- `p` — column of raw p-values, or a column name.
- `method` (default `"holm"`) — one of `"holm"`, `"hochberg"`, `"hommel"`, `"bonferroni"`,
  `"BH"` (alias `"fdr"`), `"BY"`, `"none"` (case-insensitive).

**Returns:** Bare `Float64` expression, same length as `p` (no `Struct` wrapper — see
[Elementwise Float64 Output](../outputs.md#elementwise-float64-output)).

**R equivalent:** `p.adjust(p, method = method)`.

**Notes:** Null and NaN entries are kept (null stays null, NaN stays NaN) and are excluded
from the comparison count `n`, matching R's `NA` handling. This is a **whole-column**
operation, not a true elementwise one — it adjusts the entire family of p-values together.
Use `.over(group_col)` to adjust several independent families within one DataFrame; inside
`group_by().agg()` it returns one adjusted list per group. `ValueError` is raised at build
time for an unknown `method`.

**Example:**
```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "family": ["x", "x", "x", "y", "y"],
    "p": [0.001, 0.01, 0.03, 0.04, 0.2],
})

# Whole-column adjustment
print(df.select(ps.p_adjust("p", method="holm").alias("p_adj")))

# Adjust each family independently
print(
    df.with_columns(
        ps.p_adjust("p", method="BH").over("family").alias("p_adj")
    )
)
```

---

## `ptukey`

Studentized range distribution CDF, matching R's `ptukey(q, nmeans, df, nranges)`. Used by
`tukey_hsd` internally to convert the studentized range statistic to a p-value.

```python title="Signature"
ps.ptukey(
    q: pl.Expr | str | float,
    nmeans: pl.Expr | str | float,
    df: pl.Expr | str | float,
    nranges: pl.Expr | str | float = 1,
    lower_tail: bool = True,
) -> pl.Expr
```

**Parameters:**
- `q` — studentized range quantile, column name, expression, or scalar.
- `nmeans` — number of means in the range.
- `df` — degrees of freedom.
- `nranges` (default `1`) — number of ranges.
- `lower_tail` (default `True`) — if `True`, returns `P(Q <= q)`; if `False`, returns `P(Q > q)`.

**Returns:** Bare `Float64` expression (see
[Elementwise Float64 Output](../outputs.md#elementwise-float64-output)). Each argument may be
a scalar (broadcasts against the others) or a column (all columns must share the same length).

**R equivalent:** `ptukey(q, nmeans, df, nranges, lower.tail = lower_tail)`. Agrees with R to
within `1e-10`.

**Notes:** Returns `NaN` (never an exception) for `NaN` arguments or invalid parameters
(`df < 2`, `nranges < 1`, `nmeans < 2`).

**Example:**
```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({"q": [2.0, 3.0, 4.0]})
print(df.select(ps.ptukey("q", nmeans=3, df=10)))
```

---

## `qtukey`

Studentized range distribution quantile function, matching R's `qtukey(p, nmeans, df, nranges)`.
The inverse of [`ptukey`](#ptukey); used by `tukey_hsd` to compute the critical value for a
given confidence level.

```python title="Signature"
ps.qtukey(
    p: pl.Expr | str | float,
    nmeans: pl.Expr | str | float,
    df: pl.Expr | str | float,
    nranges: pl.Expr | str | float = 1,
    lower_tail: bool = True,
) -> pl.Expr
```

**Parameters:**
- `p` — probability in `[0, 1]`, column name, expression, or scalar.
- `nmeans` — number of means in the range.
- `df` — degrees of freedom.
- `nranges` (default `1`) — number of ranges.
- `lower_tail` (default `True`) — if `True`, `p` is `P(Q <= q)`; if `False`, `p` is `P(Q > q)`.

**Returns:** Bare `Float64` expression (see
[Elementwise Float64 Output](../outputs.md#elementwise-float64-output)). Each argument may be
a scalar (broadcasts against the others) or a column (all columns must share the same length).

**R equivalent:** `qtukey(p, nmeans, df, nranges, lower.tail = lower_tail)`. Agrees with R to
within `1e-8` (`ptukey(qtukey(p, ...), ...)` round-trips to `1e-6` — `qtukey`'s Newton-Raphson
solver converges to its own `1e-4` criterion, not a precision bug).

**Notes:** Returns `NaN` (never an exception) for `NaN` arguments, invalid parameters
(`df < 2`, `nranges < 1`, `nmeans < 2`), or `p` outside `[0, 1]`.

**Example:**
```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({"p": [0.9, 0.95, 0.99]})
print(df.select(ps.qtukey("p", nmeans=3, df=10)))
```

---

## See Also

- [Parametric Tests](parametric.md) - `one_way_anova` (the omnibus test `tukey_hsd`/`pairwise_t_test` follow up on)
- [Non-Parametric Tests](nonparametric.md) - `kruskal_wallis` (the omnibus test `dunn_test` follows up on)
- [Output Structures](../outputs.md) - Full struct-field reference
