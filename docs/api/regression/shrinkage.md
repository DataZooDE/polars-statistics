# Empirical-Bayes Shrinkage

Empirical-Bayes (random-effects meta-analytic) shrinkage of per-group estimates toward a
pooled mean, matching R's `metafor::rma(method = "DL")` / `blup()`. Pulls noisy, low-precision
estimates toward the overall mean while leaving precise estimates nearly untouched — the
classic James-Stein / DerSimonian-Laird shrinkage estimator.

> **Validation:** Validated against R `metafor::rma(method = "DL")`/`blup()` on the
> `metafor::dat.bcg` reference dataset and additional synthetic fixtures.

Works with `group_by` and `.over()`.

---

## Model

Given `k` groups' point estimates `y_i` and their standard errors `se_i`, the random-effects
model is `y_i = mu + u_i + e_i`, where `u_i ~ N(0, tau_squared)` (between-group heterogeneity)
and `e_i ~ N(0, se_i^2)` (within-group sampling error, known). `eb_shrink` estimates `tau_squared`
(DerSimonian-Laird by default), then computes the pooled mean `mu` and, for each row, the
empirical-Bayes shrunken estimate (the best linear unbiased predictor, BLUP):

```text
weight_i    = 1 / (se_i^2 + tau_squared)
mu          = sum(weight_i * y_i) / sum(weight_i)          # inverse-variance pooled mean
shrunken_i  = weight_i * se_i^2 * mu + (1 - weight_i * se_i^2) * y_i
            = y_i - se_i^2 / (se_i^2 + tau_squared) * (y_i - mu)
```

A group with a large `se_i` (noisy) gets pulled strongly toward `mu`; a group with a small
`se_i` (precise) stays close to its own `y_i`. When `tau_squared == 0` every group is pulled
all the way to `mu` (complete pooling); as `tau_squared -> infinity`, `shrunken_i -> y_i` (no
pooling).

---

## `eb_shrink`

```python title="Signature"
ps.eb_shrink(
    estimate: pl.Expr | str,
    std_error: pl.Expr | str,
    method: str = "dl",
    tau_squared: float | None = None,
) -> pl.Expr
```

**Parameters:**
- `estimate` — per-row point estimate (e.g. a per-store/per-study effect size), or a column name.
- `std_error` — per-row standard error of `estimate`, or a column name. Must be positive and
  finite; non-positive or non-finite values mark the row unusable.
- `method` (default `"dl"`) — `"dl"` estimates `tau_squared` via DerSimonian-Laird (matching
  `metafor::rma(method = "DL")`); `"none"` forces `tau_squared = 0` (complete pooling — every
  `shrunken` value equals `mu`).
- `tau_squared` (default `None`) — if given, overrides the estimator entirely and uses this
  fixed value instead of estimating it from the data. Must be finite and `>= 0`.

**Returns:** `Struct{shrunken, shrunken_se, weight, mu, mu_se, tau_squared, i_squared, q, n_groups}`
— see [EB Shrinkage Output](../outputs.md#eb-shrinkage-output) for the full field table.
One row per input row (`returns_scalar=False`, like the `*_predict` expressions) — every
shrunken value belongs to an input row, unlike the one-row `List[Struct]` shape used by the
[post-hoc pairwise tests](../tests/posthoc.md). `mu`/`mu_se`/`tau_squared`/`i_squared`/`q`/
`n_groups` are whole-sample scalars, repeated identically on every row.

**R equivalent:** `rma(yi = estimate, sei = std_error, method = "DL")` for `mu`/`tau_squared`/
`q`/`i_squared`, then `blup(fit)$pred`/`blup(fit)$se` for `shrunken`/`shrunken_se`.

**Notes:**
- `i_squared` is the Higgins-Thompson `I^2` (the proportion of total variation due to
  heterogeneity rather than sampling error), not an alternative heterogeneity measure.
- `q` is Cochran's Q heterogeneity test statistic.
- Rows with a null/non-finite `estimate`, or a non-positive/non-finite `std_error`, stay
  `NaN`-aligned (all nine fields `NaN` for that row) and are excluded from `mu`/`tau_squared`/
  `n_groups`. Fewer than 2 usable rows (a single usable row, every row unusable, or an empty
  frame) give all-`NaN` rows with `n_groups == 0` — never an exception.
- `ValueError` is raised at **build** time (before `.collect()`) for an unknown `method` or
  an invalid `tau_squared`.
- Use `.over(family)` (or `group_by(family).agg(...)`) to shrink several independent families
  of estimates within one DataFrame — each family gets its own `mu`/`tau_squared`.

**Example — per-store estimates shrunk toward the chain mean:**
```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "store": ["s1", "s2", "s3", "s4", "s5"],
    "estimate": [0.12, 0.35, -0.05, 0.40, 0.08],
    "std_error": [0.10, 0.30, 0.05, 0.35, 0.09],
})

result = df.with_columns(
    ps.eb_shrink("estimate", "std_error").alias("eb")
).unnest("eb")
print(result.select("store", "estimate", "shrunken", "shrunken_se", "mu", "tau_squared"))
```

**Example — several independent families with `.over()`:**
```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "family": ["a", "a", "a", "b", "b", "b"],
    "estimate": [0.12, 0.35, -0.05, 1.2, 0.9, 1.5],
    "std_error": [0.10, 0.30, 0.05, 0.40, 0.25, 0.50],
})

result = df.with_columns(
    ps.eb_shrink("estimate", "std_error").over("family").alias("eb")
).unnest("eb")
print(result.select("family", "estimate", "shrunken", "mu"))
```

---

## Validated against R

`mu`/`mu_se`/`tau_squared`/`q`/`i_squared` and every row's `shrunken`/`shrunken_se`/`weight`
match `metafor::rma(method = "DL")`/`blup()` to `1e-10` on the `metafor::dat.bcg` 13-study
reference dataset (log risk ratios and their standard errors) and additional synthetic
fixtures, including `method="none"` (complete pooling, `tau_squared == 0` exactly) and a fixed
`tau_squared` override.

---

## See Also

- [Diagnostics](diagnostics.md) - Model diagnostics, including [`augment`](diagnostics.md#augment-per-observation-diagnostics)
- [Post-hoc Tests & Multiple Comparisons](../tests/posthoc.md) - Pairwise comparison tests after an omnibus test
- [Output Structures](../outputs.md) - Full struct field definitions
