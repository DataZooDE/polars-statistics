# Correlation Tests

Tests for measuring and testing associations between variables.

## `pearson`

Pearson correlation coefficient with hypothesis test.

```python
ps.pearson(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    conf_level: float = 0.95,
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64, statistic: Float64, p_value: Float64, ci_lower: Float64, ci_upper: Float64, n: UInt32}`

**Example:**
```python
df.select(ps.pearson("x", "y").alias("cor"))
```

---

## `spearman`

Spearman rank correlation coefficient with hypothesis test. Robust to outliers and non-linear monotonic relationships.

```python
ps.spearman(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    conf_level: float = 0.95,
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64, statistic: Float64, p_value: Float64, ci_lower: Float64, ci_upper: Float64, n: UInt32}`

---

## `kendall`

Kendall's tau correlation coefficient with hypothesis test.

```python
ps.kendall(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    variant: str = "b",  # "a", "b", or "c"
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64, statistic: Float64, p_value: Float64, ci_lower: Float64, ci_upper: Float64, n: UInt32}`

**Variants:**
- `"a"`: Does not adjust for ties
- `"b"`: Adjusts for ties (default, most common)
- `"c"`: Stuart's tau-c for rectangular tables

---

## `distance_cor`

Distance correlation with permutation test. Detects both linear and nonlinear associations.

```python
ps.distance_cor(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    n_permutations: int = 999,
    seed: int | None = None,
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64, statistic: Float64, p_value: Float64, ci_lower: Float64, ci_upper: Float64, n: UInt32}`

**Note:** Distance correlation is 0 if and only if the variables are independent (unlike Pearson/Spearman).

---

## `partial_cor`

Partial correlation controlling for covariates.

```python
ps.partial_cor(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    covariates: list[Union[pl.Expr, str]],
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64, statistic: Float64, p_value: Float64, ci_lower: Float64, ci_upper: Float64, n: UInt32}`

**Example:**
```python
# Correlation between x and y, controlling for z1 and z2
df.select(ps.partial_cor("x", "y", ["z1", "z2"]))
```

---

## `semi_partial_cor`

Semi-partial (part) correlation. Controls for covariates in y only.

```python
ps.semi_partial_cor(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    covariates: list[Union[pl.Expr, str]],
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64, statistic: Float64, p_value: Float64, ci_lower: Float64, ci_upper: Float64, n: UInt32}`

---

## `icc`

Intraclass Correlation Coefficient (ICC) for reliability and agreement.

```python
ps.icc(
    values: Union[pl.Expr, str],
    icc_type: str = "icc1",  # "icc1", "icc2", "icc3", "icc2k", "icc3k"
    conf_level: float = 0.95,
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64, statistic: Float64, p_value: Float64, ci_lower: Float64, ci_upper: Float64, n: UInt32}`

**ICC Types:**
- `"icc1"`: One-way random effects, absolute agreement
- `"icc2"`: Two-way random effects, absolute agreement
- `"icc3"`: Two-way mixed effects, consistency
- `"icc2k"`, `"icc3k"`: Average of k raters

---

## See Also

- [TOST Equivalence Tests](tost.md) - `tost_correlation` for equivalence testing
