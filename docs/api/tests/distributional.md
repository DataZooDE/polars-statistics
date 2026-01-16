# Distributional Tests

Tests for checking distributional properties of data, particularly normality.

## `shapiro_wilk`

Shapiro-Wilk test for normality. One of the most powerful normality tests for small to medium samples.

```python
ps.shapiro_wilk(
    x: Union[pl.Expr, str],
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**Example:**
```python
df.select(ps.shapiro_wilk("values").alias("normality_test"))
```

**Interpretation:**
- p > 0.05: Data is consistent with normal distribution
- p ≤ 0.05: Significant deviation from normality

---

## `dagostino`

D'Agostino-Pearson test for normality. Tests for skewness and kurtosis.

```python
ps.dagostino(
    x: Union[pl.Expr, str],
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**Note:** Requires at least 20 observations for reliable results.

---

## See Also

- [Parametric Tests](parametric.md) - Tests that assume normality
- [Non-Parametric Tests](nonparametric.md) - Distribution-free alternatives
