# Non-Parametric Tests

Distribution-free statistical tests that make no assumptions about the underlying distribution.

## `mann_whitney_u`

Mann-Whitney U test (Wilcoxon rank-sum test) for comparing two independent samples.

```python
ps.mann_whitney_u(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**Example:**
```python
df.select(ps.mann_whitney_u("treatment", "control"))
```

---

## `wilcoxon_signed_rank`

Wilcoxon signed-rank test for paired samples.

```python
ps.wilcoxon_signed_rank(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**Example:**
```python
df.select(ps.wilcoxon_signed_rank("before", "after"))
```

---

## `kruskal_wallis`

Kruskal-Wallis H test for comparing 3 or more independent groups.

```python
ps.kruskal_wallis(
    *groups: Union[pl.Expr, str],  # 3 or more groups
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**Example:**
```python
df.select(ps.kruskal_wallis("group1", "group2", "group3"))
```

---

## `brunner_munzel`

Brunner-Munzel test for stochastic equality. More robust than Mann-Whitney when variances differ.

```python
ps.brunner_munzel(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    alternative: str = "two-sided",  # "two-sided", "less", "greater"
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

---

## See Also

- [Parametric Tests](parametric.md) - Tests with distributional assumptions
- [TOST Equivalence Tests](tost.md) - Non-parametric equivalence tests
