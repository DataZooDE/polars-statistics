# Parametric Tests

Parametric statistical tests that assume specific distributional properties of the data.

## `ttest_ind`

Independent samples t-test (Welch's t-test by default).

```python
ps.ttest_ind(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    alternative: str = "two-sided",  # "two-sided", "less", "greater"
    equal_var: bool = False,         # False = Welch's, True = Student's
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**Example:**
```python
df.select(ps.ttest_ind("treatment", "control", alternative="two-sided"))
```

---

## `ttest_paired`

Paired samples t-test.

```python
ps.ttest_paired(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    alternative: str = "two-sided",  # "two-sided", "less", "greater"
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**Example:**
```python
df.select(ps.ttest_paired("before", "after"))
```

---

## `brown_forsythe`

Brown-Forsythe test for equality of variances.

```python
ps.brown_forsythe(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

---

## `yuen_test`

Yuen's test for trimmed means (robust to outliers).

```python
ps.yuen_test(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    trim: float = 0.2,  # Proportion to trim from each tail
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**Note:** Trimmed means are more robust to outliers than regular means. The default `trim=0.2` removes 20% from each tail.

---

## See Also

- [Non-Parametric Tests](nonparametric.md) - Distribution-free alternatives
- [TOST Equivalence Tests](tost.md) - Equivalence testing versions
