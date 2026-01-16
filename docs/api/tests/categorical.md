# Categorical Tests

Tests for categorical data, proportions, and contingency tables.

## Proportion Tests

### `binom_test`

Exact binomial test.

```python
ps.binom_test(
    successes: int,
    n: int,
    p0: float = 0.5,
    alternative: str = "two-sided",  # "two-sided", "less", "greater"
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64, statistic: Float64, p_value: Float64, ci_lower: Float64, ci_upper: Float64, n: UInt32}`

---

### `prop_test_one`

One-sample proportion test (normal approximation).

```python
ps.prop_test_one(
    successes: int,
    n: int,
    p0: float = 0.5,
    alternative: str = "two-sided",
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64, statistic: Float64, p_value: Float64, ci_lower: Float64, ci_upper: Float64, n: UInt32}`

---

### `prop_test_two`

Two-sample proportion test.

```python
ps.prop_test_two(
    successes1: int,
    n1: int,
    successes2: int,
    n2: int,
    alternative: str = "two-sided",
    correction: bool = False,  # Yates' continuity correction
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64, statistic: Float64, p_value: Float64, ci_lower: Float64, ci_upper: Float64, n: UInt32}`

---

## Contingency Table Tests

### `chisq_test`

Chi-square test for independence in contingency table.

```python
ps.chisq_test(
    data: Union[pl.Expr, str],  # Flattened contingency table (row-major)
    n_rows: int = 2,
    n_cols: int = 2,
    correction: bool = False,  # Yates' continuity correction
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64, df: Float64, n: UInt32}`

**Example:**
```python
# 2x2 table: [[10, 20], [30, 40]] flattened to [10, 20, 30, 40]
df.select(ps.chisq_test("counts", n_rows=2, n_cols=2))
```

---

### `chisq_goodness_of_fit`

Chi-square goodness-of-fit test.

```python
ps.chisq_goodness_of_fit(
    observed: Union[pl.Expr, str],
    expected: Union[pl.Expr, str] | None = None,  # None = uniform distribution
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64, df: Float64, n: UInt32}`

---

### `g_test`

G-test (likelihood ratio test) for independence.

```python
ps.g_test(
    data: Union[pl.Expr, str],
    n_rows: int = 2,
    n_cols: int = 2,
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64, df: Float64, n: UInt32}`

---

### `fisher_exact`

Fisher's exact test for 2x2 contingency tables.

```python
ps.fisher_exact(
    a: int, b: int, c: int, d: int,  # 2x2 table cells
    alternative: str = "two-sided",
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64 (odds ratio), p_value: Float64}`

**Table layout:**
```
     | Col1 | Col2 |
-----|------|------|
Row1 |  a   |  b   |
Row2 |  c   |  d   |
```

---

## Paired Proportion Tests

### `mcnemar_test`

McNemar's test for paired proportions.

```python
ps.mcnemar_test(
    a: int, b: int, c: int, d: int,  # 2x2 table cells
    correction: bool = False,  # Edwards' continuity correction
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64, df: Float64, n: UInt32}`

---

### `mcnemar_exact`

McNemar's exact test for paired proportions.

```python
ps.mcnemar_exact(
    a: int, b: int, c: int, d: int,
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

---

## Agreement and Association Measures

### `cohen_kappa`

Cohen's Kappa for inter-rater agreement.

```python
ps.cohen_kappa(
    data: Union[pl.Expr, str],  # Flattened confusion matrix
    n_categories: int = 2,
    weighted: bool = False,  # Linear weights
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64 (kappa), statistic: Float64 (se), p_value: Float64}`

**Interpretation:**
- κ < 0.20: Poor agreement
- 0.21-0.40: Fair agreement
- 0.41-0.60: Moderate agreement
- 0.61-0.80: Substantial agreement
- 0.81-1.00: Almost perfect agreement

---

### `cramers_v`

Cramer's V for association strength (0 to 1).

```python
ps.cramers_v(
    data: Union[pl.Expr, str],
    n_rows: int = 2,
    n_cols: int = 2,
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64, statistic: Float64, p_value: Float64}`

---

### `phi_coefficient`

Phi coefficient for 2x2 tables.

```python
ps.phi_coefficient(
    a: int, b: int, c: int, d: int,
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64, statistic: Float64, p_value: Float64}`

---

### `contingency_coef`

Contingency coefficient (Pearson's C).

```python
ps.contingency_coef(
    data: Union[pl.Expr, str],
    n_rows: int = 2,
    n_cols: int = 2,
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64, statistic: Float64, p_value: Float64}`

---

## See Also

- [TOST Equivalence Tests](tost.md) - Equivalence tests for proportions
