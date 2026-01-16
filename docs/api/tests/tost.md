# TOST Equivalence Tests

Two One-Sided Tests (TOST) for testing practical equivalence. Unlike traditional hypothesis tests that test for difference, TOST tests whether effects are small enough to be considered equivalent.

## Common Parameters

| Parameter | Description |
|-----------|-------------|
| `bounds_type` | `"symmetric"` (±delta), `"raw"` (lower/upper), or `"cohen_d"` |
| `delta` | Equivalence margin for symmetric/cohen_d bounds |
| `lower`, `upper` | Explicit bounds for raw bounds_type |
| `alpha` | Significance level (default 0.05) |

## Return Structure

All TOST tests return:
```
Struct{
    estimate: Float64,      # Point estimate
    ci_lower: Float64,      # CI lower bound
    ci_upper: Float64,      # CI upper bound
    bound_lower: Float64,   # Equivalence lower bound
    bound_upper: Float64,   # Equivalence upper bound
    tost_p_value: Float64,  # TOST p-value (max of two one-sided)
    equivalent: Boolean,    # True if equivalence established
    alpha: Float64,
    n: UInt32,
}
```

---

## T-Test Based TOST

### `tost_t_test_one_sample`

One-sample TOST equivalence test.

```python
ps.tost_t_test_one_sample(
    x: Union[pl.Expr, str],
    mu: float = 0.0,
    bounds_type: str = "symmetric",
    delta: float = 0.5,
    lower: float = -0.5,
    upper: float = 0.5,
    alpha: float = 0.05,
) -> pl.Expr
```

---

### `tost_t_test_two_sample`

Two-sample TOST equivalence test.

```python
ps.tost_t_test_two_sample(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    bounds_type: str = "symmetric",
    delta: float = 0.5,
    lower: float = -0.5,
    upper: float = 0.5,
    alpha: float = 0.05,
    pooled: bool = False,
) -> pl.Expr
```

**Example:**
```python
df.select(ps.tost_t_test_two_sample("treatment", "control", delta=0.5))
```

---

### `tost_t_test_paired`

Paired-samples TOST equivalence test.

```python
ps.tost_t_test_paired(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    bounds_type: str = "symmetric",
    delta: float = 0.5,
    lower: float = -0.5,
    upper: float = 0.5,
    alpha: float = 0.05,
) -> pl.Expr
```

---

## Correlation TOST

### `tost_correlation`

Correlation TOST equivalence test using Fisher's z-transformation. Tests whether correlation is close to a reference value.

```python
ps.tost_correlation(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    method: str = "pearson",  # "pearson" or "spearman"
    rho_null: float = 0.0,
    bounds_type: str = "symmetric",
    delta: float = 0.3,
    lower: float = -0.3,
    upper: float = 0.3,
    alpha: float = 0.05,
) -> pl.Expr
```

**Example:**
```python
# Test if correlation is equivalent to zero (negligible association)
df.select(ps.tost_correlation("x", "y", delta=0.3))
```

---

## Proportion TOST

### `tost_prop_one`

One-proportion TOST equivalence test.

```python
ps.tost_prop_one(
    successes: int,
    n: int,
    p0: float = 0.5,
    bounds_type: str = "symmetric",
    delta: float = 0.1,
    lower: float = -0.1,
    upper: float = 0.1,
    alpha: float = 0.05,
) -> pl.Expr
```

---

### `tost_prop_two`

Two-proportion TOST equivalence test.

```python
ps.tost_prop_two(
    successes1: int,
    n1: int,
    successes2: int,
    n2: int,
    bounds_type: str = "symmetric",
    delta: float = 0.1,
    lower: float = -0.1,
    upper: float = 0.1,
    alpha: float = 0.05,
) -> pl.Expr
```

---

## Non-Parametric TOST

### `tost_wilcoxon_paired`

Wilcoxon paired-samples TOST equivalence test (non-parametric).

```python
ps.tost_wilcoxon_paired(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    bounds_type: str = "symmetric",
    delta: float = 0.5,
    lower: float = -0.5,
    upper: float = 0.5,
    alpha: float = 0.05,
) -> pl.Expr
```

---

### `tost_wilcoxon_two_sample`

Wilcoxon two-sample TOST equivalence test (non-parametric).

```python
ps.tost_wilcoxon_two_sample(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    bounds_type: str = "symmetric",
    delta: float = 0.5,
    lower: float = -0.5,
    upper: float = 0.5,
    alpha: float = 0.05,
) -> pl.Expr
```

---

## Robust TOST

### `tost_bootstrap`

Bootstrap TOST equivalence test. Uses bootstrap resampling for inference.

```python
ps.tost_bootstrap(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    bounds_type: str = "symmetric",
    delta: float = 0.5,
    lower: float = -0.5,
    upper: float = 0.5,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    seed: int | None = None,
) -> pl.Expr
```

---

### `tost_yuen`

Yuen TOST equivalence test using trimmed means (robust to outliers).

```python
ps.tost_yuen(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    trim: float = 0.2,
    bounds_type: str = "symmetric",
    delta: float = 0.5,
    lower: float = -0.5,
    upper: float = 0.5,
    alpha: float = 0.05,
) -> pl.Expr
```

---

## Interpretation

- **equivalent = True**: The effect is within the equivalence bounds at the specified alpha level
- **equivalent = False**: Cannot conclude equivalence (effect may be too large or sample size too small)
- **tost_p_value**: The maximum of the two one-sided p-values; reject the null hypothesis of non-equivalence if < alpha

## See Also

- [Parametric Tests](parametric.md) - Traditional t-tests
- [Non-Parametric Tests](nonparametric.md) - Traditional non-parametric tests
