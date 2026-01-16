# Forecast Comparison Tests

Tests for comparing predictive accuracy of forecasting models.

## `diebold_mariano`

Diebold-Mariano test for equal predictive accuracy between two forecasts.

```python
ps.diebold_mariano(
    errors1: Union[pl.Expr, str],
    errors2: Union[pl.Expr, str],
    loss: str = "squared",  # "squared", "absolute"
    horizon: int = 1,       # Forecast horizon
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**Example:**
```python
df.select(ps.diebold_mariano("model1_errors", "model2_errors", horizon=1))
```

---

## `permutation_t_test`

Permutation-based t-test (non-parametric).

```python
ps.permutation_t_test(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    alternative: str = "two-sided",
    n_permutations: int = 999,
    seed: int | None = None,
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

---

## `clark_west`

Clark-West test for comparing nested forecasting models.

```python
ps.clark_west(
    restricted_errors: Union[pl.Expr, str],
    unrestricted_errors: Union[pl.Expr, str],
    horizon: int = 1,
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

---

## `spa_test`

Superior Predictive Ability (SPA) test. Tests whether any model outperforms a benchmark.

```python
ps.spa_test(
    benchmark_loss: Union[pl.Expr, str],
    *model_losses: Union[pl.Expr, str],
    n_bootstrap: int = 999,
    block_length: float = 5.0,
    seed: int | None = None,
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

---

## `model_confidence_set`

Model Confidence Set (MCS) for identifying the set of best models.

```python
ps.model_confidence_set(
    *model_losses: Union[pl.Expr, str],
    alpha: float = 0.1,
    statistic: str = "range",  # "range" or "semi-quadratic"
    n_bootstrap: int = 999,
    block_length: float = 5.0,
    seed: int | None = None,
) -> pl.Expr
```

**Returns:** `Struct{included: List[Boolean], p_values: List[Float64]}`

---

## `mspe_adjusted`

MSPE-Adjusted SPA test for nested models.

```python
ps.mspe_adjusted(
    benchmark_errors: Union[pl.Expr, str],
    *model_errors: Union[pl.Expr, str],
    n_bootstrap: int = 999,
    block_length: float = 5.0,
    seed: int | None = None,
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

---

## Modern Distribution Tests

### `energy_distance`

Energy Distance test for comparing distributions. Detects differences in both location and shape.

```python
ps.energy_distance(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    n_permutations: int = 999,
    seed: int | None = None,
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

---

### `mmd_test`

Maximum Mean Discrepancy (MMD) test with Gaussian kernel.

```python
ps.mmd_test(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    n_permutations: int = 999,
    seed: int | None = None,
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

---

## See Also

- [Parametric Tests](parametric.md) - Standard hypothesis tests
