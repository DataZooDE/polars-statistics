# Demand Classification (AID)

Automatic Identification of Demand patterns with anomaly detection.

> **Reference:** Svetunkov, I. & Sroginis, A. (2025). *Why do zeroes happen? A model-based approach for demand classification*. [arXiv:2504.05894](https://arxiv.org/abs/2504.05894)

## `aid`

Classifies demand time series as regular or intermittent, fits the best distribution, and detects anomalies.

```python
ps.aid(
    y: Union[pl.Expr, str],
    intermittent_threshold: float = 0.3,  # Zero proportion threshold
) -> pl.Expr
```

**Returns:** See [AID Output](../outputs.md#aid-output)

**Example:**

```python
# Classify demand per SKU
result = df.group_by("sku").agg(
    ps.aid("demand").alias("classification")
).unnest("classification")
```

---

## `aid_anomalies`

Returns per-observation anomaly flags. Use with `.over()` to add anomaly columns to the original DataFrame.

```python
ps.aid_anomalies(
    y: Union[pl.Expr, str],
    intermittent_threshold: float = 0.3,
) -> pl.Expr
```

**Returns:** See [AID Anomalies Output](../outputs.md#aid-anomalies-output)

---

## Anomaly Types

| Type | Description |
|------|-------------|
| `stockout` | Unexpected zero in otherwise positive demand |
| `new_product` | Leading zeros pattern (product launch) |
| `obsolete_product` | Trailing zeros pattern (product phase-out) |
| `high_outlier` | Unusually high demand value |
| `low_outlier` | Unusually low demand value |

---

## Example: Per-Row Anomaly Detection

```python
# Add per-row anomaly flags
result = df.with_columns(
    ps.aid_anomalies("demand").over("sku").alias("anomalies")
).unnest("anomalies")

# Filter to flagged observations
flagged = result.filter(pl.col("high_outlier") | pl.col("stockout"))
```

---

## Supported Distributions

| Category | Distributions |
|----------|---------------|
| Count | `poisson`, `negative_binomial`, `geometric` |
| Continuous | `normal`, `gamma`, `lognormal`, `rectified_normal` |

---

## See Also

- [ALM](alm.md) - Augmented Linear Model distributions
