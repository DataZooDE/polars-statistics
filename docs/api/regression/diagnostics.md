# Regression Diagnostics

Tools for detecting multicollinearity, quasi-separation, and other model issues.

## `condition_number`

Compute condition number diagnostics to detect multicollinearity in design matrices.

```python
ps.condition_number(
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr
```

**Returns:** See [Condition Number Output](../outputs.md#condition-number-output)

**Example:**
```python
df.select(ps.condition_number("x1", "x2", "x3").alias("diagnostics"))
```

**Interpretation:**

| Condition Number | Severity | Recommendation |
|------------------|----------|----------------|
| < 30 | Well-conditioned | Numerically stable |
| 30 - 100 | Moderate | Monitor for instability |
| 100 - 1000 | High | Consider regularization or removing predictors |
| > 1000 | Severe | Strong multicollinearity present |

---

## `check_binary_separation`

Detect quasi-separation in binary response data (logistic/probit regression).

```python
ps.check_binary_separation(
    y: Union[pl.Expr, str],      # Binary (0/1)
    *x: Union[pl.Expr, str],
) -> pl.Expr
```

**Returns:** See [Separation Check Output](../outputs.md#separation-check-output)

**Example:**
```python
df.select(ps.check_binary_separation("success", "predictor1", "predictor2"))
```

**Separation Types:**
- `Complete`: Predictor perfectly divides the classes (MLE does not exist)
- `Quasi`: Nearly perfect separation with 1-2 observations crossing
- `MonotonicResponse`: Each predictor level has all observations in same class

**When Separation is Detected:**
- Consider using penalized regression (`lambda_ > 0`)
- Remove or combine problematic predictors
- Use Firth's bias-reduced logistic regression (if available)

---

## `check_count_sparsity`

Detect sparsity-induced separation in count data (Poisson/NegBin regression).

```python
ps.check_count_sparsity(
    y: Union[pl.Expr, str],      # Non-negative counts
    *x: Union[pl.Expr, str],
) -> pl.Expr
```

**Returns:** See [Separation Check Output](../outputs.md#separation-check-output)

**Example:**
```python
df.select(ps.check_count_sparsity("count", "x1", "x2"))
```

**When Sparsity is Detected:**
- Use regularization (`lambda_ > 0`)
- Consider zero-inflated models
- Check for sparse predictor-response combinations

---

## Diagnostic Workflow

```python
import polars as pl
import polars_statistics as ps

# 1. Check multicollinearity before fitting
cond = df.select(ps.condition_number("x1", "x2", "x3"))
print(cond)

# 2. For binary outcomes, check separation
sep = df.select(ps.check_binary_separation("y", "x1", "x2"))
if sep["has_separation"][0]:
    print("Warning: separation detected")
    # Use regularization
    model = df.select(ps.logistic("y", "x1", "x2", lambda_=1.0))
else:
    model = df.select(ps.logistic("y", "x1", "x2"))

# 3. For count data, check sparsity
sparse = df.select(ps.check_count_sparsity("count", "x1", "x2"))
```

---

## See Also

- [GLM Models](glm.md) - GLMs with regularization
- [Linear Models](linear.md) - Linear regression variants
