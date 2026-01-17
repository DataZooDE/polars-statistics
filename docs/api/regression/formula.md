# Formula Syntax

Formula-based regression using R-style syntax. Supports polynomial and interaction effects that are computed per-group with `group_by`/`over`.

> **Note:** Unlike R, the intercept is controlled via the `with_intercept` parameter, not in the formula. R's `y ~ x - 1` or `y ~ 0 + x` syntax for removing the intercept is not supported. Use `with_intercept=False` instead.

## Supported Formula Syntax

| Pattern | Example | Expansion |
|---------|---------|-----------|
| Main effects | `y ~ x1 + x2` | Two variables |
| Interaction only | `y ~ x1:x2` | Product term |
| Full crossing | `y ~ x1 * x2` | `x1 + x2 + x1:x2` |
| Polynomial (centered) | `y ~ poly(x, 2)` | Centered x, x^2 |
| Polynomial (raw) | `y ~ poly(x, 2, raw=True)` | x, x^2 |
| Explicit transform | `y ~ I(x^2)` | x squared |

---

## Formula Functions

All regression models have a `*_formula` variant:

### Linear Models

```python
ps.ols_formula(formula: str, with_intercept: bool = True) -> pl.Expr
ps.ridge_formula(formula: str, lambda_: float = 1.0, with_intercept: bool = True) -> pl.Expr
ps.elastic_net_formula(formula: str, lambda_: float = 1.0, alpha: float = 0.5, with_intercept: bool = True) -> pl.Expr
ps.wls_formula(formula: str, weights: Union[pl.Expr, str], with_intercept: bool = True) -> pl.Expr
ps.rls_formula(formula: str, forgetting_factor: float = 0.99, with_intercept: bool = True) -> pl.Expr
ps.bls_formula(formula: str, lower_bound: float | None = None, upper_bound: float | None = None, with_intercept: bool = True) -> pl.Expr
ps.nnls_formula(formula: str, with_intercept: bool = True) -> pl.Expr
```

### GLM Models

```python
ps.logistic_formula(formula: str, with_intercept: bool = True) -> pl.Expr
ps.poisson_formula(formula: str, with_intercept: bool = True) -> pl.Expr
ps.negative_binomial_formula(formula: str, theta: float | None = None, with_intercept: bool = True) -> pl.Expr
ps.tweedie_formula(formula: str, var_power: float = 1.5, with_intercept: bool = True) -> pl.Expr
ps.probit_formula(formula: str, with_intercept: bool = True) -> pl.Expr
ps.cloglog_formula(formula: str, with_intercept: bool = True) -> pl.Expr
```

### ALM

```python
ps.alm_formula(formula: str, distribution: str = "normal", with_intercept: bool = True) -> pl.Expr
```

---

## Examples

### Polynomial Regression

```python
# Quadratic regression per group
df.group_by("group").agg(
    ps.ols_formula("y ~ poly(x, 2)").alias("model")
)
```

### Interaction Effects

```python
# Full interaction model
df.group_by("group").agg(
    ps.ols_formula("y ~ x1 * x2").alias("model")
)

# Equivalent to:
df.group_by("group").agg(
    ps.ols_formula("y ~ x1 + x2 + x1:x2").alias("model")
)
```

### Explicit Transformations

```python
# Square term
df.group_by("group").agg(
    ps.ols_formula("y ~ x + I(x^2)").alias("model")
)
```

---

## Summary and Prediction Formula Variants

Summary functions:
```python
ps.ols_formula_summary(formula, with_intercept=True)
ps.ridge_formula_summary(formula, lambda_=1.0, with_intercept=True)
# ... etc.
```

Prediction functions:
```python
ps.ols_formula_predict(formula, interval=None, level=0.95, with_intercept=True)
ps.ridge_formula_predict(formula, lambda_=1.0, interval=None, level=0.95, with_intercept=True)
# ... etc.
```

---

## See Also

- [Linear Models](linear.md) - Column-based interface
- [GLM Models](glm.md) - GLM column-based interface
