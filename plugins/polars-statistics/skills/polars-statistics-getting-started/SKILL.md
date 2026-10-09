---
name: polars-statistics-getting-started
description: Get started with polars-statistics — install the package, understand the two API styles (expressions vs. PyModel classes), choose the right model with the decision matrix, and migrate from sklearn.
version: "0.8.0"
user-invocable: false
---

## Overview

`polars-statistics` is a Rust-backed statistical library for Python that exposes hypothesis
tests, ANOVA, regression, and related methods as native Polars expressions *and* as
sklearn-style PyModel classes. Every operation is validated against R.

**Two orthogonal API styles:**

- **Expression API** — integrate with `df.select(...)`, `df.group_by(...).agg(...)`, and
  `.over()` for lazy, per-group statistics inside a DataFrame pipeline.
- **PyModel API** — imperative `Model().fit(X, y)` classes for fine-grained inspection,
  prediction, and sklearn-style workflows.

Both styles share the same underlying Rust engine. Choose based on where the computation
lives in your workflow.

---

## When to use this skill

Route here when the user asks about:
- Installing `polars-statistics`
- Which import alias to use (`import polars_statistics as ps`)
- How the expression API differs from the PyModel API
- Which model or test to pick for a given task
- Migrating code from sklearn to polars-statistics
- General orientation before diving into a specific family

For deeper coverage of specific capability families, see **See also** at the bottom.

---

## API Reference

### Install

```bash
pip install polars-statistics
```

**Requirements:**
- Python 3.9+ (CPython; all platforms)
- Wheels available for: Linux x86\_64/aarch64, macOS x86\_64/ARM64, Windows x86\_64
- `polars` is a required dependency (installed automatically)

**Idiomatic imports:**

```python
import polars_statistics as ps   # conventional alias used in all examples
import polars as pl
```

**Version check:**

```python
print(ps.__version__)  # "0.8.0" once released (current build may still report "0.7.0"; see Gotchas)
```

---

### Two API Styles

#### Expression API

Best for: group-level statistics inside a DataFrame pipeline, lazy evaluation, `.over()`,
`group_by(...).agg(...)`.

```python
import polars as pl
import polars_statistics as ps
import numpy as np

rng = np.random.default_rng(0)
df = pl.DataFrame({
    "group": ["A"] * 30 + ["B"] * 30,
    "y":  rng.normal(1.0, 0.5, 60).tolist(),
    "x1": rng.normal(2.0, 0.3, 60).tolist(),
    "x2": rng.normal(0.5, 0.2, 60).tolist(),
})

# Whole-frame test (no group_by)
df.select(ps.ttest_ind("y", "x1"))

# Per-group test
df.group_by("group").agg(
    ps.ttest_ind("y", "x1").alias("t")
)

# Per-row predictions with .over() — window function style
df.with_columns(
    ps.ols_predict("y", "x1", "x2").over("group").alias("pred")
)
```

Expression results are **Polars Struct columns**. Unpack them with:

```python
import polars as pl
import polars_statistics as ps

df_ab = pl.DataFrame({
    "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "b": [2.0, 2.5, 3.5, 4.5, 5.5, 7.0],
})

# Access one field
df_ab.select(ps.ttest_ind("a", "b").alias("t"))["t"].struct.field("p_value")

# Unnest all fields using the results helper
result = df_ab.select(ps.ttest_ind("a", "b").alias("t"))
ps.unnest(result, "t")   # returns DataFrame with columns: statistic, p_value
```

#### PyModel API

Best for: imperative sklearn-style workflows, fine-grained inspection, prediction,
streaming, cross-validation, and anything operating on NumPy arrays.

```python
from polars_statistics import OLS
import polars as pl
import numpy as np

rng = np.random.default_rng(42)
df = pl.DataFrame({
    "x1": rng.normal(0, 1, 80).tolist(),
    "x2": rng.normal(0, 1, 80).tolist(),
})
# y = 1.5*x1 - 0.8*x2 + noise (simulates data.csv)
y_arr = df["x1"].to_numpy() * 1.5 - df["x2"].to_numpy() * 0.8 + rng.normal(0, 0.3, 80)
df = df.with_columns(pl.Series("y", y_arr))

X = df.select("x1", "x2").to_numpy()   # shape (n, p), float64
y = df["y"].to_numpy()                  # shape (n,)

m = OLS(add_intercept=True).fit(X, y)
print(m.r_squared)
print(m.coefficients)
print(m.summary())

yhat = m.predict(X)        # numpy array
score = m.score(X, y)      # R²
```

Every PyModel class shares this interface:
- `__init__(...)` — set hyperparameters
- `.fit(X, y)` → returns self (chainable); `X` is `(n, p)` float64 array, `y` is `(n,)`
- `.predict(X)` → `numpy.ndarray`
- `.score(X, y)` → `float` (R²)
- `.is_fitted()` → `bool` — always check before accessing fitted properties
- `.summary()` → `str` — human-readable coefficient table
- `.to_dict()` → `dict` — machine-readable result map

---

### Model Selection Matrix

| Task | Expression API | PyModel API | Notes |
|------|---------------|-------------|-------|
| Compare two group means | `ps.ttest_ind("a","b")` | `TTestInd` | `equal_var=False` (Welch's) by default |
| Paired samples | `ps.ttest_paired("x","y")` | `TTestPaired` | — |
| Non-parametric two groups | `ps.mann_whitney_u("a","b")` | `MannWhitneyU` | continuity correction on by default |
| Multiple groups (parametric) | `ps.one_way_anova("g1","g2","g3")` | — | wide format: one column per group |
| Multiple groups (non-parametric) | `ps.kruskal_wallis("g1","g2","g3")` | `KruskalWallis` | same wide-format requirement |
| Two-factor design | `ps.two_way_anova("y","fac_a","fac_b")` | — | auto-encodes string factors |
| Repeated measures | `ps.repeated_measures_anova("val","subj","cond")` | — | balanced design required |
| Normality check (n < 5000) | `ps.shapiro_wilk("x")` | `ShapiroWilk` | — |
| Normality check (n >= 20) | `ps.dagostino("x")` | `DAgostino` | — |
| Pearson/Spearman correlation | `ps.pearson("x","y")` / `ps.spearman("x","y")` | — | returns estimate + CI |
| Non-linear association | `ps.distance_cor("x","y")` | — | detects non-linear dependence |
| Equivalence testing | `ps.tost_t_test_two_sample("x","y")` | — | replaces significance-only testing |
| Distribution comparison | `ps.mmd_test("x","y")` / `ps.energy_distance("x","y")` | — | detects location + scale + shape shifts |
| Linear regression (pipeline) | `ps.ols("y","x1","x2")` | — | expression form for group_by |
| Linear regression (imperative) | — | `OLS(add_intercept=True)` | PyModel form |
| Regularized regression (L2) | `ps.ridge("y","x", lambda_=1.0)` | `Ridge(lambda_=1.0)` | `lambda_` controls strength |
| Regularized regression (L1) | `ps.lasso("y","x", lambda_=1.0)` | — | `ElasticNet(alpha=1.0)` under the hood |
| Elastic net | `ps.elastic_net("y","x", lambda_=1.0, alpha=0.5)` | `ElasticNet` | `alpha` = L1 ratio |
| Robust regression | `ps.huber("y","x")` / `ps.quantile("y","x", tau=0.5)` | `Huber`, `Quantile`, `TheilSen`, `RANSAC` | outlier-resistant; SEs via `Huber`/`RANSAC(compute_inference=True)` |
| Effect size for a group comparison | `ps.cohens_d("a","b")` / `ps.rank_biserial("a","b")` | — | see polars-statistics-hypothesis-tests |
| Bootstrap CI for a mean (incl. dependent data) | `ps.bootstrap_mean_ci("x", block_length=...)` | — | `block_length=0` IID, `>0` time-series; see polars-statistics-hypothesis-tests |
| Count data | `ps.poisson("y","x")` / `ps.negative_binomial("y","x")` | `Poisson`, `NegativeBinomial` | NB for overdispersion |
| Binary outcome | `ps.logistic("y","x")` | `Logistic`, `LogisticRegression` | `Logistic`=lower-level; `LogisticRegression`=sklearn-style |
| Flexible GLM (25 families) | `ps.alm("y","x", distribution="normal")` | `ALM` | Augmented Linear Model |
| Mixed effects | — | `GLMM()` | not available as expression |
| Tidy coefficient table | `ps.ols_summary("y","x")` | `.to_dict()` | `List[Struct]` ready for `.explode().unnest()` |
| Per-row predictions + intervals | `ps.ols_predict("y","x", interval="prediction")` | `.predict(X)` | expression returns struct with bounds |
| Online / streaming fit | — | `MomentAccumulator` + `OLS.fit_from_accumulator(acc)` | exact accumulation |
| R-style formula | `ps.ols_formula("y ~ x1 + x2 + x1:x2")` | — | handles `*`, `:`, `poly()`, `I()` |

---

### sklearn Migration

```python
# verify: skip -- sklearn migration reference table (illustrative comment-annotated import; sklearn not installed)
# sklearn import                       polars-statistics equivalent
# ─────────────────────────────────────────────────────────────────
from sklearn.linear_model import (
    LinearRegression                 # → OLS(add_intercept=True)
    Ridge                            # → Ridge(lambda_=alpha)
    Lasso                            # → ElasticNet(lambda_=alpha, alpha=1.0)
    ElasticNet                       # → ElasticNet(lambda_=alpha, alpha=l1_ratio)
    HuberRegressor                   # → Huber(epsilon=epsilon)
    QuantileRegressor                # → Quantile(tau=quantile)
    TheilSenRegressor                # → TheilSen()
    RANSACRegressor                  # → RANSAC()
    BayesianRidge                    # → BayesianRidge()
    ARDRegression                    # → ARD()
    Lars                             # → LARS(method="lar")
    LassoLars                        # → LARS(method="lasso")
    PLSRegression                    # → PLS(n_components=n_components)
    IsotonicRegression               # → Isotonic()
    PassiveAggressiveRegressor       # → PassiveAggressive()
    LogisticRegression               # → LogisticRegression(C=C, penalty=penalty)
)
```

**Key parameter renames:**

| sklearn param | polars-statistics param | Notes |
|---------------|------------------------|-------|
| `fit_intercept=True` | `add_intercept=True` | Required rename |
| `alpha=` (regularization) | `lambda_=` | Renamed for mathematical clarity |
| `l1_ratio=` | `alpha=` | In `ElasticNet` — L1 ratio |
| `normalize=True` | Not available | Preprocess manually or use `PLS(scale=True)` |
| `.coef_` | `.coefficients` | Attribute rename |
| `.intercept_` | `.intercept` | Attribute rename |
| `.score(X, y)` | `.score(X, y)` | Returns R² — same semantics |

---

## Worked Examples

### Example 1: Whole-frame t-test

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "control":   [4.2, 3.8, 5.1, 4.6, 3.9],
    "treatment": [5.5, 6.1, 5.8, 6.3, 5.9],
})

result = df.select(ps.ttest_ind("control", "treatment").alias("t"))
print(ps.unnest(result, "t"))
# shape: (1, 2) with columns: statistic, p_value
```

### Example 2: Per-group OLS in a pipeline

```python
import polars as pl
import polars_statistics as ps
import numpy as np

rng = np.random.default_rng(42)
df = pl.DataFrame({
    "region": ["N"] * 50 + ["S"] * 50,
    "sales":  rng.normal(100, 10, 100).tolist(),
    "price":  rng.normal(50, 5, 100).tolist(),
})

coefs = (
    df.group_by("region")
    .agg(ps.ols("sales", "price").alias("model"))
    .with_columns(pl.col("model").struct.field("r_squared").alias("r2"))
)
print(coefs.select("region", "r2"))
```

### Example 3: OLS PyModel — fit, inspect, predict

```python
from polars_statistics import OLS
import polars as pl
import numpy as np

rng = np.random.default_rng(0)
X = rng.standard_normal((100, 2))
y = X[:, 0] * 2.0 + X[:, 1] * 0.5 + rng.standard_normal(100) * 0.1

m = OLS(add_intercept=True).fit(X, y)
if m.is_fitted():
    print(m.summary())          # coefficient table
    yhat = m.predict(X)         # numpy array, shape (n,)
    print("R²:", round(m.r_squared, 4))
    print("AIC:", round(m.aic, 2))
```

### Example 4: Formula API

```python
import polars as pl
import polars_statistics as ps
import numpy as np

rng = np.random.default_rng(1)
df = pl.DataFrame({
    "y":  rng.standard_normal(50).tolist(),
    "x1": rng.standard_normal(50).tolist(),
    "x2": rng.standard_normal(50).tolist(),
})

# Main effects
df.select(ps.ols_formula("y ~ x1 + x2"))

# Full crossing (expands to x1 + x2 + x1:x2)
df.select(ps.ols_formula("y ~ x1 * x2"))

# Polynomial
df.select(ps.ols_formula("y ~ poly(x1, 2)"))
```

### Example 5: sklearn-style Ridge migration

```python
# verify: skip -- sklearn migration example (sklearn not installed in test environment; polars-statistics Ridge API shown in other examples)
import numpy as np

# Before (sklearn)
from sklearn.linear_model import Ridge as SkRidge
m_sk = SkRidge(alpha=0.5, fit_intercept=True).fit(X, y)

# After (polars-statistics) — exact drop-in
from polars_statistics import Ridge
m = Ridge(lambda_=0.5, add_intercept=True).fit(X, y)
print(m.coefficients)   # was m_sk.coef_
print(m.intercept)      # was m_sk.intercept_
print(m.r_squared)      # replaces m_sk.score(X, y)
```

---

## Gotchas / Notes

**Use `add_intercept=`, not `with_intercept=`**
The `with_intercept` parameter is deprecated as of v0.7.0 and emits a `FutureWarning`.
All new code should use `add_intercept=True/False`.

**Use `lambda_=` for regularization strength, not `alpha=`**
In `Ridge`, `ElasticNet`, `Lasso`, and their expression counterparts, the regularization
strength parameter is `lambda_`. Passing `alpha=` for this purpose will raise
`TypeError: unexpected keyword argument 'alpha'`. Note that `alpha` *does* exist in
`ElasticNet` as the L1 ratio — a different concept.

**`normalize=` does not exist**
There is no `normalize=` parameter anywhere in the library. Standardize inputs before
fitting if needed, or use `PLS(scale=True)` for the PLS model.

**`one_way_anova` and `kruskal_wallis` require wide-format input**
Pass one column per ANOVA group. Do not use a long-format value+group pair.
Correct: `df.select(ps.one_way_anova("ctrl", "treat_a", "treat_b"))`.
Wrong: `df.group_by("group").agg(ps.one_way_anova("value"))`.

**`icc`: pass each rater as a separate positional argument**
The current API takes each rater column as a positional argument (rows = subjects).
Passing a single matrix column produces wrong results without raising an error.
Correct: `ps.icc("rater1", "rater2", "rater3", icc_type="icc2")`.

**Add `.alias("name")` before calling `ps.unnest()`**
Expression results are unnamed by default. If you aggregate multiple expressions in one
`group_by().agg()`, each needs an alias so `unnest` knows which struct column to expand.
`ps.unnest(result)` auto-detects the struct column only when exactly one struct column
exists in the DataFrame.

**Accessing unfitted properties raises `PyRuntimeError`**
Calling `.coefficients`, `.r_squared`, etc. on an unfitted model raises
`PyRuntimeError("Model not fitted")`. Always call `.is_fitted()` first, or wrap in
`try/except PyRuntimeError`.

**Version string vs. frontmatter version**
The installed wheel reports `ps.__version__ == "0.7.0"`. This skill's frontmatter pins
`version: "0.8.0"` — the target milestone release. The API surface is identical.

---

## See also

- `polars-statistics-hypothesis-tests` — t-tests, Mann-Whitney, Kruskal-Wallis, normality
  tests, correlation, chi-square, TOST equivalence tests, MMD/energy distance
- `polars-statistics-anova` — one-way, two-way, repeated-measures ANOVA; ICC
- `polars-statistics-regression` — 40+ PyModel classes, regression expressions, diagnostics,
  R-style formula parser
- `polars-statistics-results` — unnest/struct\_to\_dict helpers, `to_dict()`/`summary()` on
  PyModels, error handling, `.pyi` type stubs
