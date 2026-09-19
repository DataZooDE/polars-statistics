# Coming from scikit-learn

If you know scikit-learn, you already know most of polars-statistics. The model
classes follow the same `fit` / `predict` / `score` contract, take the same NumPy
`X` / `y` arrays, and expose comparable getters. This page maps common sklearn
workflows to their polars-statistics equivalents and highlights what you gain by
switching: **R-validated statistics**, a **native Polars expression API** for
per-group fitting, and **shipped type stubs** for full IDE autocomplete.

## The shared contract

Every regressor class supports the sklearn-style loop:

```python
from polars_statistics import OLS

model = OLS().fit(X_train, y_train)   # X: 2-D array, y: 1-D array
preds = model.predict(X_test)         # -> numpy array
r2    = model.score(X_test, y_test)   # -> R^2 for regressors
```

`score` returns **R²** for regressors and **accuracy** for classifiers, matching
scikit-learn's conventions. On top of that, every model adds ergonomic result
access that sklearn lacks:

```python
model.to_dict()    # every statistic as a plain dict
print(model.summary())   # readable text summary
model                    # informative __repr__
```

## Class-to-class map

| scikit-learn | polars-statistics | Notes |
|--------------|-------------------|-------|
| `LinearRegression` | `OLS` | Adds full inference: std errors, t-stats, p-values, CIs |
| `Ridge` | `Ridge` | `Ridge(lambda_=...)` (sklearn's `alpha`) |
| `Lasso` | `ElasticNet` / `ps.lasso` expr | Elastic net with L1 mix, or the Lasso expression |
| `ElasticNet` | `ElasticNet` | |
| `LogisticRegression` | `LogisticRegression` | `predict`, `predict_proba`, `decision_function`, `score` (accuracy) |
| `TheilSenRegressor` | `TheilSen` | Robust median-of-slopes |
| `RANSACRegressor` | `RANSAC` | `inlier_mask`, `n_inliers` |
| `BayesianRidge` | `BayesianRidge` | `alpha_`, `lambda_`, `sigma_diag` |
| `ARDRegression` | `ARD` | Sparse coefficients via per-weight precision |
| `Lars` / `LassoLars` | `LARS` | `LARS(method="lasso")` for the Lasso path |
| `PassiveAggressiveRegressor` | `PassiveAggressive` | `partial_fit` for online updates |
| `HuberRegressor` | `Huber` | Robust to mild outliers |
| `QuantileRegressor` | `Quantile` | Model any quantile |
| `PoissonRegressor` | `Poisson` | GLM, log link |
| `GammaRegressor` | `Gamma` | GLM, log link |
| `TweedieRegressor` | `Tweedie` | Compound Poisson-Gamma |
| `IsotonicRegression` | `Isotonic` | Monotone fit |
| `PLSRegression` | `PLS` | Partial least squares |

## Worked comparisons

### Linear regression

=== "scikit-learn"

    ```python
    from sklearn.linear_model import LinearRegression

    model = LinearRegression().fit(X, y)
    print(model.coef_, model.intercept_)
    print(model.score(X, y))          # R^2
    ```

=== "polars-statistics"

    ```python
    from polars_statistics import OLS

    model = OLS().fit(X, y)
    print(model.coefficients, model.intercept)
    print(model.score(X, y))          # R^2
    print(model.summary())            # + std errors, t-stats, p-values
    ```

The main naming differences: `coef_` → `coefficients`, `intercept_` → `intercept`.
You additionally get inference statistics for free.

### Ridge (note the penalty name)

=== "scikit-learn"

    ```python
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=1.0).fit(X, y)
    ```

=== "polars-statistics"

    ```python
    from polars_statistics import Ridge

    model = Ridge(lambda_=1.0).fit(X, y)   # sklearn's `alpha` is `lambda_`
    ```

### Logistic regression (classifier)

=== "scikit-learn"

    ```python
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression().fit(X, y)
    clf.predict(X_new)
    clf.predict_proba(X_new)
    clf.score(X_test, y_test)         # accuracy
    ```

=== "polars-statistics"

    ```python
    from polars_statistics import LogisticRegression

    clf = LogisticRegression().fit(X, y)
    clf.predict(X_new)
    clf.predict_proba(X_new)
    clf.decision_function(X_new)
    clf.score(X_test, y_test)         # accuracy
    ```

### Robust regression

=== "scikit-learn"

    ```python
    from sklearn.linear_model import RANSACRegressor

    r = RANSACRegressor(random_state=0).fit(X, y)
    r.inlier_mask_          # trailing underscore
    ```

=== "polars-statistics"

    ```python
    from polars_statistics import RANSAC

    r = RANSAC(random_state=0).fit(X, y)
    r.inlier_mask           # no trailing underscore
    r.n_inliers
    ```

See the [Robust & Sparse Regression cookbook](examples/robust-regression.md) for
`TheilSen`, `RANSAC`, `BayesianRidge`, `ARD`, `LARS`, and `PassiveAggressive` in
depth.

## Naming conventions

polars-statistics drops scikit-learn's trailing-underscore convention for fitted
attributes and uses plain names:

| scikit-learn | polars-statistics |
|--------------|-------------------|
| `coef_` | `coefficients` |
| `intercept_` | `intercept` |
| `n_iter_` | `n_iter` |
| `inlier_mask_` | `inlier_mask` |
| `alpha` (Ridge penalty) | `lambda_` |

## What you gain beyond sklearn

### 1. Native Polars expression API + per-group fitting

The class API is the drop-in. But the real leverage is fitting **one model per
group** as a lazy Polars expression — something sklearn can't express directly:

```python
import polars as pl
import polars_statistics as ps

# One OLS per region, computed in parallel across groups.
coefs = (
    df.group_by("region")
      .agg(ps.ols("sales", "spend", "price").alias("model"))
)
```

This scales to millions of groups and stays inside your DataFrame pipeline. See the
[Group-wise Analysis cookbook](examples/group-analysis.md).

### 2. R-validated statistics

Where the backing crates supply reference values, results are validated against R
(e.g. `stats::lm`, `irr::icc`). You get inference — standard errors, t-statistics,
p-values, confidence intervals — that scikit-learn's estimators mostly omit.

### 3. Typed API

The package ships `py.typed` and full `.pyi` stubs, so editors and type checkers
resolve every constructor, method, and getter with accurate signatures — no more
guessing whether it's `coef_` or `coefficients`.

## See Also

- [Model Selection](model-selection.md) — pick the right estimator
- [Robust & Sparse Regression](examples/robust-regression.md)
- [GLMs, Smoothers & Streaming](examples/glm-smoothers-streaming.md)
- [Migration Guide](migration.md) — `icc` and `with_intercept` changes
