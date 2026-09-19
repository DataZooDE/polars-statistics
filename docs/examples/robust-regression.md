# Robust & Sparse Regression

Real data is messy. Sensors glitch, humans fat-finger data entry, and a handful of
extreme rows can drag an ordinary least-squares line far away from the trend the
other 95% of your data agrees on. Other times you have *too many* candidate
features and need the model to tell you which ones actually matter. And sometimes
the data never stops arriving, so you need to update the model row by row.

This cookbook covers the robust and sparse regressor classes for those scenarios:

| Model | Best for |
|-------|----------|
| [`TheilSen`](#theil-sen-median-of-slopes) | Outliers in the response; want an unbiased slope |
| [`RANSAC`](#ransac-consensus-fitting) | Outliers *and* you want them flagged (inlier mask) |
| [`BayesianRidge`](#bayesianridge-regularization-with-learned-precision) | Regularized linear fit with automatically-tuned strength |
| [`ARD`](#ard-automatic-relevance-determination) | Sparse coefficients: prune irrelevant features automatically |
| [`LARS`](#lars-least-angle-regression) | Exact feature-selection path (Lasso variant) |
| [`PassiveAggressive`](#passiveaggressive-online-learning) | Online / streaming updates, one row at a time |

All of these follow the same sklearn-style contract — `fit(X, y)`, `predict(X)`,
`score(X, y)` (returns R²) — and every fitted model exposes `.summary()`,
`.to_dict()`, and a readable `__repr__`. The typed stubs shipped with the package
mean your IDE autocompletes every method and getter.

!!! tip "Runnable script"
    The complete, runnable version of this page is
    [`examples/06_robust_regression.py`](https://github.com/DataZooDE/polars-statistics/blob/main/examples/06_robust_regression.py).

## Setup

These classes take NumPy arrays (a 2-D feature matrix `X` and a 1-D target `y`),
which pair naturally with `df.to_numpy()` when your data lives in a Polars frame.

```python
import numpy as np
from polars_statistics import ARD, LARS, RANSAC, BayesianRidge, PassiveAggressive, TheilSen

rng = np.random.default_rng(42)
```

## Theil-Sen: median of slopes

Theil-Sen estimates the slope as the median of the slopes between all point pairs.
Because the median ignores extreme values, up to ~29% of the data can be corrupted
before the estimate breaks down — a far cry from OLS, where a single leverage point
can flip the sign of a coefficient.

```python
n = 100
x = np.linspace(0, 10, n).reshape(-1, 1)
y = 2.0 * x.ravel() + 1.0 + rng.normal(0, 0.5, n)

# Corrupt 15% of the points with large vertical outliers.
outlier_idx = rng.choice(n, size=15, replace=False)
y[outlier_idx] += rng.normal(0, 30, size=15)

theilsen = TheilSen(random_state=0).fit(x, y)
print(f"slope={theilsen.coefficients[0]:.3f} intercept={theilsen.intercept:.3f}")
```

```text
slope=2.007 intercept=1.028
```

The true relationship is `y = 2x + 1`. Despite 15% of the data being corrupted,
Theil-Sen recovers the slope almost exactly. A plain OLS fit on the same data would
be visibly biased toward the outliers.

**Reading the output:** `coefficients` holds one slope per feature; `intercept` is
the offset; `r_squared`, `rmse`, and `residuals` describe the fit quality on the
data as given (note that with outliers present, R² measures fit to the *corrupted*
data, so a robust model can legitimately show a lower R² than a fragile one that
chased the outliers).

## RANSAC: consensus fitting

RANSAC (RANdom SAmple Consensus) repeatedly fits on small random subsets, keeps
the fit that has the largest set of agreeing points ("inliers"), and refits on that
consensus set. Its distinguishing feature is the **inlier mask** — a boolean array
telling you exactly which rows it trusted.

```python
ransac = RANSAC(random_state=0).fit(x, y)
print(f"slope={ransac.coefficients[0]:.3f} intercept={ransac.intercept:.3f}")
print(f"inliers: {ransac.n_inliers}/{ransac.n_observations}")

# Which rows were rejected as outliers?
flagged = np.where(~ransac.inlier_mask)[0]
print(f"flagged {len(flagged)} rows as outliers")
```

```text
slope=2.007 intercept=1.038
inliers: 87/100
flagged 13 rows as outliers
```

Use `inlier_mask` to filter your data for downstream analysis, or to build an
outlier-detection pipeline. `residual_threshold` reports the cutoff RANSAC chose
(by default the median absolute deviation of the residuals). Tune `min_samples`,
`residual_threshold`, and `max_trials` if the defaults are too permissive or strict.

## BayesianRidge: regularization with learned precision

Bayesian Ridge places priors on both the noise and the coefficients and estimates
their precisions (`alpha_` for the noise, `lambda_` for the weights) from the data.
This means you get ridge-style regularization **without** having to grid-search the
penalty — the model tunes it for you.

```python
n, p = 150, 5
X = rng.normal(0, 1, size=(n, p))
beta = np.array([1.5, 0.0, -2.0, 0.0, 0.5])
y = X @ beta + rng.normal(0, 1.0, n)

br = BayesianRidge().fit(X, y)
print(br.summary())
print(f"noise precision alpha_={br.alpha_:.3f}  weight precision lambda_={br.lambda_:.3f}")
```

`alpha_` is the estimated precision (1/variance) of the observation noise, so a
larger `alpha_` means a cleaner signal. `lambda_` is the precision of the weight
prior — larger values pull coefficients harder toward zero. `sigma_diag` gives the
per-coefficient posterior variance, a built-in measure of coefficient uncertainty.

## ARD: automatic relevance determination

ARD is Bayesian Ridge's sparse cousin: it learns a **separate** precision per
coefficient, and features that don't help get their precision driven to infinity,
pushing the coefficient to essentially zero. This is automatic feature selection
with no penalty to tune.

```python
n, p = 200, 10
X = rng.normal(0, 1, size=(n, p))
true_coef = np.zeros(p)
true_coef[[0, 3, 7]] = [5.0, -3.0, 2.0]   # only 3 of 10 features matter
y = X @ true_coef + rng.normal(0, 0.5, n)

ard = ARD().fit(X, y)
for j, c in enumerate(ard.coefficients):
    marker = "  <- relevant" if abs(c) > 0.5 else ""
    print(f"x{j}: {c:8.3f}{marker}")
```

```text
x0:    4.974  <- relevant
x1:    0.020
x2:    0.000
x3:   -3.017  <- relevant
x4:    0.000
x5:    0.000
x6:    0.004
x7:    1.974  <- relevant
x8:    0.000
x9:    0.000
```

ARD correctly identified features 0, 3, and 7 and drove the other seven to zero.
The `lambdas` getter exposes the learned per-coefficient precisions if you want to
inspect the pruning directly.

## LARS: least-angle regression

LARS builds the solution one feature at a time, adding whichever feature is most
correlated with the current residual. With `method="lasso"` it traces the exact
Lasso path, giving genuinely sparse coefficients; `n_nonzero_coefs` caps how many
features enter.

```python
lars = LARS(method="lasso", n_nonzero_coefs=3).fit(X, y)
nonzero = np.where(np.abs(lars.coefficients) > 1e-8)[0]
print(f"selected features: {nonzero.tolist()}")
```

```text
selected features: [0, 3, 7]
```

The `alphas` getter returns the regularization values at each step of the path,
useful for plotting a coefficient trajectory.

## PassiveAggressive: online learning

When data streams in continuously, refitting from scratch on every new row is
wasteful. PassiveAggressive updates the model incrementally: it stays "passive"
when a prediction is within tolerance and reacts "aggressively" when it isn't.

```python
n, p = 300, 3
X = rng.normal(0, 1, size=(n, p))
beta = np.array([2.0, -1.0, 0.5])
y = X @ beta + rng.normal(0, 0.3, n)

# Warm-start on a batch, then stream the rest one row at a time.
pa = PassiveAggressive(random_state=0).fit(X[:50], y[:50])
for i in range(50, n):
    pa.partial_fit(X[i], y[i])

print(f"coefficients: {np.round(pa.coefficients, 3).tolist()}")
print(f"R^2 on full data: {pa.score(X, y):.3f}")
```

```text
coefficients: [2.051, -0.898, 0.51]
R^2 on full data: 0.979
```

`partial_fit(x_row, y_value)` applies a single online update. `predict_from_state`
scores the current weights without a full refit. This is the model to reach for in
low-latency or memory-constrained settings where the whole dataset never sits in
memory at once — see also [`MomentAccumulator`](glm-smoothers-streaming.md#momentaccumulator-streaming-sufficient-statistics)
for streaming *exact* least-squares.

## Choosing among them

| If you… | Use |
|---------|-----|
| have vertical outliers and want an unbiased slope | `TheilSen` |
| want outliers *flagged* for downstream filtering | `RANSAC` |
| want a well-regularized fit without tuning a penalty | `BayesianRidge` |
| have many features and want the model to prune them | `ARD` |
| want an interpretable feature-selection path | `LARS` (Lasso) |
| are learning from a live stream | `PassiveAggressive` |

For the full comparison against OLS, Ridge, Lasso, and the GLMs, see the
[Model Selection matrix](../model-selection.md).

## See Also

- [Model Selection](../model-selection.md) — decision matrix across all regressors
- [sklearn Migration](../sklearn-migration.md) — porting `TheilSenRegressor`, `RANSACRegressor`, `BayesianRidge`, `ARDRegression`
- [GLM, Smoothers & Streaming](glm-smoothers-streaming.md) — Gamma, GLMM, PSpline, MomentAccumulator
- [Regularized & Specialized Models](regularized-regression.md) — Lasso, Ridge, Elastic Net as Polars expressions
