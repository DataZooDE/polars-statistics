# GLMs, Smoothers & Streaming Models

Not every response variable is a well-behaved Gaussian. Insurance claims and
waiting times are strictly positive and right-skewed; measurements nested within
patients or stores are correlated; some relationships are smoothly non-linear; and
some datasets are simply too big to hold in memory. This cookbook covers four
model classes that handle those cases:

| Model | Best for |
|-------|----------|
| [`Gamma`](#gamma-glm-positive-skewed-responses) | Strictly-positive, right-skewed responses (claims, durations) |
| [`GLMM`](#glmm-random-effects-across-groups) | Data grouped by subject/store/site (random intercepts) |
| [`PSpline`](#pspline-smooth-non-linear-trends) | Smooth non-linear trends without picking a polynomial degree |
| [`MomentAccumulator`](#momentaccumulator-streaming-sufficient-statistics) | Out-of-core / streaming least squares |

Each fitted model exposes `.summary()`, `.to_dict()`, and a readable `__repr__`,
and the shipped type stubs give you full IDE autocomplete.

!!! tip "Runnable script"
    The complete, runnable version of this page is
    [`examples/07_glm_smoothers_streaming.py`](https://github.com/DataZooDE/polars-statistics/blob/main/examples/07_glm_smoothers_streaming.py).

## Setup

```python
import numpy as np
from polars_statistics import GLMM, Gamma, MomentAccumulator, PSpline, Ridge

rng = np.random.default_rng(7)
```

## Gamma GLM: positive, skewed responses

The Gamma GLM models a strictly-positive response whose variance grows with its
mean — the classic shape of insurance claim sizes, hospital stays, or rainfall.
With the default log link, the coefficients are **multiplicative**: `exp(coef)` is
the factor by which the mean changes per unit of the predictor.

```python
n = 300
x = rng.uniform(0, 3, size=(n, 1))
mu = np.exp(0.5 + 0.8 * x.ravel())      # mean grows multiplicatively with x
y = rng.gamma(shape=2.0, scale=mu / 2.0)  # Gamma noise keeps y > 0

gamma = Gamma().fit(x, y)
print(gamma.summary())
print(f"multiplicative effect per unit x: {np.exp(gamma.coefficients[0]):.3f}x")
```

```text
Gamma Results
=============
...
coefficients:            array([0.80221158])
intercept:               0.4044
aic:                     668.0365
converged:               true

multiplicative effect per unit x: 2.230x
```

**Reading the output:** the fitted slope of `0.80` recovers the true `0.8`, and
`exp(0.80) ≈ 2.23` means each unit increase in `x` multiplies the expected response
by ~2.23. Check `converged` before trusting the fit, and use `aic`/`bic` to compare
against other GLM families. `predict` returns the mean response; `predict_eta`
returns the linear predictor (before the inverse link).

!!! note "Other GLM families"
    For counts use [`Poisson`](../api/regression/glm.md) or `NegativeBinomial`;
    for binary outcomes use `LogisticRegression`; for compound Poisson-Gamma data
    use `Tweedie`. They share the same `fit`/`predict`/`.summary()` surface.

## GLMM: random effects across groups

When observations are grouped — repeated measures on the same subject, students
within schools, sales within stores — the groups have their own baselines. A
generalized linear **mixed** model adds a random intercept per group so the fixed
effects (the slopes you care about) aren't confounded by those baselines.

`GLMM` is constructed through a **family factory** — `GLMM.gaussian()`,
`GLMM.poisson()`, or `GLMM.binomial()` — not a bare constructor. `fit(X, y, group)`
takes a group-id array as its third argument.

```python
n_groups, per_group = 8, 25
group_ids = np.repeat(np.arange(n_groups), per_group)
group_offsets = rng.normal(0, 2.0, size=n_groups)  # true random intercepts

x = rng.normal(0, 1, size=(n_groups * per_group, 1))
y = 1.0 + 3.0 * x.ravel() + group_offsets[group_ids] + rng.normal(0, 0.5, len(x))

glmm = GLMM.gaussian().fit(x, y, group_ids.tolist())
fe = glmm.fixed_effects   # [intercept, slope, ...]
print(f"fixed effects [intercept, slope]: {np.round(fe, 3).tolist()}")
print(f"random-intercept variance ratio theta: {glmm.theta:.3f}")
print(f"residual sigma: {glmm.sigma:.3f}, converged: {glmm.converged}")
```

```text
fixed effects [intercept, slope]: [2.023, 3.001]
random-intercept variance ratio theta: 4.278
residual sigma: 0.480, converged: True
```

**Reading the output:** `fixed_effects[0]` is the intercept and `fixed_effects[1:]`
are the slopes — here the slope of `3.001` recovers the true `3.0` even though each
group had a different baseline. `theta` is the ratio of random-intercept variance to
residual variance (large `theta` means the groups differ a lot); `sigma` is the
residual SD. Use `predict_fixed(X)` for population-level predictions. For designs
with two crossing grouping factors, `fit_crossed(X, y, [g1, g2])` populates
`factors()` with per-factor variance components.

## PSpline: smooth non-linear trends

A penalized spline (P-spline) fits a flexible smooth curve without you having to
choose a polynomial degree or knot placement. A roughness penalty controls the
wiggliness; with `lambda_=None` the model selects that penalty automatically by
generalized cross-validation (GCV).

```python
n = 200
x = np.sort(rng.uniform(0, 2 * np.pi, size=n)).reshape(-1, 1)
y = np.sin(x.ravel()) + rng.normal(0, 0.15, n)

pspline = PSpline(penalty_order=2, lambda_=None).fit(x, y)
print(f"effective degrees of freedom (edf): {pspline.edf:.2f}")
print(f"R^2: {pspline.r_squared:.3f}")

grid = np.linspace(0, 2 * np.pi, 5).reshape(-1, 1)
print(f"fitted curve: {np.round(pspline.predict(grid), 3).tolist()}")
```

```text
effective degrees of freedom (edf): 8.80
R^2: 0.965
fitted curve: [0.077, 0.992, -0.05, -1.003, -0.076]
```

**Reading the output:** `edf` (effective degrees of freedom) measures how wiggly the
fitted curve is — `1.0` would be a straight line, higher values mean more curvature.
Here `edf ≈ 8.8` reflects the sine wave's shape. `sigma2` is the residual variance.
`predict` evaluates the smooth on any new inputs, so you can render the fitted curve
on a fine grid.

## MomentAccumulator: streaming sufficient statistics

For a linear model you don't actually need to keep all the rows — you only need the
**sufficient statistics** `XᵀX` and `Xᵀy`. `MomentAccumulator` builds those
incrementally, so you can stream a dataset that never fits in memory, and even
merge accumulators computed on separate shards (e.g. one per file or worker) before
solving once at the end.

```python
n, p = 5000, 3
X = rng.normal(0, 1, size=(n, p))
beta = np.array([1.0, -2.0, 0.5])
y = X @ beta + 3.0 + rng.normal(0, 0.5, n)

# Build two accumulators on disjoint shards, then merge.
acc_a = MomentAccumulator(n_features=p)
acc_b = MomentAccumulator(n_features=p)
for i in range(n // 2):
    acc_a.push_row(X[i], y[i])
for i in range(n // 2, n):
    acc_b.push_row(X[i], y[i])
acc_a.merge(acc_b)
print(f"merged rows: {acc_a.n}")

# Solve once from the accumulated statistics.
ridge = Ridge(lambda_=0.0).fit_from_accumulator(acc_a)
print(f"coefficients: {np.round(ridge.coefficients, 3).tolist()}  (true {beta.tolist()})")
print(f"intercept: {ridge.intercept:.3f}  (true 3.0)")
```

```text
merged rows: 5000
coefficients: [1.01, -1.997, 0.507]  (true [1.0, -2.0, 0.5])
intercept: 2.997  (true 3.0)
```

**Reading the output:** the coefficients solved from the streamed statistics match a
batch fit to numerical precision — `merge` is exact, not approximate, because summing
`XᵀX` blocks is associative. `push_row(x_row, y)` adds one observation; `n` counts
the rows seen; `xtx`, `xty`, `sum_x`, `sum_y` expose the raw statistics if you need
them. Pair this with `Ridge.fit_from_accumulator` (`lambda_=0.0` gives ordinary
least squares) for out-of-core regression. For an *online-updating* estimator that
also predicts as it learns, see [`PassiveAggressive`](robust-regression.md#passiveaggressive-online-learning).

## See Also

- [Model Selection](../model-selection.md) — decision matrix across all regressors
- [Robust & Sparse Regression](robust-regression.md) — TheilSen, RANSAC, BayesianRidge, ARD, LARS, PassiveAggressive
- [Generalized Linear Models](glm-models.md) — Poisson, NegativeBinomial, Tweedie, Logistic as Polars expressions
- [GLM Model Classes](../api/classes/glm.md) — full class reference
