#!/usr/bin/env python3
"""GLM, Smoother and Streaming Model Example

Demonstrates:

- Gamma            - GLM for strictly-positive, right-skewed responses (log link)
- GLMM             - generalized linear mixed model (random intercepts by group)
- PSpline          - penalized B-spline smoother for non-linear trends
- MomentAccumulator - streaming sufficient statistics for out-of-core fitting

Each model exposes ``.summary()`` and ``.to_dict()`` for ergonomic inspection.
"""

import numpy as np
from polars_statistics import GLMM, Gamma, MomentAccumulator, PSpline, Ridge

rng = np.random.default_rng(7)


# =============================================================================
# 1. Gamma GLM - positive, right-skewed responses (e.g. insurance claims)
# =============================================================================

print("=" * 60)
print("1. Gamma GLM (log link) for positive skewed data")
print("=" * 60)

n = 300
x = rng.uniform(0, 3, size=(n, 1))
# Mean grows multiplicatively with x; Gamma noise keeps y strictly positive.
mu = np.exp(0.5 + 0.8 * x.ravel())
shape = 2.0
y = rng.gamma(shape=shape, scale=mu / shape)

gamma = Gamma().fit(x, y)
print(gamma.summary())
print(f"converged: {gamma.converged}")
# Coefficients are on the log scale: exp(coef) is the multiplicative effect on the mean.
print(f"multiplicative effect per unit x: {np.exp(gamma.coefficients[0]):.3f}x")
print()


# =============================================================================
# 2. GLMM - random intercepts across groups (repeated measures)
# =============================================================================

print("=" * 60)
print("2. GLMM with random intercepts by group")
print("=" * 60)

n_groups = 8
per_group = 25
group_ids = np.repeat(np.arange(n_groups), per_group)
group_offsets = rng.normal(0, 2.0, size=n_groups)  # true random intercepts

x = rng.normal(0, 1, size=(n_groups * per_group, 1))
y = 1.0 + 3.0 * x.ravel() + group_offsets[group_ids] + rng.normal(0, 0.5, n_groups * per_group)

# Gaussian family with a random intercept per group. GLMM is built through a
# family factory (gaussian / poisson / binomial), never a bare constructor.
glmm = GLMM.gaussian().fit(x, y, group_ids.tolist())
# fixed_effects is [intercept, slope, ...]; the slope on x is element 1.
fe = glmm.fixed_effects
print(f"fixed effects [intercept, slope]: {np.round(fe, 3).tolist()}  (true [1.0, 3.0])")
print(f"random-intercept variance ratio theta: {glmm.theta:.3f}")
print(f"residual sigma:        {glmm.sigma:.3f}")
print(f"n_groups:              {glmm.n_groups}")
print(f"converged:             {glmm.converged} in {glmm.iterations} iterations")
print()


# =============================================================================
# 3. PSpline - penalized-spline smoother for non-linear trends
# =============================================================================

print("=" * 60)
print("3. PSpline smoother (non-linear trend)")
print("=" * 60)

n = 200
x = np.sort(rng.uniform(0, 2 * np.pi, size=n)).reshape(-1, 1)
y = np.sin(x.ravel()) + rng.normal(0, 0.15, n)

# lambda_=None lets the model select the smoothing penalty by GCV.
pspline = PSpline(penalty_order=2, lambda_=None).fit(x, y)
print(f"effective degrees of freedom (edf): {pspline.edf:.2f}")
print(f"residual variance sigma2:           {pspline.sigma2:.4f}")
print(f"R^2:                                {pspline.r_squared:.3f}")

# Predict on a fine grid to obtain the smooth curve.
grid = np.linspace(0, 2 * np.pi, 5).reshape(-1, 1)
smooth = pspline.predict(grid)
print(f"fitted curve at 5 points: {np.round(smooth, 3).tolist()}")
print()


# =============================================================================
# 4. MomentAccumulator - streaming / out-of-core sufficient statistics
# =============================================================================

print("=" * 60)
print("4. MomentAccumulator (streaming fit)")
print("=" * 60)

n, p = 5000, 3
X = rng.normal(0, 1, size=(n, p))
beta = np.array([1.0, -2.0, 0.5])
y = X @ beta + 3.0 + rng.normal(0, 0.5, n)

# Accumulate X'X and X'y one chunk at a time - never holding all rows at once.
acc = MomentAccumulator(n_features=p)
chunk = 500
for start in range(0, n, chunk):
    for i in range(start, min(start + chunk, n)):
        acc.push_row(X[i], y[i])

print(f"rows accumulated: {acc.n}")
print(f"n_features:       {acc.n_features}")

# Two accumulators built on disjoint shards can be merged, then solved once.
acc_a = MomentAccumulator(n_features=p)
acc_b = MomentAccumulator(n_features=p)
for i in range(n // 2):
    acc_a.push_row(X[i], y[i])
for i in range(n // 2, n):
    acc_b.push_row(X[i], y[i])
acc_a.merge(acc_b)
print(f"merged rows:      {acc_a.n}")

# Ridge can be fit directly from the accumulated sufficient statistics.
ridge = Ridge(lambda_=0.0).fit_from_accumulator(acc_a)
print(f"streamed coefficients: {np.round(ridge.coefficients, 3).tolist()}")
print(f"true coefficients:     {beta.tolist()}")
print(f"intercept: {ridge.intercept:.3f} (true 3.0)")
print()

print("Done!")
