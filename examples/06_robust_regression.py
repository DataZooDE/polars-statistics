#!/usr/bin/env python3
"""Robust and Sparse Regression Example

Demonstrates the robust / sparse regressor classes for real-world scenarios:

- TheilSen         - median-of-slopes estimator, resists outliers
- RANSAC           - consensus fitting, flags inliers vs outliers
- BayesianRidge    - regularized linear model with learned precision
- ARD              - automatic relevance determination (sparse coefficients)
- LARS / Lasso     - least-angle regression path for feature selection
- PassiveAggressive - online / streaming linear updates

Each model follows the sklearn-style ``fit`` / ``predict`` / ``score`` contract
and exposes ``.summary()`` and ``.to_dict()`` for ergonomic result inspection.
"""

import numpy as np
from polars_statistics import ARD, LARS, RANSAC, BayesianRidge, PassiveAggressive, TheilSen

rng = np.random.default_rng(42)


# =============================================================================
# Scenario 1: Outliers in the response - TheilSen vs RANSAC
# =============================================================================

print("=" * 60)
print("1. Robust regression with outliers (TheilSen & RANSAC)")
print("=" * 60)

n = 100
x = np.linspace(0, 10, n).reshape(-1, 1)
y = 2.0 * x.ravel() + 1.0 + rng.normal(0, 0.5, n)

# Corrupt 15% of the points with large vertical outliers.
outlier_idx = rng.choice(n, size=15, replace=False)
y[outlier_idx] += rng.normal(0, 30, size=15)

theilsen = TheilSen(random_state=0).fit(x, y)
ransac = RANSAC(random_state=0).fit(x, y)

# The true slope is 2.0 and intercept 1.0. A plain least-squares fit is dragged
# toward the outliers; robust estimators recover the underlying line.
print(f"TheilSen slope={theilsen.coefficients[0]:.3f} intercept={theilsen.intercept:.3f}")
print(f"  R^2 (on clean signal): {theilsen.r_squared:.3f}")
print(f"RANSAC   slope={ransac.coefficients[0]:.3f} intercept={ransac.intercept:.3f}")
print(f"  inliers: {ransac.n_inliers}/{ransac.n_observations}, threshold={ransac.residual_threshold:.2f}")

# The inlier mask tells you which rows RANSAC trusted - useful for downstream filtering.
flagged_outliers = np.where(~ransac.inlier_mask)[0]
print(f"  RANSAC flagged {len(flagged_outliers)} rows as outliers")
print()


# =============================================================================
# Scenario 2: Sparse features - ARD and LARS/Lasso feature selection
# =============================================================================

print("=" * 60)
print("2. Sparse feature selection (ARD & LARS)")
print("=" * 60)

# 10 features, but only features 0, 3 and 7 actually drive the response.
n, p = 200, 10
X = rng.normal(0, 1, size=(n, p))
true_coef = np.zeros(p)
true_coef[[0, 3, 7]] = [5.0, -3.0, 2.0]
y = X @ true_coef + rng.normal(0, 0.5, n)

ard = ARD().fit(X, y)
print("ARD coefficients (near-zero = pruned):")
for j, c in enumerate(ard.coefficients):
    marker = "  <- relevant" if abs(c) > 0.5 else ""
    print(f"  x{j}: {c:8.3f}{marker}")
print(f"ARD R^2: {ard.r_squared:.3f}")
print()

# LARS with the Lasso variant produces an exact-sparse coefficient path.
lars = LARS(method="lasso", n_nonzero_coefs=3).fit(X, y)
nonzero = np.where(np.abs(lars.coefficients) > 1e-8)[0]
print(f"LARS (lasso) selected features: {nonzero.tolist()}")
print(f"LARS R^2: {lars.r_squared:.3f}")
print()


# =============================================================================
# Scenario 3: Bayesian Ridge - regularization with uncertainty
# =============================================================================

print("=" * 60)
print("3. BayesianRidge (regularization + learned precision)")
print("=" * 60)

n, p = 150, 5
X = rng.normal(0, 1, size=(n, p))
beta = np.array([1.5, 0.0, -2.0, 0.0, 0.5])
y = X @ beta + rng.normal(0, 1.0, n)

br = BayesianRidge().fit(X, y)
print(br.summary())
# alpha_ is the estimated noise precision (1/variance); lambda_ the weight precision.
print(f"noise precision alpha_={br.alpha_:.3f}  weight precision lambda_={br.lambda_:.3f}")
print(f"R^2: {br.r_squared:.3f}")
print()


# =============================================================================
# Scenario 4: Online updates - PassiveAggressive streaming
# =============================================================================

print("=" * 60)
print("4. PassiveAggressive online learning (partial_fit)")
print("=" * 60)

n, p = 300, 3
X = rng.normal(0, 1, size=(n, p))
beta = np.array([2.0, -1.0, 0.5])
y = X @ beta + rng.normal(0, 0.3, n)

# Warm-start on the first batch, then stream the rest one row at a time.
pa = PassiveAggressive(random_state=0).fit(X[:50], y[:50])
for i in range(50, n):
    pa.partial_fit(X[i], y[i])

print(f"PassiveAggressive coefficients: {np.round(pa.coefficients, 3).tolist()}")
print(f"true coefficients:              {beta.tolist()}")
print(f"iterations: {pa.n_iter}")

# score() returns R^2 for regressors, mirroring sklearn.
print(f"R^2 on full data: {pa.score(X, y):.3f}")
print()

print("Done!")
