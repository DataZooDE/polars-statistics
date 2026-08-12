"""Smoke tests for RANSAC (Random Sample Consensus) robust regression PyModel."""

import numpy as np
import pytest

from polars_statistics import RANSAC


class TestRANSAC:
    def test_fit_basic(self):
        np.random.seed(42)
        X = np.random.randn(60, 2)
        y = X @ np.array([1.0, 2.0]) + np.random.randn(60) * 0.1
        model = RANSAC().fit(X, y)
        assert model.is_fitted()
        assert len(model.coefficients) == 2
        assert np.all(np.isfinite(model.coefficients))

    def test_predict_shape(self):
        np.random.seed(0)
        X = np.random.randn(40, 2)
        y = X @ np.array([1.0, -1.0]) + 0.1 * np.random.randn(40)
        model = RANSAC().fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (40,)
        assert np.all(np.isfinite(preds))

    def test_not_fitted_raises(self):
        model = RANSAC()
        with pytest.raises(Exception):
            _ = model.coefficients

    def test_inlier_mask_length(self):
        """inlier_mask must have the same length as the number of observations."""
        np.random.seed(1)
        n = 50
        X = np.random.randn(n, 2)
        y = X @ np.array([1.0, -1.0]) + 0.1 * np.random.randn(n)
        model = RANSAC().fit(X, y)
        mask = model.inlier_mask
        assert len(mask) == n

    def test_inlier_mask_dtype(self):
        """inlier_mask must be a boolean array."""
        np.random.seed(2)
        X = np.random.randn(50, 2)
        y = X @ np.array([1.0, 2.0]) + 0.1 * np.random.randn(50)
        model = RANSAC().fit(X, y)
        mask = model.inlier_mask
        assert mask.dtype == bool

    def test_n_inliers_consistent_with_mask(self):
        np.random.seed(3)
        X = np.random.randn(60, 2)
        y = X @ np.array([0.5, 1.5]) + 0.1 * np.random.randn(60)
        model = RANSAC().fit(X, y)
        assert model.n_inliers == int(model.inlier_mask.sum())

    def test_n_trials_positive(self):
        np.random.seed(4)
        X = np.random.randn(40, 2)
        y = X @ np.array([1.0, 1.0]) + 0.1 * np.random.randn(40)
        model = RANSAC().fit(X, y)
        assert model.n_trials >= 1

    def test_residual_threshold_positive(self):
        np.random.seed(5)
        X = np.random.randn(40, 2)
        y = X @ np.array([1.0, 1.0]) + 0.1 * np.random.randn(40)
        model = RANSAC().fit(X, y)
        assert model.residual_threshold > 0.0

    def test_robust_to_outliers(self):
        """RANSAC should ignore severe outliers."""
        np.random.seed(11)
        n = 80
        X = np.random.randn(n, 1)
        true_slope = 3.0
        y = 1.0 + true_slope * X[:, 0] + 0.05 * np.random.randn(n)
        # Inject 10 severe outliers
        y[:10] += 50.0
        model = RANSAC().fit(X, y)
        assert abs(model.coefficients[0] - true_slope) < 0.5

    def test_custom_residual_threshold(self):
        np.random.seed(6)
        X = np.random.randn(50, 2)
        y = X @ np.array([1.0, 2.0]) + 0.1 * np.random.randn(50)
        # Explicit threshold — should fit cleanly on well-behaved data
        model = RANSAC(residual_threshold=1.0).fit(X, y)
        assert model.is_fitted()
        assert len(model.inlier_mask) == 50

    def test_n_observations(self):
        np.random.seed(9)
        n = 55
        X = np.random.randn(n, 2)
        y = X @ np.array([1.0, -1.0]) + 0.1 * np.random.randn(n)
        model = RANSAC().fit(X, y)
        assert model.n_observations == n

    def test_recovers_known_slope_with_outliers(self):
        """RANSAC recovers the true slope even with 20% outliers.

        Ground truth: y = 2.5 * x + noise.  20 of 100 observations are
        contaminated by ±30 spikes.  RANSAC should recover the slope within 0.3.
        """
        np.random.seed(42)
        n = 100
        x = np.random.randn(n, 1)
        true_slope = 2.5
        y = true_slope * x[:, 0] + 0.1 * np.random.randn(n)
        # Plant 20 severe outliers
        y[:20] += np.random.choice([-30.0, 30.0], size=20)
        model = RANSAC().fit(x, y)
        assert abs(model.coefficients[0] - true_slope) < 0.3, (
            f"RANSAC slope {model.coefficients[0]:.4f} far from true {true_slope}"
        )

    def test_vs_sklearn_on_clean_data(self):
        """RANSAC coefficients are close to sklearn RANSACRegressor on clean data.

        sklearn is NOT a declared dependency — guard with importorskip.
        """
        sklearn_linear = pytest.importorskip("sklearn.linear_model")
        np.random.seed(77)
        n = 80
        X = np.random.randn(n, 1)
        true_slope = 2.0
        y = 1.0 + true_slope * X[:, 0] + 0.1 * np.random.randn(n)

        ps_model = RANSAC().fit(X, y)
        sk_model = sklearn_linear.RANSACRegressor(random_state=77).fit(X, y)

        # RANSAC is stochastic — allow generous tolerance
        assert abs(ps_model.coefficients[0] - float(sk_model.estimator_.coef_[0])) < 0.5, (
            f"RANSAC vs sklearn slope mismatch: "
            f"ps={ps_model.coefficients[0]:.4f}, sk={sk_model.estimator_.coef_[0]:.4f}"
        )
