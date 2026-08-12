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
