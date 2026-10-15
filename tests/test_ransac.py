"""Smoke tests for RANSAC (Random Sample Consensus) robust regression PyModel."""

import numpy as np
import pytest
from polars_statistics import OLS, RANSAC


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
        """anofox-regression 0.5.14 #35: n_observations now reports the inlier count
        (the sample size the inference statistics were fit on), not the total row
        count. residuals still covers every row regardless of inlier/outlier status."""
        np.random.seed(9)
        n = 55
        X = np.random.randn(n, 2)
        y = X @ np.array([1.0, -1.0]) + 0.1 * np.random.randn(n)
        model = RANSAC().fit(X, y)
        assert model.n_observations == model.n_inliers
        assert len(model.residuals) == n

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


class TestRANSACInference:
    """R reference: summary(lm(y[inl] ~ x[inl])); anofox-regression
    tests/issue35_ransac.rs."""

    def _data(self):
        i = np.arange(1, 21, dtype=float)
        base = 1 + 2 * i + (((i * 7) % 5) - 2) * 0.3
        y = base.copy()
        for outlier_i in (4, 11, 17):
            y[int(outlier_i) - 1] += 25.0
        return i.reshape(-1, 1), y

    def test_default_inference_is_none_but_fit_stats_finite(self):
        x, y = self._data()
        m = RANSAC(residual_threshold=2.0, random_state=42).fit(x, y)
        assert m.std_errors is None
        assert m.t_statistics is None
        assert m.p_values is None
        assert m.conf_interval_lower is None
        assert m.conf_interval_upper is None
        assert m.intercept_std_error is None
        assert m.intercept_t_statistic is None
        assert m.intercept_p_value is None
        assert m.intercept_conf_interval is None
        for attr in ("adj_r_squared", "mse", "rmse", "f_statistic", "f_pvalue", "aic", "bic", "log_likelihood"):
            assert np.isfinite(getattr(m, attr)), attr

    def test_compute_inference_matches_r_lm_on_inliers(self):
        x, y = self._data()
        m = RANSAC(residual_threshold=2.0, random_state=42, compute_inference=True).fit(
            x, y
        )
        assert m.n_inliers == 17
        assert m.rmse == pytest.approx(0.44666920869378707, abs=1e-8)
        assert m.adj_r_squared == pytest.approx(0.99860732006386344, abs=1e-8)
        assert m.std_errors[0] == pytest.approx(0.018543168475545296, abs=1e-8)
        assert m.mse == pytest.approx(m.rmse**2, rel=1e-12)
        for attr in ("f_statistic", "f_pvalue", "aic", "bic", "log_likelihood"):
            assert np.isfinite(getattr(m, attr)), attr

    def test_inlier_fit_returns_consistent_ols(self):
        x, y = self._data()
        m = RANSAC(residual_threshold=2.0, random_state=42, compute_inference=True).fit(
            x, y
        )
        inlier = m.inlier_fit()
        assert isinstance(inlier, OLS)
        assert inlier.is_fitted()
        assert inlier.coefficients[0] == pytest.approx(m.coefficients[0], abs=1e-12)
        assert inlier.intercept == pytest.approx(m.intercept, abs=1e-12)
        assert inlier.n_observations == 17
        assert len(inlier.residuals) == 17
        assert inlier.std_errors[0] == pytest.approx(m.std_errors[0], abs=1e-12)

    def test_inlier_fit_unfitted_raises(self):
        m = RANSAC()
        with pytest.raises(Exception):
            m.inlier_fit()

    def test_constant_response_fits_cleanly(self):
        """sklearn semantics (0.5.14): MAD threshold 0, |r| <= threshold inlier
        test, so a constant response fits with every row as an inlier."""
        n = 12
        x = np.column_stack(
            [np.arange(n, dtype=float), np.array([(i * 5) % 7 for i in range(n)], dtype=float)]
        )
        y = np.full(n, 5.0)
        m = RANSAC(random_state=42).fit(x, y)
        assert m.n_inliers == n
        assert m.intercept == pytest.approx(5.0, abs=1e-12)
        assert abs(m.coefficients[0]) < 1e-12
        assert abs(m.coefficients[1]) < 1e-12
