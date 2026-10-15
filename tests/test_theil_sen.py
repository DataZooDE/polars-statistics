"""Smoke tests for TheilSen (Theil-Sen robust regression) PyModel."""

import numpy as np
import pytest
from polars_statistics import TheilSen


class TestTheilSen:
    def test_fit_basic(self):
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X @ np.array([1.0, 2.0]) + np.random.randn(50) * 0.1
        model = TheilSen().fit(X, y)
        assert model.is_fitted()
        assert len(model.coefficients) == 2
        assert np.all(np.isfinite(model.coefficients))

    def test_predict_shape(self):
        np.random.seed(0)
        X = np.random.randn(30, 2)
        y = X @ np.array([1.0, -1.0]) + 0.05 * np.random.randn(30)
        model = TheilSen().fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (30,)
        assert np.all(np.isfinite(preds))

    def test_not_fitted_raises(self):
        model = TheilSen()
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.coefficients

    def test_intercept_finite(self):
        np.random.seed(1)
        X = np.random.randn(40, 2)
        y = 3.0 + X @ np.array([0.5, -0.5]) + 0.1 * np.random.randn(40)
        model = TheilSen(with_intercept=True).fit(X, y)
        assert model.intercept is not None
        assert np.isfinite(model.intercept)

    def test_r_squared_range(self):
        np.random.seed(7)
        X = np.random.randn(60, 3)
        y = X @ np.array([1.0, 2.0, -1.0]) + 0.05 * np.random.randn(60)
        model = TheilSen().fit(X, y)
        assert 0.0 <= model.r_squared <= 1.0

    def test_residuals_shape(self):
        np.random.seed(99)
        X = np.random.randn(25, 2)
        y = X @ np.array([1.0, 1.0]) + 0.1 * np.random.randn(25)
        model = TheilSen().fit(X, y)
        assert model.residuals.shape == (25,)

    def test_robust_to_outliers(self):
        """Theil-Sen should be robust to a small fraction of severe outliers."""
        np.random.seed(10)
        n = 100
        X = np.random.randn(n, 1)
        true_slope = 2.0
        y = 1.0 + true_slope * X[:, 0] + 0.1 * np.random.randn(n)
        # Inject 10 severe outliers
        y[:10] += 50.0
        model = TheilSen().fit(X, y)
        # Recovered slope should be close to the true value
        assert abs(model.coefficients[0] - true_slope) < 0.5

    def test_no_intercept(self):
        np.random.seed(5)
        X = np.random.randn(30, 2)
        y = X @ np.array([1.5, -0.5]) + 0.1 * np.random.randn(30)
        model = TheilSen(with_intercept=False).fit(X, y)
        assert model.is_fitted()
        assert model.intercept is None
        assert len(model.coefficients) == 2

    def test_n_observations(self):
        np.random.seed(3)
        X = np.random.randn(45, 2)
        y = X @ np.array([1.0, 1.0]) + 0.1 * np.random.randn(45)
        model = TheilSen().fit(X, y)
        assert model.n_observations == 45

    def test_recovers_known_slope_clean(self):
        """Theil-Sen recovers known slope on clean data within tight tolerance.

        Ground truth: y = 1.0 + 3.0 * x (single predictor, no intercept in X).
        With low noise the recovered coefficient should be within 0.15 of 3.0.
        """
        np.random.seed(21)
        n = 150
        x = np.random.randn(n, 1)
        y = 1.0 + 3.0 * x[:, 0] + 0.05 * np.random.randn(n)
        model = TheilSen(with_intercept=True).fit(x, y)
        assert abs(model.coefficients[0] - 3.0) < 0.15, (
            f"TheilSen slope {model.coefficients[0]:.4f} far from true 3.0"
        )

    def test_vs_sklearn_on_clean_data(self):
        """TheilSen coefficients are close to sklearn TheilSenRegressor on clean data.

        sklearn is NOT a declared dependency — guard with importorskip.
        """
        sklearn_linear = pytest.importorskip("sklearn.linear_model")
        np.random.seed(55)
        n = 80
        X = np.random.randn(n, 2)
        true_beta = np.array([1.5, -0.5])
        y = X @ true_beta + 0.1 * np.random.randn(n)

        ps_model = TheilSen(with_intercept=False).fit(X, y)
        sk_model = sklearn_linear.TheilSenRegressor(fit_intercept=False).fit(X, y)

        np.testing.assert_allclose(
            ps_model.coefficients, sk_model.coef_, atol=0.25,
            err_msg="TheilSen vs sklearn coefficient mismatch"
        )

    def _pairwise_median_data(self):
        x = np.arange(20, dtype=float)
        y = 3 + 0.5 * x
        y[3] += 20.0
        y[11] += 20.0
        return x, y

    def _pairwise_median_slope_intercept(self, x, y):
        slopes = []
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                if x[i] != x[j]:
                    slopes.append((y[j] - y[i]) / (x[j] - x[i]))
        slope = float(np.median(slopes))
        intercept = float(np.median(y - slope * x))
        return slope, intercept

    def test_univariate_pairwise_is_median_of_pairwise_slopes(self):
        x, y = self._pairwise_median_data()
        exp_slope, exp_intercept = self._pairwise_median_slope_intercept(x, y)

        default_model = TheilSen().fit(x.reshape(-1, 1), y)
        explicit_model = TheilSen(univariate_pairwise=True).fit(x.reshape(-1, 1), y)

        for model in (default_model, explicit_model):
            assert model.coefficients[0] == pytest.approx(exp_slope, abs=1e-12)
            assert model.intercept == pytest.approx(exp_intercept, abs=1e-12)

    def test_univariate_pairwise_false_matches_sklearn_spatial_median(self):
        sklearn_linear = pytest.importorskip("sklearn.linear_model")
        x, y = self._pairwise_median_data()
        exp_slope, _ = self._pairwise_median_slope_intercept(x, y)

        model = TheilSen(univariate_pairwise=False).fit(x.reshape(-1, 1), y)
        assert model.coefficients[0] != pytest.approx(exp_slope, abs=1e-6)

        sk_model = sklearn_linear.TheilSenRegressor(random_state=0).fit(
            x.reshape(-1, 1), y
        )
        assert model.coefficients[0] == pytest.approx(sk_model.coef_[0], abs=1e-6)
        assert model.intercept == pytest.approx(sk_model.intercept_, abs=1e-6)
