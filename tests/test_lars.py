"""Smoke tests for LARS (Least-Angle Regression) PyModel."""
import numpy as np
import pytest
from polars_statistics import LARS, OLS


class TestLARS:
    def test_fit_basic(self):
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.array([1.0, 2.0, -1.0]) + 0.1 * np.random.randn(50)
        model = LARS().fit(X, y)
        assert model.is_fitted()
        assert len(model.coefficients) == 3
        assert np.all(np.isfinite(model.coefficients))

    def test_alphas_non_empty(self):
        """REGR-06: alphas path must be non-empty after fit."""
        np.random.seed(0)
        X = np.random.randn(40, 2)
        y = X @ np.array([1.0, -1.0]) + 0.05 * np.random.randn(40)
        model = LARS().fit(X, y)
        assert len(model.alphas) > 0
        assert np.all(np.isfinite(model.alphas))

    def test_predict_shape(self):
        np.random.seed(1)
        X = np.random.randn(30, 2)
        y = X @ np.array([1.0, 2.0]) + 0.1 * np.random.randn(30)
        model = LARS().fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (30,)
        assert np.all(np.isfinite(preds))

    def test_lasso_method(self):
        np.random.seed(7)
        X = np.random.randn(50, 3)
        y = X @ np.array([1.0, 0.0, -1.0]) + 0.1 * np.random.randn(50)
        model = LARS(method="lasso").fit(X, y)
        assert model.is_fitted()
        assert len(model.alphas) > 0

    def test_not_fitted_raises(self):
        model = LARS()
        assert not model.is_fitted()
        with pytest.raises(Exception):
            _ = model.coefficients

    def test_not_fitted_alphas_raises(self):
        model = LARS()
        with pytest.raises(Exception):
            _ = model.alphas

    def test_intercept_present(self):
        np.random.seed(3)
        X = np.random.randn(40, 2)
        y = X @ np.array([1.0, 2.0]) + 5.0 + 0.1 * np.random.randn(40)
        model = LARS(fit_intercept=True).fit(X, y)
        assert model.intercept is not None
        assert np.isfinite(model.intercept)

    def test_recovers_known_coefficients(self):
        """LARS recovers known coefficients on a low-noise linear design.

        Ground truth: y = 1.0 * x1 + 2.0 * x2 - 1.0 * x3.  With tiny noise
        the full LARS path should reach the OLS solution and the final coefficients
        should be within 0.20 of the true values.
        """
        np.random.seed(15)
        n = 120
        X = np.random.randn(n, 3)
        true_beta = np.array([1.0, 2.0, -1.0])
        y = X @ true_beta + 0.05 * np.random.randn(n)
        model = LARS(fit_intercept=False).fit(X, y)
        np.testing.assert_allclose(model.coefficients, true_beta, atol=0.20,
                                   err_msg="LARS failed to recover known coefficients")

    def test_vs_sklearn_lars(self):
        """LARS coefficients are close to sklearn Lars on clean data.

        sklearn is NOT a declared dependency — guard with importorskip.
        """
        sklearn_linear = pytest.importorskip("sklearn.linear_model")
        np.random.seed(44)
        n = 100
        X = np.random.randn(n, 3)
        y = X @ np.array([1.0, -1.0, 0.5]) + 0.1 * np.random.randn(n)
        ps_model = LARS(fit_intercept=False).fit(X, y)
        sk_model = sklearn_linear.Lars(fit_intercept=False).fit(X, y)
        np.testing.assert_allclose(ps_model.coefficients, sk_model.coef_, atol=0.30,
                                   err_msg="LARS vs sklearn coefficient mismatch")


class TestLarsFitStatistics:
    """anofox-regression 0.5.21 #65: LARS.adj_r_squared/mse/rmse, R-validated.

    Issue #65 data (n=60), copied verbatim from
    ``~/.cargo/registry/src/*/anofox-regression-0.5.22/tests/issue65_lars_fit_stats.rs``.
    """

    @staticmethod
    def _data():
        n = 60
        t = np.arange(n, dtype=float)
        x1 = t / 10.0
        x2 = np.sin(t)
        X = np.column_stack([x1, x2])
        y = 2.0 + 1.5 * (t / 10.0) - 0.8 * np.sin(t) + 0.3 * np.cos(7.0 * t)
        return X, y

    def test_lar_matches_r_lm_summary(self):
        """Full LAR path == OLS on [x1, x2] -> R's summary(lm(y ~ x1 + x2))
        r.squared / adj.r.squared / sigma."""
        X, y = self._data()
        model = LARS(method="lar", fit_intercept=True).fit(X, y)
        np.testing.assert_allclose(model.r_squared, 0.993640840379574, atol=1e-9)
        np.testing.assert_allclose(model.adj_r_squared, 0.993417711971839, atol=1e-9)
        np.testing.assert_allclose(model.rmse, 0.216257642550628, atol=1e-9)
        np.testing.assert_allclose(model.mse, 0.216257642550628 ** 2, atol=1e-9)

    def test_lasso_lars_active_set_df(self):
        """LassoLars (alpha=0.05): rmse/adj_r_squared use df = active
        coefficients + intercept (not the full path's parameter count)."""
        X, y = self._data()
        model = LARS(method="lasso", alpha=0.05, fit_intercept=True).fit(X, y)

        residuals = y - model.predict(X)
        active = int(np.count_nonzero(model.coefficients))
        n = len(y)
        df_resid = n - active - 1
        rss = float(np.sum(residuals ** 2))

        assert model.rmse > 0.0
        assert model.adj_r_squared > 0.0
        np.testing.assert_allclose(model.rmse, np.sqrt(rss / df_resid), atol=1e-12)
        expected_adj = 1.0 - (1.0 - model.r_squared) * (n - 1) / df_resid
        np.testing.assert_allclose(model.adj_r_squared, expected_adj, atol=1e-12)

    def test_lar_no_intercept_matches_ols(self):
        """LAR without an intercept: rmse equals OLS(add_intercept=False)'s
        rmse on the same design (full path == OLS on the active set)."""
        X, y = self._data()
        lars_model = LARS(method="lar", fit_intercept=False).fit(X, y)
        ols_model = OLS(add_intercept=False).fit(X, y)
        np.testing.assert_allclose(lars_model.rmse, ols_model.rmse, atol=1e-8)

    def test_lars_fit_stat_getters_unfitted(self):
        """Unfitted LARS: the three getters raise RuntimeError naming LARS;
        to_dict() after fit contains them."""
        model = LARS()
        for attr in ("adj_r_squared", "mse", "rmse"):
            with pytest.raises(Exception) as exc_info:
                getattr(model, attr)
            assert "LARS" in str(exc_info.value)

        X, y = self._data()
        model.fit(X, y)
        result = model.to_dict()
        for key in ("adj_r_squared", "mse", "rmse"):
            assert key in result
