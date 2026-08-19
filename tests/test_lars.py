"""Smoke tests for LARS (Least-Angle Regression) PyModel."""
import numpy as np
import pytest
from polars_statistics import LARS


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
