"""Smoke tests for GammaRegressor PyModel."""

import numpy as np
import pytest

from polars_statistics import Gamma


class TestGamma:
    def test_fit_basic(self):
        """Gamma().fit(X, y) returns is_fitted()==True and finite coefficients."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        beta = np.array([1.0, 2.0])
        # Gamma requires strictly positive response: use exp(linear predictor)
        y = np.exp(X @ beta + 0.1 * np.random.randn(50))
        model = Gamma().fit(X, y)
        assert model.is_fitted()
        assert len(model.coefficients) == 2
        assert np.all(np.isfinite(model.coefficients))

    def test_predict_shape(self):
        """predict(X) returns array of shape (n,) with all-finite values."""
        np.random.seed(0)
        X = np.random.randn(30, 2)
        y = np.exp(X @ np.array([1.0, -1.0]) + 0.05 * np.random.randn(30))
        model = Gamma().fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (30,)
        assert np.all(np.isfinite(preds))

    def test_not_fitted_raises(self):
        """Accessing .coefficients before fit raises an exception."""
        model = Gamma()
        with pytest.raises(Exception):
            _ = model.coefficients

    def test_converged(self):
        """.converged is True on a well-conditioned Gamma fit."""
        np.random.seed(7)
        X = np.random.randn(50, 2)
        y = np.exp(X @ np.array([0.5, -0.5]) + 0.1 * np.random.randn(50))
        model = Gamma().fit(X, y)
        assert model.converged is True

    def test_predict_eta(self):
        """predict_eta(X) returns the linear predictor on the link scale."""
        np.random.seed(3)
        X = np.random.randn(40, 2)
        y = np.exp(X @ np.array([1.0, 0.5]) + 0.1 * np.random.randn(40))
        model = Gamma().fit(X, y)
        eta = model.predict_eta(X)
        mu = model.predict(X)
        assert eta.shape == (40,)
        assert np.all(np.isfinite(eta))
        # For Gamma with log link: mu = exp(eta)
        np.testing.assert_allclose(np.exp(eta), mu, rtol=1e-6)

    def test_intercept_getter(self):
        """intercept getter returns a float when with_intercept=True."""
        np.random.seed(11)
        X = np.random.randn(50, 2)
        y = np.exp(X @ np.array([1.0, 2.0]) + 0.1 * np.random.randn(50))
        model = Gamma(with_intercept=True).fit(X, y)
        intercept = model.intercept
        assert intercept is not None
        assert np.isfinite(intercept)

    def test_aic_bic_finite(self):
        """AIC and BIC are finite after a successful fit."""
        np.random.seed(5)
        X = np.random.randn(50, 2)
        y = np.exp(X @ np.array([0.3, -0.7]) + 0.05 * np.random.randn(50))
        model = Gamma().fit(X, y)
        assert model.aic is not None
        assert model.bic is not None
        assert np.isfinite(model.aic)
        assert np.isfinite(model.bic)

    def test_constructor_defaults(self):
        """Gamma() constructs with default parameters without error."""
        model = Gamma()
        assert not model.is_fitted()

    def test_constructor_custom(self):
        """Gamma accepts custom hyperparameters."""
        model = Gamma(with_intercept=False, max_iter=50, tol=1e-6, lambda_=0.01)
        assert not model.is_fitted()

    def test_predict_before_fit_raises(self):
        """predict() before fit raises an exception."""
        model = Gamma()
        X = np.ones((10, 2))
        with pytest.raises(Exception):
            model.predict(X)
