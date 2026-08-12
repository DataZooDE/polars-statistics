"""Smoke tests for GammaRegressor PyModel."""

import numpy as np
import pytest

from polars_statistics import Gamma


def _gamma_design(seed: int = 42, n: int = 100):
    """Return (X, y, true_beta) for a well-conditioned Gamma GLM with log link.

    y ~ Gamma(shape=3, scale=exp(X @ beta) / 3)  so E[y] = exp(X @ beta).
    True intercept = 0.5, true slope for x1 = 0.8.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    eta = 0.5 + 0.8 * x1
    mu = np.exp(eta)
    y = rng.gamma(shape=3.0, scale=mu / 3.0)
    X = x1.reshape(-1, 1)
    return X, y, np.array([0.8])  # true slope (intercept handled separately)


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

    def test_coefficients_vs_statsmodels(self, require_statsmodels):
        """Gamma GLM coefficients match statsmodels GLM(family=Gamma) within loose tolerance.

        Uses a 1-predictor Gamma design (log link, n=200) and asserts the slope
        recovered by polars-statistics is within 0.15 of statsmodels' MLE estimate.
        Guarded by require_statsmodels so the test skips when statsmodels is absent.
        """
        sm = require_statsmodels
        X, y, _ = _gamma_design(seed=7, n=200)
        ps_model = Gamma(with_intercept=True).fit(X, y)
        ps_slope = ps_model.coefficients[0]

        # statsmodels reference
        import statsmodels.api as sm_api

        X_sm = sm_api.add_constant(X)
        sm_model = sm_api.GLM(y, X_sm, family=sm_api.families.Gamma(link=sm_api.families.links.Log())).fit()
        sm_slope = float(sm_model.params[1])

        assert abs(ps_slope - sm_slope) < 0.15, (
            f"Gamma slope mismatch: polars-statistics={ps_slope:.4f}, "
            f"statsmodels={sm_slope:.4f} (tolerance 0.15)"
        )
        # Also check intercept direction
        ps_intercept = ps_model.intercept
        sm_intercept = float(sm_model.params[0])
        if ps_intercept is not None:
            assert abs(ps_intercept - sm_intercept) < 0.30, (
                f"Gamma intercept mismatch: ps={ps_intercept:.4f}, sm={sm_intercept:.4f}"
            )
