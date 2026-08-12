"""Smoke tests for GlmmRegressor PyModel (REGR-01)."""

import numpy as np
import pytest
from polars_statistics import GLMM


class TestGLMM:
    def test_gaussian_fit_basic(self):
        """GLMM.gaussian().fit(X, y, group) returns is_fitted==True and finite fixed_effects."""
        rng = np.random.default_rng(42)
        n = 60
        X = rng.standard_normal((n, 2))
        group = np.repeat(np.arange(6), 10).astype(np.int64)
        y = X @ np.array([1.0, -0.5]) + rng.standard_normal(n) * 0.3
        model = GLMM.gaussian().fit(X, y, group.tolist())
        assert model.is_fitted()
        fe = model.fixed_effects
        assert len(fe) >= 2
        assert np.all(np.isfinite(fe))

    def test_fit_crossed(self):
        """GLMM.gaussian().fit_crossed returns factors() with >=2 dicts each having n_levels/sd/blups."""
        rng = np.random.default_rng(7)
        n = 80
        X = rng.standard_normal((n, 2))
        y = X @ np.array([1.0, 0.5]) + rng.standard_normal(n) * 0.2
        g1 = np.repeat(np.arange(4), 20).astype(np.int64)
        g2 = np.tile(np.arange(4), 20).astype(np.int64)
        model = GLMM.gaussian().fit_crossed(X, y, [g1.tolist(), g2.tolist()])
        assert model.is_fitted()
        fs = model.factors()
        assert len(fs) >= 2
        for factor_dict in fs:
            assert "n_levels" in factor_dict
            assert "sd" in factor_dict
            assert "blups" in factor_dict
            assert np.isfinite(factor_dict["sd"])

    def test_not_fitted_raises(self):
        """Reading fixed_effects before fit raises RuntimeError."""
        model = GLMM.gaussian()
        with pytest.raises(Exception):
            _ = model.fixed_effects

    def test_predict_fixed(self):
        """predict_fixed returns marginal predictions of shape (n,)."""
        rng = np.random.default_rng(0)
        n = 50
        X = rng.standard_normal((n, 2))
        group = np.repeat(np.arange(5), 10).astype(np.int64)
        y = X @ np.array([2.0, -1.0]) + rng.standard_normal(n) * 0.5
        model = GLMM.gaussian().fit(X, y, group.tolist())
        preds = model.predict_fixed(X)
        assert preds.shape == (n,)
        assert np.all(np.isfinite(preds))

    def test_getters(self):
        """FittedGlmm getters return sensible values."""
        rng = np.random.default_rng(1)
        n = 60
        X = rng.standard_normal((n, 2))
        group = np.repeat(np.arange(6), 10).astype(np.int64)
        y = X @ np.array([1.0, -0.5]) + rng.standard_normal(n) * 0.3
        model = GLMM.gaussian().fit(X, y, group.tolist())
        assert isinstance(model.converged, bool)
        assert model.n_groups == 6
        assert model.iterations >= 1
        assert np.isfinite(model.deviance)
        assert np.isfinite(model.log_likelihood)
        assert np.isfinite(model.sigma)
        assert np.isfinite(model.theta)

    def test_poisson_factory(self):
        """GLMM.poisson() factory can be constructed."""
        model = GLMM.poisson()
        assert not model.is_fitted()

    def test_binomial_factory(self):
        """GLMM.binomial() factory can be constructed."""
        model = GLMM.binomial()
        assert not model.is_fitted()

    def test_fixed_effect_near_truth(self):
        """GLMM Gaussian random-intercept: fixed-effect estimate is near the true slope.

        Design: y = 1.5 * x + u_g + noise, where u_g ~ N(0, 0.5^2) is a
        group-specific random intercept across 10 groups of 10 observations.
        The fixed-effect slope should be recovered within 0.30 of 1.5.
        """
        rng = np.random.default_rng(42)
        n_groups, n_per = 10, 10
        n = n_groups * n_per
        X = rng.standard_normal((n, 1))
        group = np.repeat(np.arange(n_groups), n_per).astype(np.int64)
        u = rng.normal(0, 0.5, n_groups)  # random intercepts
        true_slope = 1.5
        y = true_slope * X[:, 0] + u[group] + rng.standard_normal(n) * 0.3
        model = GLMM.gaussian().fit(X, y, group.tolist())
        fe = model.fixed_effects
        assert abs(fe[0] - true_slope) < 0.30, (
            f"GLMM fixed-effect slope {fe[0]:.4f} far from true {true_slope}"
        )
        # Factor variance must be positive (random intercept is non-zero)
        fs = model.factors()
        assert len(fs) >= 1
        assert fs[0]["sd"] > 0.0, "Random intercept SD must be positive"
        assert np.isfinite(fs[0]["sd"]), "Random intercept SD must be finite"

    def test_factor_summary_populated(self):
        """factors() returns a list of dicts with expected keys after fit."""
        rng = np.random.default_rng(7)
        n = 60
        X = rng.standard_normal((n, 2))
        group = np.repeat(np.arange(6), 10).astype(np.int64)
        y = X @ np.array([1.0, -0.5]) + rng.standard_normal(n) * 0.3
        model = GLMM.gaussian().fit(X, y, group.tolist())
        fs = model.factors()
        assert len(fs) >= 1
        for fd in fs:
            assert "n_levels" in fd
            assert "sd" in fd
            assert "blups" in fd
            assert np.isfinite(fd["sd"])
