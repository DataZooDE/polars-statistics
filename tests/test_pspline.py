"""Smoke tests for PSplineRegressor PyModel (REGR-02)."""

import numpy as np
import pytest
from polars_statistics import PSpline


class TestPSpline:
    def test_fit_basic(self):
        """PSpline().fit(X_1col, y) returns is_fitted==True, finite edf."""
        rng = np.random.default_rng(42)
        x = np.linspace(0, 1, 60).reshape(-1, 1)
        y = np.sin(2 * np.pi * x.ravel()) + rng.standard_normal(60) * 0.1
        model = PSpline().fit(x, y)
        assert model.is_fitted()
        assert np.isfinite(model.edf)
        assert model.edf > 0

    def test_predict_shape(self):
        """predict() returns array of shape (n,) with finite values."""
        rng = np.random.default_rng(0)
        x = np.linspace(0, 2 * np.pi, 80).reshape(-1, 1)
        y = np.sin(x.ravel()) + rng.standard_normal(80) * 0.05
        model = PSpline().fit(x, y)
        preds = model.predict(x)
        assert preds.shape == (80,)
        assert np.all(np.isfinite(preds))

    def test_single_column_required(self):
        """PSpline.fit raises ValueError when X has more than one column."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((50, 2))
        y = rng.standard_normal(50)
        with pytest.raises(ValueError, match="single-column"):
            PSpline().fit(X, y)

    def test_not_fitted_raises(self):
        """Reading edf before fit raises RuntimeError."""
        model = PSpline()
        with pytest.raises(Exception):
            _ = model.edf

    def test_sigma2(self):
        """sigma2 getter is finite and positive after fit."""
        x = np.linspace(0, 1, 40).reshape(-1, 1)
        y = x.ravel() ** 2 + 0.01 * np.random.randn(40)
        model = PSpline().fit(x, y)
        assert np.isfinite(model.sigma2)
        assert model.sigma2 >= 0

    def test_r_squared(self):
        """r_squared getter is finite after fit."""
        x = np.linspace(0, 1, 50).reshape(-1, 1)
        y = np.sin(2 * np.pi * x.ravel()) + 0.05 * np.random.randn(50)
        model = PSpline().fit(x, y)
        assert np.isfinite(model.r_squared)
        assert 0 <= model.r_squared <= 1

    def test_with_lambda(self):
        """PSpline with fixed lambda fits without error."""
        x = np.linspace(0, 1, 60).reshape(-1, 1)
        y = np.cos(2 * np.pi * x.ravel()) + 0.1 * np.random.randn(60)
        model = PSpline(lambda_=0.01).fit(x, y)
        assert model.is_fitted()
        assert np.isfinite(model.edf)

    def test_with_n_basis(self):
        """PSpline with explicit n_basis fits without error."""
        x = np.linspace(0, 1, 60).reshape(-1, 1)
        y = x.ravel() + 0.05 * np.random.randn(60)
        model = PSpline(n_basis=10).fit(x, y)
        assert model.is_fitted()
