"""Smoke tests for PassiveAggressive regression PyModel."""
import numpy as np
import pytest
from polars_statistics import PassiveAggressive


class TestPassiveAggressive:
    def test_fit_basic(self):
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X @ np.array([1.0, -1.0, 0.5]) + 0.1 * np.random.randn(100)
        model = PassiveAggressive().fit(X, y)
        assert model.is_fitted()
        assert len(model.coefficients) == 3
        assert np.all(np.isfinite(model.coefficients))

    def test_predict_shape(self):
        np.random.seed(0)
        X = np.random.randn(50, 2)
        y = X @ np.array([2.0, -0.5]) + 0.05 * np.random.randn(50)
        model = PassiveAggressive().fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (50,)
        assert np.all(np.isfinite(preds))

    def test_partial_fit_one_sample(self):
        """REGR-06: partial_fit must succeed for a single-sample online update."""
        np.random.seed(1)
        X = np.random.randn(30, 2)
        y = X @ np.array([1.0, 2.0]) + 0.1 * np.random.randn(30)
        model = PassiveAggressive()
        # partial_fit should work without a prior batch fit
        model.partial_fit(X[0], y[0])

    def test_partial_fit_after_batch_fit(self):
        np.random.seed(2)
        X = np.random.randn(50, 2)
        y = X @ np.array([1.0, -2.0]) + 0.1 * np.random.randn(50)
        model = PassiveAggressive().fit(X, y)
        # Continue training online
        for i in range(5):
            model.partial_fit(X[i], y[i])
        # Batch predictions should still work after partial_fit
        preds = model.predict(X)
        assert preds.shape == (50,)

    def test_n_iter_present(self):
        np.random.seed(5)
        X = np.random.randn(40, 2)
        y = X @ np.array([1.0, 1.0]) + 0.05 * np.random.randn(40)
        model = PassiveAggressive().fit(X, y)
        assert model.n_iter >= 1

    def test_not_fitted_raises(self):
        model = PassiveAggressive()
        assert not model.is_fitted()
        with pytest.raises(Exception):
            _ = model.coefficients

    def test_squared_loss(self):
        np.random.seed(9)
        X = np.random.randn(60, 2)
        y = X @ np.array([1.0, 2.0]) + 0.1 * np.random.randn(60)
        model = PassiveAggressive(loss="squared_epsilon_insensitive").fit(X, y)
        assert model.is_fitted()
        assert np.all(np.isfinite(model.coefficients))
