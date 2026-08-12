"""Smoke tests for MomentAccumulator PyModel and OLS/Ridge fit_from_accumulator."""
import numpy as np
import pytest
from polars_statistics import MomentAccumulator, OLS, Ridge


class TestMomentAccumulator:
    def test_push_and_accumulate(self):
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X @ np.array([1.0, 2.0]) + 0.1 * np.random.randn(50)

        acc = MomentAccumulator(n_features=2)
        assert acc.n == 0
        for i in range(len(y)):
            acc.push_row(X[i], y[i])
        assert acc.n == 50
        assert acc.n_features == 2
        assert np.isfinite(acc.sum_y)
        assert acc.sum_x.shape == (2,)
        assert acc.xtx.shape == (2, 2)
        assert acc.xty.shape == (2,)

    def test_wrong_row_length_raises(self):
        """REGR-06 / T-04-12: push_row with wrong length raises ValueError."""
        acc = MomentAccumulator(n_features=3)
        x_wrong = np.array([1.0, 2.0])  # length 2, expects 3
        with pytest.raises(ValueError):
            acc.push_row(x_wrong, 0.0)

    def test_clear(self):
        np.random.seed(0)
        acc = MomentAccumulator(n_features=2)
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        for i in range(10):
            acc.push_row(X[i], y[i])
        assert acc.n == 10
        acc.clear()
        assert acc.n == 0

    def test_merge(self):
        np.random.seed(1)
        X = np.random.randn(40, 2)
        y = np.random.randn(40)
        acc1 = MomentAccumulator(n_features=2)
        acc2 = MomentAccumulator(n_features=2)
        for i in range(20):
            acc1.push_row(X[i], y[i])
        for i in range(20, 40):
            acc2.push_row(X[i], y[i])
        acc1.merge(acc2)
        assert acc1.n == 40


class TestOLSFitFromAccumulator:
    def test_finite_coefficients(self):
        """REGR-06: OLS().fit_from_accumulator(acc) gives finite coefficients."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X @ np.array([1.0, 2.0]) + 0.1 * np.random.randn(100)

        acc = MomentAccumulator(n_features=2)
        for i in range(len(y)):
            acc.push_row(X[i], y[i])

        model = OLS().fit_from_accumulator(acc)
        assert model.is_fitted()
        assert len(model.coefficients) == 2
        assert np.all(np.isfinite(model.coefficients))

    def test_close_to_direct_ols(self):
        """fit_from_accumulator coefficients should be close to direct OLS fit."""
        np.random.seed(7)
        X = np.random.randn(200, 2)
        y = X @ np.array([1.5, -0.5]) + 0.05 * np.random.randn(200)

        acc = MomentAccumulator(n_features=2)
        for i in range(len(y)):
            acc.push_row(X[i], y[i])

        model_acc = OLS().fit_from_accumulator(acc)
        model_direct = OLS().fit(X, y)
        np.testing.assert_allclose(
            model_acc.coefficients, model_direct.coefficients, atol=1e-6
        )

    def test_not_fitted_raises(self):
        model = OLS()
        assert not model.is_fitted()
        with pytest.raises(Exception):
            _ = model.coefficients


class TestRidgeFitFromAccumulator:
    def test_finite_coefficients(self):
        """REGR-06: Ridge().fit_from_accumulator(acc) gives finite coefficients."""
        np.random.seed(0)
        X = np.random.randn(80, 3)
        y = X @ np.array([1.0, -1.0, 0.5]) + 0.1 * np.random.randn(80)

        acc = MomentAccumulator(n_features=3)
        for i in range(len(y)):
            acc.push_row(X[i], y[i])

        model = Ridge().fit_from_accumulator(acc)
        assert model.is_fitted()
        assert len(model.coefficients) == 3
        assert np.all(np.isfinite(model.coefficients))
