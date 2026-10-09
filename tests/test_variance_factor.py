"""Numpy cross-checks for OLS/WLS/Ridge ``variance_factor()`` / ``leverage_new()``,
plus Ridge's sandwich inference getters for ``lambda_ > 0`` (anofox-regression
>= 0.5.14, #37). See `.planning/phases/16.1-upstream-crate-refresh/16.1-05-PLAN.md`.
"""

import numpy as np
import polars as pl
import pytest
from polars_statistics import OLS, WLS, MomentAccumulator, Ridge


def _d15() -> pl.DataFrame:
    """Shared n=15 dataset for the OLS logLik/AIC/BIC reference (#41)."""
    i = list(range(1, 16))
    x0 = [k / 3 for k in i]
    x1 = [((k * 5) % 7) / 2 for k in i]
    y = [2 + 1.5 * a - 0.8 * b + (((k * 3) % 5) - 2) * 0.5 for k, a, b in zip(i, x0, x1)]
    return pl.DataFrame({"y": y, "x0": x0, "x1": x1})


def _xy():
    df = _d15()
    X = df.select(["x0", "x1"]).to_numpy()
    y = df["y"].to_numpy()
    return X, y


def _augment(X):
    """Prepend an intercept column of ones."""
    return np.column_stack([np.ones(X.shape[0]), X])


class TestOlsVarianceFactor:
    def test_variance_factor_matches_inv_xtx_with_intercept(self):
        X, y = _xy()
        model = OLS().fit(X, y)
        X1 = _augment(X)
        expected = np.linalg.inv(X1.T @ X1)

        vf = model.variance_factor()
        assert vf.shape == (3, 3)
        np.testing.assert_allclose(vf, expected, rtol=1e-8)

    def test_variance_factor_matches_inv_xtx_no_intercept(self):
        X, y = _xy()
        model = OLS(with_intercept=False).fit(X, y)
        expected = np.linalg.inv(X.T @ X)

        vf = model.variance_factor()
        assert vf.shape == (2, 2)
        np.testing.assert_allclose(vf, expected, rtol=1e-8)

    def test_leverage_new_matches_hat_diagonal(self):
        X, y = _xy()
        model = OLS().fit(X, y)
        X1 = _augment(X)
        M = model.variance_factor()
        expected = np.diag(X1 @ M @ X1.T)

        leverage = model.leverage_new(X)
        np.testing.assert_allclose(leverage, expected, atol=1e-10)
        # Leverage of the training design sums to the number of parameters
        # (hat-matrix trace == rank), here 3 (intercept + 2 features).
        assert leverage.sum() == pytest.approx(3.0, abs=1e-8)

    def test_leverage_new_accepts_multi_row_input(self):
        X, y = _xy()
        model = OLS().fit(X, y)
        leverage = model.leverage_new(X[:2])
        assert leverage.shape == (2,)

    def test_leverage_new_column_mismatch_raises_value_error(self):
        X, y = _xy()
        model = OLS().fit(X, y)
        bad_x_new = np.column_stack([X, X[:, 0]])  # 3 columns, model has 2
        with pytest.raises(ValueError):
            model.leverage_new(bad_x_new)

    def test_fit_from_accumulator_has_no_variance_factor(self):
        X, y = _xy()
        acc = MomentAccumulator(n_features=2)
        for i in range(len(y)):
            acc.push_row(np.ascontiguousarray(X[i]), y[i])
        model = OLS().fit_from_accumulator(acc)

        assert model.variance_factor() is None
        leverage = model.leverage_new(X)
        assert np.all(np.isnan(leverage))


class TestWlsVarianceFactor:
    def test_variance_factor_matches_inv_xtwx(self):
        X, y = _xy()
        n = X.shape[0]
        w = np.array([1 + (i % 3) for i in range(1, n + 1)], dtype=float)
        model = WLS().fit(X, y, w)
        X1 = _augment(X)
        W = np.diag(w)
        expected = np.linalg.inv(X1.T @ W @ X1)

        vf = model.variance_factor()
        assert vf.shape == (3, 3)
        np.testing.assert_allclose(vf, expected, rtol=1e-8)


class TestRidgeVarianceFactorAndInference:
    def test_variance_factor_matches_sandwich_formula(self):
        X, y = _xy()
        model = Ridge(lambda_=1.0).fit(X, y)
        X1 = _augment(X)
        P = np.diag([0.0, 1.0, 1.0])
        A = np.linalg.inv(X1.T @ X1 + 1.0 * P)
        expected = A @ (X1.T @ X1) @ A

        vf = model.variance_factor()
        assert vf.shape == (3, 3)
        np.testing.assert_allclose(vf, expected, rtol=1e-6)

    def test_sandwich_std_errors_match_r_reference(self):
        """R reference (anofox-regression #37)."""
        X, y = _xy()
        model = Ridge(lambda_=1.0).fit(X, y)

        assert model.std_errors[0] == pytest.approx(0.13688648231466696, abs=1e-6)
        assert model.std_errors[1] == pytest.approx(0.19112708153110064, abs=1e-6)
        assert model.intercept_std_error == pytest.approx(
            0.53418431237438180, abs=1e-6
        )
        assert model.t_statistics is None
        assert model.p_values is None

    def test_lambda_zero_delegates_to_ols(self):
        X, y = _xy()
        ridge0 = Ridge(lambda_=0.0).fit(X, y)
        ols = OLS().fit(X, y)

        np.testing.assert_allclose(
            ridge0.std_errors, ols.std_errors, rtol=1e-8
        )
        assert ridge0.t_statistics is not None
        assert ridge0.p_values is not None

    def test_leverage_new_matches_variance_factor(self):
        X, y = _xy()
        model = Ridge(lambda_=1.0).fit(X, y)
        X1 = _augment(X)
        M = model.variance_factor()
        expected = np.diag(X1 @ M @ X1.T)

        leverage = model.leverage_new(X)
        np.testing.assert_allclose(leverage, expected, atol=1e-10)

    def test_leverage_new_column_mismatch_raises_value_error(self):
        X, y = _xy()
        model = Ridge(lambda_=1.0).fit(X, y)
        bad_x_new = np.column_stack([X, X[:, 0]])
        with pytest.raises(ValueError):
            model.leverage_new(bad_x_new)
