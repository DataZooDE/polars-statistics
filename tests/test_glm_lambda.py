"""Tests for the `lambda_` (penalized IRLS) kwarg on GLM PyClasses (issue #15)."""

import numpy as np
import pytest

from polars_statistics import (
    Cloglog,
    Logistic,
    NegativeBinomial,
    Poisson,
    Probit,
    Tweedie,
)


@pytest.fixture
def logistic_data():
    """Modest-signal binary outcome — pattern that converges quickly."""
    np.random.seed(42)
    n = 200
    x = np.random.randn(n, 2)
    y = ((x[:, 0] + x[:, 1] + np.random.randn(n) * 0.5) > 0).astype(float)
    return x, y


@pytest.fixture
def count_data():
    """Modest-signal Poisson counts that converge quickly."""
    np.random.seed(43)
    n = 200
    x = np.random.randn(n, 2) * 0.5
    y = np.random.poisson(np.exp(x[:, 0] * 0.3 + 0.5)).astype(float)
    return x, y


def _coef_norm(model):
    return float(np.linalg.norm(model.coefficients))


class TestLambdaShrinks:
    """For each GLM, lambda_ > 0 should shrink coefficient L2 norm vs lambda_ = 0."""

    def test_logistic(self, logistic_data):
        x, y = logistic_data
        unreg = Logistic(max_iter=100).fit(x, y)
        strong = Logistic(max_iter=100, lambda_=1.0).fit(x, y)
        assert _coef_norm(strong) < _coef_norm(unreg)

    def test_poisson(self, count_data):
        x, y = count_data
        unreg = Poisson(max_iter=100).fit(x, y)
        strong = Poisson(max_iter=100, lambda_=1.0).fit(x, y)
        assert _coef_norm(strong) < _coef_norm(unreg)

    def test_probit(self, logistic_data):
        x, y = logistic_data
        unreg = Probit(max_iter=200).fit(x, y)
        strong = Probit(max_iter=200, lambda_=1.0).fit(x, y)
        assert _coef_norm(strong) < _coef_norm(unreg)

    def test_cloglog(self, logistic_data):
        x, y = logistic_data
        unreg = Cloglog(max_iter=200).fit(x, y)
        strong = Cloglog(max_iter=200, lambda_=1.0).fit(x, y)
        assert _coef_norm(strong) < _coef_norm(unreg)

    def test_negative_binomial(self, count_data):
        x, y = count_data
        # estimate_theta=False keeps the fit deterministic for the shrinkage check
        unreg = NegativeBinomial(
            theta=1.0, estimate_theta=False, max_iter=200
        ).fit(x, y)
        strong = NegativeBinomial(
            theta=1.0, estimate_theta=False, max_iter=200, lambda_=1.0
        ).fit(x, y)
        assert _coef_norm(strong) < _coef_norm(unreg)

    def test_tweedie(self):
        # Use the same data shape as test_models.py's Tweedie tests so the
        # unreg fit converges; we only need shrinkage between two lambda values.
        np.random.seed(44)
        n = 200
        x = np.random.randn(n, 2) * 0.3
        y = np.exp(0.5 + 0.3 * x[:, 0] - 0.2 * x[:, 1]) + np.abs(
            np.random.randn(n) * 0.1
        )
        unreg = Tweedie(var_power=1.5, max_iter=200).fit(x, y)
        strong = Tweedie(var_power=1.5, max_iter=200, lambda_=1.0).fit(x, y)
        assert _coef_norm(strong) < _coef_norm(unreg)


class TestLambdaDefaultIsZero:
    """lambda_ defaults to 0.0, so default fits match an explicit lambda_=0 fit."""

    def test_logistic(self, logistic_data):
        x, y = logistic_data
        default = Logistic(max_iter=100).fit(x, y)
        explicit = Logistic(max_iter=100, lambda_=0.0).fit(x, y)
        np.testing.assert_allclose(
            default.coefficients, explicit.coefficients, rtol=1e-10
        )

    def test_poisson(self, count_data):
        x, y = count_data
        default = Poisson(max_iter=100).fit(x, y)
        explicit = Poisson(max_iter=100, lambda_=0.0).fit(x, y)
        np.testing.assert_allclose(
            default.coefficients, explicit.coefficients, rtol=1e-10
        )


class TestTweedieFactoryConstructorsAcceptLambda:
    """Tweedie's gaussian/gamma/inverse_gaussian classmethods should also accept lambda_."""

    @pytest.fixture
    def positive_data(self):
        np.random.seed(45)
        n = 200
        x = np.random.randn(n, 2) * 0.3
        y = np.exp(0.5 + 0.3 * x[:, 0] - 0.2 * x[:, 1]) + np.abs(
            np.random.randn(n) * 0.1
        )
        # Ensure strictly positive for gamma
        y = np.maximum(y, 0.01)
        return x, y

    def test_gaussian(self, positive_data):
        x, y = positive_data
        unreg = Tweedie.gaussian(max_iter=200).fit(x, y)
        strong = Tweedie.gaussian(max_iter=200, lambda_=1.0).fit(x, y)
        assert _coef_norm(strong) < _coef_norm(unreg)

    def test_gamma(self, positive_data):
        x, y = positive_data
        unreg = Tweedie.gamma(max_iter=200).fit(x, y)
        strong = Tweedie.gamma(max_iter=200, lambda_=1.0).fit(x, y)
        assert _coef_norm(strong) < _coef_norm(unreg)
