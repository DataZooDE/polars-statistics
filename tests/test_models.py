"""Tests for regression models."""

import numpy as np
import pytest

from polars_statistics import OLS, Ridge, ElasticNet, WLS, Logistic, Poisson, LmDynamic, Aid


class TestOLS:
    """Tests for OLS regression."""

    def test_fit_basic(self):
        """Test basic OLS fitting."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X @ np.array([1, 2, 3]) + np.random.randn(100) * 0.1

        model = OLS().fit(X, y)

        assert model.is_fitted()
        assert len(model.coefficients) == 3
        assert model.r_squared > 0.99

    def test_coefficients_accuracy(self):
        """Test that coefficients are accurate."""
        np.random.seed(42)
        X = np.random.randn(1000, 2)
        true_coef = np.array([1.5, -2.0])
        y = X @ true_coef + np.random.randn(1000) * 0.01

        model = OLS().fit(X, y)

        np.testing.assert_array_almost_equal(model.coefficients, true_coef, decimal=1)

    def test_with_intercept(self):
        """Test OLS with intercept."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = 5.0 + X @ np.array([1, 2]) + np.random.randn(100) * 0.1

        model = OLS(with_intercept=True).fit(X, y)

        assert model.intercept is not None
        assert abs(model.intercept - 5.0) < 0.5

    def test_without_intercept(self):
        """Test OLS without intercept."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X @ np.array([1, 2]) + np.random.randn(100) * 0.1

        model = OLS(with_intercept=False).fit(X, y)

        assert model.intercept is None

    def test_inference(self):
        """Test inference statistics."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X @ np.array([1, 2, 3]) + np.random.randn(100) * 0.1

        model = OLS(compute_inference=True).fit(X, y)

        assert model.std_errors is not None
        assert model.p_values is not None
        assert len(model.std_errors) == 3
        assert len(model.p_values) == 3

    def test_predict(self):
        """Test prediction."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X @ np.array([1, 2, 3])

        model = OLS().fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 100
        np.testing.assert_array_almost_equal(predictions, y, decimal=5)


class TestRidge:
    """Tests for Ridge regression."""

    def test_fit_basic(self):
        """Test basic Ridge fitting."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X @ np.array([1, 2, 3]) + np.random.randn(100) * 0.1

        model = Ridge(lambda_=0.1).fit(X, y)

        assert model.is_fitted()
        assert len(model.coefficients) == 3

    def test_regularization_effect(self):
        """Test that regularization shrinks coefficients."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X @ np.array([1, 2, 3]) + np.random.randn(100) * 0.1

        ols = OLS().fit(X, y)
        ridge = Ridge(lambda_=10.0).fit(X, y)

        # Ridge coefficients should be smaller in magnitude
        assert np.sum(ridge.coefficients**2) < np.sum(ols.coefficients**2)


class TestElasticNet:
    """Tests for Elastic Net regression."""

    def test_fit_basic(self):
        """Test basic Elastic Net fitting."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X @ np.array([1, 2, 3]) + np.random.randn(100) * 0.1

        model = ElasticNet(lambda_=0.1, alpha=0.5).fit(X, y)

        assert model.is_fitted()
        assert len(model.coefficients) == 3


class TestWLS:
    """Tests for Weighted Least Squares."""

    def test_fit_basic(self):
        """Test basic WLS fitting."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X @ np.array([1, 2, 3]) + np.random.randn(100) * 0.1
        weights = np.ones(100)

        model = WLS().fit(X, y, weights)

        assert model.is_fitted()
        assert len(model.coefficients) == 3

    def test_weighted_fitting(self):
        """Test that weights affect the fit."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X @ np.array([1, 2]) + np.random.randn(100)

        # Higher weights for first half
        weights = np.concatenate([np.ones(50) * 10, np.ones(50) * 0.1])

        model = WLS().fit(X, y, weights)

        assert model.is_fitted()


class TestLogistic:
    """Tests for Logistic regression."""

    def test_fit_basic(self):
        """Test basic Logistic fitting."""
        np.random.seed(42)
        X = np.random.randn(200, 2)
        y = ((X[:, 0] + X[:, 1] + np.random.randn(200) * 0.5) > 0).astype(float)

        model = Logistic().fit(X, y)

        assert model.is_fitted()
        assert len(model.coefficients) == 2

    def test_predict_proba(self):
        """Test probability prediction."""
        np.random.seed(42)
        X = np.random.randn(200, 2)
        y = ((X[:, 0] + X[:, 1] + np.random.randn(200) * 0.5) > 0).astype(float)

        model = Logistic().fit(X, y)
        probs = model.predict_proba(X)

        assert len(probs) == 200
        assert all(0 <= p <= 1 for p in probs)

    def test_predict(self):
        """Test class prediction."""
        np.random.seed(42)
        X = np.random.randn(200, 2)
        y = ((X[:, 0] + X[:, 1] + np.random.randn(200) * 0.5) > 0).astype(float)

        model = Logistic().fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 200
        assert all(p in [0.0, 1.0] for p in predictions)


class TestPoisson:
    """Tests for Poisson regression."""

    def test_fit_basic(self):
        """Test basic Poisson fitting."""
        np.random.seed(42)
        X = np.random.randn(200, 2) * 0.5
        y = np.random.poisson(np.exp(X[:, 0] * 0.3 + 0.5)).astype(float)

        model = Poisson(max_iter=100).fit(X, y)

        assert model.is_fitted()
        assert len(model.coefficients) == 2

    def test_predict(self):
        """Test count prediction."""
        np.random.seed(42)
        X = np.random.randn(200, 2) * 0.5
        y = np.random.poisson(np.exp(X[:, 0] * 0.3 + 0.5)).astype(float)

        model = Poisson(max_iter=100).fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 200
        assert all(p >= 0 for p in predictions)


class TestLmDynamic:
    """Tests for Dynamic Linear Model."""

    def test_fit_basic(self):
        """Test basic dynamic linear model fitting."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X @ np.array([1, 2]) + np.random.randn(100) * 0.1

        model = LmDynamic(ic="aicc", distribution="normal").fit(X, y)

        assert model.is_fitted()
        assert len(model.coefficients) == 2
        assert model.r_squared > 0.9

    def test_dynamic_coefficients(self):
        """Test dynamic coefficients extraction."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X @ np.array([1, 2]) + np.random.randn(100) * 0.1

        model = LmDynamic(ic="aicc").fit(X, y)

        dyn_coef = model.dynamic_coefficients
        assert dyn_coef.shape[0] == 100  # n_observations rows
        assert dyn_coef.shape[1] >= 2  # at least 2 coefficients

    def test_predict(self):
        """Test prediction."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X @ np.array([1, 2])

        model = LmDynamic().fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 100

    def test_no_smoothing(self):
        """Test model without LOWESS smoothing."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X @ np.array([1, 2]) + np.random.randn(50) * 0.1

        model = LmDynamic(lowess_span=None).fit(X, y)

        assert model.is_fitted()
        assert model.smoothed_weights is None


class TestAid:
    """Tests for AID demand classifier."""

    def test_regular_demand(self):
        """Test classification of regular demand."""
        np.random.seed(42)
        # Regular demand - Poisson with mean 10, few zeros
        demand = np.random.poisson(10, 100).astype(float)

        result = Aid().classify(demand)

        assert result.demand_type == "regular"
        assert not result.is_intermittent()
        assert result.n_observations == 100
        assert result.zero_proportion < 0.3

    def test_intermittent_demand(self):
        """Test classification of intermittent demand."""
        np.random.seed(42)
        # Intermittent demand - many zeros
        demand = np.where(np.random.random(100) < 0.5, 0, np.random.poisson(5, 100)).astype(float)

        result = Aid(intermittent_threshold=0.3).classify(demand)

        assert result.demand_type == "intermittent"
        assert result.is_intermittent()
        assert result.zero_proportion >= 0.3

    def test_distribution_selection(self):
        """Test that a distribution is selected."""
        np.random.seed(42)
        demand = np.random.poisson(10, 100).astype(float)

        result = Aid().classify(demand)

        assert result.distribution in ["poisson", "negative_binomial", "normal", "gamma", "lognormal", "geometric", "rectified_normal"]

    def test_fractional_detection(self):
        """Test fractional data detection."""
        np.random.seed(42)
        # Fractional demand
        demand = np.random.gamma(2, 5, 100)

        result = Aid().classify(demand)

        assert result.is_fractional

    def test_count_data(self):
        """Test count (integer) data detection."""
        np.random.seed(42)
        demand = np.random.poisson(10, 100).astype(float)

        result = Aid().classify(demand)

        assert not result.is_fractional

    def test_anomaly_detection(self):
        """Test anomaly detection capability."""
        np.random.seed(42)
        demand = np.random.poisson(10, 100).astype(float)

        result = Aid(detect_anomalies=True).classify(demand)

        # Should have anomaly flags (list of same length as demand)
        assert len(list(result.anomalies)) == 100
