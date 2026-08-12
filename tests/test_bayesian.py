"""Smoke tests for BayesianRidge and ARD (Automatic Relevance Determination) PyModels."""

import numpy as np
import pytest

from polars_statistics import ARD, BayesianRidge


class TestBayesianRidge:
    def test_fit_basic(self):
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X @ np.array([1.0, 2.0]) + np.random.randn(50) * 0.1
        model = BayesianRidge().fit(X, y)
        assert model.is_fitted()
        assert len(model.coefficients) == 2
        assert np.all(np.isfinite(model.coefficients))

    def test_predict_shape(self):
        np.random.seed(0)
        X = np.random.randn(30, 2)
        y = X @ np.array([1.0, -1.0]) + 0.05 * np.random.randn(30)
        model = BayesianRidge().fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (30,)
        assert np.all(np.isfinite(preds))

    def test_not_fitted_raises(self):
        model = BayesianRidge()
        with pytest.raises(Exception):
            _ = model.coefficients

    def test_alpha_finite(self):
        """alpha_ (noise precision) must be positive and finite after fitting."""
        np.random.seed(1)
        X = np.random.randn(60, 3)
        y = X @ np.array([1.0, -0.5, 2.0]) + 0.1 * np.random.randn(60)
        model = BayesianRidge().fit(X, y)
        assert np.isfinite(model.alpha_)
        assert model.alpha_ > 0.0

    def test_lambda_finite(self):
        """lambda_ (weight precision) must be positive and finite after fitting."""
        np.random.seed(2)
        X = np.random.randn(60, 3)
        y = X @ np.array([0.5, 1.5, -1.0]) + 0.1 * np.random.randn(60)
        model = BayesianRidge().fit(X, y)
        assert np.isfinite(model.lambda_)
        assert model.lambda_ > 0.0

    def test_sigma_diag_shape(self):
        """sigma_diag must have length equal to n_features."""
        np.random.seed(3)
        n_features = 4
        X = np.random.randn(80, n_features)
        y = X @ np.ones(n_features) + 0.1 * np.random.randn(80)
        model = BayesianRidge().fit(X, y)
        assert model.sigma_diag.shape == (n_features,)
        assert np.all(model.sigma_diag > 0.0)

    def test_intercept_present(self):
        np.random.seed(4)
        X = np.random.randn(50, 2)
        y = 2.0 + X @ np.array([1.0, -1.0]) + 0.1 * np.random.randn(50)
        model = BayesianRidge(fit_intercept=True).fit(X, y)
        assert model.intercept is not None
        assert np.isfinite(model.intercept)

    def test_no_intercept(self):
        np.random.seed(5)
        X = np.random.randn(50, 2)
        y = X @ np.array([1.0, -1.0]) + 0.1 * np.random.randn(50)
        model = BayesianRidge(fit_intercept=False).fit(X, y)
        assert model.intercept is None

    def test_r_squared_range(self):
        np.random.seed(6)
        X = np.random.randn(50, 2)
        y = X @ np.array([1.0, 2.0]) + 0.1 * np.random.randn(50)
        model = BayesianRidge().fit(X, y)
        assert 0.0 <= model.r_squared <= 1.0

    def test_residuals_shape(self):
        np.random.seed(7)
        n = 40
        X = np.random.randn(n, 2)
        y = X @ np.array([1.0, 1.0]) + 0.1 * np.random.randn(n)
        model = BayesianRidge().fit(X, y)
        assert model.residuals.shape == (n,)

    def test_n_observations(self):
        np.random.seed(8)
        n = 55
        X = np.random.randn(n, 2)
        y = X @ np.array([1.0, -1.0]) + 0.1 * np.random.randn(n)
        model = BayesianRidge().fit(X, y)
        assert model.n_observations == n

    def test_recovers_known_coefficients(self):
        """BayesianRidge recovers known coefficients on a well-conditioned linear design.

        Ground truth: y = 2.0 * x1 - 1.0 * x2 + tiny noise.
        The regularization is mild (default priors), so the posterior mean should
        be within 0.20 of the true values.
        """
        np.random.seed(99)
        n = 200
        X = np.random.randn(n, 2)
        true_beta = np.array([2.0, -1.0])
        y = X @ true_beta + 0.05 * np.random.randn(n)
        model = BayesianRidge(fit_intercept=False).fit(X, y)
        np.testing.assert_allclose(model.coefficients, true_beta, atol=0.20,
                                   err_msg="BayesianRidge failed to recover known coefficients")

    def test_vs_sklearn_on_clean_data(self):
        """BayesianRidge coefficients are close to sklearn BayesianRidge on clean data.

        sklearn is NOT a declared dependency — guard with importorskip.
        """
        sklearn_linear = pytest.importorskip("sklearn.linear_model")
        np.random.seed(17)
        n = 120
        X = np.random.randn(n, 2)
        y = X @ np.array([1.0, -2.0]) + 0.1 * np.random.randn(n)
        ps_model = BayesianRidge(fit_intercept=False).fit(X, y)
        sk_model = sklearn_linear.BayesianRidge(fit_intercept=False).fit(X, y)
        np.testing.assert_allclose(ps_model.coefficients, sk_model.coef_, atol=0.30,
                                   err_msg="BayesianRidge vs sklearn coefficient mismatch")


class TestARD:
    def test_fit_basic(self):
        np.random.seed(42)
        X = np.random.randn(60, 3)
        y = X @ np.array([1.0, 2.0, 0.0]) + np.random.randn(60) * 0.1
        model = ARD().fit(X, y)
        assert model.is_fitted()
        assert len(model.coefficients) == 3
        assert np.all(np.isfinite(model.coefficients))

    def test_predict_shape(self):
        np.random.seed(0)
        X = np.random.randn(40, 3)
        y = X @ np.array([1.0, -1.0, 0.5]) + 0.1 * np.random.randn(40)
        model = ARD().fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (40,)
        assert np.all(np.isfinite(preds))

    def test_not_fitted_raises(self):
        model = ARD()
        with pytest.raises(Exception):
            _ = model.coefficients

    def test_lambdas_length(self):
        """lambdas must have length equal to n_features."""
        np.random.seed(1)
        n_features = 5
        X = np.random.randn(100, n_features)
        y = X @ np.array([1.0, 0.0, 2.0, 0.0, -1.0]) + 0.1 * np.random.randn(100)
        model = ARD().fit(X, y)
        assert len(model.lambdas) == n_features

    def test_lambdas_positive(self):
        np.random.seed(2)
        X = np.random.randn(80, 3)
        y = X @ np.array([1.0, 2.0, 0.0]) + 0.1 * np.random.randn(80)
        model = ARD().fit(X, y)
        assert np.all(model.lambdas > 0.0)

    def test_alpha_finite(self):
        np.random.seed(3)
        X = np.random.randn(60, 3)
        y = X @ np.array([1.0, -0.5, 2.0]) + 0.1 * np.random.randn(60)
        model = ARD().fit(X, y)
        assert np.isfinite(model.alpha_)
        assert model.alpha_ > 0.0

    def test_intercept_present(self):
        np.random.seed(4)
        X = np.random.randn(60, 3)
        y = 2.0 + X @ np.array([1.0, -1.0, 0.5]) + 0.1 * np.random.randn(60)
        model = ARD(fit_intercept=True).fit(X, y)
        assert model.intercept is not None
        assert np.isfinite(model.intercept)

    def test_no_intercept(self):
        np.random.seed(5)
        X = np.random.randn(60, 3)
        y = X @ np.array([1.0, -1.0, 0.5]) + 0.1 * np.random.randn(60)
        model = ARD(fit_intercept=False).fit(X, y)
        assert model.intercept is None

    def test_r_squared_range(self):
        np.random.seed(6)
        X = np.random.randn(70, 3)
        y = X @ np.array([1.0, 2.0, -1.0]) + 0.1 * np.random.randn(70)
        model = ARD().fit(X, y)
        assert 0.0 <= model.r_squared <= 1.0

    def test_sparse_recovery(self):
        """ARD should produce near-zero coefficients for irrelevant features."""
        np.random.seed(10)
        n = 200
        X = np.random.randn(n, 5)
        # Only first two features are relevant
        y = X @ np.array([3.0, -2.0, 0.0, 0.0, 0.0]) + 0.1 * np.random.randn(n)
        model = ARD().fit(X, y)
        coef = model.coefficients
        # Relevant features should have larger magnitude than irrelevant ones
        assert abs(coef[0]) > abs(coef[3])
        assert abs(coef[1]) > abs(coef[4])

    def test_recovers_known_coefficients(self):
        """ARD recovers known non-zero coefficients on a clean design.

        Ground truth: y = 1.5 * x1 - 0.8 * x2 + tiny noise.
        ARD posterior mean should be within 0.20 of the true values.
        """
        np.random.seed(88)
        n = 200
        X = np.random.randn(n, 2)
        true_beta = np.array([1.5, -0.8])
        y = X @ true_beta + 0.05 * np.random.randn(n)
        model = ARD(fit_intercept=False).fit(X, y)
        np.testing.assert_allclose(model.coefficients, true_beta, atol=0.20,
                                   err_msg="ARD failed to recover known coefficients")

    def test_vs_sklearn_on_clean_data(self):
        """ARD coefficients are close to sklearn ARDRegression on clean data.

        sklearn is NOT a declared dependency — guard with importorskip.
        """
        sklearn_linear = pytest.importorskip("sklearn.linear_model")
        np.random.seed(33)
        n = 150
        X = np.random.randn(n, 2)
        y = X @ np.array([2.0, -1.0]) + 0.1 * np.random.randn(n)
        ps_model = ARD(fit_intercept=False).fit(X, y)
        sk_model = sklearn_linear.ARDRegression(fit_intercept=False).fit(X, y)
        np.testing.assert_allclose(ps_model.coefficients, sk_model.coef_, atol=0.30,
                                   err_msg="ARD vs sklearn coefficient mismatch")
