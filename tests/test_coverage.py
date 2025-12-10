"""Tests to increase code coverage for regression functions."""

import numpy as np
import polars as pl
import pytest

import polars_statistics as ps
from polars_statistics import models


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def df_regression():
    """DataFrame for regression tests."""
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = 1.0 + 2.0 * x1 - 1.0 * x2 + np.random.randn(n) * 0.5
    return pl.DataFrame({
        "group": ["A"] * 50 + ["B"] * 50,
        "y": y.tolist(),
        "x1": x1.tolist(),
        "x2": x2.tolist(),
        "weights": np.abs(np.random.randn(n)).tolist(),
    })


@pytest.fixture
def df_binary():
    """DataFrame for binary classification tests."""
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    prob = 1 / (1 + np.exp(-(0.5 + 1.0 * x1 - 0.5 * x2)))
    y = (np.random.rand(n) < prob).astype(float)
    return pl.DataFrame({
        "group": ["A"] * 50 + ["B"] * 50,
        "y": y.tolist(),
        "x1": x1.tolist(),
        "x2": x2.tolist(),
    })


@pytest.fixture
def df_count():
    """DataFrame for count regression tests."""
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    rate = np.exp(0.5 + 0.3 * x1 + 0.2 * x2)
    y = np.random.poisson(rate)
    return pl.DataFrame({
        "group": ["A"] * 50 + ["B"] * 50,
        "y": y.astype(float).tolist(),
        "x1": x1.tolist(),
        "x2": x2.tolist(),
    })


# =============================================================================
# Test models/__init__.py imports
# =============================================================================


class TestModelsImports:
    """Test that models module imports work correctly."""

    def test_linear_models_import(self):
        """Test linear model imports."""
        assert hasattr(models, "OLS")
        assert hasattr(models, "Ridge")
        assert hasattr(models, "ElasticNet")
        assert hasattr(models, "WLS")
        assert hasattr(models, "RLS")
        assert hasattr(models, "BLS")

    def test_glm_models_import(self):
        """Test GLM model imports."""
        assert hasattr(models, "Logistic")
        assert hasattr(models, "Poisson")
        assert hasattr(models, "NegativeBinomial")
        assert hasattr(models, "Tweedie")
        assert hasattr(models, "Probit")
        assert hasattr(models, "Cloglog")

    def test_bootstrap_import(self):
        """Test bootstrap imports."""
        assert hasattr(models, "StationaryBootstrap")
        assert hasattr(models, "CircularBlockBootstrap")


# =============================================================================
# Test Summary Functions
# =============================================================================


class TestSummaryFunctions:
    """Test summary functions for all regression types."""

    def test_ridge_summary(self, df_regression):
        """Test ridge_summary function."""
        result = df_regression.select(
            ps.ridge_summary("y", "x1", "x2", lambda_=1.0).alias("coef")
        )
        coefs = result["coef"][0]
        assert len(coefs) == 3  # intercept + 2 coefficients

    def test_elastic_net_summary(self, df_regression):
        """Test elastic_net_summary function."""
        result = df_regression.select(
            ps.elastic_net_summary("y", "x1", "x2", lambda_=1.0, alpha=0.5).alias("coef")
        )
        coefs = result["coef"][0]
        assert len(coefs) == 3

    def test_wls_summary(self, df_regression):
        """Test wls_summary function."""
        result = df_regression.select(
            ps.wls_summary("y", "weights", "x1", "x2").alias("coef")
        )
        coefs = result["coef"][0]
        assert len(coefs) == 3

    def test_rls_summary(self, df_regression):
        """Test rls_summary function."""
        result = df_regression.select(
            ps.rls_summary("y", "x1", "x2", forgetting_factor=0.99).alias("coef")
        )
        coefs = result["coef"][0]
        assert len(coefs) == 3

    def test_bls_summary(self, df_regression):
        """Test bls_summary function."""
        result = df_regression.select(
            ps.bls_summary("y", "x1", "x2", lower_bound=-10.0, upper_bound=10.0).alias("coef")
        )
        coefs = result["coef"][0]
        assert len(coefs) == 3

    def test_logistic_summary(self, df_binary):
        """Test logistic_summary function."""
        result = df_binary.select(
            ps.logistic_summary("y", "x1", "x2").alias("coef")
        )
        coefs = result["coef"][0]
        assert len(coefs) == 3

    def test_poisson_summary(self, df_count):
        """Test poisson_summary function."""
        result = df_count.select(
            ps.poisson_summary("y", "x1", "x2").alias("coef")
        )
        coefs = result["coef"][0]
        assert len(coefs) == 3

    def test_negative_binomial_summary(self, df_count):
        """Test negative_binomial_summary function."""
        result = df_count.select(
            ps.negative_binomial_summary("y", "x1", "x2").alias("coef")
        )
        coefs = result["coef"][0]
        assert len(coefs) == 3

    def test_tweedie_summary(self, df_count):
        """Test tweedie_summary function."""
        result = df_count.select(
            ps.tweedie_summary("y", "x1", "x2", var_power=1.5).alias("coef")
        )
        coefs = result["coef"][0]
        assert len(coefs) == 3

    def test_probit_summary(self, df_binary):
        """Test probit_summary function."""
        result = df_binary.select(
            ps.probit_summary("y", "x1", "x2").alias("coef")
        )
        coefs = result["coef"][0]
        assert len(coefs) == 3

    def test_cloglog_summary(self, df_binary):
        """Test cloglog_summary function."""
        result = df_binary.select(
            ps.cloglog_summary("y", "x1", "x2").alias("coef")
        )
        coefs = result["coef"][0]
        assert len(coefs) == 3

    def test_alm_summary(self, df_regression):
        """Test alm_summary function."""
        result = df_regression.select(
            ps.alm_summary("y", "x1", "x2", distribution="normal").alias("coef")
        )
        coefs = result["coef"][0]
        assert len(coefs) == 3


# =============================================================================
# Test Formula Functions
# =============================================================================


class TestFormulaFunctions:
    """Test formula-based regression functions."""

    def test_wls_formula(self, df_regression):
        """Test wls_formula function."""
        result = df_regression.select(
            ps.wls_formula("y ~ x1 + x2", weights="weights").alias("result")
        )
        assert result["result"][0] is not None

    def test_rls_formula(self, df_regression):
        """Test rls_formula function."""
        result = df_regression.select(
            ps.rls_formula("y ~ x1 + x2", forgetting_factor=0.99).alias("result")
        )
        assert result["result"][0] is not None

    def test_bls_formula(self, df_regression):
        """Test bls_formula function."""
        result = df_regression.select(
            ps.bls_formula("y ~ x1 + x2", lower_bound=-10.0, upper_bound=10.0).alias("result")
        )
        assert result["result"][0] is not None

    def test_nnls_formula(self, df_regression):
        """Test nnls_formula function."""
        result = df_regression.select(
            ps.nnls_formula("y ~ x1 + x2").alias("result")
        )
        assert result["result"][0] is not None

    def test_logistic_formula(self, df_binary):
        """Test logistic_formula function."""
        result = df_binary.select(
            ps.logistic_formula("y ~ x1 + x2").alias("result")
        )
        assert result["result"][0] is not None

    def test_poisson_formula(self, df_count):
        """Test poisson_formula function."""
        result = df_count.select(
            ps.poisson_formula("y ~ x1 + x2").alias("result")
        )
        assert result["result"][0] is not None

    def test_negative_binomial_formula(self, df_count):
        """Test negative_binomial_formula function."""
        result = df_count.select(
            ps.negative_binomial_formula("y ~ x1 + x2").alias("result")
        )
        assert result["result"][0] is not None

    def test_tweedie_formula(self, df_count):
        """Test tweedie_formula function."""
        result = df_count.select(
            ps.tweedie_formula("y ~ x1 + x2", var_power=1.5).alias("result")
        )
        assert result["result"][0] is not None

    def test_probit_formula(self, df_binary):
        """Test probit_formula function."""
        result = df_binary.select(
            ps.probit_formula("y ~ x1 + x2").alias("result")
        )
        assert result["result"][0] is not None

    def test_cloglog_formula(self, df_binary):
        """Test cloglog_formula function."""
        result = df_binary.select(
            ps.cloglog_formula("y ~ x1 + x2").alias("result")
        )
        assert result["result"][0] is not None

    def test_alm_formula(self, df_regression):
        """Test alm_formula function."""
        result = df_regression.select(
            ps.alm_formula("y ~ x1 + x2", distribution="laplace").alias("result")
        )
        assert result["result"][0] is not None


# =============================================================================
# Test Formula Summary Functions
# =============================================================================


class TestFormulaSummaryFunctions:
    """Test formula-based summary functions."""

    def test_ridge_formula_summary(self, df_regression):
        """Test ridge_formula_summary function."""
        result = df_regression.select(
            ps.ridge_formula_summary("y ~ x1 + x2", lambda_=1.0).alias("coef")
        )
        coefs = result["coef"][0]
        assert len(coefs) == 3

    def test_elastic_net_formula_summary(self, df_regression):
        """Test elastic_net_formula_summary function."""
        result = df_regression.select(
            ps.elastic_net_formula_summary("y ~ x1 + x2", lambda_=1.0, alpha=0.5).alias("coef")
        )
        coefs = result["coef"][0]
        assert len(coefs) == 3

    def test_logistic_formula_summary(self, df_binary):
        """Test logistic_formula_summary function."""
        result = df_binary.select(
            ps.logistic_formula_summary("y ~ x1 + x2").alias("coef")
        )
        coefs = result["coef"][0]
        assert len(coefs) == 3

    def test_poisson_formula_summary(self, df_count):
        """Test poisson_formula_summary function."""
        result = df_count.select(
            ps.poisson_formula_summary("y ~ x1 + x2").alias("coef")
        )
        coefs = result["coef"][0]
        assert len(coefs) == 3

    def test_alm_formula_summary(self, df_regression):
        """Test alm_formula_summary function."""
        result = df_regression.select(
            ps.alm_formula_summary("y ~ x1 + x2", distribution="normal").alias("coef")
        )
        coefs = result["coef"][0]
        assert len(coefs) == 3


# =============================================================================
# Test Formula Predict Functions
# =============================================================================


class TestFormulaPredictFunctions:
    """Test formula-based predict functions."""

    def test_ridge_formula_predict(self, df_regression):
        """Test ridge_formula_predict function."""
        result = df_regression.with_columns(
            ps.ridge_formula_predict("y ~ x1 + x2", lambda_=1.0).alias("pred")
        )
        assert result["pred"].len() == 100

    def test_elastic_net_formula_predict(self, df_regression):
        """Test elastic_net_formula_predict function."""
        result = df_regression.with_columns(
            ps.elastic_net_formula_predict("y ~ x1 + x2", lambda_=1.0, alpha=0.5).alias("pred")
        )
        assert result["pred"].len() == 100

    def test_logistic_formula_predict(self, df_binary):
        """Test logistic_formula_predict function."""
        result = df_binary.with_columns(
            ps.logistic_formula_predict("y ~ x1 + x2").alias("pred")
        )
        assert result["pred"].len() == 100

    def test_poisson_formula_predict(self, df_count):
        """Test poisson_formula_predict function."""
        result = df_count.with_columns(
            ps.poisson_formula_predict("y ~ x1 + x2").alias("pred")
        )
        assert result["pred"].len() == 100

    def test_alm_formula_predict(self, df_regression):
        """Test alm_formula_predict function."""
        result = df_regression.with_columns(
            ps.alm_formula_predict("y ~ x1 + x2", distribution="normal").alias("pred")
        )
        assert result["pred"].len() == 100

    def test_ridge_formula_predict_with_interval(self, df_regression):
        """Test ridge_formula_predict with prediction interval."""
        result = df_regression.with_columns(
            ps.ridge_formula_predict("y ~ x1 + x2", lambda_=1.0, interval="prediction").alias("pred")
        )
        pred = result["pred"][0]
        assert "ridge_prediction" in pred
        assert "ridge_lower" in pred
        assert "ridge_upper" in pred


# =============================================================================
# Test Modern Tests Edge Cases
# =============================================================================


class TestModernTestsEdgeCases:
    """Test edge cases for modern distribution tests."""

    def test_mmd_test_with_seed(self):
        """Test mmd_test with seed parameter."""
        np.random.seed(42)
        df = pl.DataFrame({
            "x": np.random.randn(50).tolist(),
            "y": np.random.randn(50).tolist(),
        })
        result = df.select(
            ps.mmd_test("x", "y", n_permutations=99, seed=42).alias("result")
        )
        assert result["result"][0] is not None


# =============================================================================
# Test Formula Parser Edge Cases
# =============================================================================


class TestFormulaParserEdgeCases:
    """Test edge cases for formula parser."""

    def test_poly_with_spaces(self, df_regression):
        """Test polynomial with spaces in formula."""
        result = df_regression.select(
            ps.ols_formula("y ~ poly( x1 , 2 )").alias("result")
        )
        assert result["result"][0] is not None

    def test_multiple_interactions(self, df_regression):
        """Test formula with multiple interactions."""
        df = df_regression.with_columns(
            x3=pl.lit(1.0) + pl.col("x1") * 0.5
        )
        result = df.select(
            ps.ols_formula("y ~ x1 + x2 + x3 + x1:x2").alias("result")
        )
        assert result["result"][0] is not None


# =============================================================================
# Test String Column Names in Summary Functions
# =============================================================================


class TestStringColumnNames:
    """Test that string column names work in all functions."""

    def test_ols_summary_string_cols(self, df_regression):
        """Test ols_summary with string column names."""
        result = df_regression.select(
            ps.ols_summary("y", "x1", "x2").alias("coef")
        )
        assert len(result["coef"][0]) == 3

    def test_ridge_summary_string_cols(self, df_regression):
        """Test ridge_summary with string column names."""
        result = df_regression.select(
            ps.ridge_summary("y", "x1", "x2").alias("coef")
        )
        assert len(result["coef"][0]) == 3

    def test_elastic_net_summary_string_cols(self, df_regression):
        """Test elastic_net_summary with string column names."""
        result = df_regression.select(
            ps.elastic_net_summary("y", "x1", "x2").alias("coef")
        )
        assert len(result["coef"][0]) == 3
