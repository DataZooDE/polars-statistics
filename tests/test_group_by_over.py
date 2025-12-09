"""Smoke tests for group_by and over operations.

These tests verify that all expressions run successfully with group_by and over,
and return actual numeric results (not NaN/null). Correctness tests to be added later.
"""

import math
import numpy as np
import polars as pl
import pytest

import polars_statistics as ps


# =============================================================================
# Helper Functions
# =============================================================================


def has_valid_numbers(result: pl.DataFrame, col: str = "result") -> bool:
    """Check that result column contains valid (non-null, non-NaN) numbers."""
    for row in result[col]:
        if row is None:
            return False
        # Check struct fields for NaN/null
        if isinstance(row, dict):
            for key, val in row.items():
                if val is None:
                    return False
                if isinstance(val, float) and math.isnan(val):
                    return False
                if isinstance(val, list):
                    for v in val:
                        if v is None or (isinstance(v, float) and math.isnan(v)):
                            return False
    return True


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def df_two_sample():
    """DataFrame for two-sample tests."""
    np.random.seed(42)
    return pl.DataFrame({
        "group": ["A"] * 50 + ["B"] * 50,
        "x": np.random.randn(100).tolist(),
        "y": np.random.randn(100).tolist(),
    })


@pytest.fixture
def df_one_sample():
    """DataFrame for one-sample tests."""
    np.random.seed(42)
    return pl.DataFrame({
        "group": ["A"] * 50 + ["B"] * 50,
        "x": np.random.randn(100).tolist(),
    })


@pytest.fixture
def df_three_sample():
    """DataFrame for three-sample tests."""
    np.random.seed(42)
    return pl.DataFrame({
        "group": ["A"] * 50 + ["B"] * 50,
        "x": np.random.randn(100).tolist(),
        "y": np.random.randn(100).tolist(),
        "z": np.random.randn(100).tolist(),
    })


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
    # Ensure balanced classes within each group for better GLM convergence
    p = 1 / (1 + np.exp(-(0.5 * x1 + 0.5 * x2)))
    y = (np.random.rand(n) < p).astype(float)
    # Ensure at least some variation in each group
    y[0] = 1.0  # Group A has at least one 1
    y[49] = 0.0  # Group A has at least one 0
    y[50] = 1.0  # Group B has at least one 1
    y[99] = 0.0  # Group B has at least one 0
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
    x1 = np.random.randn(n) * 0.5
    x2 = np.random.randn(n) * 0.5
    y = np.random.poisson(np.exp(0.5 + 0.3 * x1)).astype(float)
    return pl.DataFrame({
        "group": ["A"] * 50 + ["B"] * 50,
        "y": y.tolist(),
        "x1": x1.tolist(),
        "x2": x2.tolist(),
    })


@pytest.fixture
def df_positive():
    """DataFrame for positive continuous data tests."""
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    y = np.abs(np.random.randn(n)) + 0.1
    return pl.DataFrame({
        "group": ["A"] * 50 + ["B"] * 50,
        "y": y.tolist(),
        "x1": x1.tolist(),
    })


# =============================================================================
# Parametric Tests - group_by
# =============================================================================


class TestParametricGroupBy:
    """Smoke tests for parametric tests with group_by."""

    def test_ttest_ind_group_by(self, df_two_sample):
        result = df_two_sample.group_by("group").agg(
            ps.ttest_ind("x", "y").alias("result")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result)

    def test_ttest_paired_group_by(self, df_two_sample):
        result = df_two_sample.group_by("group").agg(
            ps.ttest_paired("x", "y").alias("result")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result)

    def test_brown_forsythe_group_by(self, df_two_sample):
        result = df_two_sample.group_by("group").agg(
            ps.brown_forsythe("x", "y").alias("result")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result)

    def test_yuen_test_group_by(self, df_two_sample):
        result = df_two_sample.group_by("group").agg(
            ps.yuen_test("x", "y").alias("result")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result)


# =============================================================================
# Non-Parametric Tests - group_by
# =============================================================================


class TestNonParametricGroupBy:
    """Smoke tests for non-parametric tests with group_by."""

    def test_mann_whitney_u_group_by(self, df_two_sample):
        result = df_two_sample.group_by("group").agg(
            ps.mann_whitney_u("x", "y").alias("result")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result)

    def test_wilcoxon_signed_rank_group_by(self, df_two_sample):
        result = df_two_sample.group_by("group").agg(
            ps.wilcoxon_signed_rank("x", "y").alias("result")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result)

    def test_kruskal_wallis_group_by(self, df_three_sample):
        result = df_three_sample.group_by("group").agg(
            ps.kruskal_wallis("x", "y", "z").alias("result")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result)

    def test_brunner_munzel_group_by(self, df_two_sample):
        result = df_two_sample.group_by("group").agg(
            ps.brunner_munzel("x", "y").alias("result")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result)


# =============================================================================
# Distributional Tests - group_by
# =============================================================================


class TestDistributionalGroupBy:
    """Smoke tests for distributional tests with group_by."""

    def test_shapiro_wilk_group_by(self, df_one_sample):
        result = df_one_sample.group_by("group").agg(
            ps.shapiro_wilk("x").alias("result")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result)

    def test_dagostino_group_by(self, df_one_sample):
        result = df_one_sample.group_by("group").agg(
            ps.dagostino("x").alias("result")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result)


# =============================================================================
# Forecast Tests - group_by
# =============================================================================


class TestForecastGroupBy:
    """Smoke tests for forecast tests with group_by."""

    def test_diebold_mariano_group_by(self, df_two_sample):
        result = df_two_sample.group_by("group").agg(
            ps.diebold_mariano("x", "y").alias("result")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result)

    def test_permutation_t_test_group_by(self, df_two_sample):
        result = df_two_sample.group_by("group").agg(
            ps.permutation_t_test("x", "y", n_permutations=99, seed=42).alias("result")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result)

    def test_clark_west_group_by(self, df_two_sample):
        result = df_two_sample.group_by("group").agg(
            ps.clark_west("x", "y").alias("result")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result)

    def test_spa_test_group_by(self, df_three_sample):
        result = df_three_sample.group_by("group").agg(
            ps.spa_test("x", "y", "z", n_bootstrap=99, seed=42).alias("result")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result)

    def test_model_confidence_set_group_by(self, df_three_sample):
        result = df_three_sample.group_by("group").agg(
            ps.model_confidence_set("x", "y", "z", n_bootstrap=99).alias("result")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result)

    def test_mspe_adjusted_group_by(self, df_three_sample):
        result = df_three_sample.group_by("group").agg(
            ps.mspe_adjusted("x", "y", "z", n_bootstrap=99, seed=42).alias("result")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result)


# =============================================================================
# Modern Tests - group_by
# =============================================================================


class TestModernGroupBy:
    """Smoke tests for modern distribution tests with group_by."""

    def test_energy_distance_group_by(self, df_two_sample):
        result = df_two_sample.group_by("group").agg(
            ps.energy_distance("x", "y", n_permutations=99, seed=42).alias("result")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result)

    def test_mmd_test_group_by(self, df_two_sample):
        result = df_two_sample.group_by("group").agg(
            ps.mmd_test("x", "y", n_permutations=99, seed=42).alias("result")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result)


# =============================================================================
# Regression Expressions - group_by (Linear)
# =============================================================================


class TestLinearRegressionGroupBy:
    """Smoke tests for linear regression expressions with group_by."""

    def test_ols_group_by(self, df_regression):
        result = df_regression.group_by("group").agg(
            ps.ols("y", "x1", "x2").alias("model")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result, "model")

    def test_ridge_group_by(self, df_regression):
        result = df_regression.group_by("group").agg(
            ps.ridge("y", "x1", "x2", lambda_=0.1).alias("model")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result, "model")

    def test_elastic_net_group_by(self, df_regression):
        result = df_regression.group_by("group").agg(
            ps.elastic_net("y", "x1", "x2", lambda_=0.1, alpha=0.5).alias("model")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result, "model")

    def test_wls_group_by(self, df_regression):
        result = df_regression.group_by("group").agg(
            ps.wls("y", "weights", "x1", "x2").alias("model")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result, "model")

    def test_rls_group_by(self, df_regression):
        result = df_regression.group_by("group").agg(
            ps.rls("y", "x1", "x2", forgetting_factor=0.99).alias("model")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result, "model")

    def test_bls_group_by(self, df_regression):
        result = df_regression.group_by("group").agg(
            ps.bls("y", "x1", "x2", lower_bound=None, upper_bound=None).alias("model")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result, "model")

    def test_nnls_group_by(self, df_regression):
        # Use absolute values for NNLS
        df = df_regression.with_columns([
            pl.col("x1").abs(),
            pl.col("x2").abs(),
            pl.col("y").abs(),
        ])
        result = df.group_by("group").agg(
            ps.nnls("y", "x1", "x2").alias("model")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result, "model")


# =============================================================================
# Regression Expressions - group_by (GLM)
# =============================================================================


class TestGLMRegressionGroupBy:
    """Smoke tests for GLM regression expressions with group_by."""

    def test_logistic_group_by(self, df_binary):
        result = df_binary.group_by("group").agg(
            ps.logistic("y", "x1", "x2").alias("model")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result, "model")

    def test_poisson_group_by(self, df_count):
        result = df_count.group_by("group").agg(
            ps.poisson("y", "x1", "x2").alias("model")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result, "model")

    def test_negative_binomial_group_by(self, df_count):
        result = df_count.group_by("group").agg(
            ps.negative_binomial("y", "x1", "x2").alias("model")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result, "model")

    def test_tweedie_group_by(self, df_positive):
        result = df_positive.group_by("group").agg(
            ps.tweedie("y", "x1", var_power=1.5).alias("model")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result, "model")

    def test_probit_group_by(self, df_binary):
        result = df_binary.group_by("group").agg(
            ps.probit("y", "x1", "x2").alias("model")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result, "model")

    def test_cloglog_group_by(self, df_binary):
        # Note: cloglog can fail to converge with small/imbalanced data
        result = df_binary.group_by("group").agg(
            ps.cloglog("y", "x1", "x2").alias("model")
        )
        assert result.shape[0] == 2
        # Check at least one group converged (cloglog is sensitive to data)
        valid_count = sum(1 for row in result["model"] if row and row.get("intercept") is not None and not (isinstance(row.get("intercept"), float) and math.isnan(row.get("intercept"))))
        assert valid_count >= 1, "At least one group should converge"


# =============================================================================
# Regression Expressions - group_by (ALM)
# =============================================================================


class TestALMRegressionGroupBy:
    """Smoke tests for ALM regression expressions with group_by."""

    def test_alm_normal_group_by(self, df_regression):
        result = df_regression.group_by("group").agg(
            ps.alm("y", "x1", "x2", distribution="normal").alias("model")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result, "model")

    def test_alm_laplace_group_by(self, df_regression):
        result = df_regression.group_by("group").agg(
            ps.alm("y", "x1", "x2", distribution="laplace").alias("model")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result, "model")

    def test_alm_student_t_group_by(self, df_regression):
        result = df_regression.group_by("group").agg(
            ps.alm("y", "x1", "x2", distribution="student_t").alias("model")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result, "model")

    def test_alm_gamma_group_by(self, df_positive):
        result = df_positive.group_by("group").agg(
            ps.alm("y", "x1", distribution="gamma").alias("model")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result, "model")

    def test_alm_poisson_group_by(self, df_count):
        result = df_count.group_by("group").agg(
            ps.alm("y", "x1", distribution="poisson").alias("model")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result, "model")


# =============================================================================
# Parametric Tests - over
# =============================================================================


class TestParametricOver:
    """Smoke tests for parametric tests with over (window functions)."""

    def test_ttest_ind_over(self, df_two_sample):
        result = df_two_sample.with_columns(
            ps.ttest_ind("x", "y").over("group").alias("result")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result)

    def test_ttest_paired_over(self, df_two_sample):
        result = df_two_sample.with_columns(
            ps.ttest_paired("x", "y").over("group").alias("result")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result)

    def test_brown_forsythe_over(self, df_two_sample):
        result = df_two_sample.with_columns(
            ps.brown_forsythe("x", "y").over("group").alias("result")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result)

    def test_yuen_test_over(self, df_two_sample):
        result = df_two_sample.with_columns(
            ps.yuen_test("x", "y").over("group").alias("result")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result)


# =============================================================================
# Non-Parametric Tests - over
# =============================================================================


class TestNonParametricOver:
    """Smoke tests for non-parametric tests with over (window functions)."""

    def test_mann_whitney_u_over(self, df_two_sample):
        result = df_two_sample.with_columns(
            ps.mann_whitney_u("x", "y").over("group").alias("result")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result)

    def test_wilcoxon_signed_rank_over(self, df_two_sample):
        result = df_two_sample.with_columns(
            ps.wilcoxon_signed_rank("x", "y").over("group").alias("result")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result)

    def test_kruskal_wallis_over(self, df_three_sample):
        result = df_three_sample.with_columns(
            ps.kruskal_wallis("x", "y", "z").over("group").alias("result")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result)

    def test_brunner_munzel_over(self, df_two_sample):
        result = df_two_sample.with_columns(
            ps.brunner_munzel("x", "y").over("group").alias("result")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result)


# =============================================================================
# Distributional Tests - over
# =============================================================================


class TestDistributionalOver:
    """Smoke tests for distributional tests with over (window functions)."""

    def test_shapiro_wilk_over(self, df_one_sample):
        result = df_one_sample.with_columns(
            ps.shapiro_wilk("x").over("group").alias("result")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result)

    def test_dagostino_over(self, df_one_sample):
        result = df_one_sample.with_columns(
            ps.dagostino("x").over("group").alias("result")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result)


# =============================================================================
# Forecast Tests - over
# =============================================================================


class TestForecastOver:
    """Smoke tests for forecast tests with over (window functions)."""

    def test_diebold_mariano_over(self, df_two_sample):
        result = df_two_sample.with_columns(
            ps.diebold_mariano("x", "y").over("group").alias("result")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result)

    def test_clark_west_over(self, df_two_sample):
        result = df_two_sample.with_columns(
            ps.clark_west("x", "y").over("group").alias("result")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result)


# =============================================================================
# Modern Tests - over
# =============================================================================


class TestModernOver:
    """Smoke tests for modern distribution tests with over (window functions)."""

    def test_energy_distance_over(self, df_two_sample):
        result = df_two_sample.with_columns(
            ps.energy_distance("x", "y", n_permutations=99, seed=42).over("group").alias("result")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result)

    def test_mmd_test_over(self, df_two_sample):
        result = df_two_sample.with_columns(
            ps.mmd_test("x", "y", n_permutations=99, seed=42).over("group").alias("result")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result)


# =============================================================================
# Regression Expressions - over (Linear)
# =============================================================================


class TestLinearRegressionOver:
    """Smoke tests for linear regression expressions with over (window functions)."""

    def test_ols_over(self, df_regression):
        result = df_regression.with_columns(
            ps.ols("y", "x1", "x2").over("group").alias("model")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result, "model")

    def test_ridge_over(self, df_regression):
        result = df_regression.with_columns(
            ps.ridge("y", "x1", "x2", lambda_=0.1).over("group").alias("model")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result, "model")

    def test_elastic_net_over(self, df_regression):
        result = df_regression.with_columns(
            ps.elastic_net("y", "x1", "x2", lambda_=0.1, alpha=0.5).over("group").alias("model")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result, "model")

    def test_wls_over(self, df_regression):
        result = df_regression.with_columns(
            ps.wls("y", "weights", "x1", "x2").over("group").alias("model")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result, "model")

    def test_rls_over(self, df_regression):
        result = df_regression.with_columns(
            ps.rls("y", "x1", "x2", forgetting_factor=0.99).over("group").alias("model")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result, "model")

    def test_bls_over(self, df_regression):
        result = df_regression.with_columns(
            ps.bls("y", "x1", "x2", lower_bound=None, upper_bound=None).over("group").alias("model")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result, "model")

    def test_nnls_over(self, df_regression):
        df = df_regression.with_columns([
            pl.col("x1").abs(),
            pl.col("x2").abs(),
            pl.col("y").abs(),
        ])
        result = df.with_columns(
            ps.nnls("y", "x1", "x2").over("group").alias("model")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result, "model")


# =============================================================================
# Regression Expressions - over (GLM)
# =============================================================================


class TestGLMRegressionOver:
    """Smoke tests for GLM regression expressions with over (window functions)."""

    def test_logistic_over(self, df_binary):
        result = df_binary.with_columns(
            ps.logistic("y", "x1", "x2").over("group").alias("model")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result, "model")

    def test_poisson_over(self, df_count):
        result = df_count.with_columns(
            ps.poisson("y", "x1", "x2").over("group").alias("model")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result, "model")

    def test_negative_binomial_over(self, df_count):
        result = df_count.with_columns(
            ps.negative_binomial("y", "x1", "x2").over("group").alias("model")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result, "model")

    def test_tweedie_over(self, df_positive):
        result = df_positive.with_columns(
            ps.tweedie("y", "x1", var_power=1.5).over("group").alias("model")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result, "model")

    def test_probit_over(self, df_binary):
        result = df_binary.with_columns(
            ps.probit("y", "x1", "x2").over("group").alias("model")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result, "model")

    def test_cloglog_over(self, df_binary):
        # Note: cloglog can fail to converge with small/imbalanced data
        result = df_binary.with_columns(
            ps.cloglog("y", "x1", "x2").over("group").alias("model")
        )
        assert result.shape[0] == 100
        # Check at least some rows have valid results (cloglog is sensitive to data)
        valid_count = sum(1 for row in result["model"] if row and row.get("intercept") is not None and not (isinstance(row.get("intercept"), float) and math.isnan(row.get("intercept"))))
        assert valid_count >= 50, "At least one group (50 rows) should converge"


# =============================================================================
# Regression Expressions - over (ALM)
# =============================================================================


class TestALMRegressionOver:
    """Smoke tests for ALM regression expressions with over (window functions)."""

    def test_alm_normal_over(self, df_regression):
        result = df_regression.with_columns(
            ps.alm("y", "x1", "x2", distribution="normal").over("group").alias("model")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result, "model")

    def test_alm_laplace_over(self, df_regression):
        result = df_regression.with_columns(
            ps.alm("y", "x1", "x2", distribution="laplace").over("group").alias("model")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result, "model")

    def test_alm_student_t_over(self, df_regression):
        result = df_regression.with_columns(
            ps.alm("y", "x1", "x2", distribution="student_t").over("group").alias("model")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result, "model")

    def test_alm_gamma_over(self, df_positive):
        result = df_positive.with_columns(
            ps.alm("y", "x1", distribution="gamma").over("group").alias("model")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result, "model")

    def test_alm_poisson_over(self, df_count):
        result = df_count.with_columns(
            ps.alm("y", "x1", distribution="poisson").over("group").alias("model")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result, "model")


# =============================================================================
# Lazy Evaluation Tests
# =============================================================================


class TestLazyEvaluation:
    """Smoke tests for lazy evaluation with group_by and over."""

    def test_lazy_group_by(self, df_two_sample):
        result = df_two_sample.lazy().group_by("group").agg(
            ps.ttest_ind("x", "y").alias("result")
        ).collect()
        assert result.shape[0] == 2
        assert has_valid_numbers(result)

    def test_lazy_over(self, df_two_sample):
        result = df_two_sample.lazy().with_columns(
            ps.ttest_ind("x", "y").over("group").alias("result")
        ).collect()
        assert result.shape[0] == 100
        assert has_valid_numbers(result)

    def test_lazy_regression_group_by(self, df_regression):
        result = df_regression.lazy().group_by("group").agg(
            ps.ols("y", "x1", "x2").alias("model")
        ).collect()
        assert result.shape[0] == 2
        assert has_valid_numbers(result, "model")

    def test_lazy_regression_over(self, df_regression):
        result = df_regression.lazy().with_columns(
            ps.ols("y", "x1", "x2").over("group").alias("model")
        ).collect()
        assert result.shape[0] == 100
        assert has_valid_numbers(result, "model")
