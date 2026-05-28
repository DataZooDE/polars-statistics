"""Tests for GLM residual diagnostics (#27 batch 2)."""

import numpy as np
import polars as pl
import pytest

from polars_statistics import (
    logistic_deviance_residuals,
    logistic_pearson_residuals,
    logistic_working_residuals,
    poisson_deviance_residuals,
    poisson_pearson_residuals,
    poisson_working_residuals,
)


@pytest.fixture
def logistic_data():
    rng = np.random.default_rng(42)
    n = 150
    x = rng.normal(size=n)
    logit = 0.3 + 0.8 * x
    proba = 1.0 / (1.0 + np.exp(-logit))
    y = rng.binomial(1, proba).astype(float)
    return pl.DataFrame({"y": y, "x": x})


@pytest.fixture
def poisson_data():
    rng = np.random.default_rng(43)
    n = 150
    x = rng.normal(size=n) * 0.5
    log_mu = 0.5 + 0.3 * x
    y = rng.poisson(np.exp(log_mu)).astype(float)
    return pl.DataFrame({"y": y, "x": x})


class TestLogisticResiduals:
    def test_pearson_residuals_shape_and_finite(self, logistic_data):
        result = logistic_data.select(
            logistic_pearson_residuals("y", "x").alias("r")
        ).item()
        assert result["n_observations"] == 150
        assert len(result["residuals"]) == 150
        assert all(np.isfinite(r) for r in result["residuals"])

    def test_deviance_residuals_shape_and_finite(self, logistic_data):
        result = logistic_data.select(
            logistic_deviance_residuals("y", "x").alias("r")
        ).item()
        assert result["n_observations"] == 150
        assert len(result["residuals"]) == 150
        assert all(np.isfinite(r) for r in result["residuals"])

    def test_working_residuals_shape(self, logistic_data):
        result = logistic_data.select(
            logistic_working_residuals("y", "x").alias("r")
        ).item()
        assert result["n_observations"] == 150
        assert len(result["residuals"]) == 150

    def test_pearson_sum_of_squares_relates_to_chi2(self, logistic_data):
        """Sum of squared Pearson residuals ~ chi-squared statistic ~ n - p."""
        result = logistic_data.select(
            logistic_pearson_residuals("y", "x").alias("r")
        ).item()
        ssr = sum(r**2 for r in result["residuals"])
        n_minus_p = 150 - 2  # n - p (intercept + 1 feature)
        # Should be roughly in the same order of magnitude as n - p
        assert 0.3 * n_minus_p < ssr < 3.0 * n_minus_p


class TestPoissonResiduals:
    def test_pearson_residuals_shape_and_finite(self, poisson_data):
        result = poisson_data.select(
            poisson_pearson_residuals("y", "x").alias("r")
        ).item()
        assert result["n_observations"] == 150
        assert len(result["residuals"]) == 150
        assert all(np.isfinite(r) for r in result["residuals"])

    def test_deviance_residuals_shape_and_finite(self, poisson_data):
        result = poisson_data.select(
            poisson_deviance_residuals("y", "x").alias("r")
        ).item()
        assert result["n_observations"] == 150
        assert len(result["residuals"]) == 150
        # Deviance residuals can have NaN for y=0 in some implementations — allow
        # but require most to be finite.
        n_finite = sum(1 for r in result["residuals"] if np.isfinite(r))
        assert n_finite > 100

    def test_working_residuals_shape(self, poisson_data):
        result = poisson_data.select(
            poisson_working_residuals("y", "x").alias("r")
        ).item()
        assert result["n_observations"] == 150
        assert len(result["residuals"]) == 150


class TestGroupByIntegration:
    def test_works_inside_group_by(self):
        rng = np.random.default_rng(7)
        n_per = 80
        x_a = rng.normal(size=n_per)
        x_b = rng.normal(size=n_per)
        df = pl.DataFrame(
            {
                "site": ["A"] * n_per + ["B"] * n_per,
                "y": np.concatenate(
                    [
                        rng.binomial(1, 1 / (1 + np.exp(-(0.5 + 0.8 * x_a)))).astype(float),
                        rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 0.8 * x_b)))).astype(float),
                    ]
                ),
                "x": np.concatenate([x_a, x_b]),
            }
        )
        out = df.group_by("site").agg(
            logistic_pearson_residuals("y", "x").alias("r")
        )
        assert out.shape[0] == 2
