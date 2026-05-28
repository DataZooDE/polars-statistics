"""Tests for OLS residual diagnostics (issue #27 batch 1)."""

import numpy as np
import polars as pl
import pytest

from polars_statistics import (
    externally_studentized_residuals,
    residual_outliers,
    standardized_residuals,
    studentized_residuals,
)


@pytest.fixture
def clean_data():
    """y = 1 + 2x + small noise, no outliers."""
    rng = np.random.default_rng(0)
    n = 100
    x = rng.normal(size=n)
    y = 1.0 + 2.0 * x + rng.normal(scale=0.3, size=n)
    return pl.DataFrame({"y": y, "x": x})


@pytest.fixture
def outlier_data():
    """Same as clean_data but with planted outliers at indices [3, 17, 42]."""
    rng = np.random.default_rng(1)
    n = 100
    x = rng.normal(size=n)
    y = 1.0 + 2.0 * x + rng.normal(scale=0.3, size=n)
    for idx in (3, 17, 42):
        y[idx] += 20.0
    return pl.DataFrame({"y": y, "x": x})


class TestStandardizedResiduals:
    def test_returns_one_per_row(self, clean_data):
        result = clean_data.select(
            standardized_residuals("y", "x").alias("r")
        ).item()
        assert result["n_observations"] == 100
        assert len(result["residuals"]) == 100

    def test_finite_on_well_conditioned_data(self, clean_data):
        result = clean_data.select(
            standardized_residuals("y", "x").alias("r")
        ).item()
        assert all(np.isfinite(r) for r in result["residuals"])


class TestStudentizedResiduals:
    def test_returns_one_per_row(self, clean_data):
        result = clean_data.select(
            studentized_residuals("y", "x").alias("r")
        ).item()
        assert result["n_observations"] == 100
        assert len(result["residuals"]) == 100

    def test_studentized_larger_than_standardized_in_magnitude(self, clean_data):
        """Studentized residuals divide by sqrt(1 - h_ii) so they're >= standardized."""
        s = clean_data.select(standardized_residuals("y", "x").alias("s")).item()
        t = clean_data.select(studentized_residuals("y", "x").alias("t")).item()
        for std, stu in zip(s["residuals"], t["residuals"]):
            if np.isfinite(std) and np.isfinite(stu):
                assert abs(stu) >= abs(std) - 1e-9


class TestExternallyStudentizedResiduals:
    def test_returns_one_per_row(self, clean_data):
        result = clean_data.select(
            externally_studentized_residuals("y", "x").alias("r")
        ).item()
        assert result["n_observations"] == 100
        assert len(result["residuals"]) == 100

    def test_outliers_have_large_external_residuals(self, outlier_data):
        """Planted outliers should produce |t_i| > 3 (rough guideline)."""
        result = outlier_data.select(
            externally_studentized_residuals("y", "x").alias("r")
        ).item()
        for idx in (3, 17, 42):
            assert abs(result["residuals"][idx]) > 3.0, (
                f"outlier at {idx} has externally studentized resid "
                f"{result['residuals'][idx]:.2f}"
            )


class TestResidualOutliers:
    def test_no_outliers_on_clean_data(self, clean_data):
        result = clean_data.select(
            residual_outliers("y", "x", threshold=3.0).alias("o")
        ).item()
        # Should detect very few outliers with threshold 3
        assert result["n_outliers"] < 5
        assert result["n_observations"] == 100
        assert len(result["is_outlier"]) == 100

    def test_flags_planted_outliers(self, outlier_data):
        result = outlier_data.select(
            residual_outliers("y", "x", threshold=2.0).alias("o")
        ).item()
        # Planted indices should all be flagged
        for idx in (3, 17, 42):
            assert result["is_outlier"][idx], f"missed outlier at {idx}"
        # Total flagged count is at least 3
        assert result["n_outliers"] >= 3

    def test_default_threshold_is_two(self, outlier_data):
        """Default threshold = 2.0 — verify by comparing explicit vs implicit."""
        explicit = outlier_data.select(
            residual_outliers("y", "x", threshold=2.0).alias("o")
        ).item()
        default = outlier_data.select(
            residual_outliers("y", "x").alias("o")
        ).item()
        assert explicit["n_outliers"] == default["n_outliers"]
