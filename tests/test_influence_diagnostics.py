"""Influence diagnostics tests (#27 batch 3)."""

import numpy as np
import polars as pl
import pytest

from polars_statistics import (
    dffits,
    high_leverage_points,
    influential_cooks,
    influential_dffits,
)


@pytest.fixture
def outlier_data():
    """Linear data with planted high-influence points at indices [3, 17, 42]."""
    rng = np.random.default_rng(1)
    n = 100
    x = rng.normal(size=n)
    y = 1.0 + 2.0 * x + rng.normal(scale=0.3, size=n)
    # Make these both outliers AND high-leverage by yanking x as well.
    for idx in (3, 17, 42):
        x[idx] = 6.0
        y[idx] += 25.0
    return pl.DataFrame({"y": y, "x": x})


class TestDffits:
    def test_returns_one_per_row(self, outlier_data):
        result = outlier_data.select(dffits("y", "x").alias("d")).item()
        assert result["n_observations"] == 100
        assert len(result["dffits"]) == 100

    def test_planted_indices_have_large_dffits(self, outlier_data):
        result = outlier_data.select(dffits("y", "x").alias("d")).item()
        for idx in (3, 17, 42):
            assert abs(result["dffits"][idx]) > 1.0, (
                f"DFFITS at {idx} = {result['dffits'][idx]:.2f}"
            )


class TestInfluentialCooks:
    def test_flags_planted_indices(self, outlier_data):
        result = outlier_data.select(
            influential_cooks("y", "x").alias("i")
        ).item()
        for idx in (3, 17, 42):
            assert result["is_influential"][idx], f"missed Cook's at {idx}"
        assert result["n_influential"] >= 3
        assert result["n_observations"] == 100

    def test_custom_threshold(self, outlier_data):
        """A higher threshold flags fewer points than the default."""
        default = outlier_data.select(
            influential_cooks("y", "x").alias("i")
        ).item()
        strict = outlier_data.select(
            influential_cooks("y", "x", threshold=1.0).alias("i")
        ).item()
        assert strict["n_influential"] <= default["n_influential"]


class TestInfluentialDffits:
    def test_flags_planted_indices(self, outlier_data):
        result = outlier_data.select(
            influential_dffits("y", "x").alias("i")
        ).item()
        for idx in (3, 17, 42):
            assert result["is_influential"][idx], f"missed DFFITS at {idx}"
        assert result["n_observations"] == 100


class TestHighLeveragePoints:
    def test_flags_extreme_x_values(self, outlier_data):
        """Planted x = 6.0 values should be flagged as high leverage."""
        result = outlier_data.select(
            high_leverage_points("x").alias("h")
        ).item()
        for idx in (3, 17, 42):
            assert result["is_influential"][idx], f"missed leverage at {idx}"
        assert result["n_observations"] == 100

    def test_no_y_argument_works(self):
        """high_leverage_points takes only feature columns."""
        rng = np.random.default_rng(0)
        df = pl.DataFrame({"x1": rng.normal(size=50), "x2": rng.normal(size=50)})
        result = df.select(high_leverage_points("x1", "x2").alias("h")).item()
        assert result["n_observations"] == 50
        assert len(result["is_influential"]) == 50
