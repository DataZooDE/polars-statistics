"""Multicollinearity + dispersion diagnostics tests (#27 batch 4)."""

import numpy as np
import polars as pl
import pytest

from polars_statistics import (
    generalized_vif,
    high_vif_predictors,
    pearson_chi_squared_logistic,
    pearson_chi_squared_poisson,
)


# ----------------------------------------------------------------------------
# high_vif_predictors
# ----------------------------------------------------------------------------


class TestHighVifPredictors:
    def test_flags_all_collinear(self):
        """Three highly-collinear columns: all three should be flagged."""
        rng = np.random.default_rng(0)
        n = 200
        z = rng.normal(size=n)
        # All three columns are tiny perturbations of z → strong multicollinearity.
        df = pl.DataFrame(
            {
                "x1": z + rng.normal(scale=0.01, size=n),
                "x2": z + rng.normal(scale=0.01, size=n),
                "x3": z + rng.normal(scale=0.01, size=n),
            }
        )
        result = df.select(
            high_vif_predictors("x1", "x2", "x3", threshold=5.0).alias("v")
        ).item()
        assert result["n_features"] == 3
        assert result["n_high"] == 3
        assert all(result["is_high"])

    def test_independent_columns_not_flagged(self):
        """Three independent columns at threshold 10 should flag none."""
        rng = np.random.default_rng(1)
        n = 200
        df = pl.DataFrame(
            {
                "x1": rng.normal(size=n),
                "x2": rng.normal(size=n),
                "x3": rng.normal(size=n),
            }
        )
        result = df.select(
            high_vif_predictors("x1", "x2", "x3", threshold=10.0).alias("v")
        ).item()
        assert result["n_features"] == 3
        assert result["n_high"] == 0
        assert not any(result["is_high"])

    def test_default_threshold_is_ten(self):
        """Default threshold value is 10.0."""
        rng = np.random.default_rng(2)
        n = 200
        df = pl.DataFrame(
            {
                "x1": rng.normal(size=n),
                "x2": rng.normal(size=n),
            }
        )
        default = df.select(high_vif_predictors("x1", "x2").alias("v")).item()
        explicit = df.select(
            high_vif_predictors("x1", "x2", threshold=10.0).alias("v")
        ).item()
        assert default["n_high"] == explicit["n_high"]


# ----------------------------------------------------------------------------
# generalized_vif
# ----------------------------------------------------------------------------


class TestGeneralizedVif:
    def test_two_groups_returns_two_values(self):
        """5 columns split into groups [2, 3] → 2 GVIF values."""
        rng = np.random.default_rng(3)
        n = 200
        df = pl.DataFrame(
            {f"x{i}": rng.normal(size=n) for i in range(1, 6)}
        )
        result = df.select(
            generalized_vif(
                "x1", "x2", "x3", "x4", "x5", group_sizes=[2, 3]
            ).alias("g")
        ).item()
        assert result["n_groups"] == 2
        assert len(result["gvif"]) == 2
        # Independent columns → GVIF close to 1 for each group.
        for v in result["gvif"]:
            assert v > 0.0
            assert v < 5.0

    def test_mismatched_sizes_returns_nan(self):
        """If group_sizes does not sum to n_features the result is empty."""
        rng = np.random.default_rng(4)
        n = 200
        df = pl.DataFrame(
            {f"x{i}": rng.normal(size=n) for i in range(1, 4)}
        )
        result = df.select(
            generalized_vif("x1", "x2", "x3", group_sizes=[2, 2]).alias("g")
        ).item()
        # NaN output yields n_groups == 0.
        assert result["n_groups"] == 0


# ----------------------------------------------------------------------------
# pearson_chi_squared (logistic / poisson)
# ----------------------------------------------------------------------------


def _make_logistic_data(n: int = 200, seed: int = 7):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    eta = 0.3 + 0.8 * x1 - 0.5 * x2
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p).astype(float)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2})


def _make_poisson_data(n: int = 200, seed: int = 8):
    """Modest-signal Poisson counts that converge quickly (cf. test_glm_lambda)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n) * 0.5
    x2 = rng.normal(size=n) * 0.5
    eta = 0.3 + 0.2 * x1 - 0.1 * x2
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(float)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2})


class TestPearsonChiSquaredLogistic:
    def test_struct_shape(self):
        df = _make_logistic_data()
        result = df.select(
            pearson_chi_squared_logistic("y", "x1", "x2").alias("c")
        ).item()
        assert result["n_observations"] == 200
        # df_resid = n - p where p = 3 (intercept + 2 features).
        assert result["df_resid"] == 197
        assert result["chi_squared"] > 0.0
        assert np.isfinite(result["chi_squared"])

    def test_ratio_within_order_of_magnitude(self):
        """For a well-specified model, X² / df_resid is roughly 1."""
        df = _make_logistic_data()
        result = df.select(
            pearson_chi_squared_logistic("y", "x1", "x2").alias("c")
        ).item()
        ratio = result["chi_squared"] / result["df_resid"]
        assert 0.1 < ratio < 10.0, f"ratio = {ratio}"


class TestPearsonChiSquaredPoisson:
    def test_struct_shape(self):
        df = _make_poisson_data()
        result = df.select(
            pearson_chi_squared_poisson("y", "x1", "x2").alias("c")
        ).item()
        assert result["n_observations"] == 200
        assert result["df_resid"] == 197
        assert result["chi_squared"] > 0.0
        assert np.isfinite(result["chi_squared"])

    def test_ratio_within_order_of_magnitude(self):
        df = _make_poisson_data()
        result = df.select(
            pearson_chi_squared_poisson("y", "x1", "x2").alias("c")
        ).item()
        ratio = result["chi_squared"] / result["df_resid"]
        assert 0.1 < ratio < 10.0, f"ratio = {ratio}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
