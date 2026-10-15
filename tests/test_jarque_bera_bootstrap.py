"""R-reference Jarque-Bera tests and percentile bootstrap CI behaviour/reproducibility tests."""

from __future__ import annotations

import numpy as np
import polars as pl
import polars_statistics as ps
import pytest

# R `tseries::jarque.bera.test(x)` reference values (also pinned as the upstream
# anofox-statistics doctest fixture).
JB_X = [2.1, 3.4, 1.9, 5.6, 4.4, 3.3, 2.8, 9.7, 3.0, 4.1]
JB_STATISTIC = 6.1833442919626558
JB_P_VALUE = 0.045425932072353667
JB_SKEWNESS = 1.6519836946632753
JB_KURTOSIS = 1.980864909825267


class TestJarqueBera:
    """R `tseries::jarque.bera.test` reference tests."""

    def test_matches_r_tseries(self):
        df = pl.DataFrame({"x": JB_X})
        result = df.select(ps.jarque_bera("x").alias("jb"))["jb"][0]
        assert result["statistic"] == pytest.approx(JB_STATISTIC, abs=1e-10)
        assert result["p_value"] == pytest.approx(JB_P_VALUE, abs=1e-12)
        assert result["skewness"] == pytest.approx(JB_SKEWNESS, abs=1e-10)
        assert result["kurtosis"] == pytest.approx(JB_KURTOSIS, abs=1e-10)
        assert result["n"] == 10

    def test_field_order_and_n_dtype(self):
        df = pl.DataFrame({"x": JB_X})
        schema = df.select(ps.jarque_bera("x").alias("jb")).schema["jb"]
        assert [f.name for f in schema.fields] == [
            "statistic",
            "p_value",
            "skewness",
            "kurtosis",
            "n",
        ]
        assert schema.fields[-1].dtype == pl.UInt32

    def test_fewer_than_three_values_is_nan(self):
        df = pl.DataFrame({"x": [1.0, 2.0]})
        result = df.select(ps.jarque_bera("x").alias("jb"))["jb"][0]
        assert np.isnan(result["statistic"])
        assert np.isnan(result["p_value"])
        assert np.isnan(result["skewness"])
        assert np.isnan(result["kurtosis"])
        assert result["n"] == 0

    def test_constant_data_is_nan(self):
        df = pl.DataFrame({"x": [3.0] * 10})
        result = df.select(ps.jarque_bera("x").alias("jb"))["jb"][0]
        assert np.isnan(result["statistic"])
        assert result["n"] == 0

    def test_group_by_matches_per_slice(self):
        df = pl.DataFrame(
            {
                "group": ["a"] * 10 + ["b"] * 10,
                "x": JB_X + [v * 2 for v in JB_X],
            }
        )
        result = (
            df.group_by("group", maintain_order=True)
            .agg(ps.jarque_bera("x").alias("jb"))
            .sort("group")
        )
        a_stat = result["jb"][0]["statistic"]
        b_stat = result["jb"][1]["statistic"]
        assert a_stat == pytest.approx(JB_STATISTIC, abs=1e-10)
        # Scaling x by 2 does not change skewness/kurtosis/statistic (scale-invariant).
        assert b_stat == pytest.approx(JB_STATISTIC, abs=1e-10)

    def test_exported_and_typed(self):
        assert "jarque_bera" in ps.__all__
        assert hasattr(ps, "jarque_bera")


# x = [1, 2, ..., 10] (float), used throughout the bootstrap tests below.
BOOT_X = list(range(1, 11))


class TestBootstrapMeanCi:
    """ps.bootstrap_mean_ci: reproducibility, block modes, NaN edge cases."""

    def test_basic_estimate_and_interval(self):
        df = pl.DataFrame({"x": BOOT_X})
        result = df.select(ps.bootstrap_mean_ci("x", seed=42).alias("b"))["b"][0]
        assert result["estimate"] == pytest.approx(5.5, abs=1e-12)
        assert result["se"] > 0
        assert result["ci_lower"] < 5.5 < result["ci_upper"]
        assert result["conf_level"] == 0.95
        assert result["n_bootstrap"] == 1000

    def test_same_seed_is_reproducible(self):
        df = pl.DataFrame({"x": BOOT_X})
        r1 = df.select(ps.bootstrap_mean_ci("x", seed=42).alias("b"))["b"][0]
        r2 = df.select(ps.bootstrap_mean_ci("x", seed=42).alias("b"))["b"][0]
        assert r1 == r2

    def test_different_seed_differs(self):
        df = pl.DataFrame({"x": BOOT_X})
        r1 = df.select(ps.bootstrap_mean_ci("x", seed=42).alias("b"))["b"][0]
        r2 = df.select(ps.bootstrap_mean_ci("x", seed=43).alias("b"))["b"][0]
        assert (
            r1["ci_lower"] != r2["ci_lower"]
            or r1["ci_upper"] != r2["ci_upper"]
            or r1["se"] != r2["se"]
        )

    @pytest.mark.parametrize("block_length", [0, 1, 3])
    def test_block_length_reproducible(self, block_length):
        df = pl.DataFrame({"x": BOOT_X})
        r1 = df.select(
            ps.bootstrap_mean_ci("x", block_length=block_length, seed=7).alias("b")
        )["b"][0]
        r2 = df.select(
            ps.bootstrap_mean_ci("x", block_length=block_length, seed=7).alias("b")
        )["b"][0]
        assert r1 == r2

    def test_block_length_changes_result(self):
        df = pl.DataFrame({"x": BOOT_X})
        r0 = df.select(ps.bootstrap_mean_ci("x", block_length=0, seed=7).alias("b"))["b"][0]
        r3 = df.select(ps.bootstrap_mean_ci("x", block_length=3, seed=7).alias("b"))["b"][0]
        assert r0["ci_lower"] != r3["ci_lower"] or r0["ci_upper"] != r3["ci_upper"] or (
            r0["se"] != r3["se"]
        )

    def test_invalid_params_are_nan(self):
        df = pl.DataFrame({"x": BOOT_X})
        r_n1 = df.select(ps.bootstrap_mean_ci("x", n_bootstrap=1, seed=1).alias("b"))["b"][0]
        assert np.isnan(r_n1["estimate"])
        r_cl = df.select(ps.bootstrap_mean_ci("x", conf_level=1.5, seed=1).alias("b"))["b"][0]
        assert np.isnan(r_cl["estimate"])
        df_single = pl.DataFrame({"x": [1.0]})
        r_single = df_single.select(ps.bootstrap_mean_ci("x", seed=1).alias("b"))["b"][0]
        assert np.isnan(r_single["estimate"])

    def test_group_by_estimate_equals_group_mean(self):
        df = pl.DataFrame(
            {
                "group": ["a"] * 10 + ["b"] * 10,
                "x": BOOT_X + [v + 100 for v in BOOT_X],
            }
        )
        result = (
            df.group_by("group", maintain_order=True)
            .agg(ps.bootstrap_mean_ci("x", seed=1).alias("b"))
            .sort("group")
        )
        assert result["b"][0]["estimate"] == pytest.approx(5.5, abs=1e-12)
        assert result["b"][1]["estimate"] == pytest.approx(105.5, abs=1e-12)

    def test_exported_and_typed(self):
        assert "bootstrap_mean_ci" in ps.__all__
        assert hasattr(ps, "bootstrap_mean_ci")


class TestBootstrapCi:
    """ps.bootstrap_ci: fixed statistic dispatch (mean/median/std)."""

    def test_mean_matches_bootstrap_mean_ci(self):
        df = pl.DataFrame({"x": BOOT_X})
        r_ci = df.select(
            ps.bootstrap_ci("x", statistic="mean", n_bootstrap=2000, seed=7).alias("b")
        )["b"][0]
        r_mean_ci = df.select(
            ps.bootstrap_mean_ci("x", n_bootstrap=2000, block_length=0, seed=7).alias("b")
        )["b"][0]
        assert r_ci == r_mean_ci

    def test_median(self):
        df = pl.DataFrame({"x": BOOT_X})
        result = df.select(
            ps.bootstrap_ci("x", statistic="median", seed=42).alias("b")
        )["b"][0]
        assert result["estimate"] == pytest.approx(5.5, abs=1e-12)
        assert result["ci_lower"] <= 5.5 <= result["ci_upper"]

    def test_std(self):
        df = pl.DataFrame({"x": BOOT_X})
        result = df.select(ps.bootstrap_ci("x", statistic="std", seed=42).alias("b"))["b"][0]
        assert result["estimate"] == pytest.approx(np.std(BOOT_X, ddof=1), abs=1e-12)

    def test_invalid_statistic_raises(self):
        with pytest.raises(ValueError):
            ps.bootstrap_ci("x", statistic="bogus")

    def test_exported_and_typed(self):
        assert "bootstrap_ci" in ps.__all__
        assert hasattr(ps, "bootstrap_ci")
