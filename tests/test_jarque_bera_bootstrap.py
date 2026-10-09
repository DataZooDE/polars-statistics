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
