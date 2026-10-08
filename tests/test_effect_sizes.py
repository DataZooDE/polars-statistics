"""Tests for effect-size expressions: Cohen's d and rank-biserial families.

Reference samples (shared across this file), matching R `effectsize` package values:

    X  = [5.1, 4.9, 6.2, 5.8, 6.05, 5.5, 5.3, 6.1]           (n=8)
    Y  = [6.5, 7.1, 6.8, 7.4, 6.0, 7.9, 6.6, 7.2, 6.9, 7.05]  (n=10)
    Y8 = Y[:8]                                                (n=8, for paired comparisons)

X has fewer elements than Y, so wherever both are combined into one DataFrame
(for the "x", "y" two-sample tests) X is padded with trailing Nones to the
common length; the builders' `is_finite()` filter drops the padding before the
plugin is called, so the computed estimate is unaffected.
"""

import numpy as np
import polars as pl
import polars_statistics as ps
import pytest
from polars_statistics import MannWhitneyU

X = [5.1, 4.9, 6.2, 5.8, 6.05, 5.5, 5.3, 6.1]
Y = [6.5, 7.1, 6.8, 7.4, 6.0, 7.9, 6.6, 7.2, 6.9, 7.05]
Y8 = Y[:8]
# X padded with two Nones so it aligns with the length-10 Y column.
XP = X + [None, None]


class TestCohensD:
    """R `effectsize::cohens_d` reference tests."""

    def test_student_matches_r(self):
        df = pl.DataFrame({"x": XP, "y": Y})
        result = df.select(ps.cohens_d("x", "y").alias("d"))["d"][0]
        assert result["estimate"] == pytest.approx(-2.606655378290357, abs=1e-10)

    def test_welch_matches_r(self):
        df = pl.DataFrame({"x": XP, "y": Y})
        result = df.select(ps.cohens_d("x", "y", kind="welch").alias("d"))["d"][0]
        assert result["estimate"] == pytest.approx(-2.6164932281884563, abs=1e-10)

    def test_paired_matches_r(self):
        df = pl.DataFrame({"x": X, "y8": Y8})
        result = df.select(ps.cohens_d("x", "y8", kind="paired").alias("d"))["d"][0]
        assert result["estimate"] == pytest.approx(-1.6503380487140236, abs=1e-10)

    def test_paired_with_mu_matches_r(self):
        df = pl.DataFrame({"x": X, "y8": Y8})
        result = df.select(
            ps.cohens_d("x", "y8", kind="paired", mu=-1.0).alias("d")
        )["d"][0]
        assert result["estimate"] == pytest.approx(-0.3988968743337214, abs=1e-10)

    def test_one_sample_matches_r(self):
        df = pl.DataFrame({"x": X})
        result = df.select(ps.cohens_d_one_sample("x", mu=5.0).alias("d"))["d"][0]
        assert result["estimate"] == pytest.approx(1.2593923076823721, abs=1e-10)

    def test_one_sample_constant_is_nan(self):
        df = pl.DataFrame({"x": [3.0] * 5})
        result = df.select(ps.cohens_d_one_sample("x", mu=5.0).alias("d"))["d"][0]
        assert np.isnan(result["estimate"])

    def test_fewer_than_two_observations_is_nan(self):
        df = pl.DataFrame({"x": [1.0] + [None] * 9, "y": Y})
        result = df.select(ps.cohens_d("x", "y").alias("d"))["d"][0]
        assert np.isnan(result["estimate"])

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError):
            ps.cohens_d("x", "y", kind="bogus")

    def test_null_padding_is_dropped(self):
        """Nulls padded onto x (to align column length with y) are dropped by
        the is_finite() filter, so the result matches the un-padded reference."""
        df = pl.DataFrame({"x": XP, "y": Y})
        result = df.select(ps.cohens_d("x", "y").alias("d"))["d"][0]
        assert result["estimate"] == pytest.approx(-2.606655378290357, abs=1e-10)

    def test_schema_is_single_field_estimate(self):
        df = pl.DataFrame({"x": XP, "y": Y})
        schema = df.select(ps.cohens_d("x", "y").alias("d")).schema["d"]
        assert [f.name for f in schema.fields] == ["estimate"]
        assert schema.fields[0].dtype == pl.Float64

    def test_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["a"] * 8 + ["b"] * 8,
                "x": X + X,
                "y": Y8 + Y8,
            }
        )
        result = (
            df.group_by("group", maintain_order=True)
            .agg(ps.cohens_d("x", "y").alias("d"))
            .sort("group")
        )
        expected = pl.DataFrame({"x": X, "y": Y8}).select(
            ps.cohens_d("x", "y").alias("d")
        )["d"][0]["estimate"]
        for row in result["d"]:
            assert row["estimate"] == pytest.approx(expected, abs=1e-10)

    def test_over(self):
        df = pl.DataFrame(
            {
                "group": ["a"] * 8 + ["b"] * 8,
                "x": X + X,
                "y": Y8 + Y8,
            }
        )
        result = df.select(ps.cohens_d("x", "y").over("group").alias("d"))
        values = result["d"].struct.field("estimate")
        assert values.n_unique() == 1


class TestRankBiserial:
    """R `effectsize::rank_biserial` reference tests (note: this library uses
    the opposite sign convention from R's `effectsize::rank_biserial`)."""

    def test_matches_r(self):
        df = pl.DataFrame({"x": XP, "y": Y})
        result = df.select(ps.rank_biserial("x", "y").alias("r"))["r"][0]
        assert result["estimate"] == pytest.approx(0.925, abs=1e-12)

    def test_swapped_columns_flips_sign(self):
        df = pl.DataFrame({"x": XP, "y": Y})
        result = df.select(ps.rank_biserial("y", "x").alias("r"))["r"][0]
        assert result["estimate"] == pytest.approx(-0.925, abs=1e-12)

    def test_with_mu_matches_mann_whitney_u_cross_check(self):
        df = pl.DataFrame({"x": XP, "y": Y})
        result = df.select(ps.rank_biserial("x", "y", mu=-1.0).alias("r"))["r"][0]

        mw = MannWhitneyU(mu=-1.0).fit(np.array(X), np.array(Y))
        cross_check = pl.select(
            ps.rank_biserial_from_u(mw.statistic, 8, 10).alias("r")
        )["r"][0]

        assert result["estimate"] == pytest.approx(cross_check["estimate"], abs=1e-12)

    def test_ties_give_zero(self):
        # x and y are filtered independently (unpaired test), so y is padded
        # with a null to align column heights; the null is dropped by the
        # is_finite() filter, leaving the intended x=[1.0, 1.0], y=[1.0].
        df = pl.DataFrame({"x": [1.0, 1.0], "y": [1.0, None]})
        result = df.select(ps.rank_biserial("x", "y").alias("r"))["r"][0]
        assert result["estimate"] == pytest.approx(0.0, abs=1e-12)

    def test_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["a"] * 8 + ["b"] * 8,
                "x": X + X,
                "y": Y8 + Y8,
            }
        )
        result = (
            df.group_by("group", maintain_order=True)
            .agg(ps.rank_biserial("x", "y").alias("r"))
            .sort("group")
        )
        expected = pl.DataFrame({"x": X, "y": Y8}).select(
            ps.rank_biserial("x", "y").alias("r")
        )["r"][0]["estimate"]
        for row in result["r"]:
            assert row["estimate"] == pytest.approx(expected, abs=1e-10)


class TestMatchedPairsRankBiserial:
    def test_matches_r(self):
        df = pl.DataFrame({"x": X, "y8": Y8})
        result = df.select(ps.matched_pairs_rank_biserial("x", "y8").alias("r"))["r"][0]
        assert result["estimate"] == pytest.approx(0.94444444444444442, abs=1e-12)

    def test_identical_columns_is_nan(self):
        df = pl.DataFrame({"x": X, "y": X})
        result = df.select(ps.matched_pairs_rank_biserial("x", "y").alias("r"))["r"][0]
        assert np.isnan(result["estimate"])

    def test_unequal_lengths_is_nan(self):
        """The Python builder's joint finite-filter always produces equal-length
        x/y, so exercise the Rust-side length-mismatch guard directly through
        the plugin function (bypassing the builder's filter)."""
        from pathlib import Path

        from polars.plugins import register_plugin_function

        lib = Path(ps.__file__).parent
        result = pl.select(
            register_plugin_function(
                plugin_path=lib,
                function_name="pl_matched_pairs_rank_biserial",
                args=[
                    pl.Series("x", X, dtype=pl.Float64),
                    pl.Series("y", Y, dtype=pl.Float64),
                    pl.lit(None, dtype=pl.Float64),
                ],
                returns_scalar=True,
            ).alias("r")
        )["r"][0]
        assert np.isnan(result["estimate"])

    def test_group_by(self):
        df = pl.DataFrame(
            {
                "group": ["a"] * 8 + ["b"] * 8,
                "x": X + X,
                "y8": Y8 + Y8,
            }
        )
        result = (
            df.group_by("group", maintain_order=True)
            .agg(ps.matched_pairs_rank_biserial("x", "y8").alias("r"))
            .sort("group")
        )
        for row in result["r"]:
            assert row["estimate"] == pytest.approx(0.94444444444444442, abs=1e-10)


class TestRankBiserialFromU:
    def test_matches_r(self):
        mw = MannWhitneyU().fit(np.array(X), np.array(Y))
        result = pl.select(ps.rank_biserial_from_u(mw.statistic, 8, 10).alias("r"))["r"][0]
        assert result["estimate"] == pytest.approx(0.925, abs=1e-12)

    def test_zero_sample_is_nan(self):
        result = pl.select(ps.rank_biserial_from_u(3.0, 0, 5).alias("r"))["r"][0]
        assert np.isnan(result["estimate"])
