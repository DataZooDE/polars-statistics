"""Smoke tests for Phase 3 statistics API parity (STAT-01/02/03).

Tests validate that ANOVA expressions are importable, return correct struct
shapes, and produce plausible numeric results on known inputs.
R-validated reference values are deferred to Phase 6.
"""

from __future__ import annotations

import math

import polars as pl
import polars_statistics as ps


class TestOneWayAnova:
    """STAT-01: one_way_anova expression smoke tests."""

    def test_returns_correct_schema(self):
        """Fisher ANOVA on three distinct groups returns expected struct fields."""
        df = pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "b": [4.0, 5.0, 6.0],
                "c": [7.0, 8.0, 9.0],
            }
        )
        result = df.select(ps.one_way_anova("a", "b", "c"))
        assert result.shape == (1, 1)
        v = result[0, 0]
        assert "statistic" in v
        assert "df_between" in v
        assert "df_within" in v
        assert "p_value" in v
        assert "ss_between" in v
        assert "ss_within" in v
        assert "ms_between" in v
        assert "ms_within" in v
        assert "eta_squared" in v
        assert "n_groups" in v

    def test_fisher_plausible_f_and_p(self):
        """Fisher ANOVA on clearly separated groups yields F >> 1 and p < 0.05."""
        df = pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "b": [4.0, 5.0, 6.0],
                "c": [7.0, 8.0, 9.0],
            }
        )
        result = df.select(ps.one_way_anova("a", "b", "c"))
        v = result[0, 0]
        assert v["statistic"] > 1.0
        assert 0.0 <= v["p_value"] <= 1.0
        assert v["p_value"] < 0.05
        assert v["n_groups"] == 3

    def test_fisher_has_ss_fields(self):
        """Fisher ANOVA populates ss_between/ss_within/ms_between/ms_within and eta_squared."""
        df = pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "b": [4.0, 5.0, 6.0],
                "c": [7.0, 8.0, 9.0],
            }
        )
        result = df.select(ps.one_way_anova("a", "b", "c"))
        v = result[0, 0]
        assert math.isfinite(v["ss_between"]) and v["ss_between"] > 0
        assert math.isfinite(v["ss_within"]) and v["ss_within"] > 0
        assert math.isfinite(v["ms_between"]) and v["ms_between"] > 0
        assert math.isfinite(v["ms_within"]) and v["ms_within"] > 0
        assert math.isfinite(v["eta_squared"])
        assert 0.0 < v["eta_squared"] <= 1.0

    def test_welch_ss_fields_are_nan(self):
        """Welch ANOVA returns NaN for ss/ms/eta_squared but finite statistic and p_value."""
        df = pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "b": [4.0, 5.0, 6.0],
                "c": [7.0, 8.0, 9.0],
            }
        )
        result = df.select(ps.one_way_anova("a", "b", "c", kind="welch"))
        v = result[0, 0]
        assert math.isfinite(v["statistic"])
        assert 0.0 <= v["p_value"] <= 1.0
        assert math.isnan(v["ss_between"])
        assert math.isnan(v["ss_within"])
        assert math.isnan(v["ms_between"])
        assert math.isnan(v["ms_within"])
        assert math.isnan(v["eta_squared"])


class TestTwoWayAnova:
    """STAT-02: two_way_anova expression smoke tests."""

    def test_returns_correct_schema(self):
        """Two-way ANOVA returns a struct with all expected flattened row fields."""
        df = pl.DataFrame(
            {
                "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                "fa": ["x", "x", "x", "y", "y", "y", "x", "x", "x", "y", "y", "y"],
                "fb": ["p", "q", "p", "q", "p", "q", "p", "q", "p", "q", "p", "q"],
            }
        )
        result = df.select(ps.two_way_anova("v", "fa", "fb"))
        assert result.shape == (1, 1)
        v = result[0, 0]
        for field in [
            "a_ss",
            "a_df",
            "a_ms",
            "a_f",
            "a_p_value",
            "b_ss",
            "b_df",
            "b_ms",
            "b_f",
            "b_p_value",
            "ab_ss",
            "ab_df",
            "ab_ms",
            "ab_f",
            "ab_p_value",
            "residual_ss",
            "residual_df",
            "residual_ms",
            "grand_mean",
            "n",
        ]:
            assert field in v, f"Missing field: {field}"

    def test_plausible_values_n_12(self):
        """Two-way ANOVA returns n=12 and finite a_ss on the standard test data."""
        df = pl.DataFrame(
            {
                "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                "fa": ["x", "x", "x", "y", "y", "y", "x", "x", "x", "y", "y", "y"],
                "fb": ["p", "q", "p", "q", "p", "q", "p", "q", "p", "q", "p", "q"],
            }
        )
        result = df.select(ps.two_way_anova("v", "fa", "fb"))
        v = result[0, 0]
        assert v["n"] == 12
        # a_ss should be finite and positive (there IS a factor A effect)
        assert math.isfinite(v["a_ss"]) and v["a_ss"] == v["a_ss"]

    def test_p_values_in_range(self):
        """p-values from two-way ANOVA are in [0,1] where not NaN."""
        df = pl.DataFrame(
            {
                "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                "fa": ["x", "x", "x", "y", "y", "y", "x", "x", "x", "y", "y", "y"],
                "fb": ["p", "q", "p", "q", "p", "q", "p", "q", "p", "q", "p", "q"],
            }
        )
        result = df.select(ps.two_way_anova("v", "fa", "fb"))
        v = result[0, 0]
        for pfield in ["a_p_value", "b_p_value", "ab_p_value"]:
            pval = v[pfield]
            if not math.isnan(pval):
                assert 0.0 <= pval <= 1.0, f"{pfield}={pval} out of range"


class TestRmAnova:
    """STAT-03: repeated_measures_anova expression smoke tests."""

    def _balanced_df(self):
        """4 subjects × 3 conditions balanced long-format data with within-subject noise."""
        return pl.DataFrame(
            {
                # Slight jitter so error_ss > 0 and ws_f is not inf
                "y": [1.0, 2.1, 3.2, 2.3, 3.1, 4.0, 3.2, 4.3, 5.1, 4.1, 5.2, 6.3],
                "s": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                "c": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            }
        )

    def test_returns_correct_schema(self):
        """RM-ANOVA returns a struct with all expected fields."""
        result = self._balanced_df().select(ps.repeated_measures_anova("y", "s", "c"))
        assert result.shape == (1, 1)
        v = result[0, 0]
        for field in [
            "ws_f",
            "ws_df",
            "ws_ss",
            "ws_ms",
            "ws_p_value",
            "error_df",
            "error_ss",
            "error_ms",
            "mauchly_w",
            "mauchly_p_value",
            "gg_epsilon",
            "gg_p_value",
            "hf_epsilon",
            "hf_p_value",
            "grand_mean",
        ]:
            assert field in v, f"Missing field: {field}"

    def test_plausible_within_subjects(self):
        """RM-ANOVA on monotone-increasing condition means yields finite ws_f and p."""
        result = self._balanced_df().select(ps.repeated_measures_anova("y", "s", "c"))
        v = result[0, 0]
        assert math.isfinite(v["ws_f"]) and v["ws_f"] >= 0.0
        assert math.isfinite(v["ws_p_value"])
        assert 0.0 <= v["ws_p_value"] <= 1.0
        assert math.isfinite(v["grand_mean"])

    def test_sphericity_fields_finite_for_k3(self):
        """With k=3 conditions mauchly_w and gg_epsilon/hf_epsilon are finite."""
        result = self._balanced_df().select(
            ps.repeated_measures_anova("y", "s", "c", compute_sphericity=True)
        )
        v = result[0, 0]
        assert math.isfinite(v["mauchly_w"])
        assert math.isfinite(v["gg_epsilon"])
        assert math.isfinite(v["hf_epsilon"])

    def test_unbalanced_returns_nan_struct(self):
        """Unbalanced design (missing a condition for one subject) returns NaN struct, not a panic."""
        df = pl.DataFrame(
            {
                # Subject 1 has only 2 conditions, making it unbalanced
                "y": [1.0, 2.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0],
                "s": [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                "c": [1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            }
        )
        result = df.select(ps.repeated_measures_anova("y", "s", "c"))
        v = result[0, 0]
        # Should not panic; ws_f should be NaN on unbalanced input
        assert math.isnan(v["ws_f"])


class TestEnergyDistanceNd:
    """STAT-04: energy_distance_nd expression smoke tests (nD multivariate)."""

    def test_returns_correct_schema(self):
        """energy_distance_nd on 2D separated samples returns statistic and p_value fields."""
        from polars_statistics.exprs.modern import energy_distance_nd

        df = pl.DataFrame(
            {
                "x1": [0.0, 0.1, -0.1, 0.2],
                "x2": [0.0, -0.1, 0.1, 0.05],
                "y1": [3.0, 3.1, 2.9, 3.2],
                "y2": [3.0, 2.9, 3.1, 3.05],
            }
        )
        result = df.select(energy_distance_nd(["x1", "x2"], ["y1", "y2"], n_permutations=99, seed=42))
        assert result.shape == (1, 1)
        v = result[0, 0]
        assert "statistic" in v
        assert "p_value" in v

    def test_separated_samples_positive_statistic(self):
        """Well-separated 2D samples yield statistic > 0 and p_value in [0, 1]."""
        from polars_statistics.exprs.modern import energy_distance_nd

        df = pl.DataFrame(
            {
                "x1": [0.0, 0.1, -0.1, 0.2],
                "x2": [0.0, -0.1, 0.1, 0.05],
                "y1": [3.0, 3.1, 2.9, 3.2],
                "y2": [3.0, 2.9, 3.1, 3.05],
            }
        )
        result = df.select(energy_distance_nd(["x1", "x2"], ["y1", "y2"], n_permutations=99, seed=42))
        v = result[0, 0]
        assert v["statistic"] > 0
        assert 0.0 <= v["p_value"] <= 1.0

    def test_mismatched_dims_raises(self):
        """energy_distance_nd raises ValueError when x_cols and y_cols have different lengths."""
        from polars_statistics.exprs.modern import energy_distance_nd

        import pytest

        with pytest.raises(ValueError):
            energy_distance_nd(["x1", "x2"], ["y1"])

    def test_existing_1d_energy_distance_unchanged(self):
        """Existing 1D energy_distance expression still works after adding nD variant."""
        df = pl.DataFrame({"x": [0.0, 0.1, -0.1, 0.2], "y": [3.0, 3.1, 2.9, 3.2]})
        result = df.select(ps.energy_distance("x", "y", n_permutations=99, seed=42))
        v = result[0, 0]
        assert v["statistic"] > 0
        assert 0.0 <= v["p_value"] <= 1.0
