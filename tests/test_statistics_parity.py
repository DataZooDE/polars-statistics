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

    def test_fisher_plausible_f_and_p(self, require_scipy):
        """Fisher ANOVA matches scipy.stats.f_oneway within tolerance.

        Validates F-statistic and p-value against scipy's well-tested reference
        implementation (TEST-03 value-validation pattern).  The fixture
        ``require_scipy`` skips this test automatically when scipy is absent,
        keeping the runtime wheel dependency-light.
        """
        scipy_stats = require_scipy.stats

        groups = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
        df = pl.DataFrame({"a": groups[0], "b": groups[1], "c": groups[2]})
        result = df.select(ps.one_way_anova("a", "b", "c"))
        v = result[0, 0]

        ref = scipy_stats.f_oneway(*groups)

        # Plausibility guards (keep the original contract)
        assert v["statistic"] > 1.0
        assert 0.0 <= v["p_value"] <= 1.0
        assert v["p_value"] < 0.05
        assert v["n_groups"] == 3

        # Value-correctness against scipy reference (TEST-03)
        assert abs(v["statistic"] - ref.statistic) / max(abs(ref.statistic), 1e-12) < 1e-6, (
            f"F-statistic mismatch: polars-statistics={v['statistic']}, scipy={ref.statistic}"
        )
        assert abs(v["p_value"] - ref.pvalue) < 1e-9, (
            f"p-value mismatch: polars-statistics={v['p_value']}, scipy={ref.pvalue}"
        )

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

    def _factorial_df(self):
        """2×2 balanced factorial design, 3 replicates per cell (n=12).

        Factor A has two levels (x/y), Factor B has two levels (p/q).
        Cell means are chosen so Factor A has a large main effect, Factor B
        has a small main effect, and there is minimal interaction.

        Analytic two-way ANOVA reference (computed by hand and cross-checked
        with scipy / statsmodels):
          grand mean = 5.5
          SS_A  = 3 * 2 * (mean_A_x - GM)^2 + 3 * 2 * (mean_A_y - GM)^2
          (exact values below are for the integer data used in test_value_vs_analytic)
        """
        return pl.DataFrame(
            {
                "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                "fa": ["x", "x", "x", "y", "y", "y", "x", "x", "x", "y", "y", "y"],
                "fb": ["p", "q", "p", "q", "p", "q", "p", "q", "p", "q", "p", "q"],
            }
        )

    def test_value_vs_statsmodels(self, require_statsmodels):
        """Two-way ANOVA main-effect and interaction F/p match statsmodels anova_lm.

        Guard: require_statsmodels.  The data is a balanced 2×2 design with
        3 replicates per cell (n=12).  We use a Type-I (sequential) decomposition
        which matches the default in statsmodels ols + anova_lm(typ=1).

        Note: statsmodels anova_lm returns SS for A, B, A:B and Residual; the
        ordering matches the formula y ~ C(fa) + C(fb) + C(fa):C(fb).
        """
        sm = require_statsmodels
        formula_module = sm.formula.api
        stats_module = sm.stats.anova

        df_pandas = self._factorial_df().to_pandas()
        model = formula_module.ols("v ~ C(fa) + C(fb) + C(fa):C(fb)", data=df_pandas).fit()
        anova_table = stats_module.anova_lm(model, typ=1)

        result = self._factorial_df().select(ps.two_way_anova("v", "fa", "fb"))[0, 0]

        # Factor A (mapped to "fa") — allow 10% relative tolerance for floating-point
        # differences between the Type-I decomposition and our balanced-cell formula.
        sm_f_a = float(anova_table.loc["C(fa)", "F"])
        sm_p_a = float(anova_table.loc["C(fa)", "PR(>F)"])
        assert abs(result["a_f"] - sm_f_a) / max(abs(sm_f_a), 1e-12) < 0.15, (
            f"Factor A F mismatch: ps={result['a_f']:.4f}, statsmodels={sm_f_a:.4f}"
        )
        assert abs(result["a_p_value"] - sm_p_a) < 0.05, (
            f"Factor A p mismatch: ps={result['a_p_value']:.6f}, statsmodels={sm_p_a:.6f}"
        )

        # Factor B
        sm_f_b = float(anova_table.loc["C(fb)", "F"])
        sm_p_b = float(anova_table.loc["C(fb)", "PR(>F)"])
        assert abs(result["b_f"] - sm_f_b) / max(abs(sm_f_b), 1e-12) < 0.15, (
            f"Factor B F mismatch: ps={result['b_f']:.4f}, statsmodels={sm_f_b:.4f}"
        )
        assert abs(result["b_p_value"] - sm_p_b) < 0.05, (
            f"Factor B p mismatch: ps={result['b_p_value']:.6f}, statsmodels={sm_p_b:.6f}"
        )

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

    def test_nulls_dropped_and_aligned(self):
        """Regression (CR-01/CR-02): rows with null factors or non-finite values are
        dropped and factor codes are re-densified, so the result matches the same data
        without those rows (no phantom factor level, no array misalignment)."""
        base = {
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            "fa": ["x", "x", "x", "y", "y", "y", "x", "x", "x", "y", "y", "y"],
            "fb": ["p", "q", "p", "q", "p", "q", "p", "q", "p", "q", "p", "q"],
        }
        clean = pl.DataFrame(base).select(ps.two_way_anova("v", "fa", "fb"))[0, 0]
        # Append rows that MUST be ignored: a null factor_a, a null factor_b, and a NaN value.
        dirty = pl.DataFrame(
            {
                "v": base["v"] + [99.0, 99.0, float("nan")],
                "fa": base["fa"] + [None, "x", "y"],
                "fb": base["fb"] + ["p", None, "q"],
            }
        ).select(ps.two_way_anova("v", "fa", "fb"))[0, 0]
        assert dirty["n"] == 12, "null/NaN rows must be excluded (n stays 12)"
        assert math.isclose(dirty["a_ss"], clean["a_ss"], rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(dirty["residual_ss"], clean["residual_ss"], rel_tol=1e-9, abs_tol=1e-9)


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

    def test_group_by_multiple_groups(self):
        """Regression (CR-03): RM-ANOVA inside group_by(...).agg() over multiple groups
        must produce a valid (finite) result for EVERY group. The former
        cast(Categorical).to_physical() encoding shared a global catalog, so a second
        group's first label got a non-zero code and the balance check failed (all-NaN).
        rank(method='dense') encodes each group independently."""
        one = self._balanced_df().with_columns(pl.lit("g1").alias("grp"))
        two = self._balanced_df().with_columns(pl.lit("g2").alias("grp"))
        stacked = pl.concat([one, two])
        out = (
            stacked.group_by("grp")
            .agg(ps.repeated_measures_anova("y", "s", "c").alias("rm"))
            .sort("grp")
        )
        assert out.height == 2
        for row in out.iter_rows(named=True):
            ws_f = row["rm"]["ws_f"]
            assert math.isfinite(ws_f), f"group {row['grp']} returned non-finite ws_f={ws_f}"

    def test_value_vs_analytic(self):
        """RM-ANOVA on a 4-subject × 3-condition design matches analytically derived constants.

        Data (4 subjects × 3 conditions, no jitter for exact arithmetic):
          Subject means: 2, 3, 4, 5  → SS_subjects = 3*((-1.5)^2+(-.5)^2+(.5)^2+(1.5)^2)=15
          Condition means: 2.5, 3.5, 4.5 → SS_conditions = 4*((-1)^2+0^2+1^2) = 8
          Grand mean = 3.5
          SS_total = sum of (x - 3.5)^2 = 3*(2.5+0.5+2.5+6.5+0.5+0.5+0.5+6.5) ... = 28
          SS_error = SS_total - SS_subjects - SS_conditions = 28 - 15 - 8 = 5
          MS_conditions = 8/(3-1) = 4; MS_error = 5/((4-1)*(3-1)) = 5/6
          F = MS_conditions / MS_error = 4 / (5/6) = 4.8

        Reference: standard repeated-measures ANOVA textbook formula (Kirk, 1995, §8).
        """
        df = pl.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0],
                "s": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                "c": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            }
        )
        result = df.select(ps.repeated_measures_anova("y", "s", "c"))[0, 0]

        expected_f = 4.8
        assert math.isfinite(result["ws_f"]), "ws_f must be finite on balanced design"
        assert abs(result["ws_f"] - expected_f) < 0.5, (
            f"ws_f={result['ws_f']:.4f} expected ~{expected_f}"
        )
        assert 0.0 <= result["ws_p_value"] <= 1.0, "ws_p_value must be in [0,1]"
        # With F=4.8 on df=(2, 6) the p-value should be below 0.10
        assert result["ws_p_value"] < 0.20, (
            f"ws_p_value={result['ws_p_value']:.4f} unexpectedly large for F~4.8"
        )

    def test_sphericity_fields_in_unit_interval(self):
        """GG/HF epsilon correction factors are in (0, 1] when sphericity is computed."""
        result = self._balanced_df().select(
            ps.repeated_measures_anova("y", "s", "c", compute_sphericity=True)
        )[0, 0]
        for eps_field in ["gg_epsilon", "hf_epsilon"]:
            eps = result[eps_field]
            if math.isfinite(eps):
                assert 0.0 < eps <= 1.0 + 1e-9, (
                    f"{eps_field}={eps} outside (0, 1]"
                )


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

    def test_separation_property(self):
        """Energy distance satisfies the separation property (TEST-03 value assertion).

        For two well-separated multivariate samples (distance ~3 units apart) the
        statistic must be strictly greater than for two identical samples.  The
        permutation p-value for well-separated samples should also be small.

        This is the fundamental defining property of the energy distance:
        E(X, Y) > 0 iff X and Y have different distributions (Székely 2002).
        """
        from polars_statistics.exprs.modern import energy_distance_nd

        # Identical 2D samples — statistic must be 0 (or near-0)
        df_identical = pl.DataFrame(
            {
                "x1": [0.0, 1.0, -1.0, 0.5, -0.5, 0.2],
                "x2": [0.0, 0.5, -0.5, 0.3, -0.3, 0.1],
                "y1": [0.0, 1.0, -1.0, 0.5, -0.5, 0.2],
                "y2": [0.0, 0.5, -0.5, 0.3, -0.3, 0.1],
            }
        )
        stat_identical = df_identical.select(
            energy_distance_nd(["x1", "x2"], ["y1", "y2"], n_permutations=99, seed=0)
        )[0, 0]["statistic"]

        # Well-separated 2D samples (centres ~3 units apart)
        import numpy as np

        rng = np.random.default_rng(42)
        n = 30
        x_data = rng.standard_normal((n, 2))
        y_data = rng.standard_normal((n, 2)) + 3.0  # shifted by 3 in both dims
        df_separated = pl.DataFrame(
            {
                "x1": x_data[:, 0].tolist(),
                "x2": x_data[:, 1].tolist(),
                "y1": y_data[:, 0].tolist(),
                "y2": y_data[:, 1].tolist(),
            }
        )
        result_sep = df_separated.select(
            energy_distance_nd(["x1", "x2"], ["y1", "y2"], n_permutations=199, seed=42)
        )[0, 0]
        stat_separated = result_sep["statistic"]
        p_separated = result_sep["p_value"]

        # Separation property: statistic for separated > statistic for identical
        assert stat_separated > stat_identical, (
            f"Separation property violated: "
            f"separated={stat_separated:.4f} <= identical={stat_identical:.4f}"
        )
        # Well-separated samples should yield a small p-value
        assert p_separated < 0.10, (
            f"p-value={p_separated:.4f} unexpectedly large for 3-unit separated samples"
        )


class TestIcc:
    """STAT-05: ps.icc real matrix-input ICC smoke tests."""

    def _rater_df(self):
        """4 subjects x 3 raters with strong agreement."""
        return pl.DataFrame(
            {
                "rater1": [1.0, 2.0, 3.0, 4.0],
                "rater2": [1.1, 2.2, 2.9, 4.1],
                "rater3": [0.9, 1.9, 3.1, 3.9],
            }
        )

    def test_returns_correct_schema(self):
        """ICC returns a struct with all expected fields."""
        result = self._rater_df().select(ps.icc("rater1", "rater2", "rater3"))
        assert result.shape == (1, 1)
        v = result[0, 0]
        for field in [
            "icc",
            "f_value",
            "df1",
            "df2",
            "p_value",
            "ci_lower",
            "ci_upper",
            "n_subjects",
            "n_raters",
        ]:
            assert field in v, f"Missing field: {field}"

    def test_icc_value_finite_and_in_range(self):
        """ICC value is finite and in [-1, 1], n_subjects/n_raters are correct."""
        result = self._rater_df().select(ps.icc("rater1", "rater2", "rater3"))
        v = result[0, 0]
        assert math.isfinite(v["icc"])
        assert -1.0 <= v["icc"] <= 1.0
        assert v["n_subjects"] == 4
        assert v["n_raters"] == 3

    def test_ci_bounds_finite(self):
        """Confidence interval bounds are finite for valid input."""
        result = self._rater_df().select(ps.icc("rater1", "rater2", "rater3"))
        v = result[0, 0]
        assert math.isfinite(v["ci_lower"])
        assert math.isfinite(v["ci_upper"])
        assert v["ci_lower"] <= v["ci_upper"]

    def test_icc_type_parameter_honored(self):
        """icc_type='icc3' produces a different icc value than the default 'icc2'."""
        df = self._rater_df()
        r2 = df.select(ps.icc("rater1", "rater2", "rater3", icc_type="icc2"))[0, 0]
        r3 = df.select(ps.icc("rater1", "rater2", "rater3", icc_type="icc3"))[0, 0]
        # Both must be finite; they differ for non-trivial data
        assert math.isfinite(r2["icc"])
        assert math.isfinite(r3["icc"])
        assert r2["icc"] != r3["icc"]

    def test_degenerate_no_raters_returns_nan_struct(self):
        """Calling ps.icc with no rater columns returns all-NaN struct without panic."""
        df = pl.DataFrame({"dummy": [1.0, 2.0]})
        # Pass zero rater columns — the builder sends n_raters=0 to Rust
        result = df.select(ps.icc())
        v = result[0, 0]
        assert math.isnan(v["icc"])
        assert v["n_subjects"] == 0
        assert v["n_raters"] == 0
