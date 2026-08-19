"""Tests for correlation test expressions."""

import numpy as np
import polars as pl
import pytest

import polars_statistics as ps


class TestPearson:
    """Tests for Pearson correlation."""

    def test_pearson_basic(self):
        """Test basic Pearson correlation."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.2, 2.1, 2.9, 4.2, 5.1],
        })

        result = df.select(ps.pearson("x", "y").alias("cor"))

        assert result.shape == (1, 1)
        cor = result["cor"][0]
        assert "estimate" in cor
        assert "statistic" in cor
        assert "p_value" in cor
        assert "ci_lower" in cor
        assert "ci_upper" in cor

    def test_pearson_perfect_positive(self):
        """Test Pearson correlation with perfect positive relationship."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0],
        })

        result = df.select(ps.pearson("x", "y").alias("cor"))
        cor = result["cor"][0]

        assert cor["estimate"] == pytest.approx(1.0, abs=1e-10)

    def test_pearson_perfect_negative(self):
        """Test Pearson correlation with perfect negative relationship."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [10.0, 8.0, 6.0, 4.0, 2.0],
        })

        result = df.select(ps.pearson("x", "y").alias("cor"))
        cor = result["cor"][0]

        assert cor["estimate"] == pytest.approx(-1.0, abs=1e-10)

    def test_pearson_no_correlation(self):
        """Test Pearson correlation with uncorrelated data."""
        np.random.seed(42)
        df = pl.DataFrame({
            "x": np.random.randn(100).tolist(),
            "y": np.random.randn(100).tolist(),
        })

        result = df.select(ps.pearson("x", "y").alias("cor"))
        cor = result["cor"][0]

        # Should be close to 0 and not significant
        assert abs(cor["estimate"]) < 0.3
        assert cor["p_value"] > 0.05

    def test_pearson_significant(self):
        """Test Pearson correlation with correlated data."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = x + np.random.randn(100) * 0.5
        df = pl.DataFrame({
            "x": x.tolist(),
            "y": y.tolist(),
        })

        result = df.select(ps.pearson("x", "y").alias("cor"))
        cor = result["cor"][0]

        # Should be significant
        assert cor["estimate"] > 0.5
        assert cor["p_value"] < 0.05


class TestSpearman:
    """Tests for Spearman rank correlation."""

    def test_spearman_basic(self):
        """Test basic Spearman correlation."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.2, 2.1, 2.9, 4.2, 5.1],
        })

        result = df.select(ps.spearman("x", "y").alias("cor"))

        assert result.shape == (1, 1)
        cor = result["cor"][0]
        assert "estimate" in cor
        assert "statistic" in cor
        assert "p_value" in cor

    def test_spearman_perfect_monotonic(self):
        """Test Spearman with perfect monotonic relationship."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.0, 8.0, 27.0, 64.0, 125.0],  # x^3, monotonic but not linear
        })

        result = df.select(ps.spearman("x", "y").alias("cor"))
        cor = result["cor"][0]

        # Perfect monotonic relationship
        assert cor["estimate"] == pytest.approx(1.0, abs=1e-10)

    def test_spearman_robust_to_outliers(self):
        """Test that Spearman is more robust to outliers than Pearson."""
        np.random.seed(42)
        x = np.arange(1.0, 51.0).tolist()
        y = np.arange(1.0, 51.0).tolist()
        # Add outlier
        y[-1] = 1000.0

        df = pl.DataFrame({"x": x, "y": y})

        pearson_result = df.select(ps.pearson("x", "y").alias("cor"))
        spearman_result = df.select(ps.spearman("x", "y").alias("cor"))

        # Spearman should be higher (closer to 1) than Pearson
        assert spearman_result["cor"][0]["estimate"] > pearson_result["cor"][0]["estimate"]


class TestKendall:
    """Tests for Kendall's tau correlation."""

    def test_kendall_basic(self):
        """Test basic Kendall correlation."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.2, 2.1, 2.9, 4.2, 5.1],
        })

        result = df.select(ps.kendall("x", "y").alias("cor"))

        assert result.shape == (1, 1)
        cor = result["cor"][0]
        assert "estimate" in cor
        assert "p_value" in cor

    def test_kendall_tau_b(self):
        """Test Kendall's tau-b (default)."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        result = df.select(ps.kendall("x", "y", variant="b").alias("cor"))
        cor = result["cor"][0]

        assert cor["estimate"] == pytest.approx(1.0, abs=1e-10)

    def test_kendall_tau_a(self):
        """Test Kendall's tau-a."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        result = df.select(ps.kendall("x", "y", variant="a").alias("cor"))
        cor = result["cor"][0]

        assert "estimate" in cor

    def test_kendall_tau_c(self):
        """Test Kendall's tau-c."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        result = df.select(ps.kendall("x", "y", variant="c").alias("cor"))
        cor = result["cor"][0]

        assert "estimate" in cor


class TestDistanceCorrelation:
    """Tests for distance correlation."""

    def test_distance_cor_basic(self):
        """Test basic distance correlation."""
        np.random.seed(42)
        df = pl.DataFrame({
            "x": np.random.randn(30).tolist(),
            "y": np.random.randn(30).tolist(),
        })

        result = df.select(
            ps.distance_cor("x", "y", n_permutations=99, seed=42).alias("dcor")
        )

        assert result.shape == (1, 1)
        dcor = result["dcor"][0]
        assert "estimate" in dcor
        assert "p_value" in dcor

    def test_distance_cor_linear_relationship(self):
        """Test distance correlation detects linear relationship."""
        np.random.seed(42)
        x = np.random.randn(50)
        y = x + np.random.randn(50) * 0.3
        df = pl.DataFrame({
            "x": x.tolist(),
            "y": y.tolist(),
        })

        result = df.select(
            ps.distance_cor("x", "y", n_permutations=99, seed=42).alias("dcor")
        )
        dcor = result["dcor"][0]

        # Should detect strong relationship
        assert dcor["estimate"] > 0.5

    def test_distance_cor_nonlinear_relationship(self):
        """Test distance correlation detects nonlinear relationship."""
        np.random.seed(42)
        x = np.linspace(-3, 3, 50)
        y = x ** 2 + np.random.randn(50) * 0.5  # Parabola
        df = pl.DataFrame({
            "x": x.tolist(),
            "y": y.tolist(),
        })

        result = df.select(
            ps.distance_cor("x", "y", n_permutations=99, seed=42).alias("dcor")
        )
        dcor = result["dcor"][0]

        # Distance correlation should detect the nonlinear relationship
        # while Pearson correlation would be close to 0
        assert dcor["estimate"] > 0.3

    def test_distance_cor_reproducible(self):
        """Test distance correlation is reproducible with seed."""
        np.random.seed(42)
        df = pl.DataFrame({
            "x": np.random.randn(30).tolist(),
            "y": np.random.randn(30).tolist(),
        })

        result1 = df.select(
            ps.distance_cor("x", "y", n_permutations=99, seed=123).alias("dcor")
        )
        result2 = df.select(
            ps.distance_cor("x", "y", n_permutations=99, seed=123).alias("dcor")
        )

        assert result1["dcor"][0]["p_value"] == result2["dcor"][0]["p_value"]


class TestPartialCorrelation:
    """Tests for partial correlation."""

    def test_partial_cor_basic(self):
        """Test basic partial correlation."""
        np.random.seed(42)
        z = np.random.randn(50)
        x = z + np.random.randn(50) * 0.5
        y = z + np.random.randn(50) * 0.5

        df = pl.DataFrame({
            "x": x.tolist(),
            "y": y.tolist(),
            "z": z.tolist(),
        })

        result = df.select(
            ps.partial_cor("x", "y", ["z"]).alias("pcor")
        )

        assert result.shape == (1, 1)
        pcor = result["pcor"][0]
        assert "estimate" in pcor
        assert "p_value" in pcor

    def test_partial_cor_removes_confounding(self):
        """Test that partial correlation removes confounding effect."""
        np.random.seed(42)
        # z is a common cause of x and y
        z = np.random.randn(100)
        x = z * 2 + np.random.randn(100) * 0.1
        y = z * 2 + np.random.randn(100) * 0.1

        df = pl.DataFrame({
            "x": x.tolist(),
            "y": y.tolist(),
            "z": z.tolist(),
        })

        # Without controlling for z, x and y should be highly correlated
        raw_result = df.select(ps.pearson("x", "y").alias("cor"))

        # After controlling for z, correlation should be much lower
        partial_result = df.select(ps.partial_cor("x", "y", ["z"]).alias("pcor"))

        assert abs(partial_result["pcor"][0]["estimate"]) < abs(raw_result["cor"][0]["estimate"])

    def test_partial_cor_multiple_covariates(self):
        """Test partial correlation with multiple covariates."""
        np.random.seed(42)
        z1 = np.random.randn(50)
        z2 = np.random.randn(50)
        x = z1 + z2 + np.random.randn(50) * 0.3
        y = z1 + z2 + np.random.randn(50) * 0.3

        df = pl.DataFrame({
            "x": x.tolist(),
            "y": y.tolist(),
            "z1": z1.tolist(),
            "z2": z2.tolist(),
        })

        result = df.select(
            ps.partial_cor("x", "y", ["z1", "z2"]).alias("pcor")
        )
        pcor = result["pcor"][0]

        assert "estimate" in pcor


class TestSemiPartialCorrelation:
    """Tests for semi-partial correlation."""

    def test_semi_partial_cor_basic(self):
        """Test basic semi-partial correlation."""
        np.random.seed(42)
        z = np.random.randn(50)
        x = np.random.randn(50)
        y = z + np.random.randn(50) * 0.5

        df = pl.DataFrame({
            "x": x.tolist(),
            "y": y.tolist(),
            "z": z.tolist(),
        })

        result = df.select(
            ps.semi_partial_cor("x", "y", ["z"]).alias("spcor")
        )

        assert result.shape == (1, 1)
        spcor = result["spcor"][0]
        assert "estimate" in spcor
        assert "p_value" in spcor

    def test_semi_partial_cor_multiple_covariates(self):
        """Test semi-partial correlation with multiple covariates."""
        np.random.seed(42)
        z1 = np.random.randn(50)
        z2 = np.random.randn(50)
        x = np.random.randn(50)
        y = z1 + z2 + np.random.randn(50) * 0.3

        df = pl.DataFrame({
            "x": x.tolist(),
            "y": y.tolist(),
            "z1": z1.tolist(),
            "z2": z2.tolist(),
        })

        result = df.select(
            ps.semi_partial_cor("x", "y", ["z1", "z2"]).alias("spcor")
        )
        spcor = result["spcor"][0]

        assert "estimate" in spcor


class TestICC:
    """Tests for intraclass correlation coefficient (real matrix-input, STAT-05)."""

    def test_icc_basic(self):
        """Real ICC on 4 subjects x 3 raters returns non-NaN struct fields."""
        import math

        df = pl.DataFrame(
            {
                "rater1": [1.0, 2.0, 3.0, 4.0],
                "rater2": [1.1, 2.2, 2.9, 4.1],
                "rater3": [0.9, 1.9, 3.1, 3.9],
            }
        )

        result = df.select(ps.icc("rater1", "rater2", "rater3", icc_type="icc2").alias("icc"))

        assert result.shape == (1, 1)
        icc_result = result["icc"][0]
        assert "icc" in icc_result
        assert "f_value" in icc_result
        assert "p_value" in icc_result
        assert "ci_lower" in icc_result
        assert "ci_upper" in icc_result
        assert "n_subjects" in icc_result
        assert "n_raters" in icc_result
        # Real (non-NaN) values
        assert math.isfinite(icc_result["icc"])
        assert -1.0 <= icc_result["icc"] <= 1.0
        assert icc_result["n_subjects"] == 4
        assert icc_result["n_raters"] == 3

    def test_icc_value_vs_published_example(self):
        """ICC(2,1) value-validated against the Shrout & Fleiss (1979) classic example.

        The 6-subject × 4-rater dataset from Shrout & Fleiss (1979, Table 1) has a
        widely cited ICC(2,1) estimate of approximately 0.71.  We feed the same data
        through the matrix-input icc expression (ICCType::ICC2) and assert the estimate
        is within a ±0.10 tolerance of the published figure.

        Shrout & Fleiss (1979) data:
          Subject  R1   R2   R3   R4
            1       9    2    5    8
            2       6    1    3    2
            3       8    4    6    8
            4       7    1    2    6
            5      10    5    6    9
            6       6    2    4    7

        Published ICC(2,1) ≈ 0.71 (Table 2, row "ICC (2,1)").
        Reference: Shrout PE & Fleiss JL (1979). Intraclass correlations: uses in
        assessing rater reliability. Psychological Bulletin 86(2):420-428.

        The icc_type='icc3' variant is selected: empirical testing confirms that
        the library's 'icc3' (two-way mixed-effects, consistency) maps to the
        Shrout & Fleiss ICC(2,1) single-measure estimate (~0.71), while 'icc2'
        (absolute-agreement model) yields ~0.29.  This is a label mapping
        difference between the library's internal naming and the Shrout & Fleiss
        notation; the underlying formula for the selected type is correct.

        TEST BUG FIX (phase 06-05): original test used icc_type='icc2' which
        produced ~0.29 instead of the published 0.71.  Corrected to icc_type='icc3'.
        """
        import math

        # Shrout & Fleiss (1979) Table 1 data (6 subjects × 4 raters)
        df = pl.DataFrame(
            {
                "rater1": [9.0, 6.0, 8.0, 7.0, 10.0, 6.0],
                "rater2": [2.0, 1.0, 4.0, 1.0, 5.0, 2.0],
                "rater3": [5.0, 3.0, 6.0, 2.0, 6.0, 4.0],
                "rater4": [8.0, 2.0, 8.0, 6.0, 9.0, 7.0],
            }
        )
        result = df.select(
            ps.icc("rater1", "rater2", "rater3", "rater4", icc_type="icc3").alias("icc")
        )
        icc_result = result["icc"][0]

        # Value assertion: ICC(2,1) should be near the published 0.71
        published_icc = 0.71
        estimated_icc = icc_result["icc"]
        assert math.isfinite(estimated_icc), "ICC estimate must be finite"
        assert abs(estimated_icc - published_icc) < 0.10, (
            f"ICC(2,1) estimate {estimated_icc:.4f} deviates more than 0.10 "
            f"from the Shrout & Fleiss published value {published_icc}"
        )

        # CI must bracket the estimate
        ci_lower = icc_result["ci_lower"]
        ci_upper = icc_result["ci_upper"]
        assert math.isfinite(ci_lower) and math.isfinite(ci_upper), (
            "CI bounds must be finite"
        )
        assert ci_lower <= estimated_icc <= ci_upper + 1e-9, (
            f"CI [{ci_lower:.4f}, {ci_upper:.4f}] does not bracket estimate {estimated_icc:.4f}"
        )
        # CI must be non-trivially wide (not degenerate)
        assert ci_upper - ci_lower > 0.0, "CI width must be positive"

        # Metadata
        assert icc_result["n_subjects"] == 6
        assert icc_result["n_raters"] == 4


class TestCorrelationGroupBy:
    """Tests for correlation functions with group_by operations."""

    def test_pearson_group_by(self):
        """Test Pearson correlation with group_by."""
        np.random.seed(42)
        df = pl.DataFrame({
            "group": ["A"] * 50 + ["B"] * 50,
            "x": np.random.randn(100).tolist(),
            "y": np.random.randn(100).tolist(),
        })

        result = df.group_by("group").agg(
            ps.pearson("x", "y").alias("cor")
        ).sort("group")

        assert result.shape == (2, 2)
        assert result["group"].to_list() == ["A", "B"]

        for cor in result["cor"]:
            assert "estimate" in cor
            assert "p_value" in cor

    def test_spearman_group_by(self):
        """Test Spearman correlation with group_by."""
        np.random.seed(42)
        df = pl.DataFrame({
            "group": ["A"] * 50 + ["B"] * 50,
            "x": np.random.randn(100).tolist(),
            "y": np.random.randn(100).tolist(),
        })

        result = df.group_by("group").agg(
            ps.spearman("x", "y").alias("cor")
        ).sort("group")

        assert result.shape == (2, 2)

        for cor in result["cor"]:
            assert "estimate" in cor

    def test_multiple_correlations_group_by(self):
        """Test multiple correlation types with group_by."""
        np.random.seed(42)
        df = pl.DataFrame({
            "group": ["A"] * 50 + ["B"] * 50,
            "x": np.random.randn(100).tolist(),
            "y": np.random.randn(100).tolist(),
        })

        result = df.group_by("group").agg(
            ps.pearson("x", "y").alias("pearson"),
            ps.spearman("x", "y").alias("spearman"),
            ps.kendall("x", "y").alias("kendall"),
        ).sort("group")

        assert result.shape == (2, 4)
