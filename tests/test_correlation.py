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
    """Tests for intraclass correlation coefficient."""

    def test_icc_basic(self):
        """Test basic ICC computation."""
        np.random.seed(42)
        df = pl.DataFrame({
            "values": np.random.randn(30).tolist(),
        })

        result = df.select(
            ps.icc("values", icc_type="icc1").alias("icc")
        )

        assert result.shape == (1, 1)
        icc_result = result["icc"][0]
        assert "estimate" in icc_result


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
