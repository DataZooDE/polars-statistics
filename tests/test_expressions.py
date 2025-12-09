"""Tests for statistical test expressions."""

import numpy as np
import polars as pl
import pytest

import polars_statistics as ps


class TestParametricTests:
    """Tests for parametric statistical tests."""

    def test_ttest_ind_basic(self):
        """Test independent t-test."""
        np.random.seed(42)
        df = pl.DataFrame({
            "x": np.random.randn(50).tolist(),
            "y": np.random.randn(50).tolist(),
        })

        result = df.select(ps.ttest_ind(pl.col("x"), pl.col("y")).alias("ttest"))

        assert result.shape == (1, 1)
        ttest = result["ttest"][0]
        assert "statistic" in ttest
        assert "p_value" in ttest

    def test_ttest_ind_different_means(self):
        """Test t-test detects different means."""
        np.random.seed(42)
        df = pl.DataFrame({
            "x": (np.random.randn(100) + 0).tolist(),
            "y": (np.random.randn(100) + 2).tolist(),
        })

        result = df.select(ps.ttest_ind(pl.col("x"), pl.col("y")).alias("ttest"))
        ttest = result["ttest"][0]

        # Should detect significant difference
        assert ttest["p_value"] < 0.05

    def test_ttest_paired_basic(self):
        """Test paired t-test."""
        np.random.seed(42)
        df = pl.DataFrame({
            "before": np.random.randn(50).tolist(),
            "after": (np.random.randn(50) + 0.5).tolist(),
        })

        result = df.select(ps.ttest_paired(pl.col("before"), pl.col("after")).alias("ttest"))

        assert result.shape == (1, 1)
        ttest = result["ttest"][0]
        assert "statistic" in ttest
        assert "p_value" in ttest

    def test_brown_forsythe(self):
        """Test Brown-Forsythe test."""
        np.random.seed(42)
        df = pl.DataFrame({
            "x": np.random.randn(50).tolist(),
            "y": np.random.randn(50).tolist(),
        })

        result = df.select(ps.brown_forsythe(pl.col("x"), pl.col("y")).alias("bf"))

        assert result.shape == (1, 1)
        bf = result["bf"][0]
        assert "statistic" in bf
        assert "p_value" in bf


class TestNonParametricTests:
    """Tests for non-parametric statistical tests."""

    def test_mann_whitney_u_basic(self):
        """Test Mann-Whitney U test."""
        np.random.seed(42)
        df = pl.DataFrame({
            "x": np.random.randn(50).tolist(),
            "y": np.random.randn(50).tolist(),
        })

        result = df.select(ps.mann_whitney_u(pl.col("x"), pl.col("y")).alias("mwu"))

        assert result.shape == (1, 1)
        mwu = result["mwu"][0]
        assert "statistic" in mwu
        assert "p_value" in mwu

    def test_mann_whitney_u_different_distributions(self):
        """Test Mann-Whitney U detects different distributions."""
        np.random.seed(42)
        df = pl.DataFrame({
            "x": (np.random.randn(100)).tolist(),
            "y": (np.random.randn(100) + 2).tolist(),
        })

        result = df.select(ps.mann_whitney_u(pl.col("x"), pl.col("y")).alias("mwu"))
        mwu = result["mwu"][0]

        # Should detect significant difference
        assert mwu["p_value"] < 0.05

    def test_wilcoxon_signed_rank(self):
        """Test Wilcoxon signed-rank test."""
        np.random.seed(42)
        df = pl.DataFrame({
            "x": np.random.randn(50).tolist(),
            "y": (np.random.randn(50) + 0.5).tolist(),
        })

        result = df.select(ps.wilcoxon_signed_rank(pl.col("x"), pl.col("y")).alias("wilcoxon"))

        assert result.shape == (1, 1)
        wilcoxon = result["wilcoxon"][0]
        assert "statistic" in wilcoxon
        assert "p_value" in wilcoxon

    def test_kruskal_wallis(self):
        """Test Kruskal-Wallis H test."""
        np.random.seed(42)
        df = pl.DataFrame({
            "x": np.random.randn(50).tolist(),
            "y": np.random.randn(50).tolist(),
        })

        result = df.select(ps.kruskal_wallis(pl.col("x"), pl.col("y")).alias("kw"))

        assert result.shape == (1, 1)
        kw = result["kw"][0]
        assert "statistic" in kw
        assert "p_value" in kw


class TestDistributionalTests:
    """Tests for distributional tests."""

    def test_shapiro_wilk_basic(self):
        """Test Shapiro-Wilk test."""
        np.random.seed(42)
        df = pl.DataFrame({
            "x": np.random.randn(50).tolist(),
        })

        result = df.select(ps.shapiro_wilk(pl.col("x")).alias("sw"))

        assert result.shape == (1, 1)
        sw = result["sw"][0]
        assert "statistic" in sw
        assert "p_value" in sw

    def test_shapiro_wilk_normal_data(self):
        """Test Shapiro-Wilk accepts normal data."""
        np.random.seed(42)
        df = pl.DataFrame({
            "x": np.random.randn(50).tolist(),
        })

        result = df.select(ps.shapiro_wilk(pl.col("x")).alias("sw"))
        sw = result["sw"][0]

        # Should not reject normality
        assert sw["p_value"] > 0.05

    def test_shapiro_wilk_non_normal_data(self):
        """Test Shapiro-Wilk rejects non-normal data."""
        np.random.seed(42)
        # Exponential distribution is not normal
        df = pl.DataFrame({
            "x": np.random.exponential(1, 50).tolist(),
        })

        result = df.select(ps.shapiro_wilk(pl.col("x")).alias("sw"))
        sw = result["sw"][0]

        # Should reject normality
        assert sw["p_value"] < 0.05

    def test_dagostino_basic(self):
        """Test D'Agostino-Pearson test."""
        np.random.seed(42)
        df = pl.DataFrame({
            "x": np.random.randn(100).tolist(),
        })

        result = df.select(ps.dagostino(pl.col("x")).alias("dag"))

        assert result.shape == (1, 1)
        dag = result["dag"][0]
        assert "statistic" in dag
        assert "p_value" in dag


class TestForecastTests:
    """Tests for forecast comparison tests."""

    def test_diebold_mariano_basic(self):
        """Test Diebold-Mariano test."""
        np.random.seed(42)
        df = pl.DataFrame({
            "e1": np.random.randn(100).tolist(),
            "e2": np.random.randn(100).tolist(),
        })

        result = df.select(ps.diebold_mariano(pl.col("e1"), pl.col("e2")).alias("dm"))

        assert result.shape == (1, 1)
        dm = result["dm"][0]
        assert "statistic" in dm
        assert "p_value" in dm

    def test_permutation_t_test_basic(self):
        """Test permutation t-test."""
        np.random.seed(42)
        df = pl.DataFrame({
            "x": np.random.randn(30).tolist(),
            "y": np.random.randn(30).tolist(),
        })

        result = df.select(
            ps.permutation_t_test(pl.col("x"), pl.col("y"), n_permutations=99, seed=42).alias("perm")
        )

        assert result.shape == (1, 1)
        perm = result["perm"][0]
        assert "statistic" in perm
        assert "p_value" in perm


class TestGroupByOperations:
    """Tests for group_by operations."""

    def test_ttest_group_by(self):
        """Test t-test with group_by."""
        np.random.seed(42)
        df = pl.DataFrame({
            "group": ["A"] * 50 + ["B"] * 50,
            "x": np.random.randn(100).tolist(),
            "y": np.random.randn(100).tolist(),
        })

        result = df.group_by("group").agg(
            ps.ttest_ind(pl.col("x"), pl.col("y")).alias("ttest")
        ).sort("group")

        assert result.shape == (2, 2)
        assert result["group"].to_list() == ["A", "B"]

        for ttest in result["ttest"]:
            assert "statistic" in ttest
            assert "p_value" in ttest

    def test_multiple_tests_group_by(self):
        """Test multiple tests with group_by."""
        np.random.seed(42)
        df = pl.DataFrame({
            "group": ["A"] * 50 + ["B"] * 50,
            "x": np.random.randn(100).tolist(),
            "y": np.random.randn(100).tolist(),
        })

        result = df.group_by("group").agg(
            ps.ttest_ind(pl.col("x"), pl.col("y")).alias("ttest"),
            ps.mann_whitney_u(pl.col("x"), pl.col("y")).alias("mwu"),
            ps.shapiro_wilk(pl.col("x")).alias("shapiro"),
        ).sort("group")

        assert result.shape == (2, 4)
