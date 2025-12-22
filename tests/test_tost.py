"""Tests for TOST (Two One-Sided Tests) equivalence testing expressions."""

import numpy as np
import polars as pl
import pytest

import polars_statistics as ps


class TestTostOneSample:
    """Tests for one-sample TOST equivalence tests."""

    def test_tost_t_test_one_sample_basic(self):
        """Test one-sample TOST with symmetric bounds."""
        np.random.seed(42)
        # Data centered near 0 should be equivalent to 0
        df = pl.DataFrame({
            "x": (np.random.randn(100) * 0.1).tolist(),
        })

        result = df.select(
            ps.tost_t_test_one_sample("x", mu=0.0, delta=0.5).alias("tost")
        )

        assert result.shape == (1, 1)
        tost = result["tost"][0]
        assert "estimate" in tost
        assert "tost_p_value" in tost
        assert "equivalent" in tost
        assert "ci_lower" in tost
        assert "ci_upper" in tost

    def test_tost_t_test_one_sample_equivalent(self):
        """Test that equivalent samples are detected."""
        np.random.seed(42)
        df = pl.DataFrame({
            "x": (np.random.randn(100) * 0.1).tolist(),
        })

        result = df.select(
            ps.tost_t_test_one_sample("x", mu=0.0, delta=0.5, alpha=0.05).alias("tost")
        )
        tost = result["tost"][0]

        # With small variance and wide bounds, should be equivalent
        assert tost["equivalent"] is True

    def test_tost_t_test_one_sample_not_equivalent(self):
        """Test that non-equivalent samples are detected."""
        np.random.seed(42)
        # Data far from 0
        df = pl.DataFrame({
            "x": (np.random.randn(100) + 2.0).tolist(),
        })

        result = df.select(
            ps.tost_t_test_one_sample("x", mu=0.0, delta=0.5).alias("tost")
        )
        tost = result["tost"][0]

        # Mean is ~2, which is outside (-0.5, 0.5) bounds
        assert tost["equivalent"] is False

    def test_tost_t_test_one_sample_cohen_d(self):
        """Test one-sample TOST with Cohen's d bounds."""
        np.random.seed(42)
        df = pl.DataFrame({
            "x": np.random.randn(100).tolist(),
        })

        result = df.select(
            ps.tost_t_test_one_sample("x", mu=0.0, bounds_type="cohen_d", delta=0.3).alias("tost")
        )
        tost = result["tost"][0]

        assert "tost_p_value" in tost
        assert "equivalent" in tost

    def test_tost_t_test_one_sample_raw_bounds(self):
        """Test one-sample TOST with raw bounds."""
        np.random.seed(42)
        df = pl.DataFrame({
            "x": (np.random.randn(50) * 0.1).tolist(),
        })

        result = df.select(
            ps.tost_t_test_one_sample(
                "x", mu=0.0, bounds_type="raw", lower=-1.0, upper=1.0
            ).alias("tost")
        )
        tost = result["tost"][0]

        assert "tost_p_value" in tost
        assert tost["equivalent"] is True


class TestTostTwoSample:
    """Tests for two-sample TOST equivalence tests."""

    def test_tost_t_test_two_sample_basic(self):
        """Test two-sample TOST."""
        np.random.seed(42)
        df = pl.DataFrame({
            "x": np.random.randn(50).tolist(),
            "y": np.random.randn(50).tolist(),
        })

        result = df.select(
            ps.tost_t_test_two_sample("x", "y", delta=0.5).alias("tost")
        )

        assert result.shape == (1, 1)
        tost = result["tost"][0]
        assert "estimate" in tost
        assert "tost_p_value" in tost
        assert "equivalent" in tost

    def test_tost_t_test_two_sample_equivalent(self):
        """Test two-sample TOST detects equivalence."""
        np.random.seed(42)
        # Same distribution
        df = pl.DataFrame({
            "x": np.random.randn(100).tolist(),
            "y": np.random.randn(100).tolist(),
        })

        result = df.select(
            ps.tost_t_test_two_sample("x", "y", delta=0.5).alias("tost")
        )
        tost = result["tost"][0]

        # Same distribution should be equivalent
        assert tost["equivalent"] is True

    def test_tost_t_test_two_sample_not_equivalent(self):
        """Test two-sample TOST detects non-equivalence."""
        np.random.seed(42)
        df = pl.DataFrame({
            "x": np.random.randn(100).tolist(),
            "y": (np.random.randn(100) + 2.0).tolist(),  # Shifted by 2
        })

        result = df.select(
            ps.tost_t_test_two_sample("x", "y", delta=0.5).alias("tost")
        )
        tost = result["tost"][0]

        # Different means should not be equivalent
        assert tost["equivalent"] is False

    def test_tost_t_test_two_sample_pooled(self):
        """Test two-sample TOST with pooled variance."""
        np.random.seed(42)
        df = pl.DataFrame({
            "x": np.random.randn(50).tolist(),
            "y": np.random.randn(50).tolist(),
        })

        result = df.select(
            ps.tost_t_test_two_sample("x", "y", delta=0.5, pooled=True).alias("tost")
        )
        tost = result["tost"][0]

        assert "tost_p_value" in tost


class TestTostPaired:
    """Tests for paired TOST equivalence tests."""

    def test_tost_t_test_paired_basic(self):
        """Test paired TOST."""
        np.random.seed(42)
        base = np.random.randn(50)
        df = pl.DataFrame({
            "before": base.tolist(),
            "after": (base + np.random.randn(50) * 0.1).tolist(),  # Small change
        })

        result = df.select(
            ps.tost_t_test_paired("before", "after", delta=0.5).alias("tost")
        )

        assert result.shape == (1, 1)
        tost = result["tost"][0]
        assert "estimate" in tost
        assert "tost_p_value" in tost
        assert "equivalent" in tost

    def test_tost_t_test_paired_equivalent(self):
        """Test paired TOST detects equivalence."""
        np.random.seed(42)
        base = np.random.randn(100)
        df = pl.DataFrame({
            "before": base.tolist(),
            "after": (base + np.random.randn(100) * 0.05).tolist(),
        })

        result = df.select(
            ps.tost_t_test_paired("before", "after", delta=0.5).alias("tost")
        )
        tost = result["tost"][0]

        # Small differences should be equivalent
        assert tost["equivalent"] is True


class TestTostCorrelation:
    """Tests for correlation TOST equivalence tests."""

    def test_tost_correlation_basic(self):
        """Test correlation TOST."""
        np.random.seed(42)
        # Weakly correlated variables
        x = np.random.randn(100)
        y = np.random.randn(100) + 0.1 * x
        df = pl.DataFrame({
            "x": x.tolist(),
            "y": y.tolist(),
        })

        result = df.select(
            ps.tost_correlation("x", "y", delta=0.3).alias("tost")
        )

        assert result.shape == (1, 1)
        tost = result["tost"][0]
        assert "estimate" in tost
        assert "tost_p_value" in tost
        assert "equivalent" in tost

    def test_tost_correlation_spearman(self):
        """Test correlation TOST with Spearman method."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)
        df = pl.DataFrame({
            "x": x.tolist(),
            "y": y.tolist(),
        })

        result = df.select(
            ps.tost_correlation("x", "y", method="spearman", delta=0.3).alias("tost")
        )
        tost = result["tost"][0]

        assert "tost_p_value" in tost


class TestTostProportion:
    """Tests for proportion TOST equivalence tests."""

    def test_tost_prop_one_basic(self):
        """Test one-proportion TOST."""
        result = pl.select(
            ps.tost_prop_one(successes=50, n=100, p0=0.5, delta=0.1).alias("tost")
        )

        assert result.shape == (1, 1)
        tost = result["tost"][0]
        assert "estimate" in tost
        assert "tost_p_value" in tost
        assert "equivalent" in tost

    def test_tost_prop_one_equivalent(self):
        """Test one-proportion TOST detects equivalence."""
        # 50/100 = 0.5, testing against p0=0.5
        result = pl.select(
            ps.tost_prop_one(successes=50, n=100, p0=0.5, delta=0.1).alias("tost")
        )
        tost = result["tost"][0]

        # Should be equivalent since observed = hypothesized
        assert tost["equivalent"] is True

    def test_tost_prop_two_basic(self):
        """Test two-proportion TOST."""
        result = pl.select(
            ps.tost_prop_two(
                successes1=48, n1=100,
                successes2=52, n2=100,
                delta=0.1
            ).alias("tost")
        )

        assert result.shape == (1, 1)
        tost = result["tost"][0]
        assert "estimate" in tost
        assert "tost_p_value" in tost
        assert "equivalent" in tost

    def test_tost_prop_two_equivalent(self):
        """Test two-proportion TOST detects equivalence."""
        # Use larger samples and exactly equal proportions for reliable equivalence
        result = pl.select(
            ps.tost_prop_two(
                successes1=50, n1=100,
                successes2=50, n2=100,
                delta=0.15
            ).alias("tost")
        )
        tost = result["tost"][0]

        # Equal proportions with wide bounds should be equivalent
        assert tost["equivalent"] is True


class TestTostWilcoxon:
    """Tests for Wilcoxon TOST equivalence tests."""

    def test_tost_wilcoxon_paired_basic(self):
        """Test paired Wilcoxon TOST."""
        np.random.seed(42)
        base = np.random.randn(50)
        df = pl.DataFrame({
            "x": base.tolist(),
            "y": (base + np.random.randn(50) * 0.1).tolist(),
        })

        result = df.select(
            ps.tost_wilcoxon_paired("x", "y", delta=0.5).alias("tost")
        )

        assert result.shape == (1, 1)
        tost = result["tost"][0]
        assert "estimate" in tost
        assert "tost_p_value" in tost
        assert "equivalent" in tost

    def test_tost_wilcoxon_two_sample_basic(self):
        """Test two-sample Wilcoxon TOST."""
        np.random.seed(42)
        df = pl.DataFrame({
            "x": np.random.randn(50).tolist(),
            "y": np.random.randn(50).tolist(),
        })

        result = df.select(
            ps.tost_wilcoxon_two_sample("x", "y", delta=0.5).alias("tost")
        )

        assert result.shape == (1, 1)
        tost = result["tost"][0]
        assert "estimate" in tost
        assert "tost_p_value" in tost
        assert "equivalent" in tost


class TestTostBootstrap:
    """Tests for bootstrap TOST equivalence tests."""

    def test_tost_bootstrap_basic(self):
        """Test bootstrap TOST."""
        np.random.seed(42)
        df = pl.DataFrame({
            "x": np.random.randn(50).tolist(),
            "y": np.random.randn(50).tolist(),
        })

        result = df.select(
            ps.tost_bootstrap("x", "y", delta=0.5, n_bootstrap=100, seed=42).alias("tost")
        )

        assert result.shape == (1, 1)
        tost = result["tost"][0]
        assert "estimate" in tost
        assert "tost_p_value" in tost
        assert "equivalent" in tost

    def test_tost_bootstrap_reproducible(self):
        """Test bootstrap TOST is reproducible with seed."""
        np.random.seed(42)
        df = pl.DataFrame({
            "x": np.random.randn(30).tolist(),
            "y": np.random.randn(30).tolist(),
        })

        result1 = df.select(
            ps.tost_bootstrap("x", "y", delta=0.5, n_bootstrap=100, seed=123).alias("tost")
        )
        result2 = df.select(
            ps.tost_bootstrap("x", "y", delta=0.5, n_bootstrap=100, seed=123).alias("tost")
        )

        assert result1["tost"][0]["tost_p_value"] == result2["tost"][0]["tost_p_value"]


class TestTostYuen:
    """Tests for Yuen TOST equivalence tests."""

    def test_tost_yuen_basic(self):
        """Test Yuen TOST with trimmed means."""
        np.random.seed(42)
        df = pl.DataFrame({
            "x": np.random.randn(50).tolist(),
            "y": np.random.randn(50).tolist(),
        })

        result = df.select(
            ps.tost_yuen("x", "y", trim=0.2, delta=0.5).alias("tost")
        )

        assert result.shape == (1, 1)
        tost = result["tost"][0]
        assert "estimate" in tost
        assert "tost_p_value" in tost
        assert "equivalent" in tost

    def test_tost_yuen_robust_to_outliers(self):
        """Test that Yuen TOST is robust to outliers."""
        np.random.seed(42)
        x = np.random.randn(50).tolist()
        y = np.random.randn(50).tolist()
        # Add outliers
        x[0] = 100.0
        y[0] = -100.0

        df = pl.DataFrame({"x": x, "y": y})

        result = df.select(
            ps.tost_yuen("x", "y", trim=0.2, delta=1.0).alias("tost")
        )
        tost = result["tost"][0]

        # With trimming, outliers should not dominate
        assert "tost_p_value" in tost


class TestTostGroupBy:
    """Tests for TOST with group_by operations."""

    def test_tost_two_sample_group_by(self):
        """Test two-sample TOST with group_by."""
        np.random.seed(42)
        df = pl.DataFrame({
            "group": ["A"] * 50 + ["B"] * 50,
            "x": np.random.randn(100).tolist(),
            "y": np.random.randn(100).tolist(),
        })

        result = df.group_by("group").agg(
            ps.tost_t_test_two_sample("x", "y", delta=0.5).alias("tost")
        ).sort("group")

        assert result.shape == (2, 2)
        assert result["group"].to_list() == ["A", "B"]

        for tost in result["tost"]:
            assert "tost_p_value" in tost
            assert "equivalent" in tost
