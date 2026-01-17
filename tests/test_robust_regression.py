"""Tests for robust regression and diagnostic expressions."""

import math
import numpy as np
import polars as pl
import pytest

import polars_statistics as ps


class TestAIDDemandClassification:
    """Tests for AID (Automatic Identification of Demand) functions."""

    def test_aid_basic(self):
        """Test basic AID classification."""
        np.random.seed(42)
        # Smooth demand pattern
        demand = np.random.poisson(10, 50).tolist()

        df = pl.DataFrame({"demand": demand})

        result = df.select(ps.aid("demand").alias("classification"))

        assert result.shape == (1, 1)
        classification = result["classification"][0]
        assert "demand_type" in classification
        assert "distribution" in classification

    def test_aid_intermittent(self):
        """Test AID with intermittent demand (many zeros)."""
        # Intermittent demand with many zeros
        demand = [0, 0, 5, 0, 0, 0, 3, 0, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0, 0, 0]

        df = pl.DataFrame({"demand": demand})

        result = df.select(
            ps.aid("demand", intermittent_threshold=0.3).alias("classification")
        )

        classification = result["classification"][0]
        assert "demand_type" in classification

    def test_aid_anomalies_basic(self):
        """Test AID anomaly detection."""
        np.random.seed(42)
        # Normal demand with some potential anomalies
        demand = [10, 12, 11, 9, 100, 10, 11, 0, 0, 0, 12, 10]

        df = pl.DataFrame({"demand": demand})

        result = df.select(ps.aid_anomalies("demand").alias("anomalies"))

        # Returns per-row results
        assert result.shape[0] == len(demand)
        anomalies = result["anomalies"][0]
        # Should have boolean flags for different anomaly types
        assert isinstance(anomalies, dict)

    def test_aid_group_by(self):
        """Test AID with group_by."""
        np.random.seed(42)

        df = pl.DataFrame({
            "sku": ["A"] * 20 + ["B"] * 20,
            "demand": np.random.poisson(5, 40).tolist(),
        })

        result = df.group_by("sku").agg(
            ps.aid("demand").alias("classification")
        ).sort("sku")

        assert result.shape[0] == 2


class TestLmDynamic:
    """Tests for dynamic linear models."""

    def test_lm_dynamic_basic(self):
        """Test basic dynamic linear model."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2.0 + 1.5 * x + np.random.randn(n) * 0.5

        df = pl.DataFrame({
            "y": y.tolist(),
            "x": x.tolist(),
        })

        result = df.select(ps.lm_dynamic("y", "x").alias("model"))

        assert result.shape == (1, 1)
        model = result["model"][0]
        assert isinstance(model, dict)

    def test_lm_dynamic_multiple_features(self):
        """Test dynamic linear model with multiple features."""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 1.0 + 2.0 * x1 + 0.5 * x2 + np.random.randn(n) * 0.5

        df = pl.DataFrame({
            "y": y.tolist(),
            "x1": x1.tolist(),
            "x2": x2.tolist(),
        })

        result = df.select(ps.lm_dynamic("y", "x1", "x2").alias("model"))

        assert result.shape == (1, 1)


class TestQuantileRegression:
    """Tests for quantile regression."""

    def test_quantile_basic(self):
        """Test basic quantile regression (median)."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2.0 + 1.5 * x + np.random.randn(n) * 0.5

        df = pl.DataFrame({
            "y": y.tolist(),
            "x": x.tolist(),
        })

        result = df.select(ps.quantile("y", "x", tau=0.5).alias("model"))

        assert result.shape == (1, 1)
        model = result["model"][0]
        assert "intercept" in model
        assert "coefficients" in model
        assert "tau" in model
        assert model["tau"] == 0.5

    def test_quantile_different_tau(self):
        """Test quantile regression with different tau values."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2.0 + 1.5 * x + np.random.randn(n) * 0.5

        df = pl.DataFrame({
            "y": y.tolist(),
            "x": x.tolist(),
        })

        # Lower quartile
        result_25 = df.select(ps.quantile("y", "x", tau=0.25).alias("model"))
        model_25 = result_25["model"][0]
        assert model_25["tau"] == 0.25

        # Upper quartile
        result_75 = df.select(ps.quantile("y", "x", tau=0.75).alias("model"))
        model_75 = result_75["model"][0]
        assert model_75["tau"] == 0.75

    def test_quantile_multiple_features(self):
        """Test quantile regression with multiple features."""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 1.0 + 2.0 * x1 + 0.5 * x2 + np.random.randn(n) * 0.5

        df = pl.DataFrame({
            "y": y.tolist(),
            "x1": x1.tolist(),
            "x2": x2.tolist(),
        })

        result = df.select(ps.quantile("y", "x1", "x2", tau=0.5).alias("model"))

        model = result["model"][0]
        assert len(model["coefficients"]) == 2

    def test_quantile_group_by(self):
        """Test quantile regression with group_by."""
        np.random.seed(42)
        n = 50

        df = pl.DataFrame({
            "group": ["A"] * n + ["B"] * n,
            "y": np.random.randn(n * 2).tolist(),
            "x": np.random.randn(n * 2).tolist(),
        })

        result = df.group_by("group").agg(
            ps.quantile("y", "x", tau=0.5).alias("model")
        ).sort("group")

        assert result.shape[0] == 2
        for model in result["model"]:
            assert "intercept" in model
            assert "tau" in model


class TestIsotonicRegression:
    """Tests for isotonic regression."""

    def test_isotonic_basic(self):
        """Test basic isotonic regression."""
        np.random.seed(42)
        n = 50
        x = np.linspace(0, 10, n)
        # Monotonically increasing with noise
        y = x + np.random.randn(n) * 0.5

        df = pl.DataFrame({
            "y": y.tolist(),
            "x": x.tolist(),
        })

        result = df.select(ps.isotonic("y", "x", increasing=True).alias("model"))

        assert result.shape == (1, 1)
        model = result["model"][0]
        assert "r_squared" in model
        assert "increasing" in model
        assert model["increasing"] is True

    def test_isotonic_decreasing(self):
        """Test decreasing isotonic regression."""
        np.random.seed(42)
        n = 50
        x = np.linspace(0, 10, n)
        # Monotonically decreasing with noise
        y = 10 - x + np.random.randn(n) * 0.5

        df = pl.DataFrame({
            "y": y.tolist(),
            "x": x.tolist(),
        })

        result = df.select(ps.isotonic("y", "x", increasing=False).alias("model"))

        model = result["model"][0]
        assert model["increasing"] is False

    def test_isotonic_group_by(self):
        """Test isotonic regression with group_by."""
        np.random.seed(42)
        n = 30

        df = pl.DataFrame({
            "group": ["A"] * n + ["B"] * n,
            "y": np.random.randn(n * 2).tolist(),
            "x": np.linspace(0, 10, n * 2).tolist(),
        })

        result = df.group_by("group").agg(
            ps.isotonic("y", "x").alias("model")
        ).sort("group")

        assert result.shape[0] == 2


class TestDiagnostics:
    """Tests for regression diagnostic expressions."""

    def test_condition_number_well_conditioned(self):
        """Test condition number for well-conditioned data."""
        np.random.seed(42)
        n = 100
        # Orthogonal-ish predictors
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)

        df = pl.DataFrame({
            "x1": x1.tolist(),
            "x2": x2.tolist(),
        })

        result = df.select(ps.condition_number("x1", "x2").alias("cond"))

        assert result.shape == (1, 1)
        cond = result["cond"][0]
        # Returns a struct with condition_number field
        assert "condition_number" in cond
        assert "severity" in cond
        # Well-conditioned data should have low condition number
        assert cond["condition_number"] > 0
        assert cond["severity"] == "WellConditioned"

    def test_condition_number_collinear(self):
        """Test condition number detects collinearity."""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        # x2 is almost identical to x1 (high collinearity)
        x2 = x1 + np.random.randn(n) * 0.001

        df = pl.DataFrame({
            "x1": x1.tolist(),
            "x2": x2.tolist(),
        })

        result = df.select(ps.condition_number("x1", "x2").alias("cond"))

        cond = result["cond"][0]
        # Collinear data should have high condition number
        assert cond["condition_number"] > 100
        # Severity should indicate a problem
        assert cond["severity"] in ["Moderate", "Serious", "Severe"]

    def test_check_binary_separation(self):
        """Test binary separation check."""
        np.random.seed(42)
        n = 50
        # Create data without complete separation
        x = np.random.randn(n)
        # Probability depends on x but has overlap
        prob = 1 / (1 + np.exp(-x))
        y = (np.random.rand(n) < prob).astype(int)

        df = pl.DataFrame({
            "y": y.tolist(),
            "x": x.tolist(),
        })

        result = df.select(ps.check_binary_separation("y", "x").alias("sep"))

        assert result.shape == (1, 1)
        sep = result["sep"][0]
        assert "has_separation" in sep

    def test_check_binary_separation_complete(self):
        """Test detection of complete separation."""
        # Create data with complete separation
        # All negative x -> y=0, all positive x -> y=1
        x = [-3, -2, -1, 1, 2, 3]
        y = [0, 0, 0, 1, 1, 1]

        df = pl.DataFrame({
            "y": y,
            "x": x,
        })

        result = df.select(ps.check_binary_separation("y", "x").alias("sep"))

        sep = result["sep"][0]
        # Should detect complete separation
        assert sep["has_separation"] is True

    def test_check_count_sparsity(self):
        """Test count sparsity check for non-sparse data."""
        np.random.seed(42)
        n = 100
        # Count data with reasonable counts
        y = np.random.poisson(5, n)
        x = np.random.randn(n)

        df = pl.DataFrame({
            "y": y.tolist(),
            "x": x.tolist(),
        })

        result = df.select(ps.check_count_sparsity("y", "x").alias("sparse"))

        assert result.shape == (1, 1)
        sparse = result["sparse"][0]
        assert "has_separation" in sparse
        # Normal Poisson data shouldn't have separation
        assert sparse["has_separation"] is False

    def test_check_count_sparsity_multiple_predictors(self):
        """Test sparsity check with multiple predictors."""
        np.random.seed(42)
        n = 100
        y = np.random.poisson(3, n)
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)

        df = pl.DataFrame({
            "y": y.tolist(),
            "x1": x1.tolist(),
            "x2": x2.tolist(),
        })

        result = df.select(ps.check_count_sparsity("y", "x1", "x2").alias("sparse"))

        sparse = result["sparse"][0]
        assert "has_separation" in sparse
        assert "separated_predictors" in sparse
        assert "warning" in sparse
