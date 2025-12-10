"""Tests for statistical test model classes."""

import numpy as np
import pytest

from polars_statistics import (
    TTestInd,
    TTestPaired,
    BrownForsythe,
    YuenTest,
    MannWhitneyU,
    WilcoxonSignedRank,
    KruskalWallis,
    BrunnerMunzel,
    ShapiroWilk,
    DAgostino,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_data():
    """Generate sample data for tests."""
    np.random.seed(42)
    return {
        "x": np.random.randn(50),
        "y": np.random.randn(50) + 0.5,
        "z": np.random.randn(50) + 1.0,
    }


@pytest.fixture
def paired_data():
    """Generate paired sample data."""
    np.random.seed(42)
    n = 30
    before = np.random.randn(n)
    after = before + 0.5 + np.random.randn(n) * 0.3
    return {"before": before, "after": after}


@pytest.fixture
def normal_data():
    """Generate normally distributed data."""
    np.random.seed(42)
    return np.random.randn(100)


@pytest.fixture
def nonnormal_data():
    """Generate non-normally distributed data."""
    np.random.seed(42)
    return np.random.exponential(1.0, 100)


# =============================================================================
# Parametric Tests
# =============================================================================


class TestTTestInd:
    """Tests for TTestInd class."""

    def test_basic_fit(self, sample_data):
        """Test basic fitting."""
        test = TTestInd()
        test.fit(sample_data["x"], sample_data["y"])
        assert test.is_fitted()
        assert isinstance(test.statistic, float)
        assert isinstance(test.p_value, float)

    def test_alternatives(self, sample_data):
        """Test different alternatives."""
        for alt in ["two-sided", "less", "greater"]:
            test = TTestInd(alternative=alt)
            test.fit(sample_data["x"], sample_data["y"])
            assert test.is_fitted()

    def test_equal_var(self, sample_data):
        """Test with equal variance assumption."""
        test = TTestInd(equal_var=True)
        test.fit(sample_data["x"], sample_data["y"])
        assert test.is_fitted()

    def test_summary(self, sample_data):
        """Test summary output."""
        test = TTestInd()
        test.fit(sample_data["x"], sample_data["y"])
        summary = test.summary()
        assert "T-Test" in summary
        assert "statistic" in summary.lower()
        assert "p-value" in summary.lower()

    def test_not_fitted_error(self):
        """Test error when not fitted."""
        test = TTestInd()
        with pytest.raises(RuntimeError):
            _ = test.statistic


class TestTTestPaired:
    """Tests for TTestPaired class."""

    def test_basic_fit(self, paired_data):
        """Test basic fitting."""
        test = TTestPaired()
        test.fit(paired_data["before"], paired_data["after"])
        assert test.is_fitted()
        assert isinstance(test.statistic, float)
        assert isinstance(test.p_value, float)

    def test_alternatives(self, paired_data):
        """Test different alternatives."""
        for alt in ["two-sided", "less", "greater"]:
            test = TTestPaired(alternative=alt)
            test.fit(paired_data["before"], paired_data["after"])
            assert test.is_fitted()

    def test_summary(self, paired_data):
        """Test summary output."""
        test = TTestPaired()
        test.fit(paired_data["before"], paired_data["after"])
        summary = test.summary()
        assert "Paired" in summary
        assert "statistic" in summary.lower()

    def test_unequal_lengths_error(self, sample_data):
        """Test error with unequal sample lengths."""
        test = TTestPaired()
        with pytest.raises(ValueError):
            test.fit(sample_data["x"], sample_data["y"][:40])


class TestBrownForsythe:
    """Tests for BrownForsythe class."""

    def test_basic_fit(self, sample_data):
        """Test basic fitting."""
        test = BrownForsythe()
        test.fit(sample_data["x"], sample_data["y"])
        assert test.is_fitted()
        assert isinstance(test.statistic, float)
        assert isinstance(test.p_value, float)

    def test_summary(self, sample_data):
        """Test summary output."""
        test = BrownForsythe()
        test.fit(sample_data["x"], sample_data["y"])
        summary = test.summary()
        assert "Brown-Forsythe" in summary
        assert "variance" in summary.lower()


class TestYuenTest:
    """Tests for YuenTest class."""

    def test_basic_fit(self, sample_data):
        """Test basic fitting."""
        test = YuenTest()
        test.fit(sample_data["x"], sample_data["y"])
        assert test.is_fitted()
        assert isinstance(test.statistic, float)
        assert isinstance(test.p_value, float)

    def test_custom_trim(self, sample_data):
        """Test with custom trim proportion."""
        test = YuenTest(trim=0.1)
        test.fit(sample_data["x"], sample_data["y"])
        assert test.is_fitted()

    def test_summary(self, sample_data):
        """Test summary output."""
        test = YuenTest()
        test.fit(sample_data["x"], sample_data["y"])
        summary = test.summary()
        assert "Yuen" in summary
        assert "trim" in summary.lower()


# =============================================================================
# Non-Parametric Tests
# =============================================================================


class TestMannWhitneyU:
    """Tests for MannWhitneyU class."""

    def test_basic_fit(self, sample_data):
        """Test basic fitting."""
        test = MannWhitneyU()
        test.fit(sample_data["x"], sample_data["y"])
        assert test.is_fitted()
        assert isinstance(test.statistic, float)
        assert isinstance(test.p_value, float)

    def test_summary(self, sample_data):
        """Test summary output."""
        test = MannWhitneyU()
        test.fit(sample_data["x"], sample_data["y"])
        summary = test.summary()
        assert "Mann-Whitney" in summary
        assert "U statistic" in summary


class TestWilcoxonSignedRank:
    """Tests for WilcoxonSignedRank class."""

    def test_basic_fit(self, paired_data):
        """Test basic fitting."""
        test = WilcoxonSignedRank()
        test.fit(paired_data["before"], paired_data["after"])
        assert test.is_fitted()
        assert isinstance(test.statistic, float)
        assert isinstance(test.p_value, float)

    def test_summary(self, paired_data):
        """Test summary output."""
        test = WilcoxonSignedRank()
        test.fit(paired_data["before"], paired_data["after"])
        summary = test.summary()
        assert "Wilcoxon" in summary
        assert "W statistic" in summary

    def test_unequal_lengths_error(self, sample_data):
        """Test error with unequal sample lengths."""
        test = WilcoxonSignedRank()
        with pytest.raises(ValueError):
            test.fit(sample_data["x"], sample_data["y"][:40])


class TestKruskalWallis:
    """Tests for KruskalWallis class."""

    def test_basic_fit(self, sample_data):
        """Test basic fitting with three groups."""
        test = KruskalWallis()
        test.fit(sample_data["x"], sample_data["y"], sample_data["z"])
        assert test.is_fitted()
        assert isinstance(test.statistic, float)
        assert isinstance(test.p_value, float)

    def test_two_groups(self, sample_data):
        """Test with two groups."""
        test = KruskalWallis()
        test.fit(sample_data["x"], sample_data["y"])
        assert test.is_fitted()

    def test_summary(self, sample_data):
        """Test summary output."""
        test = KruskalWallis()
        test.fit(sample_data["x"], sample_data["y"], sample_data["z"])
        summary = test.summary()
        assert "Kruskal-Wallis" in summary
        assert "H statistic" in summary

    def test_single_group_error(self, sample_data):
        """Test error with single group."""
        test = KruskalWallis()
        with pytest.raises(ValueError):
            test.fit(sample_data["x"])


class TestBrunnerMunzel:
    """Tests for BrunnerMunzel class."""

    def test_basic_fit(self, sample_data):
        """Test basic fitting."""
        test = BrunnerMunzel()
        test.fit(sample_data["x"], sample_data["y"])
        assert test.is_fitted()
        assert isinstance(test.statistic, float)
        assert isinstance(test.p_value, float)

    def test_alternatives(self, sample_data):
        """Test different alternatives."""
        for alt in ["two-sided", "less", "greater"]:
            test = BrunnerMunzel(alternative=alt)
            test.fit(sample_data["x"], sample_data["y"])
            assert test.is_fitted()

    def test_summary(self, sample_data):
        """Test summary output."""
        test = BrunnerMunzel()
        test.fit(sample_data["x"], sample_data["y"])
        summary = test.summary()
        assert "Brunner-Munzel" in summary
        assert "stochastic" in summary.lower()


# =============================================================================
# Distributional Tests
# =============================================================================


class TestShapiroWilk:
    """Tests for ShapiroWilk class."""

    def test_normal_data(self, normal_data):
        """Test with normal data (should have high p-value)."""
        test = ShapiroWilk()
        test.fit(normal_data)
        assert test.is_fitted()
        assert isinstance(test.statistic, float)
        assert isinstance(test.p_value, float)
        # W statistic should be close to 1 for normal data
        assert test.statistic > 0.9

    def test_nonnormal_data(self, nonnormal_data):
        """Test with non-normal data (should have low p-value)."""
        test = ShapiroWilk()
        test.fit(nonnormal_data)
        assert test.is_fitted()
        # Should typically reject normality
        assert test.p_value < 0.05

    def test_summary(self, normal_data):
        """Test summary output."""
        test = ShapiroWilk()
        test.fit(normal_data)
        summary = test.summary()
        assert "Shapiro-Wilk" in summary
        assert "normal" in summary.lower()

    def test_small_sample_error(self):
        """Test error with too few observations."""
        test = ShapiroWilk()
        with pytest.raises(ValueError):
            test.fit(np.array([1.0, 2.0]))


class TestDAgostino:
    """Tests for DAgostino class."""

    def test_normal_data(self, normal_data):
        """Test with normal data (should have high p-value)."""
        test = DAgostino()
        test.fit(normal_data)
        assert test.is_fitted()
        assert isinstance(test.statistic, float)
        assert isinstance(test.p_value, float)

    def test_nonnormal_data(self, nonnormal_data):
        """Test with non-normal data (should have low p-value)."""
        test = DAgostino()
        test.fit(nonnormal_data)
        assert test.is_fitted()
        # Should typically reject normality
        assert test.p_value < 0.05

    def test_summary(self, normal_data):
        """Test summary output."""
        test = DAgostino()
        test.fit(normal_data)
        summary = test.summary()
        assert "D'Agostino" in summary
        assert "normal" in summary.lower()

    def test_small_sample_error(self):
        """Test error with too few observations."""
        test = DAgostino()
        with pytest.raises(ValueError):
            test.fit(np.random.randn(10))


# =============================================================================
# Chained Fitting Tests
# =============================================================================


class TestChainedFitting:
    """Tests for chained fitting (fit returns self)."""

    def test_ttest_chained(self, sample_data):
        """Test chained fitting for TTestInd."""
        test = TTestInd().fit(sample_data["x"], sample_data["y"])
        assert test.is_fitted()

    def test_mann_whitney_chained(self, sample_data):
        """Test chained fitting for MannWhitneyU."""
        test = MannWhitneyU().fit(sample_data["x"], sample_data["y"])
        assert test.is_fitted()

    def test_shapiro_chained(self, normal_data):
        """Test chained fitting for ShapiroWilk."""
        test = ShapiroWilk().fit(normal_data)
        assert test.is_fitted()
