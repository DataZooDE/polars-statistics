"""Tests for `pl_alm` parity with `PyALM` (issue #16).

The polars `alm()` expression previously supported only 13 of the 25 ALM
distributions, and exposed no link / loss / extra_parameter kwargs.
"""

import numpy as np
import polars as pl
import pytest

from polars_statistics import alm


@pytest.fixture
def linear_data():
    """Mild-signal Gaussian-shaped data — works across loss functions."""
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(size=(n, 2)) * 0.3
    y = 0.5 + 0.4 * x[:, 0] - 0.3 * x[:, 1] + rng.normal(scale=0.1, size=n)
    return pl.DataFrame({"y": y, "x1": x[:, 0], "x2": x[:, 1]})


class TestPreviouslyMissingDistributions:
    """These were in PyALM but not in pl_alm before issue #16."""

    @pytest.mark.parametrize(
        "distribution",
        [
            "asymmetric_laplace",
            "generalised_normal",
            "folded_normal",
            "rectified_normal",
        ],
    )
    def test_distribution_now_supported(self, linear_data, distribution):
        """Each previously-rejected distribution string should now produce a non-NaN fit."""
        result = linear_data.select(
            alm("y", "x1", "x2", distribution=distribution).alias("m")
        ).item()
        # Before #16, parse_alm_distribution returned None for these strings and
        # alm_fit short-circuited to glm_nan_output → n_observations would be 0.
        # After #16, the parser accepts them and the fit produces a real result.
        assert result["n_observations"] > 0

    def test_logit_normal_with_bounded_data(self):
        """logit_normal needs y ∈ (0,1) — verify the parser now accepts the string."""
        rng = np.random.default_rng(7)
        n = 150
        x = rng.normal(size=(n, 1)) * 0.3
        logit = 0.0 + 0.5 * x[:, 0] + rng.normal(scale=0.3, size=n)
        y = 1.0 / (1.0 + np.exp(-logit))  # bounded to (0, 1)
        df = pl.DataFrame({"y": y, "x1": x[:, 0]})
        result = df.select(
            alm("y", "x1", distribution="logit_normal").alias("m")
        ).item()
        assert result["n_observations"] > 0


class TestLossAndLink:
    """The loss / link kwargs are new in #16."""

    def test_mse_loss_runs(self, linear_data):
        result = linear_data.select(
            alm("y", "x1", "x2", distribution="normal", loss="mse").alias("m")
        ).item()
        assert result["n_observations"] > 0

    def test_mae_loss_runs(self, linear_data):
        result = linear_data.select(
            alm("y", "x1", "x2", distribution="normal", loss="mae").alias("m")
        ).item()
        assert result["n_observations"] > 0

    def test_log_link_runs(self):
        """Force a log link on positive Gaussian data — should still fit."""
        rng = np.random.default_rng(0)
        n = 200
        x = rng.normal(size=(n, 1)) * 0.2
        y = np.exp(0.5 + 0.3 * x[:, 0]) + rng.normal(scale=0.05, size=n)
        df = pl.DataFrame({"y": y, "x1": x[:, 0]})
        result = df.select(
            alm("y", "x1", distribution="normal", link="log").alias("m")
        ).item()
        assert result["n_observations"] > 0

    def test_invalid_loss_yields_nan(self, linear_data):
        """Bad loss strings hit the None branch and produce an all-NaN struct."""
        result = linear_data.select(
            alm("y", "x1", "x2", distribution="normal", loss="bogus").alias("m")
        ).item()
        assert result["n_observations"] == 0


class TestDefaultsUnchanged:
    """Pre-#16 callers (distribution-only) should still work."""

    def test_normal_default(self, linear_data):
        result = linear_data.select(alm("y", "x1", "x2").alias("m")).item()
        assert result["n_observations"] > 0
        assert result["intercept"] == pytest.approx(0.5, abs=0.2)

    def test_laplace_distribution(self, linear_data):
        """Laplace was supported pre-#16 — verify no regression."""
        result = linear_data.select(
            alm("y", "x1", "x2", distribution="laplace").alias("m")
        ).item()
        assert result["n_observations"] > 0
