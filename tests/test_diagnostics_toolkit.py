"""Tests for the diagnostics toolkit expressions: VIF, leverage, Cook's distance.

These tests check the property contracts of the diagnostics rather than exact
numerical values (which are validated upstream in anofox-regression).
"""

from __future__ import annotations

import numpy as np
import polars as pl

import polars_statistics as ps


def test_vif_orthogonal_predictors_near_one():
    """Independent predictors should yield VIF values near 1.0."""
    rng = np.random.default_rng(seed=42)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)

    df = pl.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    result = df.select(ps.vif("x1", "x2", "x3").alias("vif"))

    row = result["vif"][0]
    terms = list(row["terms"])
    vif_values = list(row["vif"])
    n_obs = row["n_observations"]

    assert n_obs == n
    assert terms == ["x1", "x2", "x3"]
    assert len(vif_values) == 3
    for v in vif_values:
        # Independent draws should give VIF very close to 1.
        assert v >= 1.0 - 1e-9
        assert v < 1.5, f"VIF should be near 1 for independent predictors, got {v}"


def test_vif_collinear_predictors_high():
    """Highly collinear predictors should yield VIF > 5 (typical concern threshold)."""
    rng = np.random.default_rng(seed=123)
    n = 150
    x1 = rng.standard_normal(n)
    # x2 is x1 with a tiny perturbation -> near-perfect collinearity.
    x2 = x1 + 0.01 * rng.standard_normal(n)
    x3 = rng.standard_normal(n)  # independent

    df = pl.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    result = df.select(ps.vif("x1", "x2", "x3").alias("vif"))

    vif_values = list(result["vif"][0]["vif"])
    assert vif_values[0] > 5.0
    assert vif_values[1] > 5.0
    # x3 should be unaffected.
    assert vif_values[2] < 1.5


def test_leverage_sum_equals_n_params():
    """Sum of leverage values equals the number of parameters (hat-matrix trace)."""
    rng = np.random.default_rng(seed=7)
    n = 120
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)

    df = pl.DataFrame({"x1": x1, "x2": x2})

    # With intercept: trace(H) = p = 3 (intercept + 2 features).
    res_with = df.select(ps.leverage("x1", "x2", add_intercept=True).alias("h"))
    lev_with = list(res_with["h"][0]["leverage"])
    assert len(lev_with) == n
    assert all(0.0 <= h <= 1.0 for h in lev_with)
    assert abs(sum(lev_with) - 3.0) < 1e-6

    # Without intercept: trace(H) = p = 2.
    res_no = df.select(ps.leverage("x1", "x2", add_intercept=False).alias("h"))
    lev_no = list(res_no["h"][0]["leverage"])
    assert len(lev_no) == n
    assert abs(sum(lev_no) - 2.0) < 1e-6


def test_cooks_distance_non_negative_one_per_row():
    """Cook's distance returns one non-negative value per input row."""
    rng = np.random.default_rng(seed=99)
    n = 80
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + 0.3 * rng.standard_normal(n)

    df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})
    result = df.select(ps.cooks_distance("y", "x1", "x2").alias("cd"))

    row = result["cd"][0]
    cd_values = list(row["cooks_d"])
    n_obs = row["n_observations"]

    assert n_obs == n
    assert len(cd_values) == n
    # All Cook's distances are non-negative by construction.
    for v in cd_values:
        assert v >= 0.0, f"Cook's distance must be non-negative, got {v}"
        assert np.isfinite(v), f"Cook's distance must be finite, got {v}"


def test_cooks_distance_flags_outlier():
    """Injecting an outlier should produce a high Cook's distance at that row."""
    rng = np.random.default_rng(seed=2024)
    n = 100
    x = rng.standard_normal(n)
    y = 1.0 + 1.5 * x + 0.1 * rng.standard_normal(n)
    # Inject a big residual at the last row.
    y[-1] = y[-1] + 20.0

    df = pl.DataFrame({"y": y, "x": x})
    result = df.select(ps.cooks_distance("y", "x").alias("cd"))
    cd_values = np.asarray(list(result["cd"][0]["cooks_d"]))

    # The injected row should have the maximum Cook's distance.
    assert int(np.argmax(cd_values)) == n - 1
    assert cd_values[-1] > np.median(cd_values) * 10
