"""Tests for result ergonomics (Phase 9: ERGO-01, ERGO-02, ERGO-03).

Covers:
* ``to_dict()`` on a representative sample of model + test classes (ERGO-01).
* ``__repr__`` and ``.summary()`` behaviour, including unfitted state (ERGO-02).
* ``ps.unnest`` / ``ps.struct_to_dict`` helpers for the expression Struct-column
  surface (ERGO-01 for expressions + ERGO-03).
"""

import numpy as np
import polars as pl
import polars_statistics as ps
import pytest
from polars_statistics import (
    OLS,
    ElasticNet,
    Huber,
    Logistic,
    MannWhitneyU,
    Quantile,
    Ridge,
    ShapiroWilk,
    TTestInd,
    TTestPaired,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def reg_data():
    rng = np.random.RandomState(0)
    x = rng.randn(60, 3)
    y = x @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.randn(60)
    return x, y


@pytest.fixture
def binary_data():
    rng = np.random.RandomState(1)
    x = rng.randn(120, 2)
    logits = x @ np.array([1.2, -0.8])
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.rand(120) < probs).astype(float)
    return x, y


@pytest.fixture
def two_samples():
    rng = np.random.RandomState(2)
    return rng.randn(40), rng.randn(40) + 0.6


# =============================================================================
# ERGO-02: __repr__ never panics on unfitted models
# =============================================================================

_UNFITTED_MODELS = [
    lambda: OLS(),
    lambda: Ridge(),
    lambda: ElasticNet(),
    lambda: Huber(),
    lambda: Quantile(),
    lambda: TTestInd(),
    lambda: TTestPaired(),
    lambda: MannWhitneyU(),
    lambda: ShapiroWilk(),
]


@pytest.mark.parametrize("factory", _UNFITTED_MODELS)
def test_repr_unfitted_does_not_raise(factory):
    obj = factory()
    r = repr(obj)
    assert isinstance(r, str)
    assert obj.__class__.__name__ in r
    # Classes with an is_fitted state should say so.
    if hasattr(obj, "is_fitted"):
        assert "fitted=False" in r


@pytest.mark.parametrize("factory", _UNFITTED_MODELS)
def test_to_dict_unfitted_returns_fitted_false(factory):
    obj = factory()
    d = obj.to_dict()
    assert isinstance(d, dict)
    if hasattr(obj, "is_fitted"):
        assert d == {"fitted": False}


# =============================================================================
# ERGO-01 + ERGO-02: fitted models to_dict / repr / summary
# =============================================================================


def test_ols_to_dict_repr_summary(reg_data):
    x, y = reg_data
    m = OLS().fit(x, y)
    d = m.to_dict()
    assert "coefficients" in d
    assert "r_squared" in d
    assert isinstance(d["r_squared"], float)
    assert isinstance(d["coefficients"], np.ndarray)
    # values match the getters
    assert d["r_squared"] == m.r_squared
    r = repr(m)
    assert "OLS(fitted=True" in r
    assert "r_squared=" in r
    s = m.summary()
    assert "OLS Regression Results" in s


def test_ridge_to_dict_and_generic_summary(reg_data):
    x, y = reg_data
    m = Ridge(lambda_=1.0).fit(x, y)
    d = m.to_dict()
    assert "coefficients" in d
    assert "r_squared" in d
    # Ridge has no bespoke summary -> generic summary lists fields.
    s = m.summary()
    assert s.startswith("Ridge Results")
    assert "r_squared" in s
    assert "Ridge(fitted=True" in repr(m)


def test_logistic_to_dict(binary_data):
    x, y = binary_data
    m = Logistic().fit(x, y)
    d = m.to_dict()
    assert isinstance(d, dict)
    assert len(d) > 0
    assert "coefficients" in d


def test_ttest_ind_to_dict_repr(two_samples):
    x, y = two_samples
    t = TTestInd().fit(x, y)
    d = t.to_dict()
    assert "statistic" in d
    assert "p_value" in d
    assert d["statistic"] == t.statistic
    assert d["p_value"] == t.p_value
    r = repr(t)
    assert "statistic=" in r
    assert "p_value=" in r
    # bespoke summary retained
    assert "Independent Samples T-Test" in t.summary()


def test_mann_whitney_to_dict(two_samples):
    x, y = two_samples
    t = MannWhitneyU().fit(x, y)
    d = t.to_dict()
    assert "statistic" in d
    assert "p_value" in d


def test_shapiro_to_dict_repr():
    rng = np.random.RandomState(3)
    t = ShapiroWilk().fit(rng.randn(50))
    d = t.to_dict()
    assert "statistic" in d
    assert "p_value" in d
    assert "ShapiroWilk(fitted=True" in repr(t)


def test_to_dict_values_are_json_like(reg_data):
    """Every OLS to_dict value should be a plain scalar, None, or ndarray."""
    x, y = reg_data
    d = OLS().fit(x, y).to_dict()
    for k, v in d.items():
        assert v is None or isinstance(v, (int, float, bool, np.ndarray)), (
            f"{k} has unexpected type {type(v)}"
        )


# =============================================================================
# ERGO-01 (expressions) + ERGO-03: unnest / struct_to_dict helpers
# =============================================================================


@pytest.fixture
def ttest_result_df():
    df = pl.DataFrame(
        {"y": [1.0, 2, 3, 4, 5, 6], "x": [2.0, 4, 5, 9, 10, 13]}
    )
    return df.select(ps.ttest_ind("y", "x").alias("test"))


def test_struct_to_dict_single_row(ttest_result_df):
    d = ps.struct_to_dict(ttest_result_df)
    assert isinstance(d, dict)
    assert set(d) == {"statistic", "p_value"}
    assert isinstance(d["statistic"], float)


def test_struct_to_dict_explicit_column(ttest_result_df):
    d = ps.struct_to_dict(ttest_result_df, "test")
    assert set(d) == {"statistic", "p_value"}


def test_unnest_basic(ttest_result_df):
    out = ps.unnest(ttest_result_df)
    assert isinstance(out, pl.DataFrame)
    assert set(out.columns) == {"statistic", "p_value"}
    assert out.height == 1


def test_unnest_prefix(ttest_result_df):
    out = ps.unnest(ttest_result_df, prefix="ttest_")
    assert set(out.columns) == {"ttest_statistic", "ttest_p_value"}


def test_unnest_keeps_other_columns():
    df = pl.DataFrame(
        {
            "g": ["a", "a", "a", "b", "b", "b"],
            "y": [1.0, 2, 3, 1, 5, 9],
            "x": [2.0, 4, 6, 2, 3, 4],
        }
    )
    grouped = df.group_by("g", maintain_order=True).agg(
        ps.ttest_ind("y", "x").alias("test")
    )
    out = ps.unnest(grouped, "test")
    assert "g" in out.columns
    assert "statistic" in out.columns
    assert out.height == 2


def test_struct_to_dict_multi_row_returns_list():
    df = pl.DataFrame(
        {
            "g": ["a", "a", "a", "b", "b", "b"],
            "y": [1.0, 2, 3, 1, 5, 9],
            "x": [2.0, 4, 6, 2, 3, 4],
        }
    )
    grouped = df.group_by("g", maintain_order=True).agg(
        ps.ttest_ind("y", "x").alias("test")
    )
    rows = ps.struct_to_dict(grouped.select("test"))
    assert isinstance(rows, list)
    assert len(rows) == 2
    assert all("statistic" in r for r in rows)


def test_unnest_errors_on_non_struct():
    df = pl.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        ps.unnest(df)


def test_unnest_errors_on_wrong_column():
    df = pl.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        ps.unnest(df, "does_not_exist")


def test_unnest_auto_resolves_single_struct(ttest_result_df):
    # No column arg, single struct column -> resolves automatically.
    out = ps.unnest(ttest_result_df)
    assert set(out.columns) == {"statistic", "p_value"}
