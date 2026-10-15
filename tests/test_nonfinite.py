"""Non-finite input handling for anofox-statistics 0.4.4 #17.

Every test in this file has "nonfinite" in its name so `pytest -k nonfinite`
selects the whole set (see 16.1-VALIDATION.md). 0.4.4 converted ~20 internal
sort-site panics into catchable errors / NaN results when fed NaN/+-Inf/constant
data — this file proves no exposed test class or expression can crash the
Python process with a Rust panic (`pyo3_runtime.PanicException`) on such input,
and pins a handful of exact reference values along the way.
"""

import math

import numpy as np
import polars as pl
import polars_statistics as ps
import pytest
from polars_statistics import (
    BrownForsythe,
    BrunnerMunzel,
    DAgostino,
    KruskalWallis,
    MannWhitneyU,
    ShapiroWilk,
    TTestInd,
    TTestPaired,
    WilcoxonSignedRank,
    YuenTest,
)


def test_wilcoxon_signed_rank_nonfinite_positive_inf_in_x():
    model = WilcoxonSignedRank().fit(
        np.array([np.inf, 1.0, 2.0, 3.0, 4.0]), np.array([np.inf, 2.0, 1.0, 5.0, 6.0])
    )
    assert model.statistic == pytest.approx(1.5, abs=1e-9)
    assert model.p_value == pytest.approx(0.265205392591508, abs=1e-9)


def test_wilcoxon_signed_rank_nonfinite_negative_inf_in_y():
    model = WilcoxonSignedRank().fit(
        np.array([np.inf, 1.0, 2.0, 3.0, 4.0]), np.array([-np.inf, 2.0, 1.0, 5.0, 6.0])
    )
    assert model.statistic == pytest.approx(6.5, abs=1e-9)
    assert model.p_value == pytest.approx(0.891755853506794, abs=1e-9)


def test_wilcoxon_signed_rank_nonfinite_exact_vs_asymptotic():
    x = np.array([np.inf, 1.0, 2.0, 3.0, 4.0, 7.0])
    zeros = np.zeros(6)

    exact = WilcoxonSignedRank(exact=True).fit(x, zeros)
    assert exact.statistic == pytest.approx(21.0, abs=1e-12)
    assert exact.p_value == pytest.approx(0.03125, abs=1e-12)

    asymptotic = WilcoxonSignedRank(exact=False).fit(x, zeros)
    assert asymptotic.p_value == pytest.approx(0.0360316862182335, abs=1e-9)


def test_mann_whitney_nonfinite_exact_vs_asymptotic():
    x = np.array([np.inf, 1.5, 2.0, 3.0, 4.0])
    y = np.array([-np.inf, 2.5, 1.0, 5.0, 6.0])

    exact = MannWhitneyU(exact=True).fit(x, y)
    assert exact.statistic == pytest.approx(15.0, abs=1e-12)
    assert exact.p_value == pytest.approx(0.69047619047619, abs=1e-12)

    asymptotic = MannWhitneyU(exact=False).fit(x, y)
    assert asymptotic.p_value == pytest.approx(0.676103314023147, abs=1e-9)


def test_mann_whitney_nonfinite_estimate_is_nan():
    model = MannWhitneyU(conf_level=0.95).fit(
        np.array([np.inf, 1.0, 2.0, 3.0, 4.0]), np.array([np.inf, 2.0, 1.0, 5.0, 6.0])
    )
    assert math.isnan(model.estimate)


def test_kruskal_wallis_nonfinite_matches_r():
    model = KruskalWallis().fit(
        np.array([np.inf, 1.0, 2.0]), np.array([-np.inf, 3.0, 4.0]), np.array([5.0, 6.0, 7.0])
    )
    assert model.statistic == pytest.approx(2.75555555555555, abs=1e-9)
    assert model.p_value == pytest.approx(0.252138238153004, abs=1e-9)


def test_mann_whitney_nonfinite_nan_raises_value_error():
    with pytest.raises(ValueError, match="non-finite"):
        MannWhitneyU().fit(np.array([np.nan, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_wilcoxon_signed_rank_nonfinite_nan_raises_value_error():
    with pytest.raises(ValueError, match="non-finite"):
        WilcoxonSignedRank().fit(np.array([np.nan, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_wilcoxon_signed_rank_nonfinite_all_infinite_raises_value_error():
    with pytest.raises(ValueError):
        WilcoxonSignedRank().fit(np.array([np.inf] * 3), np.array([np.inf] * 3))


def test_ttest_ind_nonfinite_constant_data_gives_nan_p_value():
    for equal_var in (True, False):
        model = TTestInd(equal_var=equal_var).fit(np.array([3.0] * 6), np.array([3.0] * 6))
        assert math.isnan(model.p_value)


def test_ttest_paired_nonfinite_constant_data_gives_nan_p_value():
    model = TTestPaired().fit(np.array([3.0] * 6), np.array([3.0] * 6))
    assert math.isnan(model.p_value)


def test_pearson_nonfinite_constant_data_gives_nan():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [5.0] * 5})
    result = df.select(ps.pearson("x", "y").alias("r")).item()
    assert math.isnan(result["estimate"])
    assert math.isnan(result["p_value"])


# One-sample, two-sample, and k-sample dispatch helpers for the no-panic sweep below.
_ONE_SAMPLE_CLASSES = {"ShapiroWilk": ShapiroWilk, "DAgostino": DAgostino}
_TWO_SAMPLE_CLASSES = {
    "TTestInd": TTestInd,
    "TTestPaired": TTestPaired,
    "MannWhitneyU": MannWhitneyU,
    "WilcoxonSignedRank": WilcoxonSignedRank,
    "BrunnerMunzel": BrunnerMunzel,
    "YuenTest": YuenTest,
    "BrownForsythe": BrownForsythe,
}
_K_SAMPLE_CLASSES = {"KruskalWallis": KruskalWallis}

# Length 25 so DAgostino's n>=20 requirement is satisfied by every pattern.
_N = 25
_BASE = np.random.RandomState(0).randn(_N) + 10.0


def _pattern(kind: str) -> np.ndarray:
    arr = _BASE.copy()
    if kind == "nan":
        arr[5] = np.nan
    elif kind == "inf":
        arr[5] = np.inf
        arr[10] = -np.inf
    elif kind == "constant":
        arr = np.full(_N, 3.0)
    return arr


@pytest.mark.parametrize("pattern", ["nan", "inf", "constant"])
@pytest.mark.parametrize("name", sorted(_ONE_SAMPLE_CLASSES))
def test_one_sample_classes_nonfinite_sweep_never_panics(name, pattern):
    cls = _ONE_SAMPLE_CLASSES[name]
    x = _pattern(pattern)
    try:
        model = cls().fit(x)
    except ValueError:
        return
    assert isinstance(model.p_value, float)


@pytest.mark.parametrize("pattern", ["nan", "inf", "constant"])
@pytest.mark.parametrize("name", sorted(_TWO_SAMPLE_CLASSES))
def test_two_sample_classes_nonfinite_sweep_never_panics(name, pattern):
    cls = _TWO_SAMPLE_CLASSES[name]
    x = _pattern(pattern)
    y = _pattern(pattern)
    try:
        model = cls().fit(x, y)
    except ValueError:
        return
    assert isinstance(model.p_value, float)


@pytest.mark.parametrize("pattern", ["nan", "inf", "constant"])
@pytest.mark.parametrize("name", sorted(_K_SAMPLE_CLASSES))
def test_k_sample_classes_nonfinite_sweep_never_panics(name, pattern):
    cls = _K_SAMPLE_CLASSES[name]
    g1 = _pattern(pattern)
    g2 = _pattern(pattern)
    g3 = _BASE.copy()
    try:
        model = cls().fit(g1, g2, g3)
    except ValueError:
        return
    assert isinstance(model.p_value, float)


def test_mann_whitney_u_expression_nonfinite_matches_filtered_columns():
    """The Python builders filter non-finite values independently per column
    before calling the plugin, so a column containing NaN/Inf must give the
    same result as that column with non-finite rows pre-removed."""
    x = [1.0, 2.0, float("nan"), 4.0, float("inf")]
    y = [2.0, 3.0, 5.0, 6.0, 7.0, float("nan")]

    max_len = max(len(x), len(y))
    xp = x + [None] * (max_len - len(x))
    yp = y + [None] * (max_len - len(y))
    df_raw = pl.DataFrame({"x": xp, "y": yp})
    result_raw = df_raw.select(
        ps.mann_whitney_u(pl.col("x").drop_nulls(), pl.col("y").drop_nulls()).alias("r")
    ).item()

    xf = [v for v in x if math.isfinite(v)]
    yf = [v for v in y if math.isfinite(v)]
    max_len_f = max(len(xf), len(yf))
    xfp = xf + [None] * (max_len_f - len(xf))
    yfp = yf + [None] * (max_len_f - len(yf))
    df_filtered = pl.DataFrame({"x": xfp, "y": yfp})
    result_filtered = df_filtered.select(
        ps.mann_whitney_u(pl.col("x").drop_nulls(), pl.col("y").drop_nulls()).alias("r")
    ).item()

    # Phase 16.2 UPST-07: the struct now carries estimate/ci_lower/ci_upper, which are
    # NaN here (no conf_level requested) -- nan_ok=True so NaN == NaN compares equal.
    assert result_raw == pytest.approx(result_filtered, nan_ok=True)
