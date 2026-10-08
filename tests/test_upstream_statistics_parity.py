"""R-reference parity tests for anofox-statistics 0.4.3 behaviour changes.

Each test pins a value computed against R 4.6.1 so a future regression in the
backing `anofox-statistics` crate cannot pass silently. See
`.planning/phases/16.1-upstream-crate-refresh/16.1-RESEARCH.md`
("Runtime/Behaviour-Change Inventory") for the upstream changelog entries these
tests encode:

- #7: Shapiro-Wilk p-value for n<=11 is now R-exact.
- #8: Kendall's tau-b tie-correction fix.
- #9: Binomial test exact interval and p-value.
- #10: Proportion test (one- and two-sample) intervals.
- #11: Fisher's exact test p-value and sample odds ratio.
- #12: Exact two-sided Mann-Whitney p-value fix.
- #13: Rank-test continuity at the centre / all-tied edge cases.
"""

import numpy as np
import polars as pl
import polars_statistics as ps
import pytest
from polars_statistics import MannWhitneyU, ShapiroWilk, WilcoxonSignedRank


def test_shapiro_wilk_small_n_matches_r():
    """anofox-statistics 0.4.3 #7: Shapiro-Wilk p-value for n<=11 is R-exact."""
    samples = [
        ([4.1, 5.3, 3.8, 6.9, 5.0, 4.4, 9.2, 5.7, 4.9, 6.1], 0.88790406283629908, 0.16058545199963783),
        (
            [98.059831, 100.258934, 101.1010067, 99.33949313, 102.5085549,
             99.77152429, 98.47658593, 101.3298538, 98.84786198],
            0.96512149278140513,
            0.8500927903451192,
        ),
        ([1.2, 3.4, 2.2, 5.9], None, 0.74255791273476768),
    ]

    for values, expected_w, expected_p in samples:
        model = ShapiroWilk().fit(np.array(values))
        if expected_w is not None:
            assert model.statistic == pytest.approx(expected_w, abs=1e-6)
        assert model.p_value == pytest.approx(expected_p, abs=1e-6)

        df = pl.DataFrame({"x": values})
        expr_result = df.select(ps.shapiro_wilk("x").alias("r")).item()
        assert expr_result["p_value"] == pytest.approx(expected_p, abs=1e-6)


def test_kendall_tau_b_tie_corrected_matches_r():
    """anofox-statistics 0.4.3 #8: Kendall's tau-b tie-correction fix."""
    x = [1, 2, 2, 3, 4, 4, 4, 5, 6, 7, 8, 8]
    y = [2, 1, 3, 3, 5, 4, 6, 6, 5, 8, 7, 9]
    df = pl.DataFrame({"x": [float(v) for v in x], "y": [float(v) for v in y]})

    result = df.select(ps.kendall("x", "y", variant="b").alias("r")).item()

    assert result["estimate"] == pytest.approx(0.80655653082637868, abs=1e-12)
    assert result["statistic"] == pytest.approx(3.4987518043922514, abs=1e-8)
    assert result["p_value"] == pytest.approx(0.00046744148056981807, abs=1e-10)


def test_binom_test_exact_interval_and_p_match_r():
    """anofox-statistics 0.4.3 #9: binomial test exact interval and p-value."""
    r1 = pl.select(ps.binom_test(13, 20).alias("r")).item()
    assert r1["p_value"] == pytest.approx(0.26317596435546875, abs=1e-12)
    assert r1["ci_lower"] == pytest.approx(0.40781146546717190, abs=1e-8)
    assert r1["ci_upper"] == pytest.approx(0.84609079521545882, abs=1e-8)

    r2 = pl.select(ps.binom_test(4500, 10000).alias("r")).item()
    assert abs(r2["p_value"] / 1.5510640568246068e-23 - 1) < 1e-6

    r3 = pl.select(ps.binom_test(13, 20, alternative="less").alias("r")).item()
    assert r3["ci_lower"] == 0.0
    assert r3["ci_upper"] == pytest.approx(0.82268908242555083, abs=1e-8)


def test_prop_test_intervals_match_r():
    """anofox-statistics 0.4.3 #10: one- and two-sample proportion test intervals."""
    r1 = pl.select(ps.prop_test_one(13, 20, alternative="greater").alias("r")).item()
    assert r1["ci_lower"] == pytest.approx(0.46651268884847175, abs=1e-8)
    assert r1["ci_upper"] == 1.0

    r2 = pl.select(ps.prop_test_two(18, 30, 11, 28, correction=True).alias("r")).item()
    assert r2["ci_lower"] == pytest.approx(-0.079284640161607245, abs=1e-8)
    assert r2["ci_upper"] == pytest.approx(0.493570354447321502, abs=1e-8)

    r3 = pl.select(ps.prop_test_two(18, 30, 11, 28, correction=False).alias("r")).item()
    assert r3["ci_lower"] == pytest.approx(-0.044760830637797733, abs=1e-8)
    assert r3["ci_upper"] == pytest.approx(0.459046544923511990, abs=1e-8)


def test_fisher_exact_p_and_sample_odds_ratio_match_r():
    """anofox-statistics 0.4.3 #11: Fisher's exact test p-value and sample odds ratio."""
    result = pl.select(ps.fisher_exact(18, 7, 9, 16).alias("r")).item()

    assert result["p_value"] == pytest.approx(0.022241295691722542, abs=1e-9)
    assert result["statistic"] == pytest.approx(4.5714285714285712, abs=1e-12)


def test_mann_whitney_exact_two_sided_upper_tail_matches_r():
    """anofox-statistics 0.4.3 #12: exact two-sided Mann-Whitney p-value fix."""
    x = [1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30]
    y = [0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.07, 3.14, 1.28, 4.1]

    model = MannWhitneyU(exact=True).fit(np.array(x), np.array(y))
    assert model.statistic == 58.0
    assert model.p_value == pytest.approx(0.31537812033168078, abs=1e-12)

    swapped = MannWhitneyU(exact=True).fit(np.array(y), np.array(x))
    assert swapped.p_value == pytest.approx(0.31537812033168078, abs=1e-12)


def test_rank_tests_centre_and_all_tied_give_p_one():
    """anofox-statistics 0.4.3 #13: continuity at the centre / all-tied give p=1."""
    assert MannWhitneyU().fit(np.array([1.0, 4.0]), np.array([2.0, 3.0])).p_value == 1.0
    assert WilcoxonSignedRank().fit(
        np.array([1.0, 2.0, 3.0, 4.0]), np.array([4.0, 3.0, 2.0, 1.0])
    ).p_value == 1.0

    for alternative in ("two-sided", "less", "greater"):
        model = MannWhitneyU(alternative=alternative).fit(
            np.array([5.0] * 6), np.array([5.0] * 7)
        )
        assert model.p_value == 1.0
