"""R-reference parity tests for anofox-statistics 0.4.5 / 0.4.6 behaviour changes.

Each test pins a value computed against R 4.6.1 (or R-equivalent references computed
with scipy at planning time) so a future regression in the backing `anofox-statistics`
crate cannot pass silently. See
`.planning/phases/16.1-upstream-crate-refresh/16.1-RESEARCH.md`
("Retarget Addendum") for the upstream changelog entries these tests encode:

- #19 (anofox-statistics 0.4.5): `MannWhitneyU` / `WilcoxonSignedRank` Hodges-Lehmann
  `estimate` and `conf_int` now follow R `wilcox.test(..., conf.int = TRUE)` exactly
  (root-finding on the asymptotic path, one-sided intervals with an infinite bound,
  continuity correction applied to the interval). Exact null distributions are computed
  in a way that no longer overflows for larger samples, and `exact=True` is honoured
  only below a sample-size threshold.
- #20 (anofox-statistics 0.4.6): `ps.mcnemar_exact` p-value via the binomial CDF
  (R `binom.test(b, b + c, 0.5)`), no NaN for large discordant counts.

The 126-row Hodges-Lehmann reference fixture (`tests/fixtures/wilcox_ci_reference.csv`)
is a byte-identical copy of upstream `R/data/wilcox_ci_reference.csv`
(`~/projects/rust/anofox-statistics-rs`, release commit `3ab3bec`, R 4.6.1,
`R/generate_wilcox_ci_refs.R`) — that file is not shipped on crates.io.
"""

import csv
import math
from pathlib import Path

import numpy as np
import polars as pl
import polars_statistics as ps
import pytest
from polars_statistics import MannWhitneyU, WilcoxonSignedRank

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "wilcox_ci_reference.csv"
FIXTURE_SHA256_PREFIX = "e51fd41b79d5e43274cfe86378e71395b2a96461347294add8e425c52863c296"

_ALTERNATIVE_MAP = {"two.sided": "two-sided", "less": "less", "greater": "greater"}


def _load_fixture_rows():
    with open(FIXTURE_PATH, newline="") as f:
        return list(csv.DictReader(f))


def _row_id(i, row):
    return f"{i}-{row['kind']}-{row['alternative']}-exact{row['exact']}"


_FIXTURE_ROWS = _load_fixture_rows()


@pytest.mark.parametrize(
    "idx,row",
    list(enumerate(_FIXTURE_ROWS)),
    ids=[_row_id(i, row) for i, row in enumerate(_FIXTURE_ROWS)],
)
def test_hodges_lehmann_matches_r(idx, row):
    """anofox-statistics 0.4.5 #19: Hodges-Lehmann estimate/CI match R wilcox.test.

    Covers mw and wsr kinds, all three alternatives, continuity correction on/off,
    exact on/off, conf_level 0.8/0.9/0.95, and mu shifts — including one-sided
    intervals with an infinite bound.
    """
    kind = row["kind"]
    alternative = _ALTERNATIVE_MAP[row["alternative"]]
    continuity_correction = row["correct"] == "TRUE"
    exact = row["exact"] == "TRUE"
    conf_level = float(row["conf_level"])
    mu = float(row["mu"])
    estimate_ref = float(row["estimate"])
    lower_ref = float(row["lower"])
    upper_ref = float(row["upper"])
    x = np.array(row["x"].split(), dtype=float)
    y = np.array(row["y"].split(), dtype=float)

    tol = 1e-12 if exact else 1e-9

    kwargs = {
        "alternative": alternative,
        "continuity_correction": continuity_correction,
        "exact": exact,
        "conf_level": conf_level,
        "mu": mu,
    }
    if kind == "mw":
        model = MannWhitneyU(**kwargs).fit(x, y)
    else:
        model = WilcoxonSignedRank(**kwargs).fit(x, y)

    estimate = model.estimate
    conf_int = model.conf_int
    lower, upper = (conf_int[0], conf_int[1]) if conf_int is not None else (None, None)

    def _assert_matches(ref, got, name):
        if math.isinf(ref):
            assert got is not None and math.isinf(got), f"{name}: ref={ref} got={got}"
            assert math.copysign(1.0, got) == math.copysign(1.0, ref), (
                f"{name}: wrong infinite sign, ref={ref} got={got}"
            )
        else:
            assert got is not None and math.isfinite(got), f"{name}: ref={ref} got={got}"
            allowed = tol * (1 + abs(ref))
            assert abs(got - ref) <= allowed, (
                f"{name}: ref={ref} got={got} diff={abs(got - ref)} tol={allowed}"
            )

    _assert_matches(estimate_ref, estimate, "estimate")
    _assert_matches(lower_ref, lower, "lower")
    _assert_matches(upper_ref, upper, "upper")


def test_hodges_lehmann_fixture_integrity():
    """The fixture is byte-identical to the upstream 126-row R reference file."""
    import hashlib

    assert len(_FIXTURE_ROWS) == 126
    digest = hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()
    assert digest == FIXTURE_SHA256_PREFIX


def test_exact_distribution_rank_sum_no_overflow():
    """anofox-statistics 0.4.5 #19: exact rank-sum distribution beyond n1+n2=80.

    Previously overflowed for n1 + n2 > ~62. References computed with
    scipy.stats.mannwhitneyu(method="exact" / "asymptotic") on untied data,
    matching R wilcox.test(x, y, exact = TRUE / FALSE).
    """
    x = [((i * 37) % 101) / 10 for i in range(40)]
    y = [((i * 53) % 103) / 10 + 0.55 for i in range(40)]

    exact = MannWhitneyU(exact=True).fit(np.array(x), np.array(y))
    assert exact.statistic == pytest.approx(756.0, abs=1e-9)
    assert exact.p_value == pytest.approx(0.6771573878425421, abs=1e-9)

    asymptotic = MannWhitneyU(exact=False).fit(np.array(x), np.array(y))
    assert asymptotic.p_value == pytest.approx(0.6755238797807038, abs=1e-9)


def test_exact_distribution_signed_rank_no_overflow():
    """anofox-statistics 0.4.5 #19: exact signed-rank distribution for n >= 64.

    Previously overflowed for n >= 64. Reference computed with
    scipy.stats.wilcoxon(method="exact"), matching R wilcox.test(d, exact = TRUE).
    """
    d = [((i * 29) % 97) / 10 - 4.6 + 0.013 * i for i in range(70)]
    model = WilcoxonSignedRank(exact=True).fit(np.array(d), np.zeros(70))
    assert model.statistic == pytest.approx(1525.0, abs=1e-9)
    assert model.p_value == pytest.approx(0.09908340663363724, abs=1e-9)


def test_exact_gating_rank_sum_above_threshold_falls_back_to_asymptotic():
    """anofox-statistics 0.4.5 #19: exact=True honoured only for n1*n2 <= 10,000."""
    x = [((i * 37) % 211) / 10 for i in range(120)]
    y = [((i * 53) % 223) / 10 + 0.05 for i in range(100)]
    assert len(x) * len(y) > 10_000

    exact = MannWhitneyU(exact=True, continuity_correction=True)
    exact.fit(np.array(x), np.array(y))
    asymptotic = MannWhitneyU(exact=False, continuity_correction=True)
    asymptotic.fit(np.array(x), np.array(y))
    assert exact.p_value == asymptotic.p_value


def test_exact_gating_rank_sum_below_threshold_is_honoured():
    """Below the n1*n2 <= 10,000 threshold, exact and asymptotic differ meaningfully."""
    x = [((i * 37) % 101) / 10 for i in range(40)]
    y = [((i * 53) % 103) / 10 + 0.55 for i in range(40)]
    assert len(x) * len(y) <= 10_000

    exact = MannWhitneyU(exact=True).fit(np.array(x), np.array(y))
    asymptotic = MannWhitneyU(exact=False).fit(np.array(x), np.array(y))
    assert abs(exact.p_value - asymptotic.p_value) > 1e-4


def test_exact_gating_signed_rank_above_threshold_falls_back_to_asymptotic():
    """anofox-statistics 0.4.5 #19: exact=True honoured only for <= 300 non-zero diffs."""
    d = np.array([((i * 29) % 331) / 10 - 16.45 + 1e-4 * i for i in range(320)])
    assert np.count_nonzero(d) > 300

    exact = WilcoxonSignedRank(exact=True).fit(d, np.zeros(320))
    asymptotic = WilcoxonSignedRank(exact=False).fit(d, np.zeros(320))
    assert exact.p_value == asymptotic.p_value


@pytest.mark.parametrize(
    "b,c,p_ref",
    [
        (3, 7, 0.34375000000000011),
        (0, 0, 1.0),
        (1, 9, 0.021484375000000014),
        (12, 30, 0.0079158973348967352),
        (500, 600, 0.0028195449914364345),
        (540, 560, 0.56675144795804511),
        (1000, 1100, 0.030720707864242414),
        (5000, 5200, 0.048787963126822431),
        (499000, 501000, 0.045608299865382104),
        (499500, 500500, 0.31779469136330574),
    ],
)
def test_mcnemar_exact_large_counts_matches_r(b, c, p_ref):
    """anofox-statistics 0.4.6 #20: mcnemar_exact via binomial CDF, no NaN at large b+c.

    Reference: R 4.6.1 binom.test(b, b + c, 0.5)$p.value.
    """
    df = pl.DataFrame({"n": [1]})

    result = df.select(ps.mcnemar_exact(7, b, c, 9).alias("r")).item()
    assert math.isfinite(result["p_value"])
    assert result["p_value"] == pytest.approx(p_ref, rel=1e-9)
    assert result["statistic"] == pytest.approx(float(b + c))

    swapped = df.select(ps.mcnemar_exact(7, c, b, 9).alias("r")).item()
    assert swapped["p_value"] == pytest.approx(result["p_value"], rel=1e-9)
    assert math.isfinite(swapped["p_value"])


def test_large_sample_hodges_lehmann_completes_in_o_n_memory():
    """anofox-statistics 0.4.5 #19: 20,000-per-group CI no longer O(n^2) memory.

    Previously this allocated roughly 3.2 GB for the order-statistic search.
    """
    rng = np.random.default_rng(19)
    x = rng.random(20_000)
    y = rng.random(20_000) + 0.1

    mw = MannWhitneyU(conf_level=0.95).fit(x, y)
    assert mw.conf_int[0] < mw.estimate < mw.conf_int[1]
    assert abs(mw.estimate + 0.1) < 0.02

    wsr = WilcoxonSignedRank(conf_level=0.95).fit(x, y)
    assert wsr.conf_int[0] <= wsr.estimate <= wsr.conf_int[1]


def test_tost_wilcoxon_alpha_above_half_does_not_panic():
    """anofox-statistics 0.4.5 #19: TOST-Wilcoxon with alpha > 0.5 no longer panics."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [1.2, 2.1, 2.9, 4.3, 4.8]})

    paired = df.select(ps.tost_wilcoxon_paired("x", "y", alpha=0.6).alias("r")).item()
    assert paired["alpha"] == pytest.approx(0.6)

    two_sample = df.select(
        ps.tost_wilcoxon_two_sample("x", "y", alpha=0.6).alias("r")
    ).item()
    assert two_sample["alpha"] == pytest.approx(0.6)


def test_expression_schema_mann_whitney_and_wilcoxon_omit_estimate_ci():
    """The expression layer surfaces only statistic/p_value — a pre-existing gap,
    unchanged by the 0.4.5/0.4.6 upstream retarget (not part of this delta).
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.2, 2.1, 2.9]})

    mw_fields = sorted(
        df.select(ps.mann_whitney_u("x", "y", conf_level=0.95).alias("r")).item().keys()
    )
    assert mw_fields == ["p_value", "statistic"]

    wsr_fields = sorted(
        df.select(ps.wilcoxon_signed_rank("x", "y", conf_level=0.95).alias("r"))
        .item()
        .keys()
    )
    assert wsr_fields == ["p_value", "statistic"]
