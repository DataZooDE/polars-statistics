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
