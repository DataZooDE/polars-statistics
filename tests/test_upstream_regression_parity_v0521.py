"""R-reference parity tests for anofox-regression 0.5.19-0.5.21 behaviour changes.

anofox-statistics bumped 0.4.6 -> 0.4.7 (additive only, no behaviour change) and
anofox-regression bumped 0.5.16 -> 0.5.22 at the start of Phase 16.2 (user decision
2026-10-09, see `.planning/phases/16.2-penalized-glm-and-aft-survival-regression/
16.2-CONTEXT.md` "Retarget to 0.4.7 / 0.5.22"). 0.5.22 itself is performance-only
(warm-started NegBin theta, grouped digamma/trigamma sums, streaming Tweedie
log-likelihood); the R fixture and upstream validation test are byte-identical to
0.5.21's. This module locks the user-visible behaviour changes shipped in
0.5.19-0.5.21 so a future regression in the backing crate cannot pass silently:

- #62 (0.5.19): LARS/LassoLars alias the *later* of exactly collinear columns
  (NaN coefficient) instead of a platform-dependent result.
- #66 (0.5.20): intercept inference on every GLM (Binomial/Poisson/NegBin/Tweedie/
  Gamma) -- `intercept_t_statistic`/`intercept_p_value`/`intercept_conf_interval`
  are now populated (previously NaN/None); Tweedie/Gamma also report per-coefficient
  statistics/p-values; probit SEs now use statrs erfc/erfc_inv (previously off R in
  the 5th digit); GLM Wald p-values use the survival function (no underflow to 0).
- #64/#65 (0.5.21): OLS/WLS alias collinear designs like R's `lm` (dqrdc2, tol
  1e-7 relative to each column's norm) for every solve_method -- `svd` no longer
  returns a minimum-norm split, `qr`/WLS alias the LATER column of a collinear
  set (previously `qr` aliased the earlier column, `svd` split it), the decision
  is scale-independent; WLS-without-intercept SEs come from the reduced fit; LARS
  fills `adj_r_squared`/`mse`/`rmse` (previously 0.0 placeholders).

R 4.6.1 reference values are copied verbatim (with provenance comments) from the
`anofox-regression` 0.5.22 crate's own test fixtures, read from the local cargo
registry cache:
`~/.cargo/registry/src/*/anofox-regression-0.5.22/tests/fixtures/m1a_validation.rs`
(`generate_m1a_validation.R`, R 4.6.1, MASS 7.3.65, broom 1.0.13) and
`~/.cargo/registry/src/*/anofox-regression-0.5.22/tests/{r_validation_m1a.rs,
issue64_lm_aliasing.rs,issue62_lars_collinear.rs}`.
"""

import math

import numpy as np
import polars as pl
import polars_statistics as ps
from polars_statistics import LARS, OLS

# ----------------------------------------------------------------------------
# mtcars subset (R's built-in dataset), copied verbatim from
# anofox-regression-0.5.22/tests/fixtures/m1a_validation.rs (MT_* constants).
# ----------------------------------------------------------------------------
MT_MPG = [
    21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2,
    17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9,
    21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7,
    15.0, 21.4,
]
MT_WT = [
    2.620, 2.875, 2.320, 3.215, 3.440, 3.460, 3.570, 3.190, 3.150, 3.440,
    3.440, 4.070, 3.730, 3.780, 5.250, 5.424, 5.345, 2.200, 1.615, 1.835,
    2.465, 3.520, 3.435, 3.840, 3.845, 1.935, 2.140, 1.513, 3.170, 2.770,
    3.570, 2.780,
]
MT_HP = [
    110, 110, 93, 110, 175, 105, 245, 62, 95, 123,
    123, 180, 180, 180, 205, 215, 230, 66, 52, 65,
    97, 150, 150, 245, 175, 66, 91, 113, 264, 175,
    335, 109,
]
MT_VS = [
    0, 0, 1, 1, 0, 1, 0, 1, 1, 1,
    1, 0, 0, 0, 0, 0, 0, 1, 1, 1,
    1, 0, 0, 0, 0, 1, 0, 1, 0, 0,
    0, 1,
]
MT_CARB = [
    4, 4, 1, 1, 2, 1, 4, 2, 2, 4,
    4, 3, 3, 3, 4, 4, 4, 1, 2, 1,
    1, 2, 2, 4, 2, 1, 2, 2, 4, 6,
    8, 2,
]

# glm(vs ~ mpg, family = binomial) -- anofox-regression-0.5.22
# tests/fixtures/m1a_validation.rs BIN_EST/BIN_SE/BIN_STAT/BIN_P (intercept, mpg).
BIN_EST = [-8.833072576792057e00, 4.304135202626586e-01]
BIN_SE = [3.162274091193895e00, 1.584219695870185e-01]
BIN_STAT = [-2.793265960528169e00, 2.716880249530288e00]
BIN_P = [5.217877161081044e-03, 6.590044667692132e-03]

# glm(carb ~ wt + hp, family = poisson) -- anofox-regression-0.5.22
# tests/fixtures/m1a_validation.rs POIS_EST/POIS_SE/POIS_STAT/POIS_P.
POIS_EST = [1.387882909503199e-01, 4.481935795756660e-03, 5.487240249142543e-03]
POIS_SE = [3.986680920016935e-01, 1.309720440702774e-01, 1.644148051066227e-03]
POIS_STAT = [3.481299199378372e-01, 3.422055315370759e-02, 3.337436823638892e00]
POIS_P = [7.277426152147488e-01, 9.727012770916992e-01, 8.455494206187751e-04]

# glm(vs ~ mpg, family = binomial(link = "probit")) -- anofox-regression-0.5.22
# tests/fixtures/m1a_validation.rs PROBIT_EST/PROBIT_SE/PROBIT_STAT/PROBIT_P.
PROBIT_EST = [-5.093625476513099e00, 2.461362105074878e-01]
PROBIT_SE = [1.642683891813841e00, 8.254071916793369e-02]
PROBIT_STAT = [-3.100794682346796e00, 2.981997406718858e00]
PROBIT_P = [1.930020650356549e-03, 2.863744246497870e-03]

# glm(mpg ~ wt + hp, family = Gamma(link = "log")) -- anofox-regression-0.5.22
# tests/fixtures/m1a_validation.rs GAMMA_EST/GAMMA_SE/GAMMA_STAT/GAMMA_P. The
# Gamma PyModel exposes only the slope rows (no intercept_std_error/p_value
# getter) -- index [1:] below are the wt/hp slope entries, [0] is intercept.
GAMMA_EST = [3.825870595954185e00, -1.969867714955781e-01, -1.560105699921510e-03]
GAMMA_SE = [7.113840121796375e-02, 2.815361523482204e-02, 4.017789071627085e-04]
GAMMA_STAT = [5.378066600389219e01, -6.996855283151464e00, -3.882995528408149e00]
GAMMA_P = [1.326817361000176e-30, 1.078984289982329e-07, 5.491656220441162e-04]


def _rel_close(got: float, want: float, rel: float) -> None:
    scale = max(abs(want), 1.0)
    assert abs(got - want) <= rel * scale, f"got {got!r}, want {want!r} (rel={rel})"


def test_logistic_summary_intercept_row_matches_r():
    """anofox-regression 0.5.20 #66: ps.logistic_summary intercept row now reports
    a finite statistic/p_value matching R's summary.glm (previously NaN)."""
    df = pl.DataFrame({"vs": MT_VS, "mpg": MT_MPG})
    rows = df.select(ps.logistic_summary("vs", "mpg").alias("coef"))["coef"][0].to_list()
    assert len(rows) == 2
    intercept, slope = rows[0], rows[1]

    assert not math.isnan(intercept["statistic"])
    assert not math.isnan(intercept["p_value"])

    _rel_close(intercept["estimate"], BIN_EST[0], 1e-5)
    _rel_close(intercept["std_error"], BIN_SE[0], 1e-5)
    _rel_close(intercept["statistic"], BIN_STAT[0], 1e-5)
    _rel_close(intercept["p_value"], BIN_P[0], 1e-4)

    _rel_close(slope["estimate"], BIN_EST[1], 1e-5)
    _rel_close(slope["std_error"], BIN_SE[1], 1e-5)
    _rel_close(slope["statistic"], BIN_STAT[1], 1e-5)
    _rel_close(slope["p_value"], BIN_P[1], 1e-4)


def test_poisson_summary_intercept_row_matches_r():
    """anofox-regression 0.5.20 #66: ps.poisson_summary intercept row now reports
    a finite statistic/p_value matching R's summary.glm (previously NaN)."""
    df = pl.DataFrame({"carb": MT_CARB, "wt": MT_WT, "hp": MT_HP})
    rows = df.select(ps.poisson_summary("carb", "wt", "hp").alias("coef"))["coef"][0].to_list()
    assert len(rows) == 3
    intercept, slope_wt, slope_hp = rows[0], rows[1], rows[2]

    assert not math.isnan(intercept["statistic"])
    assert not math.isnan(intercept["p_value"])

    _rel_close(intercept["estimate"], POIS_EST[0], 1e-5)
    _rel_close(intercept["std_error"], POIS_SE[0], 1e-5)
    _rel_close(intercept["statistic"], POIS_STAT[0], 1e-5)
    _rel_close(intercept["p_value"], POIS_P[0], 1e-4)

    _rel_close(slope_wt["estimate"], POIS_EST[1], 1e-5)
    _rel_close(slope_wt["statistic"], POIS_STAT[1], 1e-5)
    _rel_close(slope_hp["estimate"], POIS_EST[2], 1e-5)
    _rel_close(slope_hp["statistic"], POIS_STAT[2], 1e-5)


# ----------------------------------------------------------------------------
# OLS/WLS lm-style aliasing (anofox-regression 0.5.21 #64), generators copied
# verbatim from anofox-regression-0.5.22/tests/issue64_lm_aliasing.rs (x1/x2/
# x3/y/w) so the R references below (issue64_lm_aliasing.R) apply unchanged.
# ----------------------------------------------------------------------------
_N64 = 60


def _x1_64() -> np.ndarray:
    return np.arange(_N64, dtype=float) / 10.0


def _x2_64() -> np.ndarray:
    return np.sin(np.arange(_N64, dtype=float))


def _y_64() -> np.ndarray:
    i = np.arange(_N64, dtype=float)
    return 2.0 + 1.5 * i / 10.0 - 0.8 * np.sin(i) + 0.3 * np.cos(7.0 * i)


def _w_64() -> np.ndarray:
    return 1.0 + (np.arange(_N64) % 5) / 2.0


def test_ols_lm_aliasing_exact_duplicate_aliases_later_column():
    """anofox-regression 0.5.21 #64: OLS aliases the LATER of two exactly
    duplicate columns (NaN coefficient), for every solve_method."""
    x1 = _x1_64()
    y = _y_64()
    X = np.column_stack([x1, x1])

    for solver in ("qr", "svd", "cholesky"):
        model = OLS(solve_method=solver).fit(X, y)
        assert math.isnan(model.coefficients[1])
        assert not math.isnan(model.coefficients[0])
        _rel_close(model.coefficients[0], 1.48861560234839, 1e-8)
        _rel_close(model.intercept, 2.00330174321490, 1e-8)
        _rel_close(model.r_squared, 0.951259322529991, 1e-8)


def test_ols_lm_aliasing_scaled_copy_aliases_later_column():
    """anofox-regression 0.5.21 #64: a scaled copy (2*x1) is aliased exactly
    like an exact duplicate -- the aliasing decision is scale-independent, and
    svd no longer returns a minimum-norm split of the collinear pair."""
    x1 = _x1_64()
    x2 = _x2_64()
    y = _y_64()
    b = 2.0 * x1
    X = np.column_stack([x1, b, x2])

    for solver in ("qr", "svd", "cholesky"):
        model = OLS(solve_method=solver).fit(X, y)
        assert math.isnan(model.coefficients[1])
        _rel_close(model.coefficients[0], 1.490290680634032, 1e-8)
        _rel_close(model.coefficients[2], -0.772169023108856, 1e-8)
        _rel_close(model.intercept, 2.023318557879104, 1e-8)
        _rel_close(model.r_squared, 0.993640840379574, 1e-8)


def test_wls_lm_aliasing_scaled_copy_aliases_later_column():
    """anofox-regression 0.5.21 #64: WLS aliases the later column of a scaled
    collinear pair too, and its SEs equal R's lm(weights=) reduced fit."""
    x1 = _x1_64()
    x2 = _x2_64()
    y = _y_64()
    w = _w_64()
    b = 2.0 * x1
    df = pl.DataFrame({"y": y, "x1": x1, "b": b, "x2": x2, "w": w})

    for solver in ("qr", "svd", "cholesky"):
        row = df.select(
            ps.wls("y", "w", "x1", "b", "x2", solve_method=solver).alias("m")
        )["m"][0]
        coefs = row["coefficients"]
        assert math.isnan(coefs[1])
        _rel_close(coefs[0], 1.494367216183389, 1e-8)
        _rel_close(coefs[2], -0.774931009626785, 1e-8)
        _rel_close(row["intercept"], 2.008988048468000, 1e-8)
        _rel_close(row["r_squared"], 0.993499086258852, 1e-8)

    summary_rows = df.select(
        ps.wls_summary("y", "w", "x1", "b", "x2", solve_method="qr").alias("s")
    )["s"][0].to_list()
    se_by_term = {r["term"]: r["std_error"] for r in summary_rows}
    _rel_close(se_by_term["x1"], 0.0163679078787441, 1e-5)
    _rel_close(se_by_term["x3"], 0.0402659270793710, 1e-5)
    assert math.isnan(se_by_term["x2"])


def test_wls_lm_aliasing_no_intercept_duplicate():
    """anofox-regression 0.5.21 #64: WLS-without-intercept aliasing and SEs
    come from the reduced (non-aliased) fit."""
    x1 = _x1_64()
    x2 = _x2_64()
    y = _y_64()
    w = _w_64()
    c = 3.0 * x2
    df = pl.DataFrame({"y": y, "x1": x1, "x2": x2, "c": c, "w": w})

    row = df.select(
        ps.wls("y", "w", "x1", "x2", "c", add_intercept=False).alias("m")
    )["m"][0]
    coefs = row["coefficients"]
    assert math.isnan(coefs[2])
    _rel_close(coefs[0], 1.996234614755434, 1e-8)
    _rel_close(coefs[1], -0.727862022303558, 1e-8)
    _rel_close(row["r_squared"], 0.978398095055174, 1e-8)


# ----------------------------------------------------------------------------
# LARS collinear aliasing (anofox-regression 0.5.19 #62), generator copied
# verbatim from anofox-regression-0.5.22/tests/issue62_lars_collinear.rs
# (`base` data).
# ----------------------------------------------------------------------------
def _lars_base(n: int) -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(1, n + 1)
    a = (i % 7).astype(float) + 0.5
    b = ((i * 3) % 11).astype(float) / 2.0
    y = 3.0 + 0.8 * a - 0.3 * b + ((i * 37) % 13).astype(float) / 13.0 - 0.5
    return a, y


def test_lars_collinear_aliases_later_column():
    """anofox-regression 0.5.19 #62: LARS on an exactly collinear design
    aliases the later column (NaN), with the same fit as the reduced model."""
    x1, y = _lars_base(40)

    reference = LARS().fit(x1[:, None], y)
    dup = LARS().fit(np.column_stack([x1, 2.0 * x1]), y)

    assert math.isnan(dup.coefficients[1])
    _rel_close(dup.coefficients[0], reference.coefficients[0], 1e-10)
    _rel_close(dup.intercept, reference.intercept, 1e-10)
    _rel_close(dup.r_squared, reference.r_squared, 1e-10)


# ----------------------------------------------------------------------------
# GLM intercept/coefficient inference (anofox-regression 0.5.20 #66): probit
# matches R exactly; NegBin/Gamma/Tweedie report finite statistics/p-values
# where they previously returned NaN/None.
# ----------------------------------------------------------------------------
def test_probit_summary_matches_r():
    """anofox-regression 0.5.20 #66: probit SEs now use statrs erfc/erfc_inv
    and match R's summary.glm(binomial(link="probit")) to high precision."""
    df = pl.DataFrame({"vs": MT_VS, "mpg": MT_MPG})
    rows = df.select(ps.probit_summary("vs", "mpg").alias("coef"))["coef"][0].to_list()
    intercept, slope = rows[0], rows[1]

    _rel_close(intercept["estimate"], PROBIT_EST[0], 1e-5)
    _rel_close(intercept["std_error"], PROBIT_SE[0], 1e-5)
    _rel_close(intercept["statistic"], PROBIT_STAT[0], 1e-5)
    _rel_close(intercept["p_value"], PROBIT_P[0], 1e-4)

    _rel_close(slope["estimate"], PROBIT_EST[1], 1e-5)
    _rel_close(slope["std_error"], PROBIT_SE[1], 1e-5)
    _rel_close(slope["statistic"], PROBIT_STAT[1], 1e-5)
    _rel_close(slope["p_value"], PROBIT_P[1], 1e-4)


def test_gamma_slope_statistics_match_r():
    """anofox-regression 0.5.20 #66: Gamma now reports finite per-coefficient
    std_error/p_value (previously None); slope rows match R summary.glm.
    The Gamma PyModel has no intercept_std_error/p_value getter, so only the
    slope coefficients (reachable today) are checked against R."""
    X = np.column_stack([np.array(MT_WT), np.array(MT_HP)])
    y = np.array(MT_MPG)

    from polars_statistics import Gamma

    model = Gamma().fit(X, y)
    assert model.std_errors is not None
    assert model.p_values is not None

    _rel_close(float(model.std_errors[0]), GAMMA_SE[1], 1e-5)
    _rel_close(float(model.std_errors[1]), GAMMA_SE[2], 1e-5)
    _rel_close(float(model.p_values[0]), GAMMA_P[1], 1e-4)
    _rel_close(float(model.p_values[1]), GAMMA_P[2], 1e-4)
    _rel_close(model.intercept, GAMMA_EST[0], 1e-5)


def test_tweedie_summary_reports_finite_statistics():
    """anofox-regression 0.5.20 #66: Tweedie summary intercept/slope rows now
    report a finite statistic/p_value (previously NaN)."""
    df = pl.DataFrame({"mpg": MT_MPG, "wt": MT_WT, "hp": MT_HP})
    rows = df.select(
        ps.tweedie_summary("mpg", "wt", "hp", var_power=2.0).alias("coef")
    )["coef"][0].to_list()
    assert len(rows) == 3
    for row in rows:
        assert not math.isnan(row["statistic"]), row
        assert not math.isnan(row["p_value"]), row
        # Wald p-values now use the survival function (0.5.20 #66): never 0.
        assert row["p_value"] > 0.0, row


def test_negative_binomial_summary_reports_finite_statistics():
    """anofox-regression 0.5.20 #66: NegBin summary intercept row now reports
    a finite statistic/p_value (previously NaN)."""
    df = pl.DataFrame({"mpg": MT_MPG, "wt": MT_WT, "hp": MT_HP})
    rows = df.select(
        ps.negative_binomial_summary("mpg", "wt", "hp").alias("coef")
    )["coef"][0].to_list()
    assert len(rows) == 3
    for row in rows:
        assert not math.isnan(row["statistic"]), row
        assert not math.isnan(row["p_value"]), row
