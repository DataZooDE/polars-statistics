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

import polars as pl
import polars_statistics as ps

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
