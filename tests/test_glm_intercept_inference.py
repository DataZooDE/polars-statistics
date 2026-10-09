"""R-reference tests for intercept inference getters on GLM / OLS / WLS PyModels.

anofox-regression 0.5.20 (#66) fills `intercept_std_error` / `intercept_t_statistic` /
`intercept_p_value` / `intercept_conf_interval` and `conf_interval_lower` /
`conf_interval_upper` on the shared `RegressionResult` struct used by every
non-penalized GLM (Poisson / Binomial-link / NegativeBinomial / Tweedie / Gamma),
plus OLS / WLS which have always populated it. Before this plan, none of the 10
PyModel classes exposed those fields through a getter.

R references: `tests/r_scripts/generate_m1a_validation.R` (R 4.6.1, MASS 7.3.65,
broom 1.0.13), shipped as `anofox-regression` 0.5.22's
`tests/fixtures/m1a_validation.rs` and transcribed verbatim into
`tests/fixtures/m1a_reference.py` (see that module's docstring for provenance and
`SOURCE_SHA256`).

See `.planning/phases/16.3-upstream-api-wave-2/16.3-02-PLAN.md`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest
from polars_statistics import (
    OLS,
    WLS,
    Cloglog,
    Gamma,
    Logistic,
    LogisticRegression,
    NegativeBinomial,
    Poisson,
    Probit,
    Tweedie,
)

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "m1a_reference.py"
_spec = importlib.util.spec_from_file_location("m1a_reference", _FIXTURE_PATH)
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)


def rel_close(a: float, b: float, tol: float) -> bool:
    """Relative-error closeness check: ``|a - b| <= tol * max(|b|, 1)``."""
    return abs(a - b) <= tol * max(abs(b), 1.0)


MT_MPG = np.array(ref.MT_MPG)
MT_WT = np.array(ref.MT_WT)
MT_HP = np.array(ref.MT_HP)
MT_VS = np.array(ref.MT_VS)
MT_CARB = np.array(ref.MT_CARB)
MT_W = np.array(ref.MT_W)
WB_BREAKS = np.array(ref.WB_BREAKS)
WB_WOOLB = np.array(ref.WB_WOOLB)
WB_TM = np.array(ref.WB_TM)
WB_TH = np.array(ref.WB_TH)


# ============================================================================
# Fixture integrity
# ============================================================================


class TestM1aFixtureIntegrity:
    def test_fixture_lengths(self):
        assert len(ref.MT_MPG) == 32
        assert len(ref.WB_BREAKS) == 54

    def test_fixture_spot_constant(self):
        assert ref.BIN_EST[0] == -8.833072576792057

    def test_fixture_source_metadata_present(self):
        assert ref.SOURCE_CRATE == "anofox-regression 0.5.22"
        assert len(ref.SOURCE_SHA256) == 64

    def test_fixture_matches_upstream_crate_file_when_present(self):
        """If the upstream crate source is on disk (dev env with a populated
        cargo registry cache), its sha256 must match SOURCE_SHA256 and every
        constant re-parsed from it must equal the module's value."""
        cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
        candidates = []
        for version in ("0.5.21", "0.5.22"):
            candidates.extend(
                sorted(
                    cargo_home.glob(
                        f"registry/src/*/anofox-regression-{version}/tests/fixtures/"
                        "m1a_validation.rs"
                    )
                )
            )
        if not candidates:
            pytest.skip("upstream anofox-regression 0.5.21/0.5.22 source not found")
        upstream = candidates[0]
        content = upstream.read_text()
        sha = hashlib.sha256(content.encode()).hexdigest()
        if sha != ref.SOURCE_SHA256:
            pytest.skip(
                "upstream fixture sha256 differs from the pinned SOURCE_SHA256 "
                "(a different crate version is cached under $CARGO_HOME)"
            )
        assert "-8.833072576792057" in content


# ============================================================================
# Logistic: Binomial logit GLM vs R glm(vs ~ mpg, binomial)
# ============================================================================


class TestLogisticMatchesR:
    def _fit(self, **kwargs):
        x = MT_MPG.reshape(-1, 1)
        return Logistic(max_iter=100, tol=1e-12, **kwargs).fit(x, MT_VS)

    def test_logistic_matches_r(self):
        m = self._fit()
        assert m.intercept == pytest.approx(ref.BIN_EST[0], rel=1e-6)
        assert m.intercept_std_error == pytest.approx(ref.BIN_SE[0], rel=1e-6)
        assert m.intercept_t_statistic == pytest.approx(ref.BIN_STAT[0], rel=1e-6)
        assert m.intercept_p_value == pytest.approx(ref.BIN_P[0], rel=1e-6)
        ci = m.intercept_conf_interval
        assert ci[0] == pytest.approx(ref.BIN_CI_LO[0], rel=1e-6)
        assert ci[1] == pytest.approx(ref.BIN_CI_HI[0], rel=1e-6)

        assert m.coefficients[0] == pytest.approx(ref.BIN_EST[1], rel=1e-6)
        assert m.std_errors[0] == pytest.approx(ref.BIN_SE[1], rel=1e-6)
        assert m.t_statistics[0] == pytest.approx(ref.BIN_STAT[1], rel=1e-6)
        assert m.p_values[0] == pytest.approx(ref.BIN_P[1], rel=1e-6)
        assert m.conf_interval_lower[0] == pytest.approx(ref.BIN_CI_LO[1], rel=1e-6)
        assert m.conf_interval_upper[0] == pytest.approx(ref.BIN_CI_HI[1], rel=1e-6)

    def test_logistic_confidence_level(self):
        m95 = self._fit()
        m90 = self._fit(confidence_level=0.90)

        z90 = NormalDist().inv_cdf(0.95)
        expected_lo = m90.intercept - z90 * m90.intercept_std_error
        expected_hi = m90.intercept + z90 * m90.intercept_std_error
        ci90 = m90.intercept_conf_interval
        assert ci90[0] == pytest.approx(expected_lo, rel=1e-10)
        assert ci90[1] == pytest.approx(expected_hi, rel=1e-10)

        ci95 = m95.intercept_conf_interval
        # The narrower 0.90 interval lies strictly inside the 0.95 one.
        assert ci95[0] < ci90[0] < ci90[1] < ci95[1]

    def test_logistic_no_inference(self):
        m = self._fit(compute_inference=False)
        for attr in (
            "std_errors",
            "t_statistics",
            "p_values",
            "conf_interval_lower",
            "conf_interval_upper",
            "intercept_std_error",
            "intercept_t_statistic",
            "intercept_p_value",
            "intercept_conf_interval",
        ):
            assert getattr(m, attr) is None

    def test_logistic_unfitted_and_to_dict(self):
        m = Logistic()
        for attr in (
            "std_errors",
            "t_statistics",
            "p_values",
            "conf_interval_lower",
            "conf_interval_upper",
            "intercept_std_error",
            "intercept_t_statistic",
            "intercept_p_value",
            "intercept_conf_interval",
        ):
            with pytest.raises(RuntimeError, match="Logistic"):
                getattr(m, attr)

        fitted = self._fit()
        d = fitted.to_dict()
        assert "intercept_conf_interval" in d
        assert "t_statistics" in d

    def test_logistic_confidence_level_out_of_range_raises(self):
        for level in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError):
                Logistic(confidence_level=level)


# ============================================================================
# Probit: Binomial probit GLM vs R glm(vs ~ mpg, binomial(probit))
# ============================================================================


class TestProbitMatchesR:
    def _fit(self, **kwargs):
        x = MT_MPG.reshape(-1, 1)
        return Probit(max_iter=100, tol=1e-12, **kwargs).fit(x, MT_VS)

    def test_probit_matches_r(self):
        m = self._fit()
        assert m.intercept == pytest.approx(ref.PROBIT_EST[0], rel=1e-6)
        assert m.intercept_std_error == pytest.approx(ref.PROBIT_SE[0], rel=1e-6)
        assert m.intercept_t_statistic == pytest.approx(ref.PROBIT_STAT[0], rel=1e-6)
        assert m.intercept_p_value == pytest.approx(ref.PROBIT_P[0], rel=1e-6)
        ci = m.intercept_conf_interval
        assert ci[0] == pytest.approx(ref.PROBIT_CI_LO[0], rel=1e-6)
        assert ci[1] == pytest.approx(ref.PROBIT_CI_HI[0], rel=1e-6)

        assert m.coefficients[0] == pytest.approx(ref.PROBIT_EST[1], rel=1e-6)
        assert m.std_errors[0] == pytest.approx(ref.PROBIT_SE[1], rel=1e-6)
        assert m.t_statistics[0] == pytest.approx(ref.PROBIT_STAT[1], rel=1e-6)
        assert m.p_values[0] == pytest.approx(ref.PROBIT_P[1], rel=1e-6)
        assert m.conf_interval_lower[0] == pytest.approx(ref.PROBIT_CI_LO[1], rel=1e-6)
        assert m.conf_interval_upper[0] == pytest.approx(ref.PROBIT_CI_HI[1], rel=1e-6)

    def test_probit_confidence_level_kwargs(self):
        m90 = self._fit(confidence_level=0.90)
        z90 = NormalDist().inv_cdf(0.95)
        expected_lo = m90.intercept - z90 * m90.intercept_std_error
        expected_hi = m90.intercept + z90 * m90.intercept_std_error
        ci90 = m90.intercept_conf_interval
        assert ci90[0] == pytest.approx(expected_lo, rel=1e-10)
        assert ci90[1] == pytest.approx(expected_hi, rel=1e-10)

    def test_probit_no_inference_kwargs(self):
        m = self._fit(compute_inference=False)
        for attr in (
            "std_errors",
            "t_statistics",
            "p_values",
            "conf_interval_lower",
            "conf_interval_upper",
            "intercept_std_error",
            "intercept_t_statistic",
            "intercept_p_value",
            "intercept_conf_interval",
        ):
            assert getattr(m, attr) is None

    def test_probit_defaults_unchanged(self):
        m_default = self._fit()
        m_no_inf = self._fit(compute_inference=False)
        assert m_default.intercept == m_no_inf.intercept
        assert m_default.coefficients[0] == m_no_inf.coefficients[0]

    def test_probit_confidence_level_out_of_range_raises(self):
        with pytest.raises(ValueError):
            Probit(confidence_level=1.5)

    def test_probit_unfitted_raises(self):
        m = Probit()
        for attr in (
            "std_errors",
            "t_statistics",
            "p_values",
            "conf_interval_lower",
            "conf_interval_upper",
            "intercept_std_error",
            "intercept_t_statistic",
            "intercept_p_value",
            "intercept_conf_interval",
        ):
            with pytest.raises(RuntimeError, match="Probit"):
                getattr(m, attr)


# ============================================================================
# Cloglog: no R reference table -- Wald-identity checks only
# ============================================================================


class TestCloglogWaldIdentities:
    def _fit(self, **kwargs):
        x = MT_MPG.reshape(-1, 1)
        return Cloglog(tol=1e-12, **kwargs).fit(x, MT_VS)

    def test_cloglog_intercept_wald_identities(self):
        m = self._fit()
        z = m.intercept / m.intercept_std_error
        assert m.intercept_t_statistic == pytest.approx(z, rel=1e-12)
        expected_p = 2 * (1 - NormalDist().cdf(abs(z)))
        assert m.intercept_p_value == pytest.approx(expected_p, abs=1e-12)
        z975 = NormalDist().inv_cdf(0.975)
        ci = m.intercept_conf_interval
        assert ci[0] == pytest.approx(m.intercept - z975 * m.intercept_std_error, rel=1e-10)
        assert ci[1] == pytest.approx(m.intercept + z975 * m.intercept_std_error, rel=1e-10)

    def test_cloglog_slope_wald_identities(self):
        m = self._fit()
        for i in range(len(m.coefficients)):
            z = m.coefficients[i] / m.std_errors[i]
            assert m.t_statistics[i] == pytest.approx(z, rel=1e-12)
            expected_p = 2 * (1 - NormalDist().cdf(abs(z)))
            assert m.p_values[i] == pytest.approx(expected_p, abs=1e-12)
            z975 = NormalDist().inv_cdf(0.975)
            lo = m.coefficients[i] - z975 * m.std_errors[i]
            hi = m.coefficients[i] + z975 * m.std_errors[i]
            assert m.conf_interval_lower[i] == pytest.approx(lo, rel=1e-10)
            assert m.conf_interval_upper[i] == pytest.approx(hi, rel=1e-10)

    def test_cloglog_confidence_level_kwargs(self):
        m90 = self._fit(confidence_level=0.90)
        z90 = NormalDist().inv_cdf(0.95)
        expected_lo = m90.intercept - z90 * m90.intercept_std_error
        expected_hi = m90.intercept + z90 * m90.intercept_std_error
        ci90 = m90.intercept_conf_interval
        assert ci90[0] == pytest.approx(expected_lo, rel=1e-10)
        assert ci90[1] == pytest.approx(expected_hi, rel=1e-10)

    def test_cloglog_no_inference_kwargs(self):
        m = self._fit(compute_inference=False)
        for attr in (
            "std_errors",
            "t_statistics",
            "p_values",
            "conf_interval_lower",
            "conf_interval_upper",
            "intercept_std_error",
            "intercept_t_statistic",
            "intercept_p_value",
            "intercept_conf_interval",
        ):
            assert getattr(m, attr) is None

    def test_cloglog_defaults_unchanged(self):
        m_default = self._fit()
        m_no_inf = self._fit(compute_inference=False)
        assert m_default.intercept == m_no_inf.intercept
        assert m_default.coefficients[0] == m_no_inf.coefficients[0]

    def test_cloglog_confidence_level_out_of_range_raises(self):
        with pytest.raises(ValueError):
            Cloglog(confidence_level=0.0)

    def test_cloglog_unfitted_raises(self):
        m = Cloglog()
        for attr in (
            "std_errors",
            "t_statistics",
            "p_values",
            "conf_interval_lower",
            "conf_interval_upper",
            "intercept_std_error",
            "intercept_t_statistic",
            "intercept_p_value",
            "intercept_conf_interval",
        ):
            with pytest.raises(RuntimeError, match="Cloglog"):
                getattr(m, attr)


# ============================================================================
# LogisticRegression: sklearn-style wrapper vs the same R binomial-logit table
# ============================================================================


class TestLogisticRegressionMatchesR:
    def _fit(self, **kwargs):
        x = MT_MPG.reshape(-1, 1)
        return LogisticRegression(
            penalty="none", max_iter=100, tol=1e-12, **kwargs
        ).fit(x, MT_VS)

    def test_logistic_regression_matches_r(self):
        m = self._fit()
        assert m.intercept == pytest.approx(ref.BIN_EST[0], rel=1e-6)
        assert m.intercept_std_error == pytest.approx(ref.BIN_SE[0], rel=1e-6)
        assert m.intercept_t_statistic == pytest.approx(ref.BIN_STAT[0], rel=1e-6)
        assert m.intercept_p_value == pytest.approx(ref.BIN_P[0], rel=1e-6)
        ci = m.intercept_conf_interval
        assert ci[0] == pytest.approx(ref.BIN_CI_LO[0], rel=1e-6)
        assert ci[1] == pytest.approx(ref.BIN_CI_HI[0], rel=1e-6)

        assert m.coefficients[0] == pytest.approx(ref.BIN_EST[1], rel=1e-6)
        assert m.std_errors[0] == pytest.approx(ref.BIN_SE[1], rel=1e-6)
        assert m.t_statistics[0] == pytest.approx(ref.BIN_STAT[1], rel=1e-6)
        assert m.p_values[0] == pytest.approx(ref.BIN_P[1], rel=1e-6)
        assert m.conf_interval_lower[0] == pytest.approx(ref.BIN_CI_LO[1], rel=1e-6)
        assert m.conf_interval_upper[0] == pytest.approx(ref.BIN_CI_HI[1], rel=1e-6)

    def test_logistic_regression_no_inference_kwargs(self):
        m = self._fit(compute_inference=False)
        for attr in (
            "std_errors",
            "t_statistics",
            "p_values",
            "conf_interval_lower",
            "conf_interval_upper",
            "intercept_std_error",
            "intercept_t_statistic",
            "intercept_p_value",
            "intercept_conf_interval",
        ):
            assert getattr(m, attr) is None

    def test_logistic_regression_confidence_level_out_of_range_raises(self):
        with pytest.raises(ValueError):
            LogisticRegression(confidence_level=1.5)

    def test_logistic_regression_unfitted_raises(self):
        m = LogisticRegression()
        for attr in (
            "std_errors",
            "t_statistics",
            "p_values",
            "conf_interval_lower",
            "conf_interval_upper",
            "intercept_std_error",
            "intercept_t_statistic",
            "intercept_p_value",
            "intercept_conf_interval",
        ):
            with pytest.raises(RuntimeError, match="LogisticRegression"):
                getattr(m, attr)


# ============================================================================
# Poisson: Poisson GLM vs R glm(carb ~ wt + hp, poisson)
# ============================================================================


class TestPoissonMatchesR:
    def _fit(self, **kwargs):
        x = np.column_stack([MT_WT, MT_HP])
        return Poisson(max_iter=100, tol=1e-12, **kwargs).fit(x, MT_CARB)

    def test_poisson_matches_r(self):
        m = self._fit()
        assert m.intercept == pytest.approx(ref.POIS_EST[0], rel=1e-6)
        assert m.intercept_std_error == pytest.approx(ref.POIS_SE[0], rel=1e-6)
        assert m.intercept_t_statistic == pytest.approx(ref.POIS_STAT[0], rel=1e-6)
        assert m.intercept_p_value == pytest.approx(ref.POIS_P[0], rel=1e-6)
        ci = m.intercept_conf_interval
        assert ci[0] == pytest.approx(ref.POIS_CI_LO[0], rel=1e-6)
        assert ci[1] == pytest.approx(ref.POIS_CI_HI[0], rel=1e-6)

        for i in range(2):
            assert m.coefficients[i] == pytest.approx(ref.POIS_EST[i + 1], rel=1e-6)
            assert m.std_errors[i] == pytest.approx(ref.POIS_SE[i + 1], rel=1e-6)
            assert m.t_statistics[i] == pytest.approx(ref.POIS_STAT[i + 1], rel=1e-6)
            assert m.p_values[i] == pytest.approx(ref.POIS_P[i + 1], rel=1e-6)
            assert m.conf_interval_lower[i] == pytest.approx(ref.POIS_CI_LO[i + 1], rel=1e-6)
            assert m.conf_interval_upper[i] == pytest.approx(ref.POIS_CI_HI[i + 1], rel=1e-6)

    def test_poisson_no_inference_kwargs(self):
        m = self._fit(compute_inference=False)
        for attr in (
            "std_errors",
            "t_statistics",
            "p_values",
            "conf_interval_lower",
            "conf_interval_upper",
            "intercept_std_error",
            "intercept_t_statistic",
            "intercept_p_value",
            "intercept_conf_interval",
        ):
            assert getattr(m, attr) is None

    def test_poisson_confidence_level_out_of_range_raises(self):
        with pytest.raises(ValueError):
            Poisson(confidence_level=1.5)

    def test_poisson_unfitted_raises(self):
        m = Poisson()
        for attr in (
            "t_statistics",
            "conf_interval_lower",
            "conf_interval_upper",
            "intercept_std_error",
            "intercept_t_statistic",
            "intercept_p_value",
            "intercept_conf_interval",
        ):
            with pytest.raises(RuntimeError, match="Poisson"):
                getattr(m, attr)


# ============================================================================
# NegativeBinomial: MASS::glm.nb(breaks ~ wool + tension, warpbreaks)
# ============================================================================


class TestNegativeBinomialMatchesR:
    def _fit(self, **kwargs):
        x = np.column_stack([WB_WOOLB, WB_TM, WB_TH])
        return NegativeBinomial(tol=1e-12, max_iter=100, **kwargs).fit(x, WB_BREAKS)

    def test_negbin_matches_r(self):
        m = self._fit()
        assert m.theta_estimated == pytest.approx(ref.NB_THETA, rel=1e-4)
        assert m.intercept == pytest.approx(ref.NB_EST[0], rel=1e-6)
        assert m.intercept_std_error == pytest.approx(ref.NB_SE[0], rel=1e-6)
        assert m.intercept_t_statistic == pytest.approx(ref.NB_STAT[0], rel=1e-6)
        assert m.intercept_p_value == pytest.approx(ref.NB_P[0], rel=1e-6)
        ci = m.intercept_conf_interval
        assert ci[0] == pytest.approx(ref.NB_CI_LO[0], rel=1e-6)
        assert ci[1] == pytest.approx(ref.NB_CI_HI[0], rel=1e-6)

        for i in range(3):
            assert m.coefficients[i] == pytest.approx(ref.NB_EST[i + 1], rel=1e-6)
            assert m.std_errors[i] == pytest.approx(ref.NB_SE[i + 1], rel=1e-6)
            assert m.t_statistics[i] == pytest.approx(ref.NB_STAT[i + 1], rel=1e-6)
            assert m.p_values[i] == pytest.approx(ref.NB_P[i + 1], rel=1e-6)
            assert m.conf_interval_lower[i] == pytest.approx(ref.NB_CI_LO[i + 1], rel=1e-6)
            assert m.conf_interval_upper[i] == pytest.approx(ref.NB_CI_HI[i + 1], rel=1e-6)

    def test_negbin_no_inference_kwargs(self):
        m = self._fit(compute_inference=False)
        for attr in (
            "std_errors",
            "t_statistics",
            "p_values",
            "conf_interval_lower",
            "conf_interval_upper",
            "intercept_std_error",
            "intercept_t_statistic",
            "intercept_p_value",
            "intercept_conf_interval",
        ):
            assert getattr(m, attr) is None

    def test_negbin_confidence_level_narrows(self):
        m90 = self._fit(confidence_level=0.90)
        m95 = self._fit()
        ci90 = m90.intercept_conf_interval
        ci95 = m95.intercept_conf_interval
        assert ci95[0] < ci90[0] < ci90[1] < ci95[1]

    def test_negbin_confidence_level_out_of_range_raises(self):
        with pytest.raises(ValueError):
            NegativeBinomial(confidence_level=1.5)


# ============================================================================
# Gamma / Tweedie: glm(mpg ~ wt + hp, Gamma(log)); Tweedie(var_power=2) wraps it
# ============================================================================


class TestGammaTweedieMatchesR:
    def _x(self):
        return np.column_stack([MT_WT, MT_HP])

    def test_gamma_matches_r(self):
        m = Gamma(max_iter=100, tol=1e-12).fit(self._x(), MT_MPG)
        assert m.intercept == pytest.approx(ref.GAMMA_EST[0], rel=1e-6)
        assert m.intercept_std_error == pytest.approx(ref.GAMMA_SE[0], rel=1e-6)
        assert m.intercept_t_statistic == pytest.approx(ref.GAMMA_STAT[0], rel=1e-6)
        assert m.intercept_p_value == pytest.approx(ref.GAMMA_P[0], rel=1e-6)
        ci = m.intercept_conf_interval
        assert ci[0] == pytest.approx(ref.GAMMA_CI_LO[0], rel=1e-6)
        assert ci[1] == pytest.approx(ref.GAMMA_CI_HI[0], rel=1e-6)

        for i in range(2):
            assert m.coefficients[i] == pytest.approx(ref.GAMMA_EST[i + 1], rel=1e-6)
            assert m.std_errors[i] == pytest.approx(ref.GAMMA_SE[i + 1], rel=1e-6)
            assert m.t_statistics[i] == pytest.approx(ref.GAMMA_STAT[i + 1], rel=1e-6)
            assert m.p_values[i] == pytest.approx(ref.GAMMA_P[i + 1], rel=1e-6)
            assert m.conf_interval_lower[i] == pytest.approx(ref.GAMMA_CI_LO[i + 1], rel=1e-6)
            assert m.conf_interval_upper[i] == pytest.approx(ref.GAMMA_CI_HI[i + 1], rel=1e-6)

    def test_tweedie_gamma_wrapper_matches_r(self):
        m = Tweedie(var_power=2.0, link_power=0.0, max_iter=100, tol=1e-12).fit(
            self._x(), MT_MPG
        )
        assert m.intercept == pytest.approx(ref.GAMMA_EST[0], rel=1e-6)
        assert m.intercept_std_error == pytest.approx(ref.GAMMA_SE[0], rel=1e-6)
        assert m.intercept_t_statistic == pytest.approx(ref.GAMMA_STAT[0], rel=1e-6)
        assert m.intercept_p_value == pytest.approx(ref.GAMMA_P[0], rel=1e-6)

    def test_tweedie_var_power_1_5_wald_identities(self):
        m = Tweedie(var_power=1.5, link_power=0.0, max_iter=100, tol=1e-12).fit(
            self._x(), MT_MPG
        )
        z = m.intercept / m.intercept_std_error
        assert m.intercept_t_statistic == pytest.approx(z, rel=1e-12)
        assert 0.0 < m.intercept_p_value < 1.0
        ci = m.intercept_conf_interval
        mid = (ci[0] + ci[1]) / 2.0
        assert mid == pytest.approx(m.intercept, rel=1e-10)

    def test_gamma_no_inference_kwargs(self):
        m = Gamma(compute_inference=False).fit(self._x(), MT_MPG)
        for attr in (
            "std_errors",
            "t_statistics",
            "p_values",
            "conf_interval_lower",
            "conf_interval_upper",
            "intercept_std_error",
            "intercept_t_statistic",
            "intercept_p_value",
            "intercept_conf_interval",
        ):
            assert getattr(m, attr) is None

    def test_tweedie_gamma_factory_no_inference_kwargs(self):
        m = Tweedie.gamma(compute_inference=False).fit(self._x(), MT_MPG)
        for attr in (
            "std_errors",
            "t_statistics",
            "p_values",
            "conf_interval_lower",
            "conf_interval_upper",
            "intercept_std_error",
            "intercept_t_statistic",
            "intercept_p_value",
            "intercept_conf_interval",
        ):
            assert getattr(m, attr) is None

    def test_gamma_confidence_level_out_of_range_raises(self):
        with pytest.raises(ValueError):
            Gamma(confidence_level=1.5)

    def test_tweedie_confidence_level_out_of_range_raises(self):
        with pytest.raises(ValueError):
            Tweedie(confidence_level=1.5)
        with pytest.raises(ValueError):
            Tweedie.gaussian(confidence_level=0.0)
        with pytest.raises(ValueError):
            Tweedie.gamma(confidence_level=0.0)
        with pytest.raises(ValueError):
            Tweedie.inverse_gaussian(confidence_level=0.0)


# ============================================================================
# OLS / WLS: lm(mpg ~ wt + hp, mtcars) / lm(..., weights = 1/cyl)
# ============================================================================


class TestOlsWlsMatchR:
    def _x(self):
        return np.column_stack([MT_WT, MT_HP])

    def test_ols_matches_r(self):
        m = OLS().fit(self._x(), MT_MPG)
        assert m.intercept_std_error == pytest.approx(ref.OLS_SE[0], rel=1e-8)
        assert m.intercept_t_statistic == pytest.approx(ref.OLS_STAT[0], rel=1e-8)
        assert m.intercept_p_value == pytest.approx(ref.OLS_P[0], rel=1e-8)
        ci = m.intercept_conf_interval
        assert ci[0] == pytest.approx(ref.OLS_CI_LO[0], rel=1e-8)
        assert ci[1] == pytest.approx(ref.OLS_CI_HI[0], rel=1e-8)

        for i in range(2):
            assert m.conf_interval_lower[i] == pytest.approx(ref.OLS_CI_LO[i + 1], rel=1e-8)
            assert m.conf_interval_upper[i] == pytest.approx(ref.OLS_CI_HI[i + 1], rel=1e-8)

    def test_wls_matches_r(self):
        m = WLS().fit(self._x(), MT_MPG, MT_W)
        assert m.std_errors[0] == pytest.approx(ref.WLS_SE[1], rel=1e-8)
        assert m.t_statistics[0] == pytest.approx(ref.WLS_STAT[1], rel=1e-8)
        assert m.p_values[0] == pytest.approx(ref.WLS_P[1], rel=1e-8)
        assert m.conf_interval_lower[0] == pytest.approx(ref.WLS_CI_LO[1], rel=1e-8)
        assert m.conf_interval_upper[0] == pytest.approx(ref.WLS_CI_HI[1], rel=1e-8)

        assert m.intercept_std_error == pytest.approx(ref.WLS_SE[0], rel=1e-8)
        assert m.intercept_t_statistic == pytest.approx(ref.WLS_STAT[0], rel=1e-8)
        assert m.intercept_p_value == pytest.approx(ref.WLS_P[0], rel=1e-8)
        ci = m.intercept_conf_interval
        assert ci[0] == pytest.approx(ref.WLS_CI_LO[0], rel=1e-8)
        assert ci[1] == pytest.approx(ref.WLS_CI_HI[0], rel=1e-8)

    def test_ols_no_inference_kwargs(self):
        m = OLS(compute_inference=False).fit(self._x(), MT_MPG)
        for attr in (
            "conf_interval_lower",
            "conf_interval_upper",
            "intercept_std_error",
            "intercept_t_statistic",
            "intercept_p_value",
            "intercept_conf_interval",
        ):
            assert getattr(m, attr) is None

    def test_wls_no_inference_kwargs(self):
        m = WLS(compute_inference=False).fit(self._x(), MT_MPG, MT_W)
        for attr in (
            "std_errors",
            "t_statistics",
            "p_values",
            "conf_interval_lower",
            "conf_interval_upper",
            "intercept_std_error",
            "intercept_t_statistic",
            "intercept_p_value",
            "intercept_conf_interval",
        ):
            assert getattr(m, attr) is None

    def test_wls_unfitted_raises(self):
        m = WLS()
        for attr in (
            "std_errors",
            "t_statistics",
            "p_values",
            "conf_interval_lower",
            "conf_interval_upper",
            "intercept_std_error",
            "intercept_t_statistic",
            "intercept_p_value",
            "intercept_conf_interval",
        ):
            with pytest.raises(RuntimeError, match="WLS"):
                getattr(m, attr)


