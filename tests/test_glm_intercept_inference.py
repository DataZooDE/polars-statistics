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
from polars_statistics import Logistic

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
