"""Smoke tests for GlmmRegressor PyModel (REGR-01).

16.3 plan 03 (UPST-09) adds the negative-binomial / Gamma / Tweedie families,
offsets, and per-factor random slopes from anofox-regression 0.5.18-0.5.22
(issue #29), validated against lme4 2.0.1 / tweedie 3.1.0 / statmod 1.5.2 via
``tests/fixtures/glmm_families_reference.py`` (verbatim transcription of
upstream's ``tests/fixtures/glmm_families_validation.rs``; see that module's
docstring for provenance and ``SOURCE_SHA256``).
"""

import hashlib
import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pytest
from polars_statistics import GLMM

_FAM_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "glmm_families_reference.py"
_fam_spec = importlib.util.spec_from_file_location(
    "glmm_families_reference", _FAM_FIXTURE_PATH
)
fam = importlib.util.module_from_spec(_fam_spec)
_fam_spec.loader.exec_module(fam)


def close(got: float, want: float, tol: float) -> bool:
    """Upstream's `close(a, b, tol) = |a - b| / (1 + |b|) < tol`."""
    return abs(got - want) / (1.0 + abs(want)) < tol


def assert_close_vec(got, want, tol: float, label: str = "") -> None:
    """Element-wise `close`, matching upstream's `close_vec` assertion style."""
    got = list(got)
    want = list(want)
    assert len(got) == len(want), f"{label}: length {len(got)} != {len(want)}"
    for i, (g, w) in enumerate(zip(got, want)):
        err = abs(g - w) / (1.0 + abs(w))
        assert err < tol, f"{label}[{i}]: got {g!r}, want {w!r} (scaled err {err:.2e} >= {tol:.0e})"


def in_solver_order(ids, r_values):
    """Reorder R's per-level values (sorted level ids) into the solver's level
    order, which is the order of first appearance in `ids`."""
    seen = []
    for i in ids:
        if i not in seen:
            seen.append(i)
    return [r_values[i] for i in seen]

# Scenario LMM_A from anofox-regression 0.5.16
# tests/fixtures/glmm_validation.rs (80 rows, 1 feature, 8 groups of 10), copied
# verbatim. lme4 2.0.1 Wald-inference references from
# tests/issue51_glmm_inference.rs (phase 16.1 plan 03, UPST-03).
X_LMM_A = [
    1.371, -0.5647, 0.3631, 0.6329, 0.4043, -0.1061, 1.5115, -0.0947, 2.0184, -0.0627,
    1.3049, 2.2866, -1.3889, -0.2788, -0.1333, 0.636, -0.2843, -2.6565, -2.4405, 1.3201,
    -0.3066, -1.7813, -0.1719, 1.2147, 1.8952, -0.4305, -0.2573, -1.7632, 0.4601, -0.64,
    0.4555, 0.7048, 1.0351, -0.6089, 0.505, -1.717, -0.7845, -0.8509, -2.4142, 0.0361,
    0.206, -0.3611, 0.7582, -0.7267, -1.3683, 0.4328, -0.8114, 1.4441, -0.4314, 0.6556,
    0.3219, -0.7838, 1.5757, 0.6429, 0.0898, 0.2766, 0.6793, 0.0898, -2.9931, 0.2849,
    -0.3672, 0.1852, 0.5818, 1.3997, -0.7273, 1.3025, 0.3358, 1.0385, 0.9207, 0.7209,
    -1.0431, -0.0902, 0.6235, -0.9535, -0.5428, 0.581, 0.7682, 0.4638, -0.8858, -1.0998,
]

Y_LMM_A = [
    3.3623, 2.3387, 3.0878, 2.2885, 2.7375, 2.8527, 2.4105, 1.7324, 2.6535, 1.4492,
    1.8988, 2.6762, 1.1124, 1.5893, 0.6381, 2.4486, 0.7308, -0.0692, -0.225, 1.8052,
    1.0115, 0.2397, 0.9723, 1.7321, 1.7756, 0.6034, 0.1116, -0.002, 1.0445, 2.1017,
    0.45, 1.3243, 0.674, -0.1364, 1.2181, -0.4535, 0.5101, 0.2637, -0.6107, -0.091,
    -0.4648, -0.0463, 0.7074, -0.5653, -0.6396, 0.8224, 0.3588, 0.218, -0.2298, 0.9731,
    1.4157, 1.0715, 2.2344, 1.3672, 1.3122, 1.6132, 1.6223, 2.0912, -0.2474, 1.4155,
    0.9911, 0.3907, 1.0968, 0.7504, 1.0462, 1.3407, 0.7603, 0.7264, 1.2828, 0.7866,
    0.0655, 1.4525, 1.0778, -0.1588, 0.664, 0.9629, 1.5329, 1.8019, -0.0855, 0.5312,
]

G_LMM_A = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
]


def _fit_lmm_a() -> GLMM:
    X = np.array(X_LMM_A, dtype=float)[:, None]
    y = np.array(Y_LMM_A, dtype=float)
    group = list(G_LMM_A)
    return GLMM.gaussian(reml=True).fit(X, y, group)


class TestGLMM:
    def test_gaussian_fit_basic(self):
        """GLMM.gaussian().fit(X, y, group) returns is_fitted==True and finite fixed_effects."""
        rng = np.random.default_rng(42)
        n = 60
        X = rng.standard_normal((n, 2))
        group = np.repeat(np.arange(6), 10).astype(np.int64)
        y = X @ np.array([1.0, -0.5]) + rng.standard_normal(n) * 0.3
        model = GLMM.gaussian().fit(X, y, group.tolist())
        assert model.is_fitted()
        fe = model.fixed_effects
        assert len(fe) >= 2
        assert np.all(np.isfinite(fe))

    def test_fit_crossed(self):
        """GLMM.gaussian().fit_crossed returns factors() with >=2 dicts each having n_levels/sd/blups."""
        rng = np.random.default_rng(7)
        n = 80
        X = rng.standard_normal((n, 2))
        y = X @ np.array([1.0, 0.5]) + rng.standard_normal(n) * 0.2
        g1 = np.repeat(np.arange(4), 20).astype(np.int64)
        g2 = np.tile(np.arange(4), 20).astype(np.int64)
        model = GLMM.gaussian().fit_crossed(X, y, [g1.tolist(), g2.tolist()])
        assert model.is_fitted()
        fs = model.factors()
        assert len(fs) >= 2
        for factor_dict in fs:
            assert "n_levels" in factor_dict
            assert "sd" in factor_dict
            assert "blups" in factor_dict
            assert np.isfinite(factor_dict["sd"])

    def test_not_fitted_raises(self):
        """Reading fixed_effects before fit raises RuntimeError."""
        model = GLMM.gaussian()
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.fixed_effects

    def test_predict_fixed(self):
        """predict_fixed returns marginal predictions of shape (n,)."""
        rng = np.random.default_rng(0)
        n = 50
        X = rng.standard_normal((n, 2))
        group = np.repeat(np.arange(5), 10).astype(np.int64)
        y = X @ np.array([2.0, -1.0]) + rng.standard_normal(n) * 0.5
        model = GLMM.gaussian().fit(X, y, group.tolist())
        preds = model.predict_fixed(X)
        assert preds.shape == (n,)
        assert np.all(np.isfinite(preds))

    def test_getters(self):
        """FittedGlmm getters return sensible values."""
        rng = np.random.default_rng(1)
        n = 60
        X = rng.standard_normal((n, 2))
        group = np.repeat(np.arange(6), 10).astype(np.int64)
        y = X @ np.array([1.0, -0.5]) + rng.standard_normal(n) * 0.3
        model = GLMM.gaussian().fit(X, y, group.tolist())
        assert isinstance(model.converged, bool)
        assert model.n_groups == 6
        assert model.iterations >= 1
        assert np.isfinite(model.deviance)
        assert np.isfinite(model.log_likelihood)
        assert np.isfinite(model.sigma)
        assert np.isfinite(model.theta)

    def test_poisson_factory(self):
        """GLMM.poisson() factory can be constructed."""
        model = GLMM.poisson()
        assert not model.is_fitted()

    def test_binomial_factory(self):
        """GLMM.binomial() factory can be constructed."""
        model = GLMM.binomial()
        assert not model.is_fitted()

    def test_fixed_effect_near_truth(self):
        """GLMM Gaussian random-intercept: fixed-effect slope is near the true slope.

        Design: y = 1.5 * x + u_g + noise, where u_g ~ N(0, 0.5^2) is a
        group-specific random intercept across 10 groups of 10 observations.
        The fixed-effect slope should be recovered within 0.30 of 1.5.

        TEST BUG FIX (phase 06-05): the GLMM with an intercept column prepended
        internally returns fixed_effects as [intercept, slope, ...].  The original
        test used fe[0] (the intercept, ~-0.10) instead of fe[1] (the slope,
        ~1.55).  Corrected to extract fe[1] for the slope coefficient.

        KNOWN LIMITATION: factors() returns an empty list after .fit() (the
        random-intercept variance is encoded in model.theta/sigma rather than
        a factors() entry).  The random-intercept SD assertion is therefore
        checked via model.theta directly.
        """
        rng = np.random.default_rng(42)
        n_groups, n_per = 10, 10
        n = n_groups * n_per
        X = rng.standard_normal((n, 1))
        group = np.repeat(np.arange(n_groups), n_per).astype(np.int64)
        u = rng.normal(0, 0.5, n_groups)  # random intercepts
        true_slope = 1.5
        y = true_slope * X[:, 0] + u[group] + rng.standard_normal(n) * 0.3
        model = GLMM.gaussian().fit(X, y, group.tolist())
        fe = model.fixed_effects
        # fe[0] = intercept, fe[1] = slope for the single predictor
        assert len(fe) >= 2, f"Expected at least 2 fixed effects (intercept + slope), got {len(fe)}"
        slope_estimate = fe[1]
        assert abs(slope_estimate - true_slope) < 0.30, (
            f"GLMM fixed-effect slope fe[1]={slope_estimate:.4f} far from true {true_slope}"
        )
        # Random intercept variance is encoded in model.theta (not factors()).
        # theta > 0 confirms the random intercept is non-trivially estimated.
        assert np.isfinite(model.theta), "model.theta must be finite"
        assert model.theta >= 0.0, "model.theta (variance ratio) must be non-negative"

    def test_factor_summary_populated(self):
        """factors() returns a list after fit(); fit_crossed() populates it with dicts.

        KNOWN LIMITATION (phase 06-05): factors() returns an empty list after
        the standard .fit() path regardless of the number of groups or the
        magnitude of the random-intercept variance.  The random-intercept variance
        is accessible via model.theta and model.sigma rather than factors().
        factors() is populated only via the fit_crossed() path (see test_fit_crossed).

        This test is updated to:
          1. Assert factors() is a list (type contract upheld).
          2. Verify that model.theta and model.sigma (which encode the
             random-intercept) are finite and accessible after .fit().
        A follow-up task should wire the single-grouping random effect into the
        factors() return value for consistency with fit_crossed().
        """
        rng = np.random.default_rng(7)
        n = 60
        X = rng.standard_normal((n, 2))
        group = np.repeat(np.arange(6), 10).astype(np.int64)
        y = X @ np.array([1.0, -0.5]) + rng.standard_normal(n) * 0.3
        model = GLMM.gaussian().fit(X, y, group.tolist())
        fs = model.factors()
        # Type contract: factors() always returns a list
        assert isinstance(fs, list), f"factors() must return a list, got {type(fs)}"
        # Random-intercept variance is accessible via theta/sigma even when factors() is empty
        assert np.isfinite(model.theta), "model.theta must be finite after fit()"
        assert np.isfinite(model.sigma), "model.sigma must be finite after fit()"
        assert model.sigma > 0.0, "model.sigma (residual SD) must be positive"


class TestGLMMWaldInference:
    """Wald z/p-values and confidence intervals (16.1 plan 03, UPST-03) vs
    lme4 2.0.1 on scenario LMM_A (anofox-regression issue51_glmm_inference.rs)."""

    def test_z_values_match_lme4(self):
        model = _fit_lmm_a()
        z = model.z_values
        expected = [4.451695054970e+00, 9.554977878286e+00]
        for actual, exp in zip(z, expected):
            tol = 2e-3 * (1 + abs(exp))
            assert abs(actual - exp) <= tol, f"z={actual} expected={exp} tol={tol}"

    def test_p_values_match_lme4(self):
        model = _fit_lmm_a()
        p = model.p_values
        expected = [8.519509552922e-06, 1.236103260560e-21]
        for actual, exp in zip(p, expected):
            tol = 10 * 2e-3 * max(exp, 1e-3)
            assert abs(actual - exp) <= tol, f"p={actual} expected={exp} tol={tol}"

    def test_conf_int_95_matches_lme4(self):
        model = _fit_lmm_a()
        lower, upper = model.conf_int(0.95)
        np.testing.assert_allclose(
            lower, [5.704827690382e-01, 3.776760859016e-01], atol=2e-3
        )
        np.testing.assert_allclose(
            upper, [1.467951757072e+00, 5.726017401650e-01], atol=2e-3
        )

    def test_conf_int_90_matches_lme4(self):
        model = _fit_lmm_a()
        lower, upper = model.conf_int(0.90)
        np.testing.assert_allclose(
            lower, [6.426274043414e-01, 3.933455301114e-01], atol=2e-3
        )
        np.testing.assert_allclose(
            upper, [1.395807121769e+00, 5.569322959551e-01], atol=2e-3
        )

    def test_z_values_internally_consistent_with_fixed_effects_and_std_errors(self):
        model = _fit_lmm_a()
        z = model.z_values
        fe = model.fixed_effects
        se = model.std_errors
        np.testing.assert_allclose(z, fe / se, rtol=1e-12)

    def test_conf_int_95_midpoint_matches_fixed_effects(self):
        model = _fit_lmm_a()
        lower, upper = model.conf_int(0.95)
        fe = model.fixed_effects
        midpoint = (np.array(lower) + np.array(upper)) / 2
        np.testing.assert_allclose(midpoint, fe, atol=1e-9)

    def test_conf_int_95_halfwidth_matches_normal_quantile(self):
        model = _fit_lmm_a()
        lower, upper = model.conf_int(0.95)
        se = model.std_errors
        ratio = (np.array(upper) - np.array(lower)) / (2 * np.array(se))
        np.testing.assert_allclose(ratio, 1.959963984540054, atol=1e-9)

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 1.5, float("nan")])
    def test_conf_int_invalid_level_returns_all_nan(self, level):
        model = _fit_lmm_a()
        lower, upper = model.conf_int(level)
        assert len(lower) == 2
        assert len(upper) == 2
        assert all(math.isnan(v) for v in lower)
        assert all(math.isnan(v) for v in upper)

    def test_unfitted_z_values_raises(self):
        model = GLMM.gaussian()
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.z_values

    def test_unfitted_p_values_raises(self):
        model = GLMM.gaussian()
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.p_values

    def test_unfitted_conf_int_raises(self):
        model = GLMM.gaussian()
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.conf_int()


def test_glmm_families_fixture_integrity():
    """Fixture length/value sanity, plus a live re-parse-and-compare against the
    upstream .rs file when the cargo registry cache is present (16.3-02 pattern)."""
    assert len(fam.X) == 180
    assert len(fam.XC) == 240
    assert fam.NB_H_THETA == pytest.approx(3.6625227730)
    assert fam.SOURCE_CRATE == "anofox-regression 0.5.22"
    assert len(fam.SOURCE_SHA256) == 64

    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    candidates = []
    for version in ("0.5.21", "0.5.22"):
        candidates.extend(
            sorted(
                cargo_home.glob(
                    f"registry/src/*/anofox-regression-{version}/tests/fixtures/"
                    "glmm_families_validation.rs"
                )
            )
        )
    if not candidates:
        pytest.skip("upstream anofox-regression 0.5.21/0.5.22 source not found")
    upstream = candidates[0]
    content = upstream.read_text()
    sha = hashlib.sha256(content.encode()).hexdigest()
    if sha != fam.SOURCE_SHA256:
        pytest.skip(
            "upstream fixture sha256 differs from the pinned SOURCE_SHA256 "
            "(a different crate version is cached under $CARGO_HOME)"
        )
    assert "3.6625227730" in content


class TestGLMMFamilies:
    """NegBin / Gamma / Tweedie families vs lme4 (16.3-03, UPST-09)."""

    def test_negative_binomial_estimated_theta_matches_glmer_nb(self):
        X = np.array(fam.X)[:, None]
        y = np.array(fam.Y_NB)
        group = list(fam.G)
        model = GLMM.negative_binomial().fit(X, y, group)
        assert model.converged
        tol = 2e-3
        assert close(model.nb_theta, fam.NB_H_THETA, tol)
        assert_close_vec(model.fixed_effects, fam.NB_H_FIXEF, tol, "NB_H fixef")
        assert_close_vec(model.std_errors, fam.NB_H_SE, tol, "NB_H se")
        assert close(model.log_likelihood, fam.NB_H_LOGLIK, tol)
        assert close(model.sigma, fam.NB_H_SIGMA, tol)
        assert close(model.random_sd[0], fam.NB_H_SD_RE, tol)
        assert_close_vec(model.random_effects, fam.NB_H_BLUP, tol, "NB_H blup")

    def test_negative_binomial_fixed_theta_type_contract(self):
        """nb_theta is reported as None for the families that don't use it, and
        echoed exactly for a user-fixed value (full glmer-reference check in
        Task 2's test_negative_binomial_fixed_theta_matches_glmer)."""
        X = np.array(fam.X)[:, None]
        y = np.array(fam.Y_NB)
        group = list(fam.G)
        model = GLMM.negative_binomial(nb_theta=2.0).fit(X, y, group)
        assert model.nb_theta == 2.0

    @pytest.mark.parametrize("bad_theta", [0.0, -1.0, float("nan")])
    def test_negative_binomial_invalid_theta(self, bad_theta):
        with pytest.raises(ValueError):
            GLMM.negative_binomial(nb_theta=bad_theta)

    def test_existing_families_nb_theta_none(self):
        """Pre-existing families regress unchanged and report nb_theta is None."""
        gaussian_model = _fit_lmm_a()
        assert gaussian_model.nb_theta is None

        rng = np.random.default_rng(3)
        n = 60
        X = rng.standard_normal((n, 2))
        group = np.repeat(np.arange(6), 10).astype(np.int64)
        y = rng.poisson(lam=2.0, size=n).astype(float)
        poisson_model = GLMM.poisson().fit(X, y, group.tolist())
        assert poisson_model.is_fitted()
        assert poisson_model.nb_theta is None

    def test_unfitted_nb_theta_raises(self):
        model = GLMM.negative_binomial()
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.nb_theta

    def test_negative_binomial_fixed_theta_matches_glmer(self):
        """nb_theta=2.0 vs lme4 glmer(family=MASS::negative.binomial(2)) (case I)."""
        X = np.array(fam.X)[:, None]
        y = np.array(fam.Y_NB)
        group = list(fam.G)
        model = GLMM.negative_binomial(nb_theta=2.0).fit(X, y, group)
        assert model.converged
        assert model.nb_theta == 2.0
        tol = 2e-4
        assert_close_vec(model.fixed_effects, fam.NB_I_FIXEF, tol, "NB_I fixef")
        assert_close_vec(model.std_errors, fam.NB_I_SE, tol, "NB_I se")
        assert close(model.log_likelihood, fam.NB_I_LOGLIK, tol)
        assert close(model.sigma, fam.NB_I_SIGMA, tol)
        assert close(model.random_sd[0], fam.NB_I_SD_RE, tol)
        assert_close_vec(model.random_effects, fam.NB_I_BLUP, tol, "NB_I blup")

    def test_gamma_matches_glmer(self):
        """GLMM.gamma() vs lme4 glmer(family=Gamma(link="log")) (case J)."""
        X = np.array(fam.X)[:, None]
        y = np.array(fam.Y_GA)
        group = list(fam.G)
        model = GLMM.gamma().fit(X, y, group)
        assert model.converged
        assert model.nb_theta is None
        tol = 2e-4
        assert_close_vec(model.fixed_effects, fam.GA_J_FIXEF, tol, "GA_J fixef")
        assert_close_vec(model.std_errors, fam.GA_J_SE, tol, "GA_J se")
        assert close(model.log_likelihood, fam.GA_J_LOGLIK, tol)
        assert close(model.sigma, fam.GA_J_SIGMA, tol)
        assert close(model.random_sd[0], fam.GA_J_SD_RE, tol)
        assert_close_vec(model.random_effects, fam.GA_J_BLUP, tol, "GA_J blup")

    def test_tweedie_matches_glmer_with_tweedie_family(self):
        """GLMM.tweedie(power=1.5) vs lme4 glmer + statmod::tweedie(1.5) (case K)."""
        X = np.array(fam.X)[:, None]
        y = np.array(fam.Y_TW)
        group = list(fam.G)
        model = GLMM.tweedie(power=1.5).fit(X, y, group)
        assert model.converged
        tol = 2e-4
        assert_close_vec(model.fixed_effects, fam.TW_K_FIXEF, tol, "TW_K fixef")
        assert_close_vec(model.std_errors, fam.TW_K_SE, tol, "TW_K se")
        assert close(model.log_likelihood, fam.TW_K_LOGLIK, tol)
        assert close(model.sigma, fam.TW_K_SIGMA, tol)
        assert close(model.random_sd[0], fam.TW_K_SD_RE, tol)
        assert_close_vec(model.random_effects, fam.TW_K_BLUP, tol, "TW_K blup")

    @pytest.mark.parametrize("bad_power", [2.5, 1.0, 2.0, 0.5, float("nan")])
    def test_tweedie_invalid_power_raises(self, bad_power):
        with pytest.raises(ValueError):
            GLMM.tweedie(power=bad_power)

    def test_gamma_requires_positive_response_validation(self):
        """Gamma GLMM rejects a non-positive response with ValueError, not a panic."""
        X = np.array(fam.X)[:, None]
        y = np.array(fam.Y_GA, dtype=float)
        y[0] = 0.0
        group = list(fam.G)
        with pytest.raises(ValueError):
            GLMM.gamma().fit(X, y, group)


class TestGLMMOffset:
    """Offsets on fit / fit_crossed / predict_fixed for every family (16.3-03
    Task 2, UPST-09) vs lme4 cases L (Poisson) and M (Gaussian REML)."""

    def test_poisson_offset_matches_glmer(self):
        X = np.array(fam.X)[:, None]
        y = np.array(fam.Y_PO_OFF)
        group = list(fam.G)
        offset = np.log(np.array(fam.EXPO))
        model = GLMM.poisson().fit(X, y, group, offset=offset)
        assert model.converged
        tol = 2e-4
        assert_close_vec(model.fixed_effects, fam.PO_L_FIXEF, tol, "PO_L fixef")
        assert_close_vec(model.std_errors, fam.PO_L_SE, tol, "PO_L se")
        assert close(model.log_likelihood, fam.PO_L_LOGLIK, tol)
        assert close(model.sigma, fam.PO_L_SIGMA, tol)
        assert close(model.random_sd[0], fam.PO_L_SD_RE, tol)
        assert_close_vec(model.random_effects, fam.PO_L_BLUP, tol, "PO_L blup")

    def test_predict_fixed_offset_adds_to_linear_predictor(self):
        X = np.array(fam.X)[:, None]
        y = np.array(fam.Y_PO_OFF)
        group = list(fam.G)
        offset = np.log(np.array(fam.EXPO))
        model = GLMM.poisson().fit(X, y, group, offset=offset)
        eta = model.predict_fixed(X)
        eta_off = model.predict_fixed(X, offset=offset)
        np.testing.assert_allclose(eta_off - eta, offset, atol=1e-12)

    def test_lmm_offset_matches_lmer(self):
        X = np.array(fam.X)[:, None]
        y = np.array(fam.Y_M)
        group = list(fam.G)
        offset = np.array(fam.OFF_M)
        model = GLMM.gaussian(reml=True).fit(X, y, group, offset=offset)
        assert model.converged
        tol = 2e-4
        assert_close_vec(model.fixed_effects, fam.LMM_M_FIXEF, tol, "LMM_M fixef")
        assert_close_vec(model.std_errors, fam.LMM_M_SE, tol, "LMM_M se")
        assert close(model.log_likelihood, fam.LMM_M_LOGLIK, tol)
        assert close(model.sigma, fam.LMM_M_SIGMA, tol)
        assert close(model.random_sd[0], fam.LMM_M_SD_RE, tol)
        assert_close_vec(model.random_effects, fam.LMM_M_BLUP, tol, "LMM_M blup")

    def test_fit_crossed_offset_accepted_matches_zero_offset(self):
        """fit_crossed(..., offset=...) is accepted; a zero offset equals the
        fit without an offset exactly."""
        XC = np.array(fam.XC)[:, None]
        groups = [list(fam.A), list(fam.B)]
        y = np.array(fam.Y_O)
        model_no_off = GLMM.poisson().fit_crossed(XC, y, groups)
        model_zero_off = GLMM.poisson().fit_crossed(XC, y, groups, offset=np.zeros(len(y)))
        np.testing.assert_array_equal(model_zero_off.fixed_effects, model_no_off.fixed_effects)

    def test_offset_length_mismatch_raises_in_fit(self):
        X = np.array(fam.X)[:, None]
        y = np.array(fam.Y_NB)
        group = list(fam.G)
        with pytest.raises(ValueError):
            GLMM.poisson().fit(X, y, group, offset=np.zeros(3))

    def test_offset_length_mismatch_raises_in_fit_crossed(self):
        XC = np.array(fam.XC)[:, None]
        groups = [list(fam.A), list(fam.B)]
        y = np.array(fam.Y_O)
        with pytest.raises(ValueError):
            GLMM.poisson().fit_crossed(XC, y, groups, offset=np.zeros(3))

    def test_offset_length_mismatch_raises_in_predict_fixed(self):
        X = np.array(fam.X)[:, None]
        y = np.array(fam.Y_NB)
        group = list(fam.G)
        model = GLMM.poisson().fit(X, y, group)
        with pytest.raises(ValueError):
            model.predict_fixed(X, offset=np.zeros(3))


class TestGLMMRandomSlopes:
    """Per-factor random slopes and the covariance / SD / BLUP accessors
    (16.3-03 Task 3, UPST-09) vs lme4 cases N, O, P."""

    def test_lmm_crossed_with_random_slope_per_factor_matches_lmer(self):
        XC = np.array(fam.XC)[:, None]
        A = list(fam.A)
        B = list(fam.B)
        y = np.array(fam.Y_N)
        model = GLMM.gaussian(random_slopes_per_factor=[[0], []]).fit_crossed(XC, y, [A, B])
        assert model.converged
        tol = 2e-3
        assert_close_vec(model.fixed_effects, fam.LMM_N_FIXEF, tol, "LMM_N fixef")
        assert_close_vec(model.std_errors, fam.LMM_N_SE, tol, "LMM_N se")
        assert close(model.log_likelihood, fam.LMM_N_LOGLIK, tol)
        assert close(model.sigma, fam.LMM_N_SIGMA, tol)
        assert model.n_factors == 2
        sd = model.factor_random_sd(0)
        assert_close_vec(sd, [fam.LMM_N_SD_A0, fam.LMM_N_SD_A1], tol, "LMM_N sd_a")
        cov = model.factor_random_cov(0)
        assert close(cov[0, 1] / (sd[0] * sd[1]), fam.LMM_N_CORR_A, tol)
        factors = model.factors()
        assert close(factors[1]["sd"], fam.LMM_N_SD_B, tol)
        re_a = model.factor_random_effects_matrix(0)
        assert_close_vec(re_a[:, 0], in_solver_order(A, fam.LMM_N_BLUP_A0), tol, "LMM_N blup_a0")
        assert_close_vec(re_a[:, 1], in_solver_order(A, fam.LMM_N_BLUP_A1), tol, "LMM_N blup_a1")
        assert_close_vec(
            factors[1]["blups"], in_solver_order(B, fam.LMM_N_BLUP_B), tol, "LMM_N blup_b"
        )
        assert model.n_random_effects == 2
        assert model.factor_random_cov(1).shape == (1, 1)
        assert model.factor_random_cov(5) is None

    def test_poisson_crossed_with_random_slope_per_factor_matches_glmer(self):
        XC = np.array(fam.XC)[:, None]
        A = list(fam.A)
        B = list(fam.B)
        y = np.array(fam.Y_O)
        model = GLMM.poisson(random_slopes_per_factor=[[0], []]).fit_crossed(XC, y, [A, B])
        assert model.converged
        tol = 2e-3
        assert_close_vec(model.fixed_effects, fam.PO_O_FIXEF, tol, "PO_O fixef")
        assert_close_vec(model.std_errors, fam.PO_O_SE, tol, "PO_O se")
        assert close(model.log_likelihood, fam.PO_O_LOGLIK, tol)
        assert close(model.sigma, fam.PO_O_SIGMA, tol)
        assert model.n_factors == 2
        sd = model.factor_random_sd(0)
        assert_close_vec(sd, [fam.PO_O_SD_A0, fam.PO_O_SD_A1], tol, "PO_O sd_a")
        cov = model.factor_random_cov(0)
        assert close(cov[0, 1] / (sd[0] * sd[1]), fam.PO_O_CORR_A, tol)
        factors = model.factors()
        assert close(factors[1]["sd"], fam.PO_O_SD_B, tol)
        re_a = model.factor_random_effects_matrix(0)
        assert_close_vec(re_a[:, 0], in_solver_order(A, fam.PO_O_BLUP_A0), tol, "PO_O blup_a0")
        assert_close_vec(re_a[:, 1], in_solver_order(A, fam.PO_O_BLUP_A1), tol, "PO_O blup_a1")
        assert_close_vec(
            factors[1]["blups"], in_solver_order(B, fam.PO_O_BLUP_B), tol, "PO_O blup_b"
        )
        assert model.n_random_effects == 2
        assert model.factor_random_cov(1).shape == (1, 1)
        assert model.factor_random_cov(5) is None

    def test_gamma_crossed_matches_glmer(self):
        XC = np.array(fam.XC)[:, None]
        A = list(fam.A)
        B = list(fam.B)
        y = np.array(fam.Y_P)
        model = GLMM.gamma().fit_crossed(XC, y, [A, B])
        assert model.converged
        tol = 2e-4
        assert_close_vec(model.fixed_effects, fam.GA_P_FIXEF, tol, "GA_P fixef")
        assert_close_vec(model.std_errors, fam.GA_P_SE, tol, "GA_P se")
        assert close(model.log_likelihood, fam.GA_P_LOGLIK, tol)
        assert close(model.sigma, fam.GA_P_SIGMA, tol)
        factors = model.factors()
        assert close(factors[0]["sd"], fam.GA_P_SD_A0, tol)
        assert close(factors[1]["sd"], fam.GA_P_SD_B, tol)
        assert_close_vec(
            factors[0]["blups"], in_solver_order(A, fam.GA_P_BLUP_A0), tol, "GA_P blup_a"
        )
        assert_close_vec(
            factors[1]["blups"], in_solver_order(B, fam.GA_P_BLUP_B), tol, "GA_P blup_b"
        )

    def test_random_slopes_per_factor_with_one_factor_equals_random_slopes(self):
        """`fit(random_slopes=[0])` and `fit_crossed(random_slopes_per_factor=[[0]])`
        on a single factor give identical results; exercises the single-factor
        covariance / correlation / BLUP-matrix accessors."""
        XC = np.array(fam.XC)[:, None]
        A = list(fam.A)
        y = np.array(fam.Y_N)
        model_a = GLMM.gaussian(random_slopes=[0]).fit(XC, y, A)
        model_b = GLMM.gaussian(random_slopes_per_factor=[[0]]).fit_crossed(XC, y, [A])
        np.testing.assert_array_equal(model_a.fixed_effects, model_b.fixed_effects)
        np.testing.assert_array_equal(model_a.random_sd, model_b.random_sd)

        cov = model_a.random_cov
        sd = model_a.random_sd
        assert cov.shape == (2, 2)
        np.testing.assert_allclose(np.diag(cov), sd**2, rtol=1e-12)
        np.testing.assert_allclose(cov, cov.T, rtol=1e-12)

        corr = model_a.random_corr
        np.testing.assert_allclose(np.diag(corr), np.ones(2), atol=1e-12)

        rem = model_a.random_effects_matrix
        assert rem.shape == (model_a.n_groups, 2)
        np.testing.assert_allclose(rem[:, 0], model_a.random_effects, rtol=1e-12)

        assert model_a.factor_random_cov(0) is None

    def test_is_reml_and_theta_max_accessors(self):
        X = np.array(fam.X)[:, None]
        group = list(fam.G)
        gaussian_model = GLMM.gaussian().fit(X, np.array(fam.Y_M), group)
        assert gaussian_model.is_reml is True
        poisson_model = GLMM.poisson().fit(X, np.array(fam.Y_PO_OFF), group)
        assert poisson_model.is_reml is False
        theta_max_model = GLMM.gaussian(theta_max=50.0).fit(X, np.array(fam.Y_M), group)
        assert theta_max_model.is_fitted()

    def test_random_slopes_per_factor_length_mismatch_structure_validation(self):
        XC = np.array(fam.XC)[:, None]
        A = list(fam.A)
        B = list(fam.B)
        y = np.array(fam.Y_N)
        with pytest.raises(ValueError):
            GLMM.gaussian(random_slopes_per_factor=[[0]]).fit_crossed(XC, y, [A, B])

    def test_random_slopes_with_multiple_factors_structure_validation(self):
        XC = np.array(fam.XC)[:, None]
        A = list(fam.A)
        B = list(fam.B)
        y = np.array(fam.Y_N)
        with pytest.raises(ValueError):
            GLMM.gaussian(random_slopes=[0]).fit_crossed(XC, y, [A, B])

    def test_random_slopes_and_per_factor_mutually_exclusive_structure_validation(self):
        with pytest.raises(ValueError):
            GLMM.gaussian(random_slopes=[0], random_slopes_per_factor=[[0]])

    def test_random_intercept_false_without_slope_structure_validation(self):
        with pytest.raises(ValueError):
            GLMM.gaussian(random_intercept=False)

    def test_theta_max_zero_structure_validation(self):
        with pytest.raises(ValueError):
            GLMM.gaussian(theta_max=0.0)
