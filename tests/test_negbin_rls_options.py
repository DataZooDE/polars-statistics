"""Tests for the new anofox-regression 0.5.14/0.5.15 options exposed in phase
16.1 plan 03: NegBin ``estimate_dispersion`` and RLS ``initial_p_diagonal``.

R/MASS references:

- NegBin: `MASS::glm.nb`-style fixed-theta fit, `R summary.glm` Pearson-dispersion
  SEs (`anofox-regression` tests `issue49_negbin_se.rs`).
- RLS: upstream `anofox-regression` RLS builder validation and P0 = delta * I
  semantics.

See `.planning/phases/16.1-upstream-crate-refresh/16.1-03-PLAN.md`.
"""

import numpy as np
import polars as pl
import polars_statistics as ps
import pytest
from polars_statistics import RLS, NegativeBinomial

# Shared 30-observation dataset, copied verbatim from
# tests/test_upstream_regression_parity.py (NegBin dispersion=1 reference, #49).
Z1 = [
    0.63, -0.79, -0.44, -0.34, 0.7, 0.52, 0.88, 0.08, -0.42, -0.57,
    -0.84, 0.66, 0.49, -0.66, -0.27, -0.19, 0.05, -0.92, 0.31, -0.52,
    0.66, -0.99, 0.42, 0.91, 0.06, 0.55, 0.79, 0.2, -0.93, 0.13,
]
Z2 = [
    1.68, 0.53, 0.07, 0.42, 1.87, 0.08, 1.07, 0.33, 1.41, 1.73,
    0.6, 1.9, 0.58, 1.11, 0.81, 1.43, 0.08, 0.38, 1.36, 0.97,
    1.17, 1.32, 0.7, 1.26, 0.45, 1.35, 0.97, 1.77, 0.16, 0.63,
]
NB = [
    0, 2, 0, 2, 0, 1, 4, 5, 2, 7,
    2, 4, 2, 5, 0, 8, 8, 2, 7, 2,
    9, 0, 10, 6, 3, 3, 2, 7, 0, 10,
]

THETA = 1.8294684732585813


def _nb_df() -> pl.DataFrame:
    return pl.DataFrame({"nb": NB, "z1": Z1, "z2": Z2})


class TestNegBinEstimateDispersionPyModel:
    """NegativeBinomial(estimate_dispersion=...) PyModel reference, MASS 7.3.65
    (anofox-regression tests/issue49_negbin_se.rs)."""

    def _fit(self, *, estimate_dispersion: bool) -> NegativeBinomial:
        X = np.column_stack([Z1, Z2])
        y = np.array(NB, dtype=float)
        return NegativeBinomial(
            theta=THETA,
            estimate_theta=False,
            tol=1e-12,
            estimate_dispersion=estimate_dispersion,
        ).fit(X, y)

    def test_default_dispersion_is_one(self):
        model = self._fit(estimate_dispersion=False)
        assert model.dispersion == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(
            model.std_errors, [2.953927095121e-01, 3.114970571954e-01], rtol=1e-5
        )

    def test_estimate_dispersion_true_matches_r(self):
        model = self._fit(estimate_dispersion=True)
        assert model.dispersion == pytest.approx(0.8951608253323096, rel=1e-6)
        np.testing.assert_allclose(
            model.std_errors, [0.2794797246792255, 0.2947165213629392], rtol=1e-5
        )

    def test_estimate_dispersion_does_not_change_coefficients(self):
        default_model = self._fit(estimate_dispersion=False)
        dispersion_model = self._fit(estimate_dispersion=True)
        np.testing.assert_allclose(
            dispersion_model.coefficients, default_model.coefficients, atol=1e-10
        )
        assert dispersion_model.intercept == pytest.approx(default_model.intercept, abs=1e-10)


class TestNegBinEstimateDispersionSummaryExpression:
    def test_summary_std_errors_default(self):
        df = _nb_df()
        result = df.select(
            ps.negative_binomial_summary("nb", "z1", "z2", theta=THETA).alias("coef")
        )["coef"][0]
        std_errors = [row["std_error"] for row in result]
        np.testing.assert_allclose(
            std_errors,
            [3.403116503810e-01, 2.953927095121e-01, 3.114970571954e-01],
            rtol=1e-4,
        )

    def test_summary_std_errors_estimate_dispersion(self):
        df = _nb_df()
        result = df.select(
            ps.negative_binomial_summary(
                "nb", "z1", "z2", theta=THETA, estimate_dispersion=True
            ).alias("coef")
        )["coef"][0]
        std_errors = [row["std_error"] for row in result]
        np.testing.assert_allclose(
            std_errors,
            [0.3219788548969521, 0.2794797246792255, 0.2947165213629392],
            rtol=1e-4,
        )


class TestNegBinEstimateDispersionExpression:
    def test_coefficients_unaffected_by_estimate_dispersion(self):
        df = _nb_df()
        default_result = df.select(
            ps.negative_binomial("nb", "z1", "z2", theta=THETA).alias("m")
        )["m"][0]
        dispersion_result = df.select(
            ps.negative_binomial(
                "nb", "z1", "z2", theta=THETA, estimate_dispersion=True
            ).alias("m")
        )["m"][0]
        assert dispersion_result["intercept"] == pytest.approx(
            default_result["intercept"], abs=1e-10
        )
        np.testing.assert_allclose(
            dispersion_result["coefficients"], default_result["coefficients"], atol=1e-10
        )

    def test_predict_interval_width_changes_point_prediction_unchanged(self):
        df = _nb_df()
        default_pred = df.select(
            ps.negative_binomial_predict(
                "nb", "z1", "z2", interval="confidence"
            ).alias("pred")
        ).unnest("pred")
        dispersion_pred = df.select(
            ps.negative_binomial_predict(
                "nb",
                "z1",
                "z2",
                interval="confidence",
                estimate_dispersion=True,
            ).alias("pred")
        ).unnest("pred")

        np.testing.assert_allclose(
            dispersion_pred["negative_binomial_prediction"].to_numpy(),
            default_pred["negative_binomial_prediction"].to_numpy(),
            atol=1e-10,
        )
        lower_differs = not np.allclose(
            dispersion_pred["negative_binomial_lower"].to_numpy(),
            default_pred["negative_binomial_lower"].to_numpy(),
            atol=1e-10,
        )
        upper_differs = not np.allclose(
            dispersion_pred["negative_binomial_upper"].to_numpy(),
            default_pred["negative_binomial_upper"].to_numpy(),
            atol=1e-10,
        )
        assert lower_differs or upper_differs

    def test_formula_matches_direct_call(self):
        df = _nb_df()
        formula_result = df.select(
            ps.negative_binomial_formula(
                "nb ~ z1 + z2", theta=THETA, estimate_dispersion=True
            ).alias("m")
        )["m"][0]
        direct_result = df.select(
            ps.negative_binomial(
                "nb", "z1", "z2", theta=THETA, estimate_dispersion=True
            ).alias("m")
        )["m"][0]
        assert formula_result["intercept"] == pytest.approx(
            direct_result["intercept"], abs=1e-10
        )
        np.testing.assert_allclose(
            formula_result["coefficients"], direct_result["coefficients"], atol=1e-10
        )

# RLS: x = 0..39, y = 1 + 2x exactly (no noise) so coefficient shifts from a
# tighter initial_p_diagonal prior are unambiguous.
RLS_X = list(range(40))
RLS_Y = [1.0 + 2.0 * xi for xi in RLS_X]


def _rls_df() -> pl.DataFrame:
    return pl.DataFrame({"y": RLS_Y, "x": [float(v) for v in RLS_X]})


class TestRlsInitialPDiagonalPyModel:
    def _fit(self, *, initial_p_diagonal: float = 1e6) -> RLS:
        X = np.array(RLS_X, dtype=float).reshape(-1, 1)
        y = np.array(RLS_Y, dtype=float)
        return RLS(forgetting_factor=1.0, initial_p_diagonal=initial_p_diagonal).fit(X, y)

    def test_default_matches_explicit_1e6(self):
        default_model = self._fit()
        explicit_model = self._fit(initial_p_diagonal=1e6)
        assert default_model.initial_p_diagonal_value == 1e6
        assert explicit_model.initial_p_diagonal_value == 1e6
        np.testing.assert_allclose(
            explicit_model.coefficients, default_model.coefficients, atol=1e-10
        )
        assert explicit_model.intercept == pytest.approx(default_model.intercept, abs=1e-10)

    def test_initial_p_diagonal_value_returns_passed_value(self):
        model = self._fit(initial_p_diagonal=1e-3)
        assert model.initial_p_diagonal_value == 1e-3

    def test_tight_prior_shifts_coefficients(self):
        default_model = self._fit()
        tight_model = self._fit(initial_p_diagonal=1e-3)
        diff = abs(tight_model.coefficients[0] - default_model.coefficients[0])
        assert diff > 1e-3

    @pytest.mark.parametrize("bad_value", [-1.0, float("nan")])
    def test_invalid_initial_p_diagonal_fails_safely(self, bad_value):
        X = np.array(RLS_X, dtype=float).reshape(-1, 1)
        y = np.array(RLS_Y, dtype=float)
        with pytest.raises(ValueError):
            RLS(forgetting_factor=1.0, initial_p_diagonal=bad_value).fit(X, y)


class TestRlsInitialPDiagonalExpression:
    def test_default_matches_explicit_1e6(self):
        df = _rls_df()
        default_result = df.select(
            ps.rls("y", "x", forgetting_factor=1.0).alias("m")
        )["m"][0]
        explicit_result = df.select(
            ps.rls("y", "x", forgetting_factor=1.0, initial_p_diagonal=1e6).alias("m")
        )["m"][0]
        assert explicit_result["intercept"] == pytest.approx(
            default_result["intercept"], abs=1e-10
        )
        np.testing.assert_allclose(
            explicit_result["coefficients"], default_result["coefficients"], atol=1e-10
        )

    def test_tight_prior_differs_from_default(self):
        df = _rls_df()
        default_result = df.select(
            ps.rls("y", "x", forgetting_factor=1.0).alias("m")
        )["m"][0]
        tight_result = df.select(
            ps.rls("y", "x", forgetting_factor=1.0, initial_p_diagonal=1e-3).alias("m")
        )["m"][0]
        diff = abs(tight_result["coefficients"][0] - default_result["coefficients"][0])
        assert diff > 1e-3

    def test_invalid_initial_p_diagonal_yields_nan_struct(self):
        df = _rls_df()
        result = df.select(
            ps.rls("y", "x", forgetting_factor=1.0, initial_p_diagonal=-1.0).alias("m")
        )["m"][0]
        assert np.isnan(result["intercept"])

    def test_summary_std_errors_differ_with_tight_prior(self):
        df = _rls_df()
        default_result = df.select(
            ps.rls_summary("y", "x", forgetting_factor=1.0).alias("coef")
        )["coef"][0]
        tight_result = df.select(
            ps.rls_summary(
                "y", "x", forgetting_factor=1.0, initial_p_diagonal=1e-3
            ).alias("coef")
        )["coef"][0]
        default_estimates = [row["estimate"] for row in default_result]
        tight_estimates = [row["estimate"] for row in tight_result]
        assert any(
            abs(a - b) > 1e-6 for a, b in zip(default_estimates, tight_estimates)
        )

    def test_predict_differs_with_tight_prior(self):
        df = _rls_df()
        default_pred = df.select(
            ps.rls_predict("y", "x", forgetting_factor=1.0).alias("pred")
        ).unnest("pred")
        tight_pred = df.select(
            ps.rls_predict(
                "y", "x", forgetting_factor=1.0, initial_p_diagonal=1e-3
            ).alias("pred")
        ).unnest("pred")
        default_vals = default_pred["rls_prediction"].to_numpy()
        tight_vals = tight_pred["rls_prediction"].to_numpy()
        assert not np.allclose(default_vals, tight_vals, atol=1e-6)

    def test_formula_matches_direct_call(self):
        df = _rls_df()
        formula_result = df.select(
            ps.rls_formula("y ~ x", initial_p_diagonal=1e-3).alias("m")
        )["m"][0]
        direct_result = df.select(
            ps.rls("y", "x", initial_p_diagonal=1e-3).alias("m")
        )["m"][0]
        assert formula_result["intercept"] == pytest.approx(
            direct_result["intercept"], abs=1e-10
        )
        np.testing.assert_allclose(
            formula_result["coefficients"], direct_result["coefficients"], atol=1e-10
        )
