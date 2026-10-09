"""Tests for the family-generic penalized GLM engine (`PenalizedGLM` /
`ps.penalized_glm` / `ps.penalized_glm_formula`), phase 16.2 plans 02-03.

R references: `glm`, `MASS::glm.nb`, `MASS::negative.binomial`, a Newton
posterior-mode fit written in R for the priors, all shipped as
`anofox-regression` 0.5.22's `tests/fixtures/penalized_glm_aft_reference.rs`
and transcribed verbatim into `tests/fixtures/penalized_glm_aft_reference.py`
(see that module's docstring for provenance and `SOURCE_SHA256`).

See `.planning/phases/16.2-penalized-glm-and-aft-survival-regression/16.2-02-PLAN.md`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np
import polars as pl
import polars_statistics as ps
import pytest

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "penalized_glm_aft_reference.py"
_spec = importlib.util.spec_from_file_location("penalized_glm_aft_reference", _FIXTURE_PATH)
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)


def rel_close(a: float, b: float, tol: float) -> bool:
    """Relative-error closeness check, matching upstream's Rust test helper:
    ``|a - b| / max(|b|, 1) < tol``."""
    err = abs(a - b) / max(abs(b), 1.0)
    return err < tol


def _fixture_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x1": ref.X1,
            "x2": ref.X2,
            "off": ref.OFFSET,
            "y_pois": ref.Y_POIS,
            "y_bin": ref.Y_BIN,
            "y_nb": ref.Y_NB,
            "y_gamma": ref.Y_GAMMA,
            "time": ref.TIME,
            "event": [e == 1.0 for e in ref.EVENT],
        }
    )


DF = _fixture_df()


# ============================================================================
# Fixture integrity
# ============================================================================


class TestFixtureIntegrity:
    def test_fixture_column_lengths_are_60(self):
        for name in ("X1", "X2", "OFFSET", "Y_POIS", "Y_BIN", "Y_NB", "Y_GAMMA", "TIME", "EVENT"):
            assert len(getattr(ref, name)) == 60, name

    def test_fixture_spot_constants(self):
        assert ref.POIS_COEF[0] == 0.5900286106754643
        assert ref.QPOIS_OVER_DISP == 1.5336159230646829

    def test_fixture_source_metadata_present(self):
        assert ref.SOURCE_CRATE == "anofox-regression 0.5.22"
        assert len(ref.SOURCE_SHA256) == 64

    def test_fixture_matches_upstream_crate_file_when_present(self):
        """If the upstream crate source is on disk (dev env with a populated
        cargo registry cache), its sha256 must match SOURCE_SHA256 and every
        constant re-parsed from it must equal the module's value."""
        cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
        candidates = sorted(
            cargo_home.glob(
                "registry/src/*/anofox-regression-0.5.22/tests/fixtures/"
                "penalized_glm_aft_reference.rs"
            )
        )
        if not candidates:
            pytest.skip("upstream anofox-regression 0.5.22 source not found under $CARGO_HOME")
        upstream = candidates[0]
        content = upstream.read_text()
        sha = hashlib.sha256(content.encode()).hexdigest()
        assert sha == ref.SOURCE_SHA256
        # Spot-check a handful of constants are present verbatim in the source.
        assert "0.5900286106754643" in content
        assert "1.5336159230646829" in content


# ============================================================================
# Expression: Poisson (log link) matches R glm
# ============================================================================


class TestPenalizedGlmExprPoisson:
    def test_expr_poisson_matches_r(self):
        row = DF.select(
            ps.penalized_glm("y_pois", "x1", "x2", family="poisson", tol=1e-10).alias("r")
        )["r"][0]
        assert rel_close(row["intercept"], ref.POIS_COEF[0], 1e-6)
        for j, c in enumerate(row["coefficients"]):
            assert rel_close(c, ref.POIS_COEF[j + 1], 1e-6)
        assert rel_close(row["deviance"], ref.POIS_DEV[0], 1e-7)
        assert rel_close(row["null_deviance"], ref.POIS_DEV[1], 1e-7)
        assert rel_close(row["log_likelihood"], ref.POIS_LL_AIC[0], 1e-7)
        assert rel_close(row["aic"], ref.POIS_LL_AIC[1], 1e-7)
        assert row["dispersion"] == 1.0
        assert row["converged"] is True
        assert row["n_observations"] == 60
        assert np.isnan(row["theta"])
        assert np.isnan(row["accuracy"])
        assert row["has_separation"] is False

    def test_expr_poisson_offset_matches_r(self):
        row = DF.select(
            ps.penalized_glm(
                "y_pois", "x1", "x2", family="poisson", offset="off", tol=1e-10
            ).alias("r")
        )["r"][0]
        assert rel_close(row["intercept"], ref.POIS_OFF_COEF[0], 1e-6)
        for j, c in enumerate(row["coefficients"]):
            assert rel_close(c, ref.POIS_OFF_COEF[j + 1], 1e-6)
        assert rel_close(row["null_deviance"], ref.POIS_OFF_DEV[1], 1e-7)
        assert rel_close(row["null_deviance"], 102.28707998763498, 1e-7)
        assert rel_close(row["log_likelihood"], ref.POIS_OFF_LL_AIC[0], 1e-7)
        assert rel_close(row["aic"], ref.POIS_OFF_LL_AIC[1], 1e-7)

    def test_offset_expr_equals_offset_string(self):
        a = DF.select(
            ps.penalized_glm("y_pois", "x1", "x2", offset="off", tol=1e-10).alias("r")
        )["r"][0]
        b = DF.select(
            ps.penalized_glm("y_pois", "x1", "x2", offset=pl.col("off"), tol=1e-10).alias("r")
        )["r"][0]
        assert a["intercept"] == b["intercept"]
        assert a["coefficients"] == b["coefficients"]
        assert a["deviance"] == b["deviance"]

    def test_omitting_offset_equals_lit_zero(self):
        a = DF.select(ps.penalized_glm("y_pois", "x1", "x2", tol=1e-10).alias("r"))["r"][0]
        b = DF.select(
            ps.penalized_glm(
                "y_pois", "x1", "x2", offset=pl.lit(0.0, dtype=pl.Float64), tol=1e-10
            ).alias("r")
        )["r"][0]
        assert a["intercept"] == b["intercept"]
        assert a["coefficients"] == b["coefficients"]


# ============================================================================
# Output schema
# ============================================================================


class TestPenalizedGlmSchema:
    def test_struct_field_order_and_types(self):
        expr = ps.penalized_glm("y_pois", "x1", "x2").alias("r")
        schema = DF.select(expr).schema
        fields = schema["r"].fields
        names = [f.name for f in fields]
        assert names == [
            "intercept",
            "coefficients",
            "deviance",
            "null_deviance",
            "pseudo_r_squared",
            "log_likelihood",
            "aic",
            "bic",
            "dispersion",
            "theta",
            "accuracy",
            "has_separation",
            "iterations",
            "converged",
            "n_observations",
        ]
        by_name = {f.name: f.dtype for f in fields}
        assert by_name["iterations"] == pl.UInt32
        assert by_name["n_observations"] == pl.UInt32
        assert by_name["has_separation"] == pl.Boolean
        assert by_name["converged"] == pl.Boolean
        assert by_name["coefficients"] == pl.List(pl.Float64)


# ============================================================================
# Build-time validation errors
# ============================================================================


class TestPenalizedGlmInvalidConfig:
    def test_invalid_family_raises_before_collect(self):
        with pytest.raises(ValueError):
            ps.penalized_glm("y", "x", family="poison")

    def test_estimate_dispersion_non_poisson_raises_before_collect(self):
        with pytest.raises(ValueError):
            ps.penalized_glm("y", "x", family="gamma", estimate_dispersion=True)


# ============================================================================
# PenalizedGLM PyModel — R reference, inference, vcov scaling, errors
# ============================================================================

X = np.column_stack([ref.X1, ref.X2])


class TestPenalizedGLMModel:
    def test_model_poisson_matches_glm(self):
        model = ps.PenalizedGLM(family="poisson", tol=1e-10).fit(X, np.array(ref.Y_POIS))
        assert rel_close(model.intercept, ref.POIS_COEF[0], 1e-6)
        for j, c in enumerate(model.coefficients):
            assert rel_close(c, ref.POIS_COEF[j + 1], 1e-6)
        assert rel_close(model.deviance, ref.POIS_DEV[0], 1e-7)
        assert rel_close(model.null_deviance, ref.POIS_DEV[1], 1e-7)
        assert rel_close(model.log_likelihood, ref.POIS_LL_AIC[0], 1e-7)
        assert rel_close(model.aic, ref.POIS_LL_AIC[1], 1e-7)

        for j, se in enumerate(model.std_errors):
            assert rel_close(se, ref.POIS_SE[j + 1], 1e-5)
        assert rel_close(model.intercept_std_error, ref.POIS_SE[0], 1e-5)
        assert rel_close(
            model.intercept_z_value, model.intercept / model.intercept_std_error, 1e-12
        )

        b1 = model.coefficients[0]
        se1 = model.std_errors[0]
        ci1_upper = model.conf_interval_upper[0]
        zq = (ci1_upper - b1) / se1
        b0 = model.intercept
        se0 = model.intercept_std_error
        lo, hi = model.intercept_conf_interval
        assert rel_close(lo, b0 - zq * se0, 1e-10)
        assert rel_close(hi, b0 + zq * se0, 1e-10)
        assert rel_close(zq, 1.959963984540054, 1e-9)

        model_90 = ps.PenalizedGLM(family="poisson", tol=1e-10, confidence_level=0.90).fit(
            X, np.array(ref.Y_POIS)
        )
        b1_90 = model_90.coefficients[0]
        se1_90 = model_90.std_errors[0]
        zq_90 = (model_90.conf_interval_upper[0] - b1_90) / se1_90
        assert rel_close(zq_90, 1.6448536269514722, 1e-9)

        assert 0.0 <= model.intercept_p_value <= 1.0
        assert model.family == "poisson"
        assert model.link == "log"
        assert model.dispersion == 1.0
        assert model.theta is None
        assert model.accuracy is None
        assert model.converged is True
        assert model.n_observations == 60
        assert model.iterations >= 1

    def test_model_overdispersed_poisson_dispersion_one(self):
        model = ps.PenalizedGLM(family="poisson", tol=1e-10).fit(X, np.array(ref.Y_NB))
        for j, c in enumerate(model.coefficients):
            assert rel_close(c, ref.POIS_OVER_COEF[j + 1], 1e-6)
        assert rel_close(model.intercept, ref.POIS_OVER_COEF[0], 1e-6)
        for j, se in enumerate(model.std_errors):
            assert rel_close(se, ref.POIS_OVER_SE[j + 1], 1e-5)
        assert rel_close(model.intercept_std_error, ref.POIS_OVER_SE[0], 1e-5)
        assert rel_close(model.deviance, ref.POIS_OVER_DEV[0], 1e-7)
        assert rel_close(model.null_deviance, ref.POIS_OVER_DEV[1], 1e-7)
        assert rel_close(model.log_likelihood, ref.POIS_OVER_LL_AIC[0], 1e-7)
        assert rel_close(model.aic, ref.POIS_OVER_LL_AIC[1], 1e-7)
        assert model.dispersion == 1.0

    def test_model_poisson_estimate_dispersion_matches_quasipoisson(self):
        base = ps.PenalizedGLM(family="poisson", tol=1e-10).fit(X, np.array(ref.Y_NB))
        model = ps.PenalizedGLM(
            family="poisson", tol=1e-10, estimate_dispersion=True
        ).fit(X, np.array(ref.Y_NB))
        assert rel_close(model.dispersion, ref.QPOIS_OVER_DISP, 1e-7)
        for j, se in enumerate(model.std_errors):
            assert rel_close(se, ref.QPOIS_OVER_SE[j + 1], 1e-5)
        assert rel_close(model.intercept_std_error, ref.QPOIS_OVER_SE[0], 1e-5)
        assert rel_close(model.intercept, base.intercept, 1e-12)
        for a, b in zip(model.coefficients, base.coefficients):
            assert rel_close(a, b, 1e-12)

    def test_model_offset_matches_glm_and_predict_consistent(self):
        y = np.array(ref.Y_POIS)
        off = np.array(ref.OFFSET)
        model = ps.PenalizedGLM(family="poisson", tol=1e-10).fit(X, y, offset=off)
        assert rel_close(model.intercept, ref.POIS_OFF_COEF[0], 1e-6)
        for j, c in enumerate(model.coefficients):
            assert rel_close(c, ref.POIS_OFF_COEF[j + 1], 1e-6)

        preds = model.predict(X, offset=off)
        for a, b in zip(preds, model.fitted_values):
            assert rel_close(a, b, 1e-12)

        model_no_off = ps.PenalizedGLM(family="poisson", tol=1e-10).fit(X, y)
        preds_no_off = model_no_off.predict(X)
        for a, b in zip(preds_no_off, model_no_off.fitted_values):
            assert rel_close(a, b, 1e-12)
        lin = model_no_off.predict_linear(X)
        for p, lg in zip(preds_no_off, lin):
            assert rel_close(np.log(p), lg, 1e-12)

    def test_model_vcov_scaling_gamma(self):
        model = ps.PenalizedGLM(family="gamma", tol=1e-10).fit(X, np.array(ref.Y_GAMMA))
        vcov = model.vcov
        assert vcov.shape == (3, 3)
        assert np.allclose(vcov, vcov.T)
        assert model.matrix_parameters == [None, 0, 1]
        se_from_vcov = np.sqrt(np.diag(vcov))
        for a, b in zip(se_from_vcov[1:], model.std_errors):
            assert rel_close(a, b, 1e-12)
        assert rel_close(np.sqrt(vcov[0, 0]), model.intercept_std_error, 1e-12)
        for j, se in enumerate(model.std_errors):
            assert rel_close(se, ref.GAMMA_SE[j + 1], 1e-5)
        assert model.information.shape == (3, 3)

    def test_model_vcov_scaling_negative_binomial(self):
        model = ps.PenalizedGLM(family="negative_binomial", tol=1e-10).fit(X, np.array(ref.Y_NB))
        vcov = model.vcov
        assert vcov.shape == (3, 3)
        se_from_vcov = np.sqrt(np.diag(vcov))
        for a, b in zip(se_from_vcov[1:], model.std_errors):
            assert rel_close(a, b, 1e-12)
        assert rel_close(np.sqrt(vcov[0, 0]), model.intercept_std_error, 1e-12)

    def test_model_vcov_scaling_quasipoisson(self):
        model = ps.PenalizedGLM(
            family="poisson", tol=1e-10, estimate_dispersion=True
        ).fit(X, np.array(ref.Y_NB))
        vcov = model.vcov
        se_from_vcov = np.sqrt(np.diag(vcov))
        for a, b in zip(se_from_vcov[1:], model.std_errors):
            assert rel_close(a, b, 1e-12)
        assert rel_close(np.sqrt(vcov[0, 0]), model.intercept_std_error, 1e-12)

    def test_model_binomial_information_vcov_are_inverses(self):
        model = ps.PenalizedGLM(family="binomial", tol=1e-10).fit(X, np.array(ref.Y_BIN))
        product = model.information @ model.vcov
        assert np.allclose(product, np.eye(3), atol=1e-8)

    def test_model_no_intercept_matches_glm(self):
        model = ps.PenalizedGLM(family="poisson", tol=1e-10, add_intercept=False).fit(
            X, np.array(ref.Y_POIS)
        )
        for j, c in enumerate(model.coefficients):
            assert rel_close(c, ref.POIS_NOINT_COEF[j], 1e-6)
        for j, se in enumerate(model.std_errors):
            assert rel_close(se, ref.POIS_NOINT_SE[j], 1e-5)
        assert rel_close(model.deviance, ref.POIS_NOINT_DEV[0], 1e-7)
        assert rel_close(model.null_deviance, ref.POIS_NOINT_DEV[1], 1e-7)
        assert rel_close(model.log_likelihood, ref.POIS_NOINT_LL_AIC[0], 1e-7)
        assert rel_close(model.aic, ref.POIS_NOINT_LL_AIC[1], 1e-7)
        assert model.intercept is None
        assert model.intercept_std_error is None
        assert model.intercept_z_value is None
        assert model.intercept_p_value is None
        assert model.intercept_conf_interval is None
        assert model.matrix_parameters == [0, 1]

    def test_model_no_intercept_offset_matches_glm(self):
        model = ps.PenalizedGLM(family="poisson", tol=1e-10, add_intercept=False).fit(
            X, np.array(ref.Y_POIS), offset=np.array(ref.OFFSET)
        )
        for j, c in enumerate(model.coefficients):
            assert rel_close(c, ref.POIS_NOINT_OFF_COEF[j], 1e-6)
        assert rel_close(model.deviance, ref.POIS_NOINT_OFF_DEV[0], 1e-7)
        assert rel_close(model.null_deviance, ref.POIS_NOINT_OFF_DEV[1], 1e-7)

    def test_model_separation_getters(self):
        model = ps.PenalizedGLM(family="binomial", tol=1e-10).fit(X, np.array(ref.Y_BIN))
        assert model.has_separation is False
        assert model.separated_predictors == []
        assert model.separation_warning is None

    def test_model_valid_rows_and_dropped_nan_row(self):
        model = ps.PenalizedGLM(family="poisson", tol=1e-10).fit(X, np.array(ref.Y_POIS))
        assert model.valid_rows == list(range(60))

        y_with_nan = np.array(ref.Y_POIS, dtype=float)
        y_with_nan[5] = np.nan
        model_nan = ps.PenalizedGLM(family="poisson", tol=1e-10).fit(X, y_with_nan)
        assert model_nan.n_observations == 59
        assert 5 not in model_nan.valid_rows
        assert len(model_nan.fitted_values) == 59

    def test_model_compute_inference_false(self):
        model = ps.PenalizedGLM(family="poisson", tol=1e-10, compute_inference=False).fit(
            X, np.array(ref.Y_POIS)
        )
        assert model.std_errors is None
        assert model.z_values is None
        assert model.p_values is None
        assert model.conf_interval_lower is None
        assert model.conf_interval_upper is None
        assert model.vcov is None
        assert model.information is None
        assert model.matrix_parameters is None
        assert model.intercept_std_error is None
        assert model.intercept_z_value is None
        assert model.intercept_p_value is None
        assert model.intercept_conf_interval is None

    def test_model_errors_invalid_config(self):
        with pytest.raises(ValueError):
            ps.PenalizedGLM(family="poison")
        with pytest.raises(ValueError):
            ps.PenalizedGLM(family="poisson", link="probit")
        with pytest.raises(ValueError):
            ps.PenalizedGLM(family="tweedie", power=2.5)
        with pytest.raises(ValueError):
            ps.PenalizedGLM(family="gamma", theta=2.0)
        with pytest.raises(ValueError):
            ps.PenalizedGLM(family="gamma", estimate_dispersion=True)
        with pytest.raises(ValueError):
            ps.PenalizedGLM(priors=[("cauchy", 0.0, 1.0)])
        with pytest.raises(ValueError):
            ps.PenalizedGLM(priors=[("normal", 0.0, -1.0)])
        with pytest.raises(ValueError):
            ps.PenalizedGLM(vcov_type="robust")

    def test_model_errors_fit_time(self):
        with pytest.raises(ValueError):
            ps.PenalizedGLM(family="poisson", tol=1e-10).fit(X, -np.array(ref.Y_POIS))
        with pytest.raises(ValueError):
            ps.PenalizedGLM(
                family="poisson", tol=1e-10, priors=[("flat",)] * 5
            ).fit(X, np.array(ref.Y_POIS))
        with pytest.raises(ValueError):
            ps.PenalizedGLM(family="poisson", tol=1e-10).fit(
                X, np.array(ref.Y_POIS), offset=np.array(ref.OFFSET[:-1])
            )

    def test_model_unfitted_raises_runtime_error(self):
        model = ps.PenalizedGLM()
        for attr in (
            "coefficients",
            "intercept",
            "family",
            "link",
            "deviance",
            "null_deviance",
            "pseudo_r_squared",
            "log_likelihood",
            "aic",
            "bic",
            "dispersion",
            "theta",
            "iterations",
            "converged",
            "n_observations",
            "accuracy",
            "std_errors",
            "z_values",
            "p_values",
            "conf_interval_lower",
            "conf_interval_upper",
            "intercept_std_error",
            "intercept_z_value",
            "intercept_p_value",
            "intercept_conf_interval",
            "vcov",
            "information",
            "matrix_parameters",
            "has_separation",
            "separated_predictors",
            "separation_warning",
            "fitted_values",
            "valid_rows",
        ):
            with pytest.raises(RuntimeError, match="PenalizedGLM"):
                getattr(model, attr)
        with pytest.raises(RuntimeError, match="PenalizedGLM"):
            model.predict(X)
        with pytest.raises(RuntimeError, match="PenalizedGLM"):
            model.predict_linear(X)

    def test_model_ergonomics(self):
        model = ps.PenalizedGLM(family="poisson", tol=1e-10)
        assert model.to_dict() == {"fitted": False}
        model.fit(X, np.array(ref.Y_POIS))
        assert repr(model).startswith("PenalizedGLM(")
        d = model.to_dict()
        assert "coefficients" in d
        assert "vcov" in d
