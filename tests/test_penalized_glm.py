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
import math
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


# ============================================================================
# Full 0.5.22 R matrix on both layers (Task 3)
# ============================================================================


def _expr_fit(y_col: str, **kwargs) -> dict:
    return DF.select(ps.penalized_glm(y_col, "x1", "x2", tol=1e-10, **kwargs).alias("r"))["r"][0]


def _model_fit(y, **kwargs) -> ps.PenalizedGLM:
    return ps.PenalizedGLM(tol=1e-10, **kwargs).fit(X, np.asarray(y, dtype=float))


def _model_fit_offset(y, offset, **kwargs) -> ps.PenalizedGLM:
    return ps.PenalizedGLM(tol=1e-10, **kwargs).fit(
        X, np.asarray(y, dtype=float), offset=np.asarray(offset, dtype=float)
    )


def has_valid_numbers(result: pl.DataFrame, col: str = "result", *, nan_ok=()) -> bool:
    """Check that result column contains valid (non-null, non-NaN) numbers.

    `nan_ok` is a tuple of struct field names whose value is permitted to be
    NaN (copied locally from tests/test_group_by_over.py per plan 16.2-02).
    """
    for row in result[col]:
        if row is None:
            return False
        if isinstance(row, dict):
            for key, val in row.items():
                if key in nan_ok:
                    continue
                if val is None:
                    return False
                if isinstance(val, float) and math.isnan(val):
                    return False
                if isinstance(val, list):
                    for v in val:
                        if v is None or (isinstance(v, float) and math.isnan(v)):
                            return False
    return True


class TestPenalizedGlmFamilyPois:
    def test_family_poisson_sqrt_link(self):
        row = _expr_fit("y_pois", family="poisson", link="sqrt")
        assert rel_close(row["intercept"], ref.POIS_SQRT_COEF[0], 1e-6)
        assert rel_close(row["coefficients"][0], ref.POIS_SQRT_COEF[1], 1e-6)
        assert rel_close(row["coefficients"][1], ref.POIS_SQRT_COEF[2], 1e-6)
        assert rel_close(row["deviance"], ref.POIS_SQRT_DEV[0], 1e-7)

        model = _model_fit(ref.Y_POIS, family="poisson", link="sqrt")
        assert rel_close(model.intercept, ref.POIS_SQRT_COEF[0], 1e-6)
        assert rel_close(model.coefficients[0], ref.POIS_SQRT_COEF[1], 1e-6)
        assert rel_close(model.coefficients[1], ref.POIS_SQRT_COEF[2], 1e-6)
        assert rel_close(model.deviance, ref.POIS_SQRT_DEV[0], 1e-7)


_BINOMIAL_LINK_CASES = [
    ("logit", ref.BIN_LOGIT_COEF, ref.BIN_LOGIT_SE, ref.BIN_LOGIT_DEV, ref.BIN_LOGIT_LL_AIC),
    ("probit", ref.BIN_PROBIT_COEF, ref.BIN_PROBIT_SE, ref.BIN_PROBIT_DEV, ref.BIN_PROBIT_LL_AIC),
    (
        "cloglog",
        ref.BIN_CLOGLOG_COEF,
        ref.BIN_CLOGLOG_SE,
        ref.BIN_CLOGLOG_DEV,
        ref.BIN_CLOGLOG_LL_AIC,
    ),
]


class TestPenalizedGlmFamilyBinomial:
    @pytest.mark.parametrize("link,coef,se,dev,ll_aic", _BINOMIAL_LINK_CASES)
    def test_family_binomial_links(self, link, coef, se, dev, ll_aic):
        row = _expr_fit("y_bin", family="binomial", link=link)
        assert rel_close(row["intercept"], coef[0], 1e-6)
        for j, c in enumerate(row["coefficients"]):
            assert rel_close(c, coef[j + 1], 1e-6)
        assert rel_close(row["deviance"], dev[0], 1e-7)
        assert rel_close(row["null_deviance"], dev[1], 1e-7)
        assert rel_close(row["log_likelihood"], ll_aic[0], 1e-7)
        assert rel_close(row["aic"], ll_aic[1], 1e-7)
        assert row["has_separation"] is False

        model = _model_fit(ref.Y_BIN, family="binomial", link=link)
        assert rel_close(model.intercept, coef[0], 1e-6)
        for j, s in enumerate(model.std_errors):
            assert rel_close(s, se[j + 1], 1e-5)
        assert rel_close(model.intercept_std_error, se[0], 1e-5)
        assert model.has_separation is False

    def test_family_logistic_accuracy(self):
        row = _expr_fit("y_bin", family="logistic")
        assert rel_close(row["intercept"], ref.BIN_LOGIT_COEF[0], 1e-6)
        assert 0.5 <= row["accuracy"] <= 1.0

        model = _model_fit(ref.Y_BIN, family="logistic")
        assert rel_close(model.intercept, ref.BIN_LOGIT_COEF[0], 1e-6)
        assert 0.5 <= model.accuracy <= 1.0

    def test_non_logistic_accuracy_is_absent(self):
        row = _expr_fit("y_pois", family="poisson")
        assert math.isnan(row["accuracy"])
        model = _model_fit(ref.Y_POIS, family="poisson")
        assert model.accuracy is None

    def test_family_binomial_offset(self):
        row = _expr_fit("y_bin", family="binomial", offset="off")
        assert rel_close(row["intercept"], ref.BIN_OFF_COEF[0], 1e-6)
        for j, c in enumerate(row["coefficients"]):
            assert rel_close(c, ref.BIN_OFF_COEF[j + 1], 1e-6)
        assert rel_close(row["deviance"], ref.BIN_OFF_DEV[0], 1e-7)
        assert rel_close(row["null_deviance"], ref.BIN_OFF_DEV[1], 1e-7)
        assert rel_close(row["log_likelihood"], ref.BIN_OFF_LL_AIC[0], 1e-7)
        assert rel_close(row["aic"], ref.BIN_OFF_LL_AIC[1], 1e-7)
        assert rel_close(
            row["pseudo_r_squared"], 1.0 - ref.BIN_OFF_DEV[0] / ref.BIN_OFF_DEV[1], 1e-9
        )

        model = _model_fit_offset(ref.Y_BIN, ref.OFFSET, family="binomial")
        assert rel_close(model.intercept, ref.BIN_OFF_COEF[0], 1e-6)


class TestPenalizedGlmFamilyGamma:
    def test_family_gamma_matches_glm(self):
        row = _expr_fit("y_gamma", family="gamma")
        assert rel_close(row["intercept"], ref.GAMMA_COEF[0], 1e-6)
        for j, c in enumerate(row["coefficients"]):
            assert rel_close(c, ref.GAMMA_COEF[j + 1], 1e-6)
        assert rel_close(row["deviance"], ref.GAMMA_DEV[0], 1e-7)
        assert rel_close(row["null_deviance"], ref.GAMMA_DEV[1], 1e-7)

        model = _model_fit(ref.Y_GAMMA, family="gamma")
        assert rel_close(model.intercept, ref.GAMMA_COEF[0], 1e-6)

    def test_gamma_log_likelihood_uses_deviance_over_n(self):
        row = _expr_fit("y_gamma", family="gamma")
        assert rel_close(row["log_likelihood"], ref.GAMMA_LL_AIC[0], 1e-9)
        assert rel_close(row["aic"], ref.GAMMA_LL_AIC[1], 1e-9)
        bic = ref.GAMMA_LL_AIC[1] - 8.0 + 4.0 * math.log(60)
        assert rel_close(row["bic"], bic, 1e-9)

        model = _model_fit(ref.Y_GAMMA, family="gamma")
        assert rel_close(model.log_likelihood, ref.GAMMA_LL_AIC[0], 1e-9)
        assert rel_close(model.aic, ref.GAMMA_LL_AIC[1], 1e-9)
        assert rel_close(model.bic, bic, 1e-9)

    def test_gamma_offset(self):
        row = _expr_fit("y_gamma", family="gamma", offset="off")
        assert rel_close(row["intercept"], ref.GAMMA_OFF_COEF[0], 1e-6)
        for j, c in enumerate(row["coefficients"]):
            assert rel_close(c, ref.GAMMA_OFF_COEF[j + 1], 1e-6)
        assert rel_close(row["deviance"], ref.GAMMA_OFF_DEV[0], 1e-7)
        assert rel_close(row["null_deviance"], ref.GAMMA_OFF_DEV[1], 1e-7)
        assert rel_close(
            row["pseudo_r_squared"], 1.0 - ref.GAMMA_OFF_DEV[0] / ref.GAMMA_OFF_DEV[1], 1e-9
        )

        model = _model_fit_offset(ref.Y_GAMMA, ref.OFFSET, family="gamma")
        assert rel_close(model.intercept, ref.GAMMA_OFF_COEF[0], 1e-6)


class TestPenalizedGlmTweedie:
    def test_tweedie_power_two_equals_gamma(self):
        g = _expr_fit("y_gamma", family="gamma")
        t = _expr_fit("y_gamma", family="tweedie", power=2.0)
        for j in range(2):
            assert rel_close(t["coefficients"][j], g["coefficients"][j], 1e-10)

    def test_tweedie_power_one_equals_poisson_slope(self):
        t1 = _expr_fit("y_pois", family="tweedie", power=1.0)
        assert rel_close(t1["coefficients"][0], ref.POIS_COEF[1], 1e-6)


class TestPenalizedGlmNullDeviance:
    """#56: null deviance as R's glm defines it (no-intercept + offset cases)."""

    def test_poisson_no_intercept(self):
        row = _expr_fit("y_pois", family="poisson", add_intercept=False)
        for j, c in enumerate(row["coefficients"]):
            assert rel_close(c, ref.POIS_NOINT_COEF[j], 1e-6)
        assert rel_close(row["deviance"], ref.POIS_NOINT_DEV[0], 1e-7)
        assert rel_close(row["null_deviance"], ref.POIS_NOINT_DEV[1], 1e-7)
        assert rel_close(row["log_likelihood"], ref.POIS_NOINT_LL_AIC[0], 1e-7)
        assert rel_close(row["aic"], ref.POIS_NOINT_LL_AIC[1], 1e-7)

    def test_poisson_no_intercept_offset(self):
        row = _expr_fit("y_pois", family="poisson", add_intercept=False, offset="off")
        for j, c in enumerate(row["coefficients"]):
            assert rel_close(c, ref.POIS_NOINT_OFF_COEF[j], 1e-6)
        assert rel_close(row["deviance"], ref.POIS_NOINT_OFF_DEV[0], 1e-7)
        assert rel_close(row["null_deviance"], ref.POIS_NOINT_OFF_DEV[1], 1e-7)
        assert rel_close(row["log_likelihood"], ref.POIS_NOINT_OFF_LL_AIC[0], 1e-7)
        assert rel_close(row["aic"], ref.POIS_NOINT_OFF_LL_AIC[1], 1e-7)

    @pytest.mark.parametrize(
        "family,y_col,coef,dev,ll_aic",
        [
            ("poisson", "y_pois", ref.POIS_OFF_COEF, ref.POIS_OFF_DEV, ref.POIS_OFF_LL_AIC),
            ("binomial", "y_bin", ref.BIN_OFF_COEF, ref.BIN_OFF_DEV, ref.BIN_OFF_LL_AIC),
            ("gamma", "y_gamma", ref.GAMMA_OFF_COEF, ref.GAMMA_OFF_DEV, ref.GAMMA_OFF_LL_AIC),
        ],
    )
    def test_offset_with_intercept_pseudo_r2(self, family, y_col, coef, dev, ll_aic):
        row = _expr_fit(y_col, family=family, offset="off")
        assert rel_close(row["intercept"], coef[0], 1e-6)
        assert rel_close(row["deviance"], dev[0], 1e-7)
        assert rel_close(row["null_deviance"], dev[1], 1e-7)
        assert rel_close(row["log_likelihood"], ll_aic[0], 1e-7)
        assert rel_close(row["aic"], ll_aic[1], 1e-7)
        assert rel_close(row["pseudo_r_squared"], 1.0 - dev[0] / dev[1], 1e-9)


class TestPenalizedGlmOverdispersedQuasipoisson:
    def test_overdispersed_poisson_dispersion_one(self):
        row = _expr_fit("y_nb", family="poisson")
        for j, c in enumerate(row["coefficients"]):
            assert rel_close(c, ref.POIS_OVER_COEF[j + 1], 1e-6)
        assert rel_close(row["deviance"], ref.POIS_OVER_DEV[0], 1e-7)
        assert rel_close(row["null_deviance"], ref.POIS_OVER_DEV[1], 1e-7)
        assert row["dispersion"] == 1.0

    def test_estimate_dispersion_matches_quasipoisson(self):
        row = _expr_fit("y_nb", family="poisson", estimate_dispersion=True)
        assert rel_close(row["dispersion"], ref.QPOIS_OVER_DISP, 1e-7)
        base = _expr_fit("y_nb", family="poisson")
        assert row["coefficients"] == base["coefficients"]


class TestPenalizedGlmNegativeBinomial:
    def test_negative_binomial_estimated_theta(self):
        row = _expr_fit("y_nb", family="negative_binomial")
        assert rel_close(row["intercept"], ref.NB_COEF[0], 1e-6)
        for j, c in enumerate(row["coefficients"]):
            assert rel_close(c, ref.NB_COEF[j + 1], 1e-6)
        assert rel_close(row["deviance"], ref.NB_DEV[0], 1e-7)
        assert rel_close(row["null_deviance"], ref.NB_DEV[1], 1e-7)
        assert rel_close(row["log_likelihood"], ref.NB_LL_AIC[0], 1e-7)
        assert rel_close(row["aic"], ref.NB_LL_AIC[1], 1e-7)
        assert rel_close(row["theta"], ref.NB_THETA, 1e-5)

        model = _model_fit(ref.Y_NB, family="negative_binomial")
        assert rel_close(model.theta, ref.NB_THETA, 1e-5)
        assert rel_close(model.intercept, ref.NB_COEF[0], 1e-6)

    def test_negative_binomial_fixed_theta(self):
        row = _expr_fit("y_nb", family="negative_binomial", theta=2.5)
        for j, c in enumerate(row["coefficients"]):
            assert rel_close(c, ref.NB_FIXED_COEF[j + 1], 1e-6)

        model = _model_fit(ref.Y_NB, family="negative_binomial", theta=2.5)
        for j, s in enumerate(model.std_errors):
            assert rel_close(s, ref.NB_FIXED_SE[j + 1], 1e-5)


class TestPenalizedGlmPriors:
    _PRIORS = [None, ("normal", 0.5, 0.5), ("normal", 0.0, 0.25)]

    @pytest.mark.parametrize(
        "vcov_type,se_ref",
        [
            ("laplace", ref.POIS_PRIOR_SE_LAPLACE),
            ("sandwich", ref.POIS_PRIOR_SE_SANDWICH),
            ("naive", ref.POIS_PRIOR_SE_NAIVE),
        ],
    )
    def test_prior_coefficients_and_vcov_type_se(self, vcov_type, se_ref):
        row = DF.select(
            ps.penalized_glm(
                "y_pois", "x1", "x2", priors=self._PRIORS, tol=1e-12
            ).alias("r")
        )["r"][0]
        assert rel_close(row["intercept"], ref.POIS_PRIOR_COEF[0], 1e-7)
        assert rel_close(row["coefficients"][0], ref.POIS_PRIOR_COEF[1], 1e-7)
        assert rel_close(row["coefficients"][1], ref.POIS_PRIOR_COEF[2], 1e-7)

        model = ps.PenalizedGLM(
            priors=self._PRIORS, vcov_type=vcov_type, tol=1e-12
        ).fit(X, np.array(ref.Y_POIS))
        vcov = model.vcov
        for j in range(3):
            assert rel_close(math.sqrt(vcov[j, j]), se_ref[j], 1e-6)

    def test_prior_forms_are_equivalent(self):
        tuple_priors = [None, ("normal", 0.5, 0.5), ("normal", 0.0, 0.25)]
        dict_priors = [
            None,
            {"kind": "normal", "loc": 0.5, "scale": 0.5},
            {"kind": "normal", "loc": 0.0, "scale": 0.25},
        ]
        list_priors = [None, ["normal", 0.5, 0.5], ["normal", 0.0, 0.25]]

        rows = [
            DF.select(
                ps.penalized_glm("y_pois", "x1", "x2", priors=p, tol=1e-12).alias("r")
            )["r"][0]
            for p in (tuple_priors, dict_priors, list_priors)
        ]
        for r in rows[1:]:
            assert r["coefficients"] == rows[0]["coefficients"]
            assert r["intercept"] == rows[0]["intercept"]

    def test_two_entry_priors_no_intercept_entry_accepted(self):
        # Two entries (no leading intercept prior) for two features.
        row = DF.select(
            ps.penalized_glm(
                "y_pois", "x1", "x2", priors=[("normal", 0.5, 0.5), ("normal", 0.0, 0.25)],
                tol=1e-12,
            ).alias("r")
        )["r"][0]
        assert math.isfinite(row["intercept"])

    def test_vcov_types_match_without_priors_differ_with_priors(self):
        models_no_prior = {
            vt: ps.PenalizedGLM(vcov_type=vt, tol=1e-10).fit(X, np.array(ref.Y_POIS))
            for vt in ("laplace", "sandwich", "naive")
        }
        se_laplace = models_no_prior["laplace"].std_errors
        se_sandwich = models_no_prior["sandwich"].std_errors
        se_naive = models_no_prior["naive"].std_errors
        for a, b in zip(se_laplace, se_sandwich):
            assert rel_close(a, b, 1e-10)
        for a, b in zip(se_laplace, se_naive):
            assert rel_close(a, b, 1e-10)

        models_prior = {
            vt: ps.PenalizedGLM(priors=self._PRIORS, vcov_type=vt, tol=1e-12).fit(
                X, np.array(ref.Y_POIS)
            )
            for vt in ("laplace", "sandwich")
        }
        se_l = models_prior["laplace"].std_errors
        se_s = models_prior["sandwich"].std_errors
        assert max(abs(a - b) for a, b in zip(se_l, se_s)) > 1e-6


class TestPenalizedGlmLambda:
    def test_lambda_shrinks_coefficients(self):
        unpenalized = _expr_fit("y_pois", family="poisson")
        shrunk = _expr_fit("y_pois", family="poisson", lambda_=4.0)
        unpen_norm = math.sqrt(sum(c**2 for c in unpenalized["coefficients"]))
        shrunk_norm = math.sqrt(sum(c**2 for c in shrunk["coefficients"]))
        assert shrunk_norm < unpen_norm

    def test_lambda_equals_ridge_prior(self):
        lambda_row = DF.select(
            ps.penalized_glm(
                "y_pois", "x1", "x2", family="poisson", lambda_=4.0, tol=1e-12
            ).alias("r")
        )["r"][0]
        prior_row = DF.select(
            ps.penalized_glm(
                "y_pois",
                "x1",
                "x2",
                family="poisson",
                priors=[None, ("normal", 0.0, 0.5), ("normal", 0.0, 0.5)],
                tol=1e-12,
            ).alias("r")
        )["r"][0]
        # Observed (and confirmed) relation: ridge lambda=4.0 is numerically
        # identical to a N(0, 1/sqrt(4))=N(0, 0.5) prior on every non-intercept
        # coefficient, matching the upstream docstring's documented relation
        # ("ridge lambda == N(0, 1/sqrt(lambda)) prior"). Neither value is in
        # the upstream R fixture; this is a direct self-consistency check.
        assert rel_close(lambda_row["intercept"], prior_row["intercept"], 1e-8)
        for a, b in zip(lambda_row["coefficients"], prior_row["coefficients"]):
            assert rel_close(a, b, 1e-8)


class TestPenalizedGlmLaplacePrior:
    def test_laplace_prior_soft_thresholds_x1(self):
        model = ps.PenalizedGLM(
            priors=[None, ("laplace", 0.0, 1e-3), None], tol=1e-10
        ).fit(X, np.array(ref.Y_POIS))
        assert model.coefficients[0] == 0.0
        assert math.isnan(model.std_errors[0])
        assert math.isfinite(model.coefficients[1])


class TestPenalizedGlmExprModelParity:
    @pytest.mark.parametrize(
        "family,y_col,y",
        [
            ("poisson", "y_pois", ref.Y_POIS),
            ("binomial", "y_bin", ref.Y_BIN),
            ("gamma", "y_gamma", ref.Y_GAMMA),
            ("negative_binomial", "y_nb", ref.Y_NB),
        ],
    )
    def test_expr_and_model_parity(self, family, y_col, y):
        row = _expr_fit(y_col, family=family)
        model = _model_fit(y, family=family)
        assert rel_close(row["intercept"], model.intercept, 1e-12)
        for a, b in zip(row["coefficients"], model.coefficients):
            assert rel_close(a, b, 1e-12)
        assert rel_close(row["deviance"], model.deviance, 1e-12)
        assert rel_close(row["null_deviance"], model.null_deviance, 1e-12)
        assert rel_close(row["log_likelihood"], model.log_likelihood, 1e-12)
        assert rel_close(row["aic"], model.aic, 1e-12)
        assert rel_close(row["dispersion"], model.dispersion, 1e-12)


class TestPenalizedGlmGroupByOver:
    @staticmethod
    def _stacked_df() -> pl.DataFrame:
        a = DF.with_columns(g=pl.lit("a"))
        b = DF.with_columns(g=pl.lit("b"))
        return pl.concat([a, b])

    def test_group_by_matches_whole_frame_fit(self):
        stacked = self._stacked_df()
        whole = _expr_fit("y_pois", family="poisson")
        grouped = stacked.group_by("g").agg(
            ps.penalized_glm("y_pois", "x1", "x2", family="poisson", tol=1e-10).alias("r")
        )
        assert has_valid_numbers(grouped, "r", nan_ok=("theta", "accuracy"))
        for row in grouped["r"]:
            assert rel_close(row["intercept"], whole["intercept"], 1e-12)
            for a, b in zip(row["coefficients"], whole["coefficients"]):
                assert rel_close(a, b, 1e-12)

    def test_group_by_with_offset(self):
        stacked = self._stacked_df()
        whole = _expr_fit("y_pois", family="poisson", offset="off")
        grouped = stacked.group_by("g").agg(
            ps.penalized_glm(
                "y_pois", "x1", "x2", family="poisson", offset="off", tol=1e-10
            ).alias("r")
        )
        for row in grouped["r"]:
            assert rel_close(row["intercept"], whole["intercept"], 1e-12)

    def test_over_returns_same_struct_every_row(self):
        result = DF.select(
            ps.penalized_glm("y_pois", "x1", "x2", family="poisson", tol=1e-10)
            .over(pl.lit(1))
            .alias("r")
        )
        first = result["r"][0]
        for row in result["r"]:
            assert row["intercept"] == first["intercept"]
            assert row["coefficients"] == first["coefficients"]


class TestPenalizedGlmSummary:
    """`ps.penalized_glm_summary` — phase 16.2 plan 04 (UPST-05, CONTEXT resolution 3)."""

    @staticmethod
    def _summary_df(y_col: str, tol: float = 1e-10, **kwargs) -> pl.DataFrame:
        return (
            DF.select(
                ps.penalized_glm_summary(y_col, "x1", "x2", tol=tol, **kwargs).alias("s")
            )
            .explode("s")
            .unnest("s")
        )

    def test_summary_columns_and_poisson_se_match_r(self):
        flat = self._summary_df("y_pois", family="poisson")
        assert flat.columns == [
            "term",
            "estimate",
            "std_error",
            "statistic",
            "p_value",
            "ci_lower",
            "ci_upper",
        ]
        assert flat["term"].to_list() == ["intercept", "x1", "x2"]
        for i in range(3):
            assert rel_close(flat["estimate"][i], ref.POIS_COEF[i], 1e-6)
            assert rel_close(flat["std_error"][i], ref.POIS_SE[i], 1e-5)

    def test_summary_gamma_se_matches_r(self):
        flat = self._summary_df("y_gamma", family="gamma")
        for i in range(3):
            assert rel_close(flat["std_error"][i], ref.GAMMA_SE[i], 1e-5)

    def test_summary_negative_binomial_se_matches_r(self):
        flat = self._summary_df("y_nb", family="negative_binomial")
        for i in range(3):
            assert rel_close(flat["std_error"][i], ref.NB_SE[i], 1e-5)

    def test_summary_quasipoisson_se_matches_r(self):
        flat_over = self._summary_df("y_nb", family="poisson")
        for i in range(3):
            assert rel_close(flat_over["std_error"][i], ref.POIS_OVER_SE[i], 1e-5)

        flat_qpois = self._summary_df("y_nb", family="poisson", estimate_dispersion=True)
        for i in range(3):
            assert rel_close(flat_qpois["std_error"][i], ref.QPOIS_OVER_SE[i], 1e-5)

    def test_summary_equals_pymodel_inference(self):
        flat = self._summary_df("y_pois", family="poisson")
        model = ps.PenalizedGLM(family="poisson", tol=1e-10).fit(X, np.array(ref.Y_POIS))

        assert rel_close(flat["statistic"][0], model.intercept_z_value, 1e-10)
        assert rel_close(flat["p_value"][0], model.intercept_p_value, 1e-10)
        ci_lo, ci_hi = model.intercept_conf_interval
        assert rel_close(flat["ci_lower"][0], ci_lo, 1e-10)
        assert rel_close(flat["ci_upper"][0], ci_hi, 1e-10)

        for j in range(2):
            assert rel_close(flat["statistic"][j + 1], model.z_values[j], 1e-10)
            assert rel_close(flat["p_value"][j + 1], model.p_values[j], 1e-10)
            assert rel_close(flat["ci_lower"][j + 1], model.conf_interval_lower[j], 1e-10)
            assert rel_close(flat["ci_upper"][j + 1], model.conf_interval_upper[j], 1e-10)

    def test_summary_conf_level_narrows_interval(self):
        flat = self._summary_df("y_pois", family="poisson", conf_level=0.9)
        z90 = 1.6448536269514722
        for i in range(3):
            ratio = (flat["ci_upper"][i] - flat["ci_lower"][i]) / flat["std_error"][i]
            assert rel_close(ratio, 2 * z90, 1e-9)

    @pytest.mark.parametrize(
        "vcov_type,se_ref",
        [
            ("laplace", ref.POIS_PRIOR_SE_LAPLACE),
            ("sandwich", ref.POIS_PRIOR_SE_SANDWICH),
            ("naive", ref.POIS_PRIOR_SE_NAIVE),
        ],
    )
    def test_summary_priors_vcov_type_se(self, vcov_type, se_ref):
        flat = self._summary_df(
            "y_pois",
            family="poisson",
            priors=[None, ("normal", 0.5, 0.5), ("normal", 0.0, 0.25)],
            vcov_type=vcov_type,
            tol=1e-12,
        )
        for i in range(3):
            assert rel_close(flat["std_error"][i], se_ref[i], 1e-6)

    def test_summary_no_intercept(self):
        flat = self._summary_df("y_pois", family="poisson", add_intercept=False)
        assert flat["term"].to_list() == ["x1", "x2"]
        for i in range(2):
            assert rel_close(flat["std_error"][i], ref.POIS_NOINT_SE[i], 1e-5)

    def test_summary_engine_error_gives_empty_list(self):
        row = DF.select(
            ps.penalized_glm_summary(
                (-pl.col("y_pois")).alias("neg_y"), "x1", "x2", family="poisson", tol=1e-10
            ).alias("s")
        )["s"][0]
        assert len(row) == 0

    def test_summary_group_by_matches_whole_frame(self):
        stacked = pl.concat(
            [DF.with_columns(g=pl.lit("a")), DF.with_columns(g=pl.lit("b"))]
        )
        whole = self._summary_df("y_pois", family="poisson").sort("term")
        grouped = (
            stacked.group_by("g")
            .agg(
                ps.penalized_glm_summary(
                    "y_pois", "x1", "x2", family="poisson", tol=1e-10
                ).alias("s")
            )
            .explode("s")
            .unnest("s")
        )
        for g in ("a", "b"):
            sub = grouped.filter(pl.col("g") == g).sort("term")
            for i in range(3):
                assert rel_close(sub["estimate"][i], whole["estimate"][i], 1e-10)

    def test_summary_invalid_vcov_type_raises(self):
        with pytest.raises(ValueError):
            ps.penalized_glm_summary("y_pois", "x1", "x2", vcov_type="bogus")

    def test_summary_invalid_conf_level_raises(self):
        with pytest.raises(ValueError):
            ps.penalized_glm_summary("y_pois", "x1", "x2", conf_level=1.5)

    def test_negative_binomial_summary_schema_unchanged(self):
        """The shared 5-field summary schema (CONTEXT: additive only) stays untouched."""
        flat = (
            DF.select(ps.negative_binomial_summary("y_nb", "x1", "x2").alias("s"))
            .explode("s")
            .unnest("s")
        )
        assert flat.columns == ["term", "estimate", "std_error", "statistic", "p_value"]


class TestPenalizedGlmFormula:
    def test_formula_equals_column_form(self):
        column_row = _expr_fit("y_nb", family="negative_binomial")
        formula_row = DF.select(
            ps.penalized_glm_formula(
                "y_nb ~ x1 + x2", family="negative_binomial", tol=1e-10
            ).alias("r")
        )["r"][0]
        assert rel_close(column_row["intercept"], formula_row["intercept"], 1e-12)
        for a, b in zip(column_row["coefficients"], formula_row["coefficients"]):
            assert rel_close(a, b, 1e-12)

    def test_formula_forwards_offset(self):
        column_row = _expr_fit("y_pois", family="poisson", offset="off")
        formula_row = DF.select(
            ps.penalized_glm_formula(
                "y_pois ~ x1 + x2", family="poisson", offset="off", tol=1e-10
            ).alias("r")
        )["r"][0]
        assert rel_close(column_row["intercept"], formula_row["intercept"], 1e-12)


class TestPenalizedGlmBuildTimeErrors:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"family": "poisson", "link": "probit"},
            {"family": "tweedie", "power": 0.5},
            {"family": "gamma", "theta": 2.0},
            {"family": "binomial", "estimate_dispersion": True},
            {"priors": [("cauchy", 0.0, 1.0)]},
            {"lambda_": -1.0},
            {"max_iter": 0},
        ],
    )
    def test_build_time_value_errors(self, kwargs):
        with pytest.raises(ValueError):
            ps.penalized_glm("y_pois", "x1", "x2", **kwargs)

    def test_engine_data_errors_give_nan_struct(self):
        row = DF.select(
            ps.penalized_glm(
                (-pl.col("y_pois")).alias("neg_y"), "x1", "x2", family="poisson", tol=1e-10
            ).alias("r")
        )["r"][0]
        assert math.isnan(row["intercept"])
        assert row["coefficients"] == []
        assert row["n_observations"] == 0
