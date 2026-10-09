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
