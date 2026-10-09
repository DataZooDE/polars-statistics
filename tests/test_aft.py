"""Tests for the AFT survival regression engine (`AFT` / `ps.aft` /
`ps.aft_summary` / `ps.aft_predict_quantile` et al.), phase 16.2 plan 05.

R reference: `survival::survreg`, shipped as `anofox-regression` 0.5.22's
`tests/fixtures/penalized_glm_aft_reference.rs` and transcribed verbatim into
`tests/fixtures/penalized_glm_aft_reference.py` (see that module's docstring
for provenance and `SOURCE_SHA256`).

See `.planning/phases/16.2-penalized-glm-and-aft-survival-regression/16.2-05-PLAN.md`.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

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
            "time": ref.TIME,
            "event": [e == 1.0 for e in ref.EVENT],
            "event_float": ref.EVENT,
        }
    )


class TestAftExprWeibull:
    """`ps.aft(..., distribution="weibull")` matches `survreg`."""

    def test_expr_weibull_matches_r(self):
        df = _fixture_df()
        row = df.select(
            ps.aft("time", "event", "x1", "x2", distribution="weibull").alias("r")
        ).unnest("r")

        assert rel_close(row["intercept"][0], ref.AFT_WEIBULL_COEF[0], 1e-6)
        coefs = row["coefficients"][0].to_list()
        assert rel_close(coefs[0], ref.AFT_WEIBULL_COEF[1], 1e-6)
        assert rel_close(coefs[1], ref.AFT_WEIBULL_COEF[2], 1e-6)
        assert rel_close(row["scale"][0], ref.AFT_WEIBULL_SCALE, 1e-6)
        assert rel_close(row["null_log_likelihood"][0], ref.AFT_WEIBULL_LL[0], 1e-7)
        assert rel_close(row["log_likelihood"][0], ref.AFT_WEIBULL_LL[1], 1e-7)
        assert row["converged"][0] is True
        assert row["n_observations"][0] == 60
        n_events = int(sum(ref.EVENT))
        assert row["n_events"][0] == n_events
        assert row["n_censored"][0] == 60 - n_events

    def test_expr_weibull_schema(self):
        df = _fixture_df()
        schema = df.select(ps.aft("time", "event", "x1", "x2").alias("r")).unnest("r").schema
        assert list(schema.keys()) == [
            "intercept",
            "coefficients",
            "scale",
            "log_likelihood",
            "null_log_likelihood",
            "aic",
            "bic",
            "iterations",
            "converged",
            "n_observations",
            "n_events",
            "n_censored",
        ]
        assert schema["n_observations"] == pl.UInt32
        assert schema["n_events"] == pl.UInt32
        assert schema["n_censored"] == pl.UInt32
        assert schema["converged"] == pl.Boolean
        assert schema["coefficients"] == pl.List(pl.Float64)


class TestAftEventCoding:
    """Event indicator accepts Boolean or numeric 0/1; any other value fails
    safely rather than being silently misread (T-16.2-14)."""

    def test_event_coding_float_matches_boolean(self):
        df = _fixture_df()
        r_bool = df.select(ps.aft("time", "event", "x1", "x2").alias("r")).unnest("r")
        r_float = df.select(ps.aft("time", "event_float", "x1", "x2").alias("r")).unnest("r")
        assert r_bool["intercept"][0] == pytest.approx(r_float["intercept"][0])
        assert r_bool["coefficients"][0].to_list() == pytest.approx(
            r_float["coefficients"][0].to_list()
        )

    def test_event_coding_int_matches_boolean(self):
        df = _fixture_df().with_columns(
            pl.col("event_float").cast(pl.Int64).alias("event_int")
        )
        r_bool = df.select(ps.aft("time", "event", "x1", "x2").alias("r")).unnest("r")
        r_int = df.select(ps.aft("time", "event_int", "x1", "x2").alias("r")).unnest("r")
        assert r_bool["intercept"][0] == pytest.approx(r_int["intercept"][0])

    def test_event_coding_invalid_gives_nan_struct(self):
        df = _fixture_df().with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(2.0)
            .otherwise(pl.col("event_float"))
            .alias("bad_event")
        )
        r = df.select(ps.aft("time", "bad_event", "x1", "x2").alias("r")).unnest("r")
        assert math.isnan(r["intercept"][0])

    def test_time_zero_gives_nan_struct(self):
        df = _fixture_df().with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(0.0)
            .otherwise(pl.col("time"))
            .alias("time")
        )
        r = df.select(ps.aft("time", "event", "x1", "x2").alias("r")).unnest("r")
        assert math.isnan(r["intercept"][0])

    def test_null_time_drops_row(self):
        df = _fixture_df().with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(None)
            .otherwise(pl.col("time"))
            .alias("time")
        )
        r = df.select(ps.aft("time", "event", "x1", "x2").alias("r")).unnest("r")
        assert r["n_observations"][0] == 59

    def test_null_event_drops_row(self):
        df = _fixture_df().with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(None)
            .otherwise(pl.col("event"))
            .alias("event")
        )
        r = df.select(ps.aft("time", "event", "x1", "x2").alias("r")).unnest("r")
        assert r["n_observations"][0] == 59


class TestAftInvalid:
    def test_invalid_distribution_raises_value_error(self):
        with pytest.raises(ValueError):
            ps.aft("time", "event", "x1", distribution="gompertz")
