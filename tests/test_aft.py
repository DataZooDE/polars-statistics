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

    def test_predict_event_coding_invalid_gives_nan(self):
        """CR-01 regression: the four `aft_predict_*` expressions must reject
        invalid event codes (e.g. R's 1/2 coding) exactly like `aft`/
        `aft_summary` do -- an all-NaN prediction struct, not a silently
        wrong value from misreading the bad code as "censored"."""
        df = _fixture_df().with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(2.0)
            .otherwise(pl.col("event_float"))
            .alias("bad_event")
        )

        q = df.select(
            ps.aft_predict_quantile("time", "bad_event", "x1", "x2", p=0.5).alias("r")
        ).unnest("r")
        assert all(math.isnan(v) for v in q["aft_predict_quantile_prediction"])

        m = df.select(
            ps.aft_predict_median("time", "bad_event", "x1", "x2").alias("r")
        ).unnest("r")
        assert all(math.isnan(v) for v in m["aft_predict_median_prediction"])

        s = df.select(
            ps.aft_predict_survival("time", "bad_event", "x1", "x2", t=5.0).alias("r")
        ).unnest("r")
        assert all(math.isnan(v) for v in s["aft_predict_survival_prediction"])

        c = df.select(
            ps.aft_predict_cdf("time", "bad_event", "x1", "x2", t=5.0).alias("r")
        ).unnest("r")
        assert all(math.isnan(v) for v in c["aft_predict_cdf_prediction"])

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


# ============================================================================
# AFT PyModel (Task 2)
# ============================================================================

X = np.column_stack([ref.X1, ref.X2])
TIME = np.array(ref.TIME)
EVENT = np.array(ref.EVENT)

_DIST_REFS = [
    ("weibull", ref.AFT_WEIBULL_COEF, ref.AFT_WEIBULL_SCALE, ref.AFT_WEIBULL_LL, ref.AFT_WEIBULL_SE, ref.AFT_WEIBULL_Q),
    (
        "lognormal",
        ref.AFT_LOGNORMAL_COEF,
        ref.AFT_LOGNORMAL_SCALE,
        ref.AFT_LOGNORMAL_LL,
        ref.AFT_LOGNORMAL_SE,
        ref.AFT_LOGNORMAL_Q,
    ),
    (
        "loglogistic",
        ref.AFT_LOGLOGISTIC_COEF,
        ref.AFT_LOGLOGISTIC_SCALE,
        ref.AFT_LOGLOGISTIC_LL,
        ref.AFT_LOGLOGISTIC_SE,
        ref.AFT_LOGLOGISTIC_Q,
    ),
    (
        "exponential",
        ref.AFT_EXPONENTIAL_COEF,
        ref.AFT_EXPONENTIAL_SCALE,
        ref.AFT_EXPONENTIAL_LL,
        ref.AFT_EXPONENTIAL_SE,
        ref.AFT_EXPONENTIAL_Q,
    ),
]

X0 = np.array([[0.3, -0.2]])


class TestAFTModel:
    """`AFT` matches `survival::survreg` for all four distributions,
    including upstream intercept inference and the four predict methods."""

    @pytest.mark.parametrize("dist,coef,scale,ll,se,q", _DIST_REFS, ids=[d[0] for d in _DIST_REFS])
    def test_model_matches_survreg(self, dist, coef, scale, ll, se, q):
        model = ps.AFT(distribution=dist).fit(X, TIME, EVENT)

        assert rel_close(model.intercept, coef[0], 1e-6)
        for j, c in enumerate(model.coefficients):
            assert rel_close(c, coef[j + 1], 1e-6)
        assert rel_close(model.scale, scale, 1e-6)
        assert rel_close(model.null_log_likelihood, ll[0], 1e-7)
        assert rel_close(model.log_likelihood, ll[1], 1e-7)

        assert rel_close(model.intercept_std_error, se[0], 1e-5)
        for j, s in enumerate(model.std_errors):
            assert rel_close(s, se[j + 1], 1e-5)
        if dist == "exponential":
            assert model.log_scale_std_error is None
            assert model.scale == 1.0
        else:
            assert rel_close(model.log_scale_std_error, se[3], 1e-5)

        assert model.converged is True
        assert model.distribution == dist

        # Upstream intercept inference, read directly (never recomputed).
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
        assert 0.0 <= model.intercept_p_value <= 1.0

        # Quantiles.
        for p, qv in zip((0.25, 0.5, 0.9), q):
            assert rel_close(model.predict_quantile(X0, p)[0], qv, 1e-6)
        assert model.predict_quantile(X0, 0.5)[0] == pytest.approx(
            model.predict_median(X0)[0]
        )
        med = model.predict_median(X0)[0]
        assert model.predict_survival(X0, med)[0] == pytest.approx(0.5, abs=1e-9)
        assert model.predict_cdf(X0, med)[0] == pytest.approx(0.5, abs=1e-9)
        linear = model.predict_linear(X0)[0]
        assert linear == pytest.approx(
            model.intercept + X0[0] @ model.coefficients, rel=1e-12
        )

    def test_model_event_array_variants_identical(self):
        model_float = ps.AFT(distribution="weibull").fit(X, TIME, EVENT)
        model_bool = ps.AFT(distribution="weibull").fit(X, TIME, EVENT.astype(bool))
        model_list = ps.AFT(distribution="weibull").fit(X, TIME, list(EVENT))
        assert model_float.intercept == pytest.approx(model_bool.intercept)
        assert model_float.intercept == pytest.approx(model_list.intercept)

    def test_model_vcov_and_information_consistent(self):
        model = ps.AFT(distribution="weibull").fit(X, TIME, EVENT)
        vcov = model.vcov
        information = model.information
        assert vcov.shape == (4, 4)
        se_diag = np.sqrt(np.diag(vcov))
        expected = np.array(
            [model.intercept_std_error, *model.std_errors, model.log_scale_std_error]
        )
        np.testing.assert_allclose(se_diag, expected, rtol=1e-12)
        identity = information @ vcov
        np.testing.assert_allclose(identity, np.eye(4), atol=1e-8)

    def test_model_exponential_vcov_shape(self):
        model = ps.AFT(distribution="exponential").fit(X, TIME, EVENT)
        assert model.vcov.shape == (3, 3)

    def test_model_priors_shrink_toward_zero_and_vcov_type_differs(self):
        base = ps.AFT(distribution="weibull").fit(X, TIME, EVENT)
        priored = ps.AFT(
            distribution="weibull", priors=[None, ("normal", 0.0, 0.05), None]
        ).fit(X, TIME, EVENT)
        assert abs(priored.coefficients[0]) < abs(base.coefficients[0])

        sandwich = ps.AFT(
            distribution="weibull",
            priors=[None, ("normal", 0.0, 0.05), None],
            vcov_type="sandwich",
        ).fit(X, TIME, EVENT)
        assert abs(sandwich.std_errors[0] - priored.std_errors[0]) > 1e-6

        base_laplace = ps.AFT(distribution="weibull", vcov_type="laplace").fit(X, TIME, EVENT)
        base_sandwich = ps.AFT(distribution="weibull", vcov_type="sandwich").fit(X, TIME, EVENT)
        base_naive = ps.AFT(distribution="weibull", vcov_type="naive").fit(X, TIME, EVENT)
        assert base_laplace.std_errors[0] == pytest.approx(base_sandwich.std_errors[0], rel=1e-10)
        assert base_laplace.std_errors[0] == pytest.approx(base_naive.std_errors[0], rel=1e-10)

    def test_model_dict_and_tuple_prior_forms_identical(self):
        tuple_model = ps.AFT(
            distribution="weibull", priors=[None, ("normal", 0.0, 0.05), None]
        ).fit(X, TIME, EVENT)
        dict_model = ps.AFT(
            distribution="weibull",
            priors=[None, {"kind": "normal", "loc": 0.0, "scale": 0.05}, None],
        ).fit(X, TIME, EVENT)
        assert tuple_model.coefficients[0] == pytest.approx(dict_model.coefficients[0])

    def test_model_build_time_and_fit_time_errors(self):
        with pytest.raises(ValueError):
            ps.AFT(distribution="gompertz")
        with pytest.raises(ValueError):
            ps.AFT(distribution="weibull", vcov_type="bogus")
        with pytest.raises(ValueError):
            ps.AFT(distribution="weibull", max_iter=0)
        with pytest.raises(ValueError):
            ps.AFT(distribution="weibull", tol=0.0)
        with pytest.raises(ValueError):
            ps.AFT(distribution="weibull", confidence_level=1.5)

        model = ps.AFT(distribution="weibull")
        with pytest.raises(ValueError):
            model.fit(X, TIME, np.where(EVENT == 1.0, 2.0, EVENT))
        with pytest.raises(ValueError):
            model.fit(X, TIME, EVENT[:-1])
        with pytest.raises(ValueError):
            model.fit(X, np.where(np.arange(len(TIME)) == 0, 0.0, TIME), EVENT)
        with pytest.raises(ValueError):
            model.fit(X, TIME, np.zeros_like(EVENT))

        fitted = ps.AFT(distribution="weibull").fit(X, TIME, EVENT)
        with pytest.raises(ValueError):
            fitted.predict_quantile(X0, p=1.5)
        with pytest.raises(ValueError):
            fitted.predict_quantile(X0, p=-0.1)
        with pytest.raises(ValueError, match="priors"):
            ps.AFT(distribution="weibull", priors=[None]).fit(X, TIME, EVENT)
        with pytest.raises(ValueError):
            ps.AFT(distribution="weibull", vcov_type="made_up").fit(X, TIME, EVENT)

    def test_model_unfitted_getters_and_methods_raise_runtime_error(self):
        model = ps.AFT(distribution="weibull")
        with pytest.raises(RuntimeError, match="AFT"):
            _ = model.intercept
        with pytest.raises(RuntimeError, match="AFT"):
            model.predict_median(X0)

    def test_model_ergonomics(self):
        model = ps.AFT(distribution="weibull").fit(X, TIME, EVENT)
        assert repr(model).startswith("AFT(")
        d = model.to_dict()
        assert "scale" in d
        assert "vcov" in d


# ============================================================================
# aft_summary (Task 3)
# ============================================================================


def _summary_rows(df: pl.DataFrame, distribution: str) -> pl.DataFrame:
    return (
        df.select(
            ps.aft_summary("time", "event", "x1", "x2", distribution=distribution).alias("s")
        )
        .explode("s")
        .unnest("s")
    )


class TestAftSummary:
    @pytest.mark.parametrize("dist,se,scale", [
        ("weibull", ref.AFT_WEIBULL_SE, ref.AFT_WEIBULL_SCALE),
        ("lognormal", ref.AFT_LOGNORMAL_SE, ref.AFT_LOGNORMAL_SCALE),
        ("loglogistic", ref.AFT_LOGLOGISTIC_SE, ref.AFT_LOGLOGISTIC_SCALE),
        ("exponential", ref.AFT_EXPONENTIAL_SE, ref.AFT_EXPONENTIAL_SCALE),
    ])
    def test_summary_terms_and_se_match_r(self, dist, se, scale):
        df = _fixture_df()
        rows = _summary_rows(df, dist)
        expected_terms = ["intercept", "x1", "x2"] + ([] if dist == "exponential" else ["log_scale"])
        assert rows["term"].to_list() == expected_terms

        for j, term in enumerate(("intercept", "x1", "x2")):
            row = rows.filter(pl.col("term") == term)
            assert rel_close(row["std_error"][0], se[j], 1e-5)

        if dist != "exponential":
            log_scale_row = rows.filter(pl.col("term") == "log_scale")
            assert rel_close(log_scale_row["estimate"][0], math.log(scale), 1e-6)
            assert rel_close(log_scale_row["std_error"][0], se[3], 1e-5)

    def test_summary_intercept_and_features_equal_model_inference(self):
        df = _fixture_df()
        rows = _summary_rows(df, "weibull")
        model = ps.AFT(distribution="weibull").fit(X, TIME, EVENT)

        intercept_row = rows.filter(pl.col("term") == "intercept")
        assert intercept_row["statistic"][0] == pytest.approx(model.intercept_z_value)
        assert intercept_row["p_value"][0] == pytest.approx(model.intercept_p_value)
        assert intercept_row["ci_lower"][0] == pytest.approx(model.intercept_conf_interval[0])
        assert intercept_row["ci_upper"][0] == pytest.approx(model.intercept_conf_interval[1])

        for j, term in enumerate(("x1", "x2")):
            row = rows.filter(pl.col("term") == term)
            assert row["statistic"][0] == pytest.approx(model.z_values[j])
            assert row["p_value"][0] == pytest.approx(model.p_values[j])
            assert row["ci_lower"][0] == pytest.approx(model.conf_interval_lower[j])
            assert row["ci_upper"][0] == pytest.approx(model.conf_interval_upper[j])

    def test_summary_conf_level_half_width(self):
        df = _fixture_df()
        rows = (
            df.select(
                ps.aft_summary(
                    "time", "event", "x1", "x2", distribution="weibull", conf_level=0.9
                ).alias("s")
            )
            .explode("s")
            .unnest("s")
        )
        for row in rows.iter_rows(named=True):
            half_width = (row["ci_upper"] - row["estimate"]) / row["std_error"]
            assert rel_close(half_width, 1.6448536269514722, 1e-9)


# ============================================================================
# aft_predict_quantile / _median / _survival / _cdf (Task 3)
# ============================================================================


class TestAftPredict:
    @pytest.mark.parametrize("dist,q", [
        ("weibull", ref.AFT_WEIBULL_Q),
        ("lognormal", ref.AFT_LOGNORMAL_Q),
        ("loglogistic", ref.AFT_LOGLOGISTIC_Q),
        ("exponential", ref.AFT_EXPONENTIAL_Q),
    ])
    def test_predict_quantile_matches_r_at_held_out_row(self, dist, q):
        df = _fixture_df().vstack(
            pl.DataFrame(
                {
                    "x1": [0.3],
                    "x2": [-0.2],
                    "time": [None],
                    "event": [None],
                    "event_float": [None],
                }
            )
        )
        for p, qv in zip((0.25, 0.5, 0.9), q):
            r = df.select(
                ps.aft_predict_quantile(
                    "time", "event", "x1", "x2", p=p, distribution=dist, null_policy="drop_y_zero_x"
                ).alias("pred")
            ).unnest("pred")
            assert rel_close(r["aft_predict_quantile_prediction"][-1], qv, 1e-6)

    def test_predict_median_equals_quantile_half(self):
        df = _fixture_df().vstack(
            pl.DataFrame(
                {"x1": [0.3], "x2": [-0.2], "time": [None], "event": [None], "event_float": [None]}
            )
        )
        q = df.select(
            ps.aft_predict_quantile(
                "time", "event", "x1", "x2", p=0.5, null_policy="drop_y_zero_x"
            ).alias("pred")
        ).unnest("pred")
        m = df.select(
            ps.aft_predict_median(
                "time", "event", "x1", "x2", null_policy="drop_y_zero_x"
            ).alias("pred")
        ).unnest("pred")
        assert q["aft_predict_quantile_prediction"][-1] == pytest.approx(
            m["aft_predict_median_prediction"][-1]
        )

    def test_predict_survival_and_cdf_identity(self):
        df = _fixture_df().vstack(
            pl.DataFrame(
                {"x1": [0.3], "x2": [-0.2], "time": [None], "event": [None], "event_float": [None]}
            )
        )
        med = df.select(
            ps.aft_predict_median(
                "time", "event", "x1", "x2", null_policy="drop_y_zero_x"
            ).alias("pred")
        ).unnest("pred")["aft_predict_median_prediction"][-1]

        surv = df.select(
            ps.aft_predict_survival(
                "time", "event", "x1", "x2", t=med, null_policy="drop_y_zero_x"
            ).alias("pred")
        ).unnest("pred")
        cdf = df.select(
            ps.aft_predict_cdf(
                "time", "event", "x1", "x2", t=med, null_policy="drop_y_zero_x"
            ).alias("pred")
        ).unnest("pred")
        assert surv["aft_predict_survival_prediction"][-1] == pytest.approx(0.5, abs=1e-9)
        assert cdf["aft_predict_cdf_prediction"][-1] == pytest.approx(0.5, abs=1e-9)

    def test_predict_default_policy_gives_nan_for_held_out_row(self):
        df = _fixture_df().vstack(
            pl.DataFrame(
                {"x1": [0.3], "x2": [-0.2], "time": [None], "event": [None], "event_float": [None]}
            )
        )
        r = df.select(
            ps.aft_predict_quantile("time", "event", "x1", "x2", p=0.5).alias("pred")
        ).unnest("pred")
        assert math.isnan(r["aft_predict_quantile_prediction"][-1])
        assert math.isnan(r["aft_predict_quantile_lower"][-1])
        assert math.isnan(r["aft_predict_quantile_upper"][-1])

    def test_predict_default_prefixes_and_custom_name(self):
        df = _fixture_df()
        r = df.select(ps.aft_predict_quantile("time", "event", "x1", "x2").alias("pred")).unnest(
            "pred"
        )
        assert "aft_predict_quantile_prediction" in r.columns
        r_named = df.select(
            ps.aft_predict_quantile("time", "event", "x1", "x2", name="custom").alias("pred")
        ).unnest("pred")
        assert "custom_prediction" in r_named.columns

    def test_predict_in_sample_matches_pymodel(self):
        df = _fixture_df()
        expr_median = df.select(
            ps.aft_predict_quantile("time", "event", "x1", "x2", p=0.5).alias("pred")
        ).unnest("pred")["aft_predict_quantile_prediction"].to_numpy()
        model = ps.AFT(distribution="weibull").fit(X, TIME, EVENT)
        model_median = model.predict_median(X)
        np.testing.assert_allclose(expr_median, model_median, rtol=1e-12)

    def test_predict_build_time_errors(self):
        with pytest.raises(ValueError):
            ps.aft_predict_quantile("time", "event", "x1", p=1.5)
        with pytest.raises(ValueError):
            ps.aft_predict_quantile("time", "event", "x1", p=-0.1)
        with pytest.raises(ValueError):
            ps.aft_predict_survival("time", "event", "x1", t=float("nan"))
        with pytest.raises(ValueError):
            ps.aft_predict_survival("time", "event", "x1", t=0.0)
        with pytest.raises(ValueError):
            ps.aft_predict_quantile("time", "event", "x1", distribution="gompertz")
        with pytest.raises(ValueError):
            ps.aft_predict_quantile("time", "event", "x1", null_policy="bogus")


# ============================================================================
# group_by / over (Task 3)
# ============================================================================


class TestAftGroupByOver:
    def test_group_by_matches_whole_frame_fit(self):
        df = _fixture_df()
        stacked = pl.concat(
            [df.with_columns(g=pl.lit(0)), df.with_columns(g=pl.lit(1))]
        )
        whole = df.select(ps.aft("time", "event", "x1", "x2").alias("r")).unnest("r")
        grouped = (
            stacked.group_by("g")
            .agg(ps.aft("time", "event", "x1", "x2").alias("r"))
            .unnest("r")
            .sort("g")
        )
        for g in (0, 1):
            row = grouped.filter(pl.col("g") == g)
            assert row["intercept"][0] == pytest.approx(whole["intercept"][0])

    def test_over_broadcasts_struct(self):
        df = _fixture_df().with_columns(g=pl.lit(0))
        r = df.with_columns(ps.aft("time", "event", "x1", "x2").over("g").alias("r")).unnest("r")
        assert r["intercept"].n_unique() == 1

    def test_group_by_summary_equals_whole_frame(self):
        df = _fixture_df()
        whole = _summary_rows(df, "weibull")
        stacked = df.with_columns(g=pl.lit(0))
        grouped = (
            stacked.group_by("g")
            .agg(ps.aft_summary("time", "event", "x1", "x2").alias("s"))
            .explode("s")
            .unnest("s")
        )
        assert grouped["estimate"].to_list() == pytest.approx(whole["estimate"].to_list())


# ============================================================================
# aft_formula / aft_formula_summary (Task 3)
# ============================================================================


class TestAftFormula:
    def test_formula_equals_column_form(self):
        df = _fixture_df()
        a = df.select(ps.aft("time", "event", "x1", "x2", distribution="lognormal").alias("r")).unnest(
            "r"
        )
        b = df.select(
            ps.aft_formula("time ~ x1 + x2", event="event", distribution="lognormal").alias("r")
        ).unnest("r")
        assert a["intercept"][0] == pytest.approx(b["intercept"][0])

    def test_formula_summary_equals_column_form(self):
        df = _fixture_df()
        a = _summary_rows(df, "lognormal")
        b = (
            df.select(
                ps.aft_formula_summary(
                    "time ~ x1 + x2", event="event", distribution="lognormal"
                ).alias("s")
            )
            .explode("s")
            .unnest("s")
        )
        assert a["estimate"].to_list() == pytest.approx(b["estimate"].to_list())
