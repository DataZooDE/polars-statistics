"""R-reference tests for `ps.augment` and `OLS.augment`/`WLS.augment` (anofox-regression
0.5.20 #66, R 4.6.1 / broom 1.0.13, M1-A `mtcars`/`warpbreaks` fixture).

`diagnostics::augment` is exposed as one combined, row-aligned expression
(`ps.augment`) for every `Augmentable` model (ols, wls, logistic, probit,
cloglog, poisson, negative_binomial, tweedie, gamma) and as an `.augment()`
PyModel method on OLS / WLS (GLMs follow in 16.3-06). Values are checked
against R's `fitted`/`residuals`/`rstandard`/`rstudent`/`hatvalues`/
`cooks.distance`/`dffits` (via `broom::augment`) on the same fixture used by
`tests/test_glm_intercept_inference.py`, and cross-checked numerically against
the pre-existing piecewise diagnostics (`ps.leverage`, `ps.cooks_distance`,
`ps.dffits`, `ps.studentized_residuals`, `ps.externally_studentized_residuals`).

See `.planning/phases/16.3-upstream-api-wave-2/16.3-05-PLAN.md`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import polars as pl
import polars_statistics as ps
import pytest
from polars_statistics import OLS, WLS, Poisson, Tweedie

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "m1a_reference.py"
_spec = importlib.util.spec_from_file_location("m1a_reference", _FIXTURE_PATH)
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)


def rel_close(a, b, tol: float) -> bool:
    """Relative-error closeness check: ``|a - b| <= tol * max(|b|, 1)`` elementwise."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return bool(np.all(np.abs(a - b) <= tol * np.maximum(np.abs(b), 1.0)))


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


def _mt_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "mpg": MT_MPG,
            "wt": MT_WT,
            "hp": MT_HP,
            "vs": MT_VS,
            "carb": MT_CARB,
            "w": MT_W,
        }
    )


def _wb_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "breaks": WB_BREAKS,
            "woolB": WB_WOOLB,
            "tensionM": WB_TM,
            "tensionH": WB_TH,
        }
    )


class TestOlsExpressionMatchesR:
    def test_ols_expression_matches_r(self):
        df = _mt_df()
        out = df.select(ps.augment("mpg", "wt", "hp").alias("a")).unnest("a")

        assert rel_close(out["fitted"].to_numpy(), ref.OLS_FITTED, 1e-8)
        assert rel_close(out["residual"].to_numpy(), ref.OLS_RESID, 1e-8)
        assert rel_close(out["std_residual"].to_numpy(), ref.OLS_RSTD, 1e-8)
        assert rel_close(out["stud_residual"].to_numpy(), ref.OLS_RSTUD, 1e-8)
        assert rel_close(out["leverage"].to_numpy(), ref.OLS_HAT, 1e-8)
        assert rel_close(out["cooks_d"].to_numpy(), ref.OLS_COOK, 1e-8)
        assert rel_close(out["dffits"].to_numpy(), ref.OLS_DFFITS, 1e-8)

    def test_ols_schema(self):
        df = _mt_df()
        out = df.select(ps.augment("mpg", "wt", "hp").alias("a")).unnest("a")
        assert out.height == 32
        assert out.columns == [
            "fitted",
            "residual",
            "std_residual",
            "stud_residual",
            "leverage",
            "cooks_d",
            "dffits",
        ]
        for col in out.columns:
            assert out[col].dtype == pl.Float64


class TestOlsCrossCheckExistingDiagnostics:
    """Cross-check `ps.augment` against the pre-existing piecewise OLS diagnostics."""

    def _augment_df(self):
        df = _mt_df()
        return df.select(ps.augment("mpg", "wt", "hp").alias("a")).unnest("a")

    def test_leverage_matches_ps_leverage(self):
        aug = self._augment_df()
        df = _mt_df()
        lev = df.select(ps.leverage("wt", "hp").alias("lev")).unnest("lev")
        lev_values = np.array(lev["leverage"][0].to_list())
        assert rel_close(aug["leverage"].to_numpy(), lev_values, 1e-8)

    def test_cooks_d_matches_ps_cooks_distance(self):
        aug = self._augment_df()
        df = _mt_df()
        cd = df.select(ps.cooks_distance("mpg", "wt", "hp").alias("cd")).unnest("cd")
        cd_values = np.array(cd["cooks_d"][0].to_list())
        assert rel_close(aug["cooks_d"].to_numpy(), cd_values, 1e-8)

    def test_dffits_matches_ps_dffits(self):
        aug = self._augment_df()
        df = _mt_df()
        d = df.select(ps.dffits("mpg", "wt", "hp").alias("d")).unnest("d")
        d_values = np.array(d["dffits"][0].to_list())
        assert rel_close(aug["dffits"].to_numpy(), d_values, 1e-8)

    def test_std_residual_matches_ps_studentized_residuals(self):
        aug = self._augment_df()
        df = _mt_df()
        s = df.select(ps.studentized_residuals("mpg", "wt", "hp").alias("s")).unnest("s")
        s_values = np.array(s["residuals"][0].to_list())
        assert rel_close(aug["std_residual"].to_numpy(), s_values, 1e-8)

    def test_stud_residual_matches_ps_externally_studentized_residuals(self):
        aug = self._augment_df()
        df = _mt_df()
        s = df.select(
            ps.externally_studentized_residuals("mpg", "wt", "hp").alias("s")
        ).unnest("s")
        s_values = np.array(s["residuals"][0].to_list())
        assert rel_close(aug["stud_residual"].to_numpy(), s_values, 1e-8)


class TestOlsPymodelMatchesR:
    def test_ols_pymodel_matches_r(self):
        x = np.column_stack([MT_WT, MT_HP])
        model = OLS().fit(x, MT_MPG)
        d = model.augment(x, MT_MPG)

        assert d["resid_type"] == "response"
        assert rel_close(d["fitted"], ref.OLS_FITTED, 1e-8)
        assert rel_close(d["residual"], ref.OLS_RESID, 1e-8)
        assert rel_close(d["std_residual"], ref.OLS_RSTD, 1e-8)
        assert rel_close(d["stud_residual"], ref.OLS_RSTUD, 1e-8)
        assert rel_close(d["leverage"], ref.OLS_HAT, 1e-8)
        assert rel_close(d["cooks_d"], ref.OLS_COOK, 1e-8)
        assert rel_close(d["dffits"], ref.OLS_DFFITS, 1e-8)


class TestOlsPymodelErrors:
    def test_mismatched_length_raises_value_error(self):
        x = np.column_stack([MT_WT, MT_HP])
        model = OLS().fit(x, MT_MPG)
        with pytest.raises(ValueError):
            model.augment(x, MT_MPG[:10])

    def test_unfitted_raises_runtime_error_naming_class(self):
        model = OLS()
        x = np.column_stack([MT_WT, MT_HP])
        with pytest.raises(RuntimeError, match="OLS"):
            model.augment(x, MT_MPG)


class TestAugmentBuildErrors:
    def test_unknown_residual_type_raises_before_collect(self):
        with pytest.raises(ValueError):
            ps.augment("mpg", "wt", "hp", residual_type="working")

    def test_unknown_model_raises_before_collect(self):
        with pytest.raises(ValueError):
            ps.augment("mpg", "wt", "hp", model="lasso")

    def test_weights_without_wls_raises(self):
        with pytest.raises(ValueError):
            ps.augment("mpg", "wt", "hp", weights="hp", model="ols")

    def test_wls_without_weights_raises(self):
        with pytest.raises(ValueError):
            ps.augment("mpg", "wt", "hp", model="wls")

    def test_no_x_raises(self):
        with pytest.raises(ValueError):
            ps.augment("mpg", model="ols")

    def test_theta_without_negative_binomial_raises(self):
        with pytest.raises(ValueError):
            ps.augment("mpg", "wt", "hp", theta=1.0, model="ols")

    def test_link_power_without_tweedie_raises(self):
        with pytest.raises(ValueError):
            ps.augment("mpg", "wt", "hp", link_power=0.0, model="ols")


class TestWlsMatchesR:
    def test_wls_expression_matches_r(self):
        df = _mt_df()
        out = df.select(
            ps.augment("mpg", "wt", "hp", model="wls", weights="w").alias("a")
        ).unnest("a")
        assert rel_close(out["fitted"].to_numpy(), ref.WLS_FITTED, 1e-8)
        assert rel_close(out["residual"].to_numpy(), ref.WLS_RESID, 1e-8)
        assert rel_close(out["std_residual"].to_numpy(), ref.WLS_RSTD, 1e-8)
        assert rel_close(out["stud_residual"].to_numpy(), ref.WLS_RSTUD, 1e-8)
        assert rel_close(out["leverage"].to_numpy(), ref.WLS_HAT, 1e-8)
        assert rel_close(out["cooks_d"].to_numpy(), ref.WLS_COOK, 1e-8)
        assert rel_close(out["dffits"].to_numpy(), ref.WLS_DFFITS, 1e-8)


class TestLogisticMatchesR:
    def test_logistic_deviance_matches_r(self):
        df = _mt_df()
        out = df.select(ps.augment("vs", "mpg", model="logistic", tol=1e-12).alias("a")).unnest(
            "a"
        )
        assert rel_close(out["fitted"].to_numpy(), ref.BIN_FITTED, 1e-6)
        assert rel_close(out["residual"].to_numpy(), ref.BIN_RESID_DEV, 1e-6)
        assert rel_close(out["std_residual"].to_numpy(), ref.BIN_RSTD_DEV, 1e-6)
        assert rel_close(out["stud_residual"].to_numpy(), ref.BIN_RSTUD, 1e-6)
        assert rel_close(out["leverage"].to_numpy(), ref.BIN_HAT, 1e-6)
        assert rel_close(out["cooks_d"].to_numpy(), ref.BIN_COOK, 1e-6)
        assert rel_close(out["dffits"].to_numpy(), ref.BIN_DFFITS, 1e-6)

    def test_logistic_pearson_matches_r(self):
        df = _mt_df()
        out = df.select(
            ps.augment(
                "vs", "mpg", model="logistic", residual_type="pearson", tol=1e-12
            ).alias("a")
        ).unnest("a")
        assert rel_close(out["residual"].to_numpy(), ref.BIN_RESID_PEAR, 1e-6)
        assert rel_close(out["std_residual"].to_numpy(), ref.BIN_RSTD_PEAR, 1e-6)


class TestPoissonMatchesR:
    def test_poisson_deviance_matches_r(self):
        df = _mt_df()
        out = df.select(
            ps.augment("carb", "wt", "hp", model="poisson", tol=1e-12).alias("a")
        ).unnest("a")
        assert rel_close(out["fitted"].to_numpy(), ref.POIS_FITTED, 1e-6)
        assert rel_close(out["residual"].to_numpy(), ref.POIS_RESID_DEV, 1e-6)
        assert rel_close(out["std_residual"].to_numpy(), ref.POIS_RSTD_DEV, 1e-6)
        assert rel_close(out["stud_residual"].to_numpy(), ref.POIS_RSTUD, 1e-6)
        assert rel_close(out["leverage"].to_numpy(), ref.POIS_HAT, 1e-6)
        assert rel_close(out["cooks_d"].to_numpy(), ref.POIS_COOK, 1e-6)
        assert rel_close(out["dffits"].to_numpy(), ref.POIS_DFFITS, 1e-6)

    def test_poisson_pearson_matches_r(self):
        df = _mt_df()
        out = df.select(
            ps.augment(
                "carb", "wt", "hp", model="poisson", residual_type="pearson", tol=1e-12
            ).alias("a")
        ).unnest("a")
        assert rel_close(out["residual"].to_numpy(), ref.POIS_RESID_PEAR, 1e-6)
        assert rel_close(out["std_residual"].to_numpy(), ref.POIS_RSTD_PEAR, 1e-6)


class TestGammaTweedieMatchesR:
    def test_gamma_deviance_matches_r(self):
        df = _mt_df()
        out = df.select(
            ps.augment("mpg", "wt", "hp", model="gamma", tol=1e-12, max_iter=100).alias("a")
        ).unnest("a")
        assert rel_close(out["fitted"].to_numpy(), ref.GAMMA_FITTED, 1e-6)
        assert rel_close(out["residual"].to_numpy(), ref.GAMMA_RESID_DEV, 1e-6)
        assert rel_close(out["std_residual"].to_numpy(), ref.GAMMA_RSTD_DEV, 1e-6)
        assert rel_close(out["stud_residual"].to_numpy(), ref.GAMMA_RSTUD, 1e-6)
        assert rel_close(out["leverage"].to_numpy(), ref.GAMMA_HAT, 1e-6)
        assert rel_close(out["cooks_d"].to_numpy(), ref.GAMMA_COOK, 1e-6)
        assert rel_close(out["dffits"].to_numpy(), ref.GAMMA_DFFITS, 1e-6)

    def test_tweedie_var_power_2_matches_gamma(self):
        # Only the log link (`link_power=0.0`) reproduces GammaRegressor's Cook's
        # distance at var_power=2.0 — the canonical-link default (var_power=2.0
        # implies link_power=1-2=-1, the inverse link) does not. This cross-check
        # exercises that explicitly; it is not the default behaviour of
        # `ps.augment(model="tweedie")` (see test_tweedie_default_link_matches_ps_tweedie).
        df = _mt_df()
        out = df.select(
            ps.augment(
                "mpg",
                "wt",
                "hp",
                model="tweedie",
                var_power=2.0,
                link_power=0.0,
                tol=1e-12,
                max_iter=100,
            ).alias("a")
        ).unnest("a")
        assert rel_close(out["cooks_d"].to_numpy(), ref.GAMMA_COOK, 1e-6)

        out_resp = df.select(
            ps.augment(
                "mpg",
                "wt",
                "hp",
                model="tweedie",
                var_power=2.0,
                link_power=0.0,
                tol=1e-12,
                max_iter=100,
                residual_type="response",
            ).alias("a")
        ).unnest("a")
        assert rel_close(
            out_resp["residual"].to_numpy(), MT_MPG - np.array(ref.GAMMA_FITTED), 1e-6
        )

    def test_tweedie_default_link_matches_ps_tweedie(self):
        # ps.augment(model="tweedie") with link_power=None (the default) must
        # describe the same model `ps.tweedie(...)`/`Tweedie(link_power=None)`
        # fits with the same arguments — not a hard-coded log link.
        df = _mt_df()
        var_power = 1.5
        out = df.select(
            ps.augment(
                "mpg", "wt", "hp", model="tweedie", var_power=var_power, tol=1e-12, max_iter=100
            ).alias("a")
        ).unnest("a")

        x = np.column_stack([MT_WT, MT_HP]).astype(np.float64)
        y = MT_MPG.astype(np.float64)
        model = Tweedie(var_power=var_power, link_power=None, max_iter=100, tol=1e-12)
        model.fit(x, y)
        expected_fitted = model.predict(x)

        assert rel_close(out["fitted"].to_numpy(), expected_fitted, 1e-8)


class TestNegativeBinomialMatchesR:
    def test_negative_binomial_deviance_matches_r(self):
        df = _wb_df()
        out = df.select(
            ps.augment(
                "breaks",
                "woolB",
                "tensionM",
                "tensionH",
                model="negative_binomial",
                tol=1e-12,
            ).alias("a")
        ).unnest("a")
        assert rel_close(out["fitted"].to_numpy(), ref.NB_FITTED, 1e-6)
        assert rel_close(out["residual"].to_numpy(), ref.NB_RESID_DEV, 1e-6)
        assert rel_close(out["std_residual"].to_numpy(), ref.NB_RSTD_DEV, 1e-6)
        assert rel_close(out["stud_residual"].to_numpy(), ref.NB_RSTUD, 1e-6)
        assert rel_close(out["leverage"].to_numpy(), ref.NB_HAT, 1e-6)
        assert rel_close(out["cooks_d"].to_numpy(), ref.NB_COOK, 1e-6)
        assert rel_close(out["dffits"].to_numpy(), ref.NB_DFFITS, 1e-6)

    def test_negative_binomial_fixed_theta_matches_estimated(self):
        df = _wb_df()
        est = df.select(
            ps.augment(
                "breaks",
                "woolB",
                "tensionM",
                "tensionH",
                model="negative_binomial",
                tol=1e-12,
            ).alias("a")
        ).unnest("a")
        fixed = df.select(
            ps.augment(
                "breaks",
                "woolB",
                "tensionM",
                "tensionH",
                model="negative_binomial",
                theta=float(ref.NB_THETA),
                tol=1e-12,
            ).alias("a")
        ).unnest("a")
        assert rel_close(fixed["fitted"].to_numpy(), est["fitted"].to_numpy(), 1e-4)


class TestProbitCloglogFinite:
    def test_probit_cloglog_leverage_and_response_residual(self):
        df = _mt_df()
        for model in ("probit", "cloglog"):
            out = df.select(
                ps.augment(
                    "vs", "mpg", model=model, tol=1e-12, residual_type="response"
                ).alias("a")
            ).unnest("a")
            for col in out.columns:
                assert np.all(
                    np.isfinite(out[col].to_numpy())
                ), f"{model}.{col} has non-finite values"
            assert out["leverage"].to_numpy().sum() == pytest.approx(2.0, abs=1e-8)
            assert rel_close(
                out["residual"].to_numpy(), MT_VS - out["fitted"].to_numpy(), 1e-12
            )


class TestNaAlignment:
    @pytest.mark.parametrize("model", ["ols", "wls", "poisson"])
    def test_null_row_becomes_nan_rest_unchanged(self, model):
        df = _mt_df()
        kwargs = {"weights": "w"} if model == "wls" else {}
        y_col = "carb" if model == "poisson" else "mpg"

        # The "run without it" reference: the same fit with row 0 dropped
        # entirely (not merely nulled) — dropping a row changes the fit for
        # every remaining row, so this (not the full-dataset fit) is the
        # correct comparison.
        without_row0 = (
            df.slice(1)
            .select(ps.augment(y_col, "wt", "hp", model=model, **kwargs).alias("a"))
            .unnest("a")
        )

        # A null y row is excluded from the fit; the rest equal the
        # row-dropped run.
        null_df = df.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(None)
            .otherwise(pl.col(y_col))
            .alias(y_col)
        )
        null_out = null_df.select(
            ps.augment(y_col, "wt", "hp", model=model, **kwargs).alias("a")
        ).unnest("a")
        assert np.all(np.isnan(null_out.to_numpy()[0, :].astype(float)))
        for col in without_row0.columns:
            assert rel_close(null_out[col][1:].to_numpy(), without_row0[col].to_numpy(), 1e-12)

        # A NaN x row is excluded from the fit too; same comparison.
        nan_df = df.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(float("nan"))
            .otherwise(pl.col("wt"))
            .alias("wt")
        )
        nan_out = nan_df.select(
            ps.augment(y_col, "wt", "hp", model=model, **kwargs).alias("a")
        ).unnest("a")
        assert np.all(np.isnan(nan_out.to_numpy()[0, :].astype(float)))
        for col in without_row0.columns:
            assert rel_close(nan_out[col][1:].to_numpy(), without_row0[col].to_numpy(), 1e-12)


class TestOverGroupBy:
    def test_over_matches_per_subset_runs(self):
        df = _mt_df()
        g_a = df.with_columns(pl.lit("a").alias("g"))
        g_b = df.with_columns((pl.col("mpg") + 1.0).alias("mpg"), pl.lit("b").alias("g"))
        stacked = pl.concat([g_a, g_b])

        over_out = stacked.with_columns(
            ps.augment("mpg", "wt", "hp").over("g").alias("a")
        ).unnest("a")

        subset_a = df.select(ps.augment("mpg", "wt", "hp").alias("a")).unnest("a")
        subset_b = (
            df.with_columns((pl.col("mpg") + 1.0).alias("mpg"))
            .select(ps.augment("mpg", "wt", "hp").alias("a"))
            .unnest("a")
        )

        a_rows = over_out.filter(pl.col("g") == "a")
        b_rows = over_out.filter(pl.col("g") == "b")
        fields = [
            "fitted",
            "residual",
            "std_residual",
            "stud_residual",
            "leverage",
            "cooks_d",
            "dffits",
        ]
        for col in fields:
            assert rel_close(a_rows[col].to_numpy(), subset_a[col].to_numpy(), 1e-8)
            assert rel_close(b_rows[col].to_numpy(), subset_b[col].to_numpy(), 1e-8)

    def test_group_by_agg_returns_list_of_structs(self):
        df = _mt_df()
        g_a = df.with_columns(pl.lit("a").alias("g"))
        g_b = df.with_columns((pl.col("mpg") + 1.0).alias("mpg"), pl.lit("b").alias("g"))
        stacked = pl.concat([g_a, g_b])

        grouped = stacked.group_by("g", maintain_order=True).agg(
            ps.augment("mpg", "wt", "hp").alias("a")
        )
        grouped_a = grouped.filter(pl.col("g") == "a")["a"][0]
        grouped_b = grouped.filter(pl.col("g") == "b")["a"][0]
        assert len(grouped_a) == df.height
        assert len(grouped_b) == df.height


class TestWlsPymodelMatchesR:
    def _x(self):
        return np.column_stack([MT_WT, MT_HP])

    def test_wls_pymodel_default_weights_matches_r(self):
        model = WLS().fit(self._x(), MT_MPG, MT_W)
        d = model.augment(self._x(), MT_MPG)

        assert d["resid_type"] == "response"
        assert rel_close(d["fitted"], ref.WLS_FITTED, 1e-8)
        assert rel_close(d["residual"], ref.WLS_RESID, 1e-8)
        assert rel_close(d["std_residual"], ref.WLS_RSTD, 1e-8)
        assert rel_close(d["stud_residual"], ref.WLS_RSTUD, 1e-8)
        assert rel_close(d["leverage"], ref.WLS_HAT, 1e-8)
        assert rel_close(d["cooks_d"], ref.WLS_COOK, 1e-8)
        assert rel_close(d["dffits"], ref.WLS_DFFITS, 1e-8)

    def test_wls_pymodel_explicit_weights_identical(self):
        model = WLS().fit(self._x(), MT_MPG, MT_W)
        d_default = model.augment(self._x(), MT_MPG)
        d_explicit = model.augment(self._x(), MT_MPG, weights=MT_W)
        for key in d_default:
            if key == "resid_type":
                assert d_default[key] == d_explicit[key]
            else:
                assert rel_close(d_default[key], d_explicit[key], 1e-12)


class TestPymodelExpressionParity:
    def test_ols_pymodel_matches_expression(self):
        x = np.column_stack([MT_WT, MT_HP])
        model = OLS().fit(x, MT_MPG)
        d = model.augment(x, MT_MPG)

        df = _mt_df()
        expr_out = df.select(ps.augment("mpg", "wt", "hp").alias("a")).unnest("a")
        for key in ("fitted", "residual", "std_residual", "stud_residual", "leverage", "cooks_d", "dffits"):
            assert rel_close(d[key], expr_out[key].to_numpy(), 1e-12)

    def test_wls_pymodel_matches_expression(self):
        x = np.column_stack([MT_WT, MT_HP])
        model = WLS().fit(x, MT_MPG, MT_W)
        d = model.augment(x, MT_MPG)

        df = _mt_df()
        expr_out = df.select(
            ps.augment("mpg", "wt", "hp", model="wls", weights="w").alias("a")
        ).unnest("a")
        for key in ("fitted", "residual", "std_residual", "stud_residual", "leverage", "cooks_d", "dffits"):
            assert rel_close(d[key], expr_out[key].to_numpy(), 1e-12)


class TestWlsPymodelErrors:
    def test_wls_unfitted_augment_raises_runtime_error_naming_class(self):
        model = WLS()
        x = np.column_stack([MT_WT, MT_HP])
        with pytest.raises(RuntimeError, match="WLS"):
            model.augment(x, MT_MPG)

    def test_wls_augment_mismatched_length_raises_value_error(self):
        model = WLS().fit(np.column_stack([MT_WT, MT_HP]), MT_MPG, MT_W)
        x = np.column_stack([MT_WT, MT_HP])
        with pytest.raises(ValueError):
            model.augment(x, MT_MPG[:10])


class TestGlmPyModelAugment:
    """``.augment()`` on the eight GLM PyModel classes (16.3-06): R references
    (where the m1a fixture supplies them) and row-for-row parity with
    ``ps.augment(..., model=<same model>)`` at the same settings."""

    # -- Poisson (Task 1) --------------------------------------------------

    def test_poisson_pymodel_matches_r(self):
        x = np.column_stack([MT_WT, MT_HP])
        model = Poisson(max_iter=100, tol=1e-12).fit(x, MT_CARB)
        d = model.augment(x, MT_CARB)

        assert d["resid_type"] == "deviance"
        assert rel_close(d["fitted"], ref.POIS_FITTED, 1e-6)
        assert rel_close(d["residual"], ref.POIS_RESID_DEV, 1e-6)
        assert rel_close(d["std_residual"], ref.POIS_RSTD_DEV, 1e-6)
        assert rel_close(d["stud_residual"], ref.POIS_RSTUD, 1e-6)
        assert rel_close(d["leverage"], ref.POIS_HAT, 1e-6)
        assert rel_close(d["cooks_d"], ref.POIS_COOK, 1e-6)
        assert rel_close(d["dffits"], ref.POIS_DFFITS, 1e-6)

    def test_poisson_pymodel_pearson(self):
        x = np.column_stack([MT_WT, MT_HP])
        model = Poisson(max_iter=100, tol=1e-12).fit(x, MT_CARB)
        d = model.augment(x, MT_CARB, residual_type="pearson")

        assert d["resid_type"] == "pearson"
        assert rel_close(d["residual"], ref.POIS_RESID_PEAR, 1e-6)
        assert rel_close(d["std_residual"], ref.POIS_RSTD_PEAR, 1e-6)

    def test_poisson_pymodel_expression_parity(self):
        x = np.column_stack([MT_WT, MT_HP])
        model = Poisson(max_iter=100, tol=1e-12).fit(x, MT_CARB)
        d = model.augment(x, MT_CARB)

        df = _mt_df()
        expr_out = df.select(
            ps.augment("carb", "wt", "hp", model="poisson", tol=1e-12).alias("a")
        ).unnest("a")
        for key in (
            "fitted",
            "residual",
            "std_residual",
            "stud_residual",
            "leverage",
            "cooks_d",
            "dffits",
        ):
            assert rel_close(d[key], expr_out[key].to_numpy(), 1e-10)

    def test_poisson_pymodel_errors(self):
        x = np.column_stack([MT_WT, MT_HP])
        model = Poisson()
        with pytest.raises(RuntimeError, match="Poisson"):
            model.augment(x, MT_CARB)

        fitted_model = Poisson(max_iter=100, tol=1e-12).fit(x, MT_CARB)
        with pytest.raises(ValueError):
            fitted_model.augment(x, MT_CARB[:5])
        with pytest.raises(ValueError):
            fitted_model.augment(x, MT_CARB, residual_type="working")
