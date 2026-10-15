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
from polars_statistics import OLS

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


def _mt_df() -> pl.DataFrame:
    return pl.DataFrame({"mpg": MT_MPG, "wt": MT_WT, "hp": MT_HP})


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
