"""Smoke tests for Gamma GLM diagnostic expressions and Ridge/WLS HC inference.

These tests assert that the new expressions produce finite, structurally correct
outputs on a small positive-response design.  They are executed by the Plan 06
phase gate (maturin develop + pytest).

Note: ``import polars_statistics as ps`` loads the compiled extension; if the
extension has not been built yet the imports will fail — the CI gate runs
``maturin develop`` first.
"""

import numpy as np
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _gamma_df(n: int = 60, p: int = 2) -> pl.DataFrame:
    """Small Gamma-distributed design: positive y, two predictors."""
    X = RNG.standard_normal((n, p))
    eta = 0.5 + X @ np.array([0.8, -0.4])
    y = RNG.gamma(shape=2.0, scale=np.exp(eta) / 2.0)
    data = {"y": y.tolist()}
    for j in range(p):
        data[f"x{j}"] = X[:, j].tolist()
    return pl.DataFrame(data)


DF = _gamma_df()


# ---------------------------------------------------------------------------
# Task 1 — Gamma GLM dispersion (deviance method)
# ---------------------------------------------------------------------------


class TestGammaDispersionDeviance:
    def test_scalar_finite(self):
        ps = pytest.importorskip("polars_statistics")
        result = DF.select(
            ps.gamma_dispersion_deviance("y", "x0", "x1").alias("d")
        )
        val = result["d"].struct.field("dispersion")[0]
        assert np.isfinite(float(val)), f"Expected finite dispersion, got {val}"

    def test_positive(self):
        ps = pytest.importorskip("polars_statistics")
        result = DF.select(
            ps.gamma_dispersion_deviance("y", "x0", "x1").alias("d")
        )
        val = float(result["d"].struct.field("dispersion")[0])
        assert val > 0.0, f"Dispersion must be positive, got {val}"


# ---------------------------------------------------------------------------
# Task 1 — Gamma GLM dispersion (Pearson method)
# ---------------------------------------------------------------------------


class TestGammaDispersionPearson:
    def test_scalar_finite(self):
        ps = pytest.importorskip("polars_statistics")
        result = DF.select(
            ps.gamma_dispersion_pearson("y", "x0", "x1").alias("d")
        )
        val = result["d"].struct.field("dispersion")[0]
        assert np.isfinite(float(val)), f"Expected finite dispersion, got {val}"

    def test_positive(self):
        ps = pytest.importorskip("polars_statistics")
        result = DF.select(
            ps.gamma_dispersion_pearson("y", "x0", "x1").alias("d")
        )
        val = float(result["d"].struct.field("dispersion")[0])
        assert val > 0.0, f"Dispersion must be positive, got {val}"


# ---------------------------------------------------------------------------
# Task 1 — Gamma Pearson chi-squared goodness-of-fit
# ---------------------------------------------------------------------------


class TestGammaPearsonChiSquared:
    def test_struct_fields(self):
        ps = pytest.importorskip("polars_statistics")
        result = DF.select(
            ps.gamma_pearson_chi_squared("y", "x0", "x1").alias("s")
        )
        chi2 = float(result["s"].struct.field("chi_squared")[0])
        df_resid = int(result["s"].struct.field("df_resid")[0])
        n_obs = int(result["s"].struct.field("n_observations")[0])
        assert np.isfinite(chi2) and chi2 > 0.0
        assert df_resid > 0
        assert n_obs == len(DF)


# ---------------------------------------------------------------------------
# Task 1 — Gamma standardized Pearson residuals
# ---------------------------------------------------------------------------


class TestGammaStandardizedPearsonResiduals:
    def test_residual_count(self):
        ps = pytest.importorskip("polars_statistics")
        result = DF.select(
            ps.gamma_standardized_pearson_residuals("y", "x0", "x1").alias("r")
        )
        resids = result["r"].struct.field("residuals")[0].to_list()
        assert len(resids) == len(DF), "One residual per observation expected"

    def test_residuals_finite(self):
        ps = pytest.importorskip("polars_statistics")
        result = DF.select(
            ps.gamma_standardized_pearson_residuals("y", "x0", "x1").alias("r")
        )
        resids = np.array(result["r"].struct.field("residuals")[0].to_list())
        assert np.all(np.isfinite(resids)), "All standardized Pearson residuals must be finite"


# ---------------------------------------------------------------------------
# Task 1 — Gamma standardized deviance residuals
# ---------------------------------------------------------------------------


class TestGammaStandardizedDevianceResiduals:
    def test_residual_count(self):
        ps = pytest.importorskip("polars_statistics")
        result = DF.select(
            ps.gamma_standardized_deviance_residuals("y", "x0", "x1").alias("r")
        )
        resids = result["r"].struct.field("residuals")[0].to_list()
        assert len(resids) == len(DF), "One residual per observation expected"

    def test_residuals_finite(self):
        ps = pytest.importorskip("polars_statistics")
        result = DF.select(
            ps.gamma_standardized_deviance_residuals("y", "x0", "x1").alias("r")
        )
        resids = np.array(result["r"].struct.field("residuals")[0].to_list())
        assert np.all(np.isfinite(resids)), "All standardized deviance residuals must be finite"


# ---------------------------------------------------------------------------
# Task 2 — Ridge hc_inference
# ---------------------------------------------------------------------------


class TestRidgeHcInference:
    def _xy(self, n: int = 50, p: int = 2):
        X = RNG.standard_normal((n, p))
        # length-p coefficient so p != 2 still conforms (linspace(1,-0.5,2) == [1,-0.5])
        y = X @ np.linspace(1.0, -0.5, p) + 0.1 * RNG.standard_normal(n)
        return X, y

    def test_std_errors_finite(self):
        Ridge = pytest.importorskip("polars_statistics").Ridge
        X, y = self._xy()
        model = Ridge(lambda_=0.01).fit(X, y)
        hc = model.hc_inference(X)
        assert np.all(np.isfinite(hc["std_errors"])), "HC std_errors must be finite"

    def test_dict_keys_present(self):
        Ridge = pytest.importorskip("polars_statistics").Ridge
        X, y = self._xy()
        hc = Ridge(lambda_=0.01).fit(X, y).hc_inference(X, hc_type="hc3")
        required = {"std_errors", "t_statistics", "p_values",
                    "conf_interval_lower", "conf_interval_upper"}
        assert required.issubset(hc.keys()), f"Missing keys: {required - hc.keys()}"

    def test_shape_matches_features(self):
        Ridge = pytest.importorskip("polars_statistics").Ridge
        X, y = self._xy(p=3)
        hc = Ridge(lambda_=0.1).fit(X, y).hc_inference(X)
        assert len(hc["std_errors"]) == 3

    def test_not_fitted_raises(self):
        Ridge = pytest.importorskip("polars_statistics").Ridge
        X, _ = self._xy()
        with pytest.raises(Exception):
            Ridge().hc_inference(X)


# ---------------------------------------------------------------------------
# Task 2 — WLS hc_inference
# ---------------------------------------------------------------------------


class TestWlsHcInference:
    def _xyw(self, n: int = 50, p: int = 2):
        X = RNG.standard_normal((n, p))
        # length-p coefficient so p != 2 still conforms (linspace(2,-1,2) == [2,-1])
        y = X @ np.linspace(2.0, -1.0, p) + 0.2 * RNG.standard_normal(n)
        weights = np.abs(RNG.standard_normal(n)) + 0.5
        return X, y, weights

    def test_std_errors_finite(self):
        WLS = pytest.importorskip("polars_statistics").WLS
        X, y, w = self._xyw()
        model = WLS().fit(X, y, w)
        hc = model.hc_inference(X)
        assert np.all(np.isfinite(hc["std_errors"])), "HC std_errors must be finite"

    def test_dict_keys_present(self):
        WLS = pytest.importorskip("polars_statistics").WLS
        X, y, w = self._xyw()
        hc = WLS().fit(X, y, w).hc_inference(X, hc_type="hc2")
        required = {"std_errors", "t_statistics", "p_values",
                    "conf_interval_lower", "conf_interval_upper"}
        assert required.issubset(hc.keys())

    def test_shape_matches_features(self):
        WLS = pytest.importorskip("polars_statistics").WLS
        X, y, w = self._xyw(p=4)
        hc = WLS().fit(X, y, w).hc_inference(X)
        assert len(hc["std_errors"]) == 4

    def test_not_fitted_raises(self):
        WLS = pytest.importorskip("polars_statistics").WLS
        X, _, _ = self._xyw()
        with pytest.raises(Exception):
            WLS().hc_inference(X)
