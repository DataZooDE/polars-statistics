"""Summary / predict completeness for Quantile, Isotonic, LmDynamic (issue #18).

Scope notes:
- `Quantile`: both summary and predict.
- `Isotonic`: predict only — no per-coefficient inference exists.
- `LmDynamic`: predict only — coefficients vary over time, so a single summary
  row would be misleading.
- `Aid`: neither — AidClassifier is not a fitted regressor and exposes no
  natural predict surface.
"""

import numpy as np
import polars as pl
import pytest

from polars_statistics import (
    isotonic_predict,
    lm_dynamic_predict,
    quantile,
    quantile_predict,
    quantile_summary,
)


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(size=n)
    y = 1.0 + 2.0 * x + rng.normal(scale=0.3, size=n)
    return pl.DataFrame({"y": y, "x": x})


class TestQuantileSummary:
    def _summary_df(self, df: pl.DataFrame) -> pl.DataFrame:
        """Materialize the List<Struct> output as a flat DataFrame."""
        return (
            df.select(quantile_summary("y", "x", tau=0.5).alias("s"))
            .explode("s")
            .unnest("s")
        )

    def test_returns_one_row_per_term(self, linear_data):
        flat = self._summary_df(linear_data)
        # intercept + x
        assert flat.shape[0] == 2
        assert set(flat.columns) >= {"term", "estimate", "std_error", "statistic", "p_value"}

    def test_std_errors_are_nan(self, linear_data):
        """Quantile regression has no analytic SEs — should be NaN."""
        flat = self._summary_df(linear_data)
        assert flat["std_error"].is_nan().all()
        assert flat["statistic"].is_nan().all()
        assert flat["p_value"].is_nan().all()

    def test_estimate_matches_quantile_fit(self, linear_data):
        """The summary's estimate column should match the base quantile fit."""
        base = linear_data.select(quantile("y", "x", tau=0.5).alias("q")).item()
        flat = self._summary_df(linear_data)
        estimates = flat["estimate"].to_list()
        assert estimates[0] == pytest.approx(base["intercept"], rel=1e-6)
        assert estimates[1] == pytest.approx(base["coefficients"][0], rel=1e-6)


class TestQuantilePredict:
    def test_predict_shape(self, linear_data):
        out = linear_data.with_columns(
            quantile_predict("y", "x", tau=0.5).alias("pred")
        )
        assert out.shape[0] == linear_data.shape[0]

    def test_predict_lower_upper_nan(self, linear_data):
        out = linear_data.with_columns(
            quantile_predict("y", "x", tau=0.5).alias("pred")
        ).unnest("pred")
        # Interval columns should all be NaN — no analytic intervals.
        assert out["quantile_lower"].is_null().sum() + out[
            "quantile_lower"
        ].is_nan().sum() == out.shape[0]

    def test_median_predict_close_to_truth(self, linear_data):
        out = linear_data.with_columns(
            quantile_predict("y", "x", tau=0.5).alias("pred")
        ).unnest("pred")
        # Predictions should track 1 + 2*x within reason.
        residuals = (out["y"] - out["quantile_prediction"]).to_numpy()
        assert abs(np.mean(residuals)) < 0.5

    def test_custom_name_prefix(self, linear_data):
        out = linear_data.with_columns(
            quantile_predict("y", "x", tau=0.5, name="q50").alias("pred")
        ).unnest("pred")
        assert "q50_prediction" in out.columns


class TestIsotonicPredict:
    def test_predict_monotonic(self):
        """Predictions on monotone data should be monotone."""
        rng = np.random.default_rng(7)
        n = 100
        x = np.linspace(-2.0, 2.0, n)
        y = np.tanh(x) + rng.normal(scale=0.05, size=n)
        df = pl.DataFrame({"y": y, "x": x})
        out = df.with_columns(
            isotonic_predict("y", "x", increasing=True).alias("pred")
        ).unnest("pred")
        preds = out["isotonic_prediction"].to_numpy()
        # Drop NaNs (rows where the predict mask might exclude data isn't an issue here)
        preds = preds[~np.isnan(preds)]
        assert (np.diff(preds) >= -1e-12).all(), "expected non-decreasing predictions"

    def test_interval_columns_nan(self):
        rng = np.random.default_rng(8)
        n = 50
        x = np.linspace(0.0, 1.0, n)
        y = x**2 + rng.normal(scale=0.01, size=n)
        df = pl.DataFrame({"y": y, "x": x})
        out = df.with_columns(
            isotonic_predict("y", "x", increasing=True).alias("pred")
        ).unnest("pred")
        assert (
            out["isotonic_lower"].is_nan().sum()
            + out["isotonic_lower"].is_null().sum()
            == out.shape[0]
        )


class TestLmDynamicPredict:
    def test_predict_shape(self):
        """LmDynamic predict should produce one prediction per row."""
        rng = np.random.default_rng(9)
        n = 150
        x = rng.normal(size=n)
        y = 0.5 + 1.5 * x + rng.normal(scale=0.2, size=n)
        df = pl.DataFrame({"y": y, "x": x})
        out = df.with_columns(
            lm_dynamic_predict(
                "y", "x", ic="aicc", distribution="normal", lowess_span=0.0
            ).alias("pred")
        ).unnest("pred")
        assert out.shape[0] == n
        # Predictions should be finite for at least most rows.
        n_finite = out["lm_dynamic_prediction"].is_finite().sum()
        assert n_finite > n // 2

    def test_predict_intervals_nan(self):
        rng = np.random.default_rng(10)
        n = 100
        x = rng.normal(size=n)
        y = 1.0 + x + rng.normal(scale=0.3, size=n)
        df = pl.DataFrame({"y": y, "x": x})
        out = df.with_columns(
            lm_dynamic_predict(
                "y", "x", lowess_span=0.0, distribution="normal"
            ).alias("pred")
        ).unnest("pred")
        # Intervals always NaN.
        assert (
            out["lm_dynamic_lower"].is_nan().sum()
            + out["lm_dynamic_lower"].is_null().sum()
            == n
        )
