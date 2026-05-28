"""Tests for the PLS wrapper (issue #19)."""

import numpy as np
import polars as pl
import pytest

from polars_statistics import PLS, pls


@pytest.fixture
def collinear_data():
    """Three highly-correlated features driving y — PLS's natural use case."""
    rng = np.random.default_rng(0)
    n = 200
    x_base = rng.normal(size=(n, 1))
    x = np.hstack(
        [
            x_base,
            x_base + 0.05 * rng.normal(size=(n, 1)),
            x_base + 0.05 * rng.normal(size=(n, 1)),
        ]
    )
    y = 1.0 + 2.0 * x[:, 0] + rng.normal(scale=0.1, size=n)
    return x, y


class TestPLSClass:
    def test_fit_recovers_intercept_and_explains_variance(self, collinear_data):
        x, y = collinear_data
        m = PLS(n_components=1).fit(x, y)
        assert m.is_fitted()
        # Intercept ~ 1.0 (true value)
        assert abs(m.intercept - 1.0) < 0.2
        # R^2 should be very high — features are essentially x_base
        assert m.r_squared > 0.95
        # n_components matches the request
        assert m.n_components == 1

    def test_explained_variance_ratio_shape(self, collinear_data):
        x, y = collinear_data
        m = PLS(n_components=2).fit(x, y)
        evr = m.explained_variance_ratio
        assert evr.shape == (2,)
        # Each ratio in [0, 1]
        assert (evr >= 0).all() and (evr <= 1).all()

    def test_predict_shape(self, collinear_data):
        x, y = collinear_data
        m = PLS().fit(x, y)
        preds = m.predict(x)
        assert preds.shape == (200,)

    def test_transform_to_latent_space(self, collinear_data):
        x, y = collinear_data
        m = PLS(n_components=2).fit(x, y)
        scores = m.transform(x)
        assert scores.shape == (200, 2)

    def test_n_components_default(self, collinear_data):
        x, y = collinear_data
        m = PLS().fit(x, y)
        assert m.n_components == 2

    def test_n_observations(self, collinear_data):
        x, y = collinear_data
        m = PLS().fit(x, y)
        assert m.n_observations == 200


class TestPLSExpression:
    def test_matches_class_api(self, collinear_data):
        x, y = collinear_data
        m = PLS(n_components=1).fit(x, y)
        df = pl.DataFrame({"y": y, "x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2]})
        result = df.select(
            pls("y", "x1", "x2", "x3", n_components=1).alias("m")
        ).item()
        assert result["intercept"] == pytest.approx(m.intercept, rel=1e-5)
        for got, want in zip(result["coefficients"], m.coefficients):
            assert got == pytest.approx(want, rel=1e-5)

    def test_group_by_works(self):
        rng = np.random.default_rng(1)
        n_per = 100
        x_a = rng.normal(size=n_per)
        x_b = rng.normal(size=n_per)
        df = pl.DataFrame(
            {
                "site": ["A"] * n_per + ["B"] * n_per,
                "y": np.concatenate(
                    [1.0 + 2.0 * x_a, 3.0 - 1.0 * x_b]
                ),
                "x1": np.concatenate([x_a, x_b]),
            }
        )
        out = df.group_by("site").agg(pls("y", "x1", n_components=1).alias("m"))
        assert out.shape[0] == 2
