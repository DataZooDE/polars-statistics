"""Tests for Huber and sklearn-style LogisticRegression (issue #14)."""

import numpy as np
import polars as pl
import pytest

from polars_statistics import (
    Huber,
    LogisticRegression,
    huber,
    logistic_regression,
)


class TestHuber:
    def test_recovers_slope_with_outliers(self):
        """Huber should down-weight planted outliers and recover the true slope."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(size=(n, 1))
        y = 2.0 * x[:, 0] + 1.0 + rng.normal(scale=0.5, size=n)
        outliers = rng.choice(n, size=20, replace=False)
        y[outliers] += 20.0

        m = Huber().fit(x, y)
        assert m.is_fitted()
        assert abs(m.intercept - 1.0) < 0.5
        assert abs(m.coefficients[0] - 2.0) < 0.3

    def test_outlier_mask_marks_planted_outliers(self):
        rng = np.random.default_rng(0)
        n = 100
        x = rng.normal(size=(n, 1))
        y = 1.0 + 2.0 * x[:, 0] + rng.normal(scale=0.3, size=n)
        # Plant 10 huge outliers
        y[:10] += 50.0

        m = Huber().fit(x, y)
        assert m.n_outliers >= 10
        assert m.outliers.dtype == bool
        # The planted outliers should all be flagged.
        assert m.outliers[:10].all()

    def test_predict_shape(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(50, 2))
        y = x.sum(axis=1) + rng.normal(scale=0.1, size=50)
        m = Huber().fit(x, y)
        preds = m.predict(x)
        assert preds.shape == (50,)

    def test_scale_and_epsilon(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(80, 1))
        y = x[:, 0] + rng.normal(scale=0.5, size=80)
        m = Huber(epsilon=1.5).fit(x, y)
        assert m.epsilon == pytest.approx(1.5)
        assert m.scale > 0

    def test_expression_matches_class(self):
        """The huber polars expression should produce the same fit as the class."""
        rng = np.random.default_rng(3)
        n = 150
        x = rng.normal(size=n)
        y = 0.5 + 1.5 * x + rng.normal(scale=0.3, size=n)
        # Inject a few outliers
        y[::20] += 30.0

        m = Huber().fit(x.reshape(-1, 1), y)
        df = pl.DataFrame({"y": y, "x1": x})
        expr_result = df.select(huber("y", "x1").alias("h")).item()

        assert expr_result["intercept"] == pytest.approx(m.intercept, rel=1e-6)
        assert expr_result["coefficients"][0] == pytest.approx(
            m.coefficients[0], rel=1e-6
        )


class TestLogisticRegression:
    def _data(self, n=300, seed=42):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, 2))
        logit = 0.5 + 1.5 * x[:, 0] - 1.0 * x[:, 1]
        proba = 1.0 / (1.0 + np.exp(-logit))
        y = (proba > 0.5).astype(float)
        return x, y

    def test_high_accuracy_on_separable_data(self):
        x, y = self._data()
        lr = LogisticRegression(penalty="l2", C=10.0).fit(x, y)
        assert lr.score(x, y) > 0.95

    def test_predict_proba_in_unit_interval(self):
        x, y = self._data()
        lr = LogisticRegression().fit(x, y)
        proba = lr.predict_proba(x)
        assert proba.shape == (x.shape[0],)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_decision_function_is_logit(self):
        x, y = self._data()
        lr = LogisticRegression().fit(x, y)
        proba = lr.predict_proba(x)
        decision = lr.decision_function(x)
        # sigmoid(decision) should match proba
        expected = 1.0 / (1.0 + np.exp(-decision))
        np.testing.assert_allclose(expected, proba, rtol=1e-5)

    def test_predict_uses_threshold(self):
        x, y = self._data()
        lr = LogisticRegression(threshold=0.5).fit(x, y)
        proba = lr.predict_proba(x)
        preds = lr.predict(x)
        # Class labels should agree with proba >= threshold
        expected = (proba >= 0.5).astype(float)
        np.testing.assert_array_equal(preds, expected)

    def test_strong_l2_shrinks_coefficients(self):
        """Stronger L2 (smaller C) should shrink coefficient magnitudes."""
        x, y = self._data(n=100, seed=7)
        strong = LogisticRegression(penalty="l2", C=0.01).fit(x, y)
        weak = LogisticRegression(penalty="l2", C=100.0).fit(x, y)
        assert np.linalg.norm(strong.coefficients) < np.linalg.norm(weak.coefficients)

    def test_invalid_penalty_raises(self):
        x, y = self._data(n=50)
        with pytest.raises(ValueError, match="penalty"):
            LogisticRegression(penalty="l1").fit(x, y)

    def test_expression_matches_class(self):
        x, y = self._data(n=200, seed=9)
        lr = LogisticRegression(penalty="l2", C=10.0).fit(x, y)

        df = pl.DataFrame({"y": y, "x1": x[:, 0], "x2": x[:, 1]})
        expr_result = df.select(
            logistic_regression("y", "x1", "x2", penalty="l2", C=10.0).alias("lr")
        ).item()

        assert expr_result["intercept"] == pytest.approx(lr.intercept, rel=1e-5)
        for got, want in zip(expr_result["coefficients"], lr.coefficients):
            assert got == pytest.approx(want, rel=1e-5)

    def test_expression_groups_by(self):
        """logistic_regression should work inside group_by().agg()."""
        rng = np.random.default_rng(11)
        n_per_group = 100
        sites = ["A"] * n_per_group + ["B"] * n_per_group
        x1_a = rng.normal(size=n_per_group)
        x1_b = rng.normal(size=n_per_group)
        x1 = np.concatenate([x1_a, x1_b])
        # Different decision boundary per group
        logit_a = 1.0 + 2.0 * x1_a
        logit_b = -1.0 + 2.0 * x1_b
        y = np.concatenate(
            [
                (1.0 / (1.0 + np.exp(-logit_a)) > 0.5).astype(float),
                (1.0 / (1.0 + np.exp(-logit_b)) > 0.5).astype(float),
            ]
        )

        df = pl.DataFrame({"site": sites, "y": y, "x1": x1})
        out = df.group_by("site").agg(
            logistic_regression("y", "x1", penalty="l2", C=10.0).alias("model")
        )
        assert out.shape[0] == 2
