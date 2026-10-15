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


class TestHuberInference:
    """R reference: MASS::rlm(y ~ x, psi = psi.huber, k = 1.35, scale.est = "MAD"),
    summary(method = "XtX"); anofox-regression tests/issue36_huber_inference.rs."""

    def _data(self):
        xv = np.array([k + 1 for k in range(20)], dtype=float)
        y = np.array(
            [1 + 2 * x + (((x * 7) % 5) - 2) * 0.4 for x in xv], dtype=float
        )
        y[5] = 40.0
        y[14] = 5.0
        return xv.reshape(-1, 1), y

    def test_default_inference_is_none(self):
        x, y = self._data()
        m = Huber().fit(x, y)
        assert m.std_errors is None
        assert m.t_statistics is None
        assert m.p_values is None
        assert m.conf_interval_lower is None
        assert m.conf_interval_upper is None
        assert m.intercept_std_error is None
        assert m.intercept_t_statistic is None
        assert m.intercept_p_value is None
        assert m.intercept_conf_interval is None

    def test_compute_inference_matches_mass_rlm(self):
        x, y = self._data()
        m = Huber(
            epsilon=1.35, alpha=0.0, max_iter=500, tol=1e-12, compute_inference=True
        ).fit(x, y)

        assert m.coefficients[0] == pytest.approx(1.9769296831089778, abs=1e-6)
        assert m.intercept == pytest.approx(1.2866827718001757, abs=1e-5)
        assert m.std_errors[0] == pytest.approx(0.029647934601485383, abs=1e-6)
        assert m.intercept_std_error == pytest.approx(
            0.355157012812956197, abs=1e-5
        )
        assert m.t_statistics[0] == pytest.approx(
            m.coefficients[0] / m.std_errors[0], rel=1e-9
        )
        # |t| ~= 66.7 with df=18 underflows the two-sided p-value to exactly 0.0
        # in f64 (astronomically significant) rather than a representable
        # positive float — assert the valid-probability range, not strict > 0.
        assert 0.0 <= m.p_values[0] < 1.0

    def test_confidence_level_narrows_interval(self):
        x, y = self._data()
        m95 = Huber(
            epsilon=1.35, alpha=0.0, max_iter=500, tol=1e-12, compute_inference=True
        ).fit(x, y)
        m90 = Huber(
            epsilon=1.35,
            alpha=0.0,
            max_iter=500,
            tol=1e-12,
            compute_inference=True,
            confidence_level=0.90,
        ).fit(x, y)

        width95 = m95.conf_interval_upper[0] - m95.conf_interval_lower[0]
        width90 = m90.conf_interval_upper[0] - m90.conf_interval_lower[0]
        assert width90 < width95

        mid95 = (m95.conf_interval_upper[0] + m95.conf_interval_lower[0]) / 2
        mid90 = (m90.conf_interval_upper[0] + m90.conf_interval_lower[0]) / 2
        assert mid95 == pytest.approx(m95.coefficients[0], abs=1e-9)
        assert mid90 == pytest.approx(m90.coefficients[0], abs=1e-9)

    def test_unfitted_inference_getters_raise(self):
        m = Huber()
        for attr in (
            "std_errors",
            "t_statistics",
            "p_values",
            "conf_interval_lower",
            "conf_interval_upper",
            "intercept_std_error",
            "intercept_t_statistic",
            "intercept_p_value",
            "intercept_conf_interval",
        ):
            with pytest.raises(Exception):
                getattr(m, attr)


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
