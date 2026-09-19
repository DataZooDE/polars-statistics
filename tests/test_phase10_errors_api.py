"""Phase 10: contextual errors + API consistency (ERR-01/02, API-01/02)."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import polars_statistics as ps
import pytest
from polars_statistics import (
    OLS,
    ElasticNet,
    Huber,
    Logistic,
    LogisticRegression,
    Probit,
    Quantile,
    Ridge,
    TheilSen,
)

# ---------------------------------------------------------------------------
# ERR-01: unfitted error names the model + the .fit call
# ---------------------------------------------------------------------------


class TestUnfittedNamesModel:
    @pytest.mark.parametrize(
        "ctor,name",
        [
            (OLS, "OLS"),
            (Ridge, "Ridge"),
            (ElasticNet, "ElasticNet"),
            (Huber, "Huber"),
            (Quantile, "Quantile"),
            (TheilSen, "TheilSen"),
            (Logistic, "Logistic"),
            (LogisticRegression, "LogisticRegression"),
        ],
    )
    def test_getter_raises_named_runtime_error(self, ctor, name):
        model = ctor()
        with pytest.raises(RuntimeError) as excinfo:
            _ = model.coefficients
        msg = str(excinfo.value)
        assert name in msg, f"error should name the model, got: {msg!r}"
        assert ".fit(" in msg, f"error should mention .fit(...), got: {msg!r}"

    def test_predict_raises_named_runtime_error(self):
        model = Ridge()
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(RuntimeError) as excinfo:
            model.predict(x)
        assert "Ridge is not fitted" in str(excinfo.value)

    def test_score_raises_named_runtime_error(self):
        model = OLS()
        x = np.array([[1.0], [2.0]])
        y = np.array([1.0, 2.0])
        with pytest.raises(RuntimeError) as excinfo:
            model.score(x, y)
        assert "OLS is not fitted" in str(excinfo.value)


# ---------------------------------------------------------------------------
# ERR-02: shape mismatch / degenerate input -> actionable ValueError
# ---------------------------------------------------------------------------


class TestShapeValidation:
    def test_row_mismatch_raises_valueerror(self):
        x = np.random.rand(10, 2)
        y = np.random.rand(8)
        with pytest.raises(ValueError) as excinfo:
            OLS().fit(x, y)
        msg = str(excinfo.value)
        assert "10" in msg and "8" in msg
        assert "OLS" in msg

    def test_empty_input_raises_valueerror(self):
        x = np.empty((0, 2))
        y = np.empty((0,))
        with pytest.raises(ValueError) as excinfo:
            Ridge().fit(x, y)
        assert "empty" in str(excinfo.value).lower()

    def test_too_few_samples_raises_valueerror(self):
        # 2 rows, 3 features -> rank deficient by construction
        x = np.random.rand(2, 3)
        y = np.random.rand(2)
        with pytest.raises(ValueError) as excinfo:
            OLS().fit(x, y)
        msg = str(excinfo.value)
        assert "2" in msg and "3" in msg

    def test_valid_shapes_fit_ok(self):
        x = np.random.rand(20, 2)
        y = np.random.rand(20)
        model = OLS().fit(x, y)
        assert model.is_fitted()


# ---------------------------------------------------------------------------
# API-01: add_intercept / with_intercept deprecation (classes AND exprs)
# ---------------------------------------------------------------------------


class TestInterceptDeprecationClasses:
    def test_with_intercept_emits_future_warning(self):
        with pytest.warns(FutureWarning, match="with_intercept"):
            OLS(with_intercept=False)

    def test_add_intercept_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = OLS(add_intercept=False)
        assert model is not None

    def test_both_kwargs_raises(self):
        with pytest.raises(ValueError, match="both"):
            OLS(add_intercept=True, with_intercept=True)

    def test_default_is_intercept_true(self):
        x = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([3.0, 5.0, 7.0, 9.0])  # y = 2x + 1
        model = OLS().fit(x, y)
        assert model.intercept is not None
        assert abs(model.intercept - 1.0) < 1e-6

    def test_add_intercept_false_drops_intercept(self):
        x = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0])
        model = OLS(add_intercept=False).fit(x, y)
        assert model.intercept is None

    def test_static_factory_deprecation(self):
        # ALM factory methods and GLMM factories also route through the helper.
        from polars_statistics import ALM

        with pytest.warns(FutureWarning, match="with_intercept"):
            ALM.normal(with_intercept=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ALM.normal(add_intercept=True)


class TestInterceptDeprecationExprs:
    def test_expr_with_intercept_warns(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0]})
        with pytest.warns(FutureWarning, match="with_intercept"):
            df.select(ps.ols("y", "x", with_intercept=True))

    def test_expr_both_kwargs_raises(self):
        with pytest.raises(ValueError, match="both"):
            ps.ols("y", "x", add_intercept=True, with_intercept=True)

    def test_expr_add_intercept_no_warning(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0]})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            df.select(ps.ols("y", "x", add_intercept=True))


# ---------------------------------------------------------------------------
# API-02: uniform fit/predict/score with correct semantics
# ---------------------------------------------------------------------------


class TestScoreSemantics:
    def test_regressor_score_is_r_squared(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(100, 2))
        beta = np.array([2.0, -1.0])
        y = x @ beta + 3.0 + rng.normal(scale=0.01, size=100)
        model = OLS().fit(x, y)
        r2 = model.score(x, y)
        assert 0.99 < r2 <= 1.0

    def test_perfect_fit_score_one(self):
        x = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([3.0, 5.0, 7.0, 9.0])
        model = OLS().fit(x, y)
        assert abs(model.score(x, y) - 1.0) < 1e-9

    def test_classifier_score_is_accuracy(self):
        # Perfectly separable-ish data; accuracy should be high and in [0, 1].
        rng = np.random.default_rng(1)
        x0 = rng.normal(loc=-2.0, size=(50, 1))
        x1 = rng.normal(loc=2.0, size=(50, 1))
        x = np.vstack([x0, x1])
        y = np.array([0.0] * 50 + [1.0] * 50)
        model = LogisticRegression().fit(x, y)
        acc = model.score(x, y)
        assert 0.0 <= acc <= 1.0
        assert acc > 0.9

    def test_logistic_expr_class_score_accuracy(self):
        # Overlapping classes so IRLS converges (perfect separation diverges).
        rng = np.random.default_rng(2)
        x = np.vstack([rng.normal(-1.0, size=(40, 1)), rng.normal(1.0, size=(40, 1))])
        y = np.array([0.0] * 40 + [1.0] * 40)
        model = Logistic(lambda_=1.0).fit(x, y)
        acc = model.score(x, y)
        assert 0.0 <= acc <= 1.0

    def test_score_available_across_regressors(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(50, 2))
        y = x @ np.array([1.0, 2.0]) + rng.normal(scale=0.1, size=50)
        for ctor in (OLS, Ridge, ElasticNet, Huber, Quantile, TheilSen):
            model = ctor().fit(x, y)
            s = model.score(x, y)
            assert isinstance(s, float)

    def test_probit_score_accuracy(self):
        rng = np.random.default_rng(4)
        x = np.vstack([rng.normal(-1.0, size=(40, 1)), rng.normal(1.0, size=(40, 1))])
        y = np.array([0.0] * 40 + [1.0] * 40)
        model = Probit(lambda_=1.0).fit(x, y)
        acc = model.score(x, y)
        assert 0.0 <= acc <= 1.0
