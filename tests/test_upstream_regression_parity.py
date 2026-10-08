"""R-reference parity tests for anofox-regression 0.5.14/0.5.15 behaviour changes.

Each test pins a value from this plan's R reference computation (R 4.6.1 / MASS /
lme4) so a future regression in the backing `anofox-regression` crate cannot pass
silently. See `.planning/phases/16.1-upstream-crate-refresh/16.1-RESEARCH.md`
("Runtime/Behaviour-Change Inventory") for the upstream changelog entries these
tests encode:

- #35 (0.5.14): RANSAC `n_observations`/`n_inliers`/inference now describe the
  inlier-only fit, not the full sample.
- #37 (0.5.14): Ridge inference for `lambda_ > 0` is a sandwich SE with no
  t-test (covered separately in `tests/test_models.py::TestRidge`).
- #38 (0.5.14): No-intercept R2/adjusted-R2/F now use uncentred TSS.
- #39 (0.5.14): A constant column without an intercept is no longer aliased to
  NaN — it behaves exactly like an intercept term.
- #41 (0.5.14): OLS `log_likelihood`/`aic`/`bic` now match R's `logLik`/`AIC`/`BIC`.
- #42 (0.5.14): NegBin/Poisson `aic`/`bic` now match R's `glm`/`glm.nb` values.
- #43 (0.5.14): Gamma GLM dispersion now matches R's `summary.glm` Pearson
  dispersion estimate.
- #49 (0.5.15): NegBin standard errors default to dispersion=1 (previously
  scaled by the Pearson chi-square/df when > 1).
- #50 (0.5.15): Theil-Sen on a rank-deficient (collinear) design aliases the
  redundant coefficient to NaN instead of reporting a ~1e14 coefficient.
"""

import math

import numpy as np
import polars as pl
import polars_statistics as ps
import pytest
from polars_statistics import OLS, RANSAC, Gamma, NegativeBinomial, Poisson, TheilSen

# Shared 30-observation dataset for the NegBin dispersion=1 (#49) reference test.
Z1 = [
    0.63, -0.79, -0.44, -0.34, 0.7, 0.52, 0.88, 0.08, -0.42, -0.57,
    -0.84, 0.66, 0.49, -0.66, -0.27, -0.19, 0.05, -0.92, 0.31, -0.52,
    0.66, -0.99, 0.42, 0.91, 0.06, 0.55, 0.79, 0.2, -0.93, 0.13,
]
Z2 = [
    1.68, 0.53, 0.07, 0.42, 1.87, 0.08, 1.07, 0.33, 1.41, 1.73,
    0.6, 1.9, 0.58, 1.11, 0.81, 1.43, 0.08, 0.38, 1.36, 0.97,
    1.17, 1.32, 0.7, 1.26, 0.45, 1.35, 0.97, 1.77, 0.16, 0.63,
]
NB = [
    0, 2, 0, 2, 0, 1, 4, 5, 2, 7,
    2, 4, 2, 5, 0, 8, 8, 2, 7, 2,
    9, 0, 10, 6, 3, 3, 2, 7, 0, 10,
]


def _d15() -> pl.DataFrame:
    """Shared n=15 dataset for the OLS logLik/AIC/BIC reference (#41)."""
    i = list(range(1, 16))
    x0 = [k / 3 for k in i]
    x1 = [((k * 5) % 7) / 2 for k in i]
    y = [2 + 1.5 * a - 0.8 * b + (((k * 3) % 5) - 2) * 0.5 for k, a, b in zip(i, x0, x1)]
    return pl.DataFrame({"y": y, "x0": x0, "x1": x1})


def test_ols_loglik_aic_bic_match_r():
    """anofox-regression 0.5.14 #41: OLS log_likelihood/aic/bic match R's logLik/AIC/BIC."""
    df = _d15()
    X = df.select(["x0", "x1"]).to_numpy()
    y = df["y"].to_numpy()

    model = OLS().fit(X, y)

    assert model.log_likelihood == pytest.approx(-15.875039497906823, abs=1e-9)
    assert model.aic == pytest.approx(39.750078995813645, abs=1e-8)
    assert model.bic == pytest.approx(42.582279800222487, abs=1e-8)


def test_ols_no_intercept_uncentred_r2_matches_r():
    """anofox-regression 0.5.14 #38: no-intercept OLS R2/adj-R2/F use uncentred TSS."""
    x = np.arange(1, 11, dtype=float)
    y = np.array([3.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1, 18.2, 19.9]) + 5
    X = x.reshape(-1, 1)

    model = OLS(add_intercept=False).fit(X, y)

    assert model.r_squared == pytest.approx(0.97794411910884449, abs=1e-9)
    assert model.adj_r_squared == pytest.approx(0.97549346567649386, abs=1e-9)
    assert model.f_statistic == pytest.approx(399.05443429870200589, abs=1e-6)


def test_wls_no_intercept_uncentred_r2_matches_r():
    """anofox-regression 0.5.14 #38: the uncentred-TSS fix also applies to WLS."""
    x = np.arange(1, 11, dtype=float)
    y = np.array([3.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1, 18.2, 19.9]) + 5
    weights = [1.0 if i % 2 == 0 else 2.0 for i in range(10)]
    df = pl.DataFrame({"y": y.tolist(), "x": x.tolist(), "w": weights})

    result = df.select(ps.wls("y", "w", "x", add_intercept=False).alias("m"))["m"][0]

    assert result["r_squared"] == pytest.approx(0.98089334384990512, abs=1e-9)
    assert result["adj_r_squared"] == pytest.approx(0.97877038205545008, abs=1e-9)
    assert result["f_statistic"] == pytest.approx(462.04003595915855840, abs=1e-6)


def test_ols_constant_column_without_intercept_is_the_intercept():
    """anofox-regression 0.5.14 #39: a constant column without add_intercept is no
    longer aliased to NaN — it behaves exactly like an intercept term."""
    rng = np.random.RandomState(0)
    x = np.arange(1, 21, dtype=float)
    y = 2.0 + 3.0 * x + rng.randn(20) * 0.01
    X = np.column_stack([np.ones(20), x])

    model_no_intercept = OLS(add_intercept=False).fit(X, y)
    model_with_intercept = OLS().fit(x.reshape(-1, 1), y)

    assert not math.isnan(model_no_intercept.coefficients[0])
    expected = [model_with_intercept.intercept, model_with_intercept.coefficients[0]]
    np.testing.assert_allclose(model_no_intercept.coefficients, expected, atol=1e-8)


def test_ransac_reports_inlier_fit_statistics():
    """anofox-regression 0.5.14 #35: n_observations now reports the inlier count
    the fit statistics were computed on; residuals still covers every row."""
    i = np.arange(1, 21, dtype=float)
    y = 1 + 2 * i + (((i * 7).astype(int) % 5) - 2) * 0.3
    for k in (4, 11, 17):
        y[k - 1] += 25.0
    X = i.reshape(-1, 1)

    model = RANSAC(residual_threshold=2.0, random_state=42).fit(X, y)

    assert model.n_inliers == 17
    assert model.n_observations == 17
    assert model.intercept == pytest.approx(1.0909975669099783, abs=1e-8)
    assert model.coefficients[0] == pytest.approx(1.9862530413625301, abs=1e-8)
    assert len(model.residuals) == 20


def test_negbin_standard_errors_use_dispersion_one():
    """anofox-regression 0.5.15 #49: NegBin SEs default to dispersion=1."""
    X = np.column_stack([Z1, Z2])
    y = np.array(NB, dtype=float)

    model = NegativeBinomial(theta=10.0, estimate_theta=False, tol=1e-12).fit(X, y)

    se = model.std_errors
    pv = model.p_values
    assert se is not None and pv is not None
    np.testing.assert_allclose(se, [2.005140224941e-01, 2.087673148773e-01], rtol=1e-5)
    np.testing.assert_allclose(pv, [5.403953102318e-02, 5.830720242408e-01], rtol=1e-5)


def test_negbin_and_poisson_loglik_aic_match_r():
    """anofox-regression 0.5.14 #42: NegBin/Poisson aic/bic match R's glm/glm.nb."""
    x = np.arange(1, 11, dtype=float)
    y = np.array([0, 1, 0, 3, 2, 0, 5, 1, 4, 0], dtype=float)
    X = x.reshape(-1, 1)

    fixed_theta = NegativeBinomial(theta=2.0, estimate_theta=False).fit(X, y)
    assert fixed_theta.aic is not None
    assert fixed_theta.aic == pytest.approx(37.750971989658396, abs=1e-5)

    estimated_theta = NegativeBinomial().fit(X, y)
    assert estimated_theta.aic is not None and estimated_theta.bic is not None
    assert estimated_theta.aic == pytest.approx(39.648903878105529, abs=1e-4)
    assert estimated_theta.bic == pytest.approx(40.556659157087665, abs=1e-4)

    poisson_model = Poisson().fit(X, y)
    assert poisson_model.aic == pytest.approx(40.137278555218266, abs=1e-5)


def test_gamma_pearson_dispersion_matches_summary_glm():
    """anofox-regression 0.5.14 #43: Gamma dispersion matches R's summary.glm
    Pearson dispersion estimate."""
    n = 40
    x = np.array([i / 10 for i in range(n)])
    y = np.array([1 + ((i * 7) % 5) + i / 10 for i in range(n)])
    X = x.reshape(-1, 1)

    model = Gamma().fit(X, y)

    assert model.std_errors is not None
    assert model.coefficients[0] == pytest.approx(0.22705616796414885, abs=1e-5)
    assert model.std_errors[0] == pytest.approx(0.045137636902525077, abs=1e-6)
    assert model.aic is not None
    assert model.aic == pytest.approx(155.559639764314568, abs=1e-4)


def test_glm_constant_column_with_intercept_is_aliased():
    """A constant column alongside add_intercept=True is genuinely rank-deficient
    (two constant columns) and must alias to NaN, unlike the #39 no-intercept case."""
    x = np.arange(1, 11, dtype=float)
    y = np.array([0, 1, 0, 3, 2, 0, 5, 1, 4, 0], dtype=float)
    X = np.column_stack([x, np.ones(10)])

    model = Poisson().fit(X, y)

    assert not math.isnan(model.coefficients[0])
    assert math.isnan(model.coefficients[1])


def test_theil_sen_collinear_column_is_aliased():
    """anofox-regression 0.5.15 #50: Theil-Sen on a rank-deficient (collinear)
    design aliases the redundant coefficient to NaN instead of reporting a
    ~1e14 coefficient."""
    i = np.arange(1, 41)
    x1 = (i % 7) + 0.5
    x2 = 2 * x1
    y = 3 + 0.8 * x1 - 0.3 * (((3 * i) % 11) / 2) + ((37 * i) % 13) / 13 - 0.5
    X = np.column_stack([x1, x2])

    model = TheilSen().fit(X, y)

    assert math.isnan(model.coefficients[1])
    assert model.coefficients[0] == pytest.approx(0.7692307692307692, abs=1e-9)
    assert model.intercept == pytest.approx(2.265384615384616, abs=1e-9)
