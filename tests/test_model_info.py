"""Tests for ``model_info`` (UPST-09, anofox-regression 0.5.20 ``HasModelInfo`` /
``ModelInfo``).

Every fitted ``PyModel`` whose upstream fitted type implements
``HasModelInfo`` exposes a ``model_info`` getter returning
``{"model_type": str, "family": str | None, "link": str}``. The expected
strings are sourced verbatim from anofox-regression's own upstream test
suite (``tests/model_info.rs`` in the anofox-regression 0.5.22 source), not
invented here.

``CASES`` is built up across 16.3-08's three tasks: Task 1 adds OLS /
Poisson / GLMM (one linear model, one GLM, the mixed model); Task 2 adds the
16 linear / robust / sparse / smoother classes; Task 3 adds the remaining 11
GLM / ALM / dynamic / penalized-GLM / AFT classes, for 30 total.
"""

from __future__ import annotations

import numpy as np
import pytest
from polars_statistics import (
    ARD,
    BLS,
    GLMM,
    LARS,
    OLS,
    PLS,
    RANSAC,
    RLS,
    WLS,
    BayesianRidge,
    ElasticNet,
    Huber,
    Isotonic,
    PassiveAggressive,
    Poisson,
    PSpline,
    Quantile,
    Ridge,
    TheilSen,
)


def _reg_xy():
    """Continuous positive-ish response, 3 features, n=60."""
    rng = np.random.RandomState(0)
    x = rng.randn(60, 3)
    y = x @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.randn(60)
    return x, y


def _count_xy():
    """Count response, 2 features, n=80."""
    rng = np.random.RandomState(4)
    x = rng.randn(80, 2) * 0.3
    lam = np.exp(0.5 + x @ np.array([0.3, -0.2]))
    y = rng.poisson(lam).astype(float)
    return x, y


def _group_xy_gaussian():
    """Grouped continuous response, 2 features, 6 groups of 10 (n=60)."""
    rng = np.random.RandomState(5)
    x = rng.randn(60, 2)
    group = list(np.repeat(np.arange(6), 10))
    y = x @ np.array([1.0, -0.5]) + np.array(group) * 0.2 + rng.randn(60) * 0.3
    return x, y, group


def _group_xy_poisson():
    """Grouped count response, 2 features, 6 groups of 10 (n=60)."""
    rng = np.random.RandomState(7)
    x = rng.randn(60, 2) * 0.3
    group = list(np.repeat(np.arange(6), 10))
    lam = np.exp(0.5 + x @ np.array([0.2, -0.1]))
    y = rng.poisson(lam).astype(float)
    return x, y, group


def _group_xy_binomial():
    """Grouped binary response, 2 features, 6 groups of 10 (n=60)."""
    rng = np.random.RandomState(8)
    x = rng.randn(60, 2)
    group = list(np.repeat(np.arange(6), 10))
    logits = x @ np.array([0.8, -0.5])
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.rand(60) < probs).astype(float)
    return x, y, group


def _group_xy_negative_binomial():
    """Grouped overdispersed count response, 2 features, 6 groups of 10 (n=60)."""
    rng = np.random.RandomState(9)
    x = rng.randn(60, 2) * 0.3
    group = list(np.repeat(np.arange(6), 10))
    lam = np.exp(0.5 + x @ np.array([0.2, -0.1]))
    y = rng.poisson(lam).astype(float) + rng.poisson(1.0, size=60).astype(float)
    return x, y, group


def _group_xy_gamma():
    """Grouped positive continuous response, 2 features, 6 groups of 10 (n=60)."""
    rng = np.random.RandomState(10)
    x = rng.randn(60, 2) * 0.3
    group = list(np.repeat(np.arange(6), 10))
    y = np.exp(0.5 + x @ np.array([0.2, -0.1]) + 0.1 * rng.randn(60))
    return x, y, group


def _reg_xy_1f():
    """Continuous response, single feature, n=60 (Isotonic / PSpline)."""
    rng = np.random.RandomState(11)
    x = rng.randn(60, 1)
    y = 2.0 * x[:, 0] + 0.1 * rng.randn(60)
    return x, y


def _wls_xyw():
    """Continuous response + positive weights, 3 features, n=60."""
    rng = np.random.RandomState(12)
    x, y = _reg_xy()
    weights = 1.0 + rng.rand(60)
    return x, y, weights


# (case_id, factory, expected_dict)
CASES: list[tuple[str, object, dict]] = [
    (
        "ols",
        lambda: OLS().fit(*_reg_xy()),
        {"model_type": "ols", "family": "gaussian", "link": "identity"},
    ),
    (
        "poisson",
        lambda: Poisson().fit(*_count_xy()),
        {"model_type": "poisson", "family": "poisson", "link": "log"},
    ),
    (
        "glmm_gaussian",
        lambda: GLMM.gaussian().fit(*_group_xy_gaussian()),
        {"model_type": "glmm", "family": "gaussian", "link": "identity"},
    ),
    (
        "glmm_poisson",
        lambda: GLMM.poisson().fit(*_group_xy_poisson()),
        {"model_type": "glmm", "family": "poisson", "link": "log"},
    ),
    (
        "glmm_binomial",
        lambda: GLMM.binomial().fit(*_group_xy_binomial()),
        {"model_type": "glmm", "family": "binomial", "link": "logit"},
    ),
    (
        "glmm_negative_binomial",
        lambda: GLMM.negative_binomial().fit(*_group_xy_negative_binomial()),
        {"model_type": "glmm", "family": "negative_binomial", "link": "log"},
    ),
    (
        "glmm_gamma",
        lambda: GLMM.gamma().fit(*_group_xy_gamma()),
        {"model_type": "glmm", "family": "gamma", "link": "log"},
    ),
    # Task 2: linear, robust, sparse and smoother classes (16).
    (
        "wls",
        lambda: WLS().fit(*_wls_xyw()),
        {"model_type": "wls", "family": "gaussian", "link": "identity"},
    ),
    (
        "ridge",
        lambda: Ridge().fit(*_reg_xy()),
        {"model_type": "ridge", "family": "gaussian", "link": "identity"},
    ),
    (
        "elastic_net",
        lambda: ElasticNet().fit(*_reg_xy()),
        {"model_type": "elastic_net", "family": "gaussian", "link": "identity"},
    ),
    (
        "rls",
        lambda: RLS().fit(*_reg_xy()),
        {"model_type": "rls", "family": "gaussian", "link": "identity"},
    ),
    (
        "bls_nnls",
        lambda: BLS.nnls().fit(*_reg_xy()),
        {"model_type": "nnls", "family": "gaussian", "link": "identity"},
    ),
    (
        "bls_bounded",
        lambda: BLS(lower_bound=0.0, upper_bound=10.0).fit(*_reg_xy()),
        {"model_type": "bls", "family": "gaussian", "link": "identity"},
    ),
    (
        "pls",
        lambda: PLS().fit(*_reg_xy()),
        {"model_type": "pls", "family": "gaussian", "link": "identity"},
    ),
    (
        "quantile",
        lambda: Quantile().fit(*_reg_xy()),
        {"model_type": "quantile", "family": None, "link": "identity"},
    ),
    (
        "isotonic",
        lambda: Isotonic().fit(_reg_xy_1f()[0][:, 0], _reg_xy_1f()[1]),
        {"model_type": "isotonic", "family": None, "link": "identity"},
    ),
    (
        "huber",
        lambda: Huber().fit(*_reg_xy()),
        {"model_type": "huber", "family": None, "link": "identity"},
    ),
    (
        "theil_sen",
        lambda: TheilSen().fit(*_reg_xy()),
        {"model_type": "theil_sen", "family": None, "link": "identity"},
    ),
    (
        "ransac",
        lambda: RANSAC(random_state=1).fit(*_reg_xy()),
        {"model_type": "ransac", "family": None, "link": "identity"},
    ),
    (
        "bayesian_ridge",
        lambda: BayesianRidge().fit(*_reg_xy()),
        {"model_type": "bayesian_ridge", "family": "gaussian", "link": "identity"},
    ),
    (
        "ard",
        lambda: ARD().fit(*_reg_xy()),
        {"model_type": "ard", "family": "gaussian", "link": "identity"},
    ),
    (
        "lars",
        lambda: LARS().fit(*_reg_xy()),
        {"model_type": "lars", "family": "gaussian", "link": "identity"},
    ),
    (
        "passive_aggressive",
        lambda: PassiveAggressive().fit(*_reg_xy()),
        {"model_type": "passive_aggressive", "family": None, "link": "identity"},
    ),
    (
        "pspline",
        lambda: PSpline().fit(*_reg_xy_1f()),
        {"model_type": "pspline", "family": "gaussian", "link": "identity"},
    ),
]

# Classes exercised by test_model_info_unfitted_raises: (class_name, factory).
_UNFITTED: list[tuple[str, object]] = [
    ("OLS", OLS),
    ("Poisson", Poisson),
    ("GLMM", GLMM.gaussian),
    ("WLS", WLS),
    ("Ridge", Ridge),
    ("ElasticNet", ElasticNet),
    ("RLS", RLS),
    ("BLS", BLS),
    ("PLS", PLS),
    ("Quantile", Quantile),
    ("Isotonic", Isotonic),
    ("Huber", Huber),
    ("TheilSen", TheilSen),
    ("RANSAC", RANSAC),
    ("BayesianRidge", BayesianRidge),
    ("ARD", ARD),
    ("LARS", LARS),
    ("PassiveAggressive", PassiveAggressive),
    ("PSpline", PSpline),
]


@pytest.mark.parametrize("case_id, factory, expected", CASES, ids=[c[0] for c in CASES])
def test_model_info_matches_upstream(case_id, factory, expected):
    model = factory()
    assert model.model_info == expected


@pytest.mark.parametrize("name, factory", _UNFITTED, ids=[c[0] for c in _UNFITTED])
def test_model_info_unfitted_raises(name, factory):
    model = factory()
    with pytest.raises(RuntimeError, match=name):
        _ = model.model_info


def test_model_info_in_to_dict():
    model = OLS().fit(*_reg_xy())
    assert "model_info" in model.to_dict()
    assert OLS().to_dict() == {"fitted": False}


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ElasticNet(),
        lambda: RLS(),
        lambda: BLS(),
    ],
    ids=["elastic_net", "rls", "bls"],
)
def test_model_info_boxed_refit_keeps_own_type(factory):
    """A type-erased (`Box<dyn FittedRegressor>`) class refitted keeps
    reporting its own `model_info` — proves the per-fit-site capture (16.3-08
    Task 2) isn't stale from a prior fit."""
    model = factory()
    x, y = _reg_xy()
    model.fit(x, y)
    first = model.model_info
    model.fit(x, y)
    second = model.model_info
    assert first == second
    assert second["model_type"] == first["model_type"]
