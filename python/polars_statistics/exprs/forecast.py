"""Forecast comparison test expressions."""

from __future__ import annotations
from pathlib import Path
from typing import Literal, Union

import polars as pl
from polars.plugins import register_plugin_function

LIB = Path(__file__).parent.parent


def _to_expr(x: Union[pl.Expr, str]) -> pl.Expr:
    """Convert string column name to expression."""
    if isinstance(x, str):
        return pl.col(x)
    return x


def diebold_mariano(
    e1: Union[pl.Expr, str],
    e2: Union[pl.Expr, str],
    loss: Literal["squared", "absolute"] = "squared",
    horizon: int = 1,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    varestimator: Literal["acf", "bartlett"] = "acf",
) -> pl.Expr:
    """Diebold-Mariano test for comparing forecast accuracy.

    Tests the null hypothesis that two forecasts have equal predictive accuracy.

    Parameters
    ----------
    e1 : pl.Expr
        Forecast errors from model 1.
    e2 : pl.Expr
        Forecast errors from model 2.
    loss : {"squared", "absolute"}, default "squared"
        Loss function to use.
    horizon : int, default 1
        Forecast horizon (for variance adjustment).
    alternative : {"two-sided", "less", "greater"}, default "two-sided"
        Alternative hypothesis direction.
    varestimator : {"acf", "bartlett"}, default "acf"
        Variance estimator to use.

    Returns
    -------
    pl.Expr
        Struct containing 'statistic' and 'p_value'.

    References
    ----------
    Diebold, F.X. and Mariano, R.S. (1995) "Comparing Predictive Accuracy"
    """
    e1_clean = _to_expr(e1).cast(pl.Float64)
    e2_clean = _to_expr(e2).cast(pl.Float64)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_diebold_mariano",
        args=[
            e1_clean,
            e2_clean,
            pl.lit(loss, dtype=pl.String),
            pl.lit(horizon, dtype=pl.UInt32),
            pl.lit(alternative, dtype=pl.String),
            pl.lit(varestimator, dtype=pl.String),
        ],
        returns_scalar=True,
    )


def permutation_t_test(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    n_permutations: int = 999,
    seed: int | None = None,
) -> pl.Expr:
    """Permutation t-test for comparing two samples.

    Non-parametric alternative to the t-test that makes no distributional assumptions.

    Parameters
    ----------
    x : pl.Expr
        First sample.
    y : pl.Expr
        Second sample.
    alternative : {"two-sided", "less", "greater"}, default "two-sided"
        Alternative hypothesis.
    n_permutations : int, default 999
        Number of permutations.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pl.Expr
        Struct containing 'statistic' and 'p_value'.
    """
    x_clean = _to_expr(x).cast(pl.Float64)
    y_clean = _to_expr(y).cast(pl.Float64)

    seed_expr = pl.lit(seed, dtype=pl.UInt64) if seed is not None else pl.lit(None, dtype=pl.UInt64)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_permutation_t_test",
        args=[
            x_clean,
            y_clean,
            pl.lit(alternative, dtype=pl.String),
            pl.lit(n_permutations, dtype=pl.UInt32),
            seed_expr,
        ],
        returns_scalar=True,
    )


def clark_west(
    e1: Union[pl.Expr, str],
    e2: Union[pl.Expr, str],
    horizon: int = 1,
) -> pl.Expr:
    """Clark-West test for nested model comparison.

    Tests whether a larger (unrestricted) model provides significantly
    better forecasts than a smaller (restricted) nested model.

    Parameters
    ----------
    e1 : pl.Expr
        Forecast errors from the restricted (smaller) model.
    e2 : pl.Expr
        Forecast errors from the unrestricted (larger) model.
    horizon : int, default 1
        Forecast horizon (for HAC variance adjustment).

    Returns
    -------
    pl.Expr
        Struct containing 'statistic' and 'p_value'.

    References
    ----------
    Clark, T.E. and West, K.D. (2007) "Approximately Normal Tests for Equal
    Predictive Accuracy in Nested Models"
    """
    e1_clean = _to_expr(e1).cast(pl.Float64)
    e2_clean = _to_expr(e2).cast(pl.Float64)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_clark_west",
        args=[
            e1_clean,
            e2_clean,
            pl.lit(horizon, dtype=pl.UInt32),
        ],
        returns_scalar=True,
    )


def spa_test(
    benchmark: Union[pl.Expr, str],
    *models: Union[pl.Expr, str],
    n_bootstrap: int = 999,
    block_length: float = 5.0,
    seed: int | None = None,
) -> pl.Expr:
    """Superior Predictive Ability (SPA) test.

    Tests whether any alternative model significantly outperforms a benchmark.

    Parameters
    ----------
    benchmark : pl.Expr
        Losses from the benchmark model.
    *models : pl.Expr
        Losses from alternative models (one or more).
    n_bootstrap : int, default 999
        Number of bootstrap replications.
    block_length : float, default 5.0
        Average block length for stationary bootstrap.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pl.Expr
        Struct containing 'statistic', 'p_value_consistent', 'p_value_upper', 'best_model_idx'.

    References
    ----------
    Hansen, P.R. (2005) "A Test for Superior Predictive Ability"
    """
    benchmark_clean = _to_expr(benchmark).cast(pl.Float64)
    seed_expr = pl.lit(seed, dtype=pl.UInt64) if seed is not None else pl.lit(None, dtype=pl.UInt64)

    model_args = [_to_expr(m).cast(pl.Float64) for m in models]

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_spa_test",
        args=[
            benchmark_clean,
            pl.lit(n_bootstrap, dtype=pl.UInt32),
            pl.lit(block_length, dtype=pl.Float64),
            seed_expr,
            *model_args,
        ],
        returns_scalar=True,
    )


def model_confidence_set(
    *models: Union[pl.Expr, str],
    alpha: float = 0.1,
    statistic: Literal["range", "max"] = "range",
    n_bootstrap: int = 999,
    block_length: float = 5.0,
    seed: int | None = None,
) -> pl.Expr:
    """Model Confidence Set (MCS) procedure.

    Identifies a set of models that contains the best model(s) with
    a given confidence level.

    Parameters
    ----------
    *models : pl.Expr
        Losses from each candidate model (two or more).
    alpha : float, default 0.1
        Significance level (1-alpha confidence).
    statistic : {"range", "max"}, default "range"
        Test statistic type.
    n_bootstrap : int, default 999
        Number of bootstrap replications.
    block_length : float, default 5.0
        Average block length for stationary bootstrap.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pl.Expr
        Struct containing 'included_models' (list of indices) and 'mcs_p_value'.

    References
    ----------
    Hansen, Lunde, and Nason (2011) "The Model Confidence Set"
    """
    seed_expr = pl.lit(seed, dtype=pl.UInt64) if seed is not None else pl.lit(None, dtype=pl.UInt64)
    model_args = [_to_expr(m).cast(pl.Float64) for m in models]

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_model_confidence_set",
        args=[
            pl.lit(alpha, dtype=pl.Float64),
            pl.lit(statistic, dtype=pl.String),
            pl.lit(n_bootstrap, dtype=pl.UInt32),
            pl.lit(block_length, dtype=pl.Float64),
            seed_expr,
            *model_args,
        ],
        returns_scalar=True,
    )


def mspe_adjusted(
    benchmark: Union[pl.Expr, str],
    *models: Union[pl.Expr, str],
    n_bootstrap: int = 999,
    block_length: float = 5.0,
    seed: int | None = None,
) -> pl.Expr:
    """MSPE-Adjusted SPA test for nested models.

    A variant of the SPA test that adjusts for the MSPE bias when comparing
    nested models.

    Parameters
    ----------
    benchmark : pl.Expr
        Errors from the benchmark model.
    *models : pl.Expr
        Errors from alternative nested models.
    n_bootstrap : int, default 999
        Number of bootstrap replications.
    block_length : float, default 5.0
        Average block length for stationary bootstrap.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pl.Expr
        Struct containing 'statistic', 'p_value_consistent', 'p_value_upper', 'best_model_idx'.
    """
    benchmark_clean = _to_expr(benchmark).cast(pl.Float64)
    seed_expr = pl.lit(seed, dtype=pl.UInt64) if seed is not None else pl.lit(None, dtype=pl.UInt64)

    model_args = [_to_expr(m).cast(pl.Float64) for m in models]

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_mspe_adjusted",
        args=[
            benchmark_clean,
            pl.lit(n_bootstrap, dtype=pl.UInt32),
            pl.lit(block_length, dtype=pl.Float64),
            seed_expr,
            *model_args,
        ],
        returns_scalar=True,
    )
