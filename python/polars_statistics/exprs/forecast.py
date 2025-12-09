"""Forecast comparison test expressions."""

from __future__ import annotations
from pathlib import Path
from typing import Literal

import polars as pl
from polars.plugins import register_plugin_function

LIB = Path(__file__).parent.parent


def diebold_mariano(
    e1: pl.Expr,
    e2: pl.Expr,
    loss: Literal["squared", "absolute"] = "squared",
    horizon: int = 1,
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

    Returns
    -------
    pl.Expr
        Struct containing 'statistic' and 'p_value'.

    References
    ----------
    Diebold, F.X. and Mariano, R.S. (1995) "Comparing Predictive Accuracy"
    """
    e1_clean = e1.cast(pl.Float64)
    e2_clean = e2.cast(pl.Float64)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_diebold_mariano",
        args=[
            e1_clean,
            e2_clean,
            pl.lit(loss, dtype=pl.String),
            pl.lit(horizon, dtype=pl.UInt32),
        ],
        returns_scalar=True,
    )


def permutation_t_test(
    x: pl.Expr,
    y: pl.Expr,
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
    x_clean = x.cast(pl.Float64)
    y_clean = y.cast(pl.Float64)

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
