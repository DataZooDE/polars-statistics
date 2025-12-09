"""Modern statistical test expressions (distribution comparison)."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import polars as pl
from polars.plugins import register_plugin_function

LIB = Path(__file__).parent.parent


def _to_expr(x: Union[pl.Expr, str]) -> pl.Expr:
    """Convert string column name to expression."""
    if isinstance(x, str):
        return pl.col(x)
    return x


def energy_distance(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    n_permutations: int = 999,
    seed: int | None = None,
) -> pl.Expr:
    """Energy Distance test for comparing distributions.

    A permutation test based on the energy distance statistic, which measures
    the distance between two distributions. It is capable of detecting
    differences in location, scale, and shape.

    Parameters
    ----------
    x : pl.Expr
        First sample.
    y : pl.Expr
        Second sample.
    n_permutations : int, default 999
        Number of permutations.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pl.Expr
        Struct containing 'statistic' and 'p_value'.

    References
    ----------
    Szekely, G.J. and Rizzo, M.L. (2004) "Testing for Equal Distributions in
    High Dimension"
    """
    x_clean = _to_expr(x).cast(pl.Float64)
    y_clean = _to_expr(y).cast(pl.Float64)

    seed_expr = pl.lit(seed, dtype=pl.UInt64) if seed is not None else pl.lit(None, dtype=pl.UInt64)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_energy_distance",
        args=[
            x_clean,
            y_clean,
            pl.lit(n_permutations, dtype=pl.UInt32),
            seed_expr,
        ],
        returns_scalar=True,
    )


def mmd_test(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    n_permutations: int = 999,
    seed: int | None = None,
) -> pl.Expr:
    """Maximum Mean Discrepancy (MMD) test for comparing distributions.

    A kernel-based two-sample test that uses the maximum mean discrepancy
    statistic with a Gaussian kernel (using median heuristic bandwidth).

    Parameters
    ----------
    x : pl.Expr
        First sample.
    y : pl.Expr
        Second sample.
    n_permutations : int, default 999
        Number of permutations.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pl.Expr
        Struct containing 'statistic' and 'p_value'.

    References
    ----------
    Gretton, A. et al. (2012) "A Kernel Two-Sample Test"
    """
    x_clean = _to_expr(x).cast(pl.Float64)
    y_clean = _to_expr(y).cast(pl.Float64)

    seed_expr = pl.lit(seed, dtype=pl.UInt64) if seed is not None else pl.lit(None, dtype=pl.UInt64)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_mmd_test",
        args=[
            x_clean,
            y_clean,
            pl.lit(n_permutations, dtype=pl.UInt32),
            seed_expr,
        ],
        returns_scalar=True,
    )
