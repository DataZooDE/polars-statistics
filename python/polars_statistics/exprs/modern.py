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


def energy_distance_nd(
    x_cols: list,
    y_cols: list,
    n_permutations: int = 999,
    seed: int | None = None,
) -> pl.Expr:
    """Multi-dimensional Energy Distance test for comparing multivariate distributions.

    A permutation test based on the energy distance statistic generalised to d-dimensional
    observations. Each observation is represented by d feature columns (one per dimension).
    The test detects differences in location, scale, and shape between two multivariate samples.

    Parameters
    ----------
    x_cols : list of (pl.Expr or str)
        Feature columns for sample X. Length must equal len(y_cols) (the dimension d).
    y_cols : list of (pl.Expr or str)
        Feature columns for sample Y. Must be the same length as x_cols.
    n_permutations : int, default 999
        Number of permutations for the permutation test.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pl.Expr
        Struct containing 'statistic' and 'p_value'.

    Raises
    ------
    ValueError
        If len(x_cols) != len(y_cols).

    References
    ----------
    Szekely, G.J. and Rizzo, M.L. (2004) "Testing for Equal Distributions in High Dimension"

    Examples
    --------
    >>> import polars as pl
    >>> from polars_statistics.exprs.modern import energy_distance_nd
    >>> df = pl.DataFrame({
    ...     "x1": [0.0, 0.1, -0.1], "x2": [0.0, -0.1, 0.1],
    ...     "y1": [3.0, 3.1, 2.9],  "y2": [3.0, 2.9, 3.1],
    ... })
    >>> result = df.select(energy_distance_nd(["x1", "x2"], ["y1", "y2"], seed=42))
    """
    if len(x_cols) != len(y_cols):
        raise ValueError(
            f"x_cols and y_cols must have the same length (dimension d), "
            f"got len(x_cols)={len(x_cols)} and len(y_cols)={len(y_cols)}"
        )
    d = len(x_cols)
    x_exprs = [_to_expr(c).cast(pl.Float64) for c in x_cols]
    y_exprs = [_to_expr(c).cast(pl.Float64) for c in y_cols]

    # Apply shared finite mask across all x columns, and separately across all y columns,
    # so that rows stay aligned within each sample (pitfall 1: per-column filtering skews rows).
    x_mask = pl.all_horizontal([col.is_finite() for col in x_exprs])
    y_mask = pl.all_horizontal([col.is_finite() for col in y_exprs])
    x_clean = [col.filter(x_mask) for col in x_exprs]
    y_clean = [col.filter(y_mask) for col in y_exprs]

    seed_expr = pl.lit(seed, dtype=pl.UInt64) if seed is not None else pl.lit(None, dtype=pl.UInt64)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_energy_distance_nd",
        args=[
            pl.lit(d, dtype=pl.UInt32),
            pl.lit(n_permutations, dtype=pl.UInt32),
            seed_expr,
            *x_clean,
            *y_clean,
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
