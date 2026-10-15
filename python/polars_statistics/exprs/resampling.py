"""Percentile bootstrap confidence intervals as Polars expressions."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl
from polars.plugins import register_plugin_function

LIB = Path(__file__).parent.parent


def bootstrap_mean_ci(
    x: pl.Expr | str,
    n_bootstrap: int = 1000,
    conf_level: float = 0.95,
    block_length: int = 0,
    seed: int | None = None,
) -> pl.Expr:
    """
    Percentile bootstrap confidence interval for the mean.

    Draws ``n_bootstrap`` resamples, computes the mean on each, and reports
    the percentile interval ``[R(floor(alpha/2*B)), R(min(ceil((1-alpha/2)*B), B-1))]``
    of the sorted replicates (0-based, ``alpha = 1 - conf_level``) together
    with the bootstrap standard error (the sample SD of the replicates).

    Parameters
    ----------
    x : pl.Expr or str
        Sample expression or column name.
    n_bootstrap : int, default 1000
        Number of bootstrap resamples.
    conf_level : float, default 0.95
        Confidence level for the percentile interval, in (0, 1).
    block_length : int, default 0
        Resampling scheme:

        - ``0``: IID (nonparametric) bootstrap — resamples of size n drawn
          with replacement, independently per observation.
        - ``1``: stationary bootstrap with expected block length 1.
        - ``> 1``: circular block bootstrap with that fixed block length,
          for dependent (e.g. time-series) data.
    seed : int, optional
        Random seed for reproducibility. Without one, each call draws fresh
        resamples (non-deterministic).

    Returns
    -------
    pl.Expr
        Expression returning struct{estimate: f64, se: f64, ci_lower: f64,
        ci_upper: f64, conf_level: f64, n_bootstrap: u32}.

    Notes
    -----
    Non-finite values are dropped before resampling. For ``block_length > 0``
    this closes the gaps left by dropped values, which changes which
    observations are adjacent for block resampling.

    ``n_bootstrap < 2``, ``conf_level`` outside (0, 1), or fewer than 2 finite
    observations yield an all-NaN struct (never an exception).

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({"x": list(range(1, 11))})
    >>> df.select(ps.bootstrap_mean_ci("x", seed=42))
    """
    if isinstance(x, str):
        x = pl.col(x)
    x = x.cast(pl.Float64)

    x_clean = x.filter(x.is_finite())

    seed_lit = pl.lit(seed, dtype=pl.UInt64) if seed is not None else pl.lit(None, dtype=pl.UInt64)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_bootstrap_mean_ci",
        args=[
            x_clean,
            pl.lit(n_bootstrap, dtype=pl.UInt32),
            pl.lit(conf_level, dtype=pl.Float64),
            pl.lit(block_length, dtype=pl.UInt32),
            seed_lit,
        ],
        returns_scalar=True,
    )


def bootstrap_ci(
    x: pl.Expr | str,
    statistic: Literal["mean", "median", "std"] = "mean",
    n_bootstrap: int = 1000,
    conf_level: float = 0.95,
    seed: int | None = None,
) -> pl.Expr:
    """
    Percentile bootstrap confidence interval for a fixed statistic.

    IID (nonparametric) bootstrap: draws ``n_bootstrap`` resamples of size n
    with replacement, evaluates ``statistic`` on each, and reports the
    percentile interval of the sorted replicates together with the bootstrap
    standard error (the sample SD of the replicates).

    Parameters
    ----------
    x : pl.Expr or str
        Sample expression or column name.
    statistic : {"mean", "median", "std"}, default "mean"
        The statistic to bootstrap. "std" is the sample standard deviation
        (denominator n - 1).
    n_bootstrap : int, default 1000
        Number of bootstrap resamples.
    conf_level : float, default 0.95
        Confidence level for the percentile interval, in (0, 1).
    seed : int, optional
        Random seed for reproducibility. Without one, each call draws fresh
        resamples (non-deterministic).

    Returns
    -------
    pl.Expr
        Expression returning struct{estimate: f64, se: f64, ci_lower: f64,
        ci_upper: f64, conf_level: f64, n_bootstrap: u32}.

    Notes
    -----
    ``statistic="mean"`` is bit-identical to
    ``bootstrap_mean_ci(block_length=0)`` for the same seed (both use the
    same mean closure in the Rust backend).

    This is a deliberate scope limit, not every bootstrap statistic: the
    upstream `anofox_statistics::bootstrap_ci` is generic over a native Rust
    closure, which cannot cross the Python FFI boundary, and calling back
    into Python from inside a Polars `#[polars_expr]` function running on a
    worker thread risks a GIL-reentrancy deadlock. Only this fixed
    ``{"mean", "median", "std"}`` dispatch is exposed; arbitrary Python
    callables are not accepted.

    Non-finite values are dropped before resampling. `n_bootstrap < 2`,
    `conf_level` outside (0, 1), or fewer than 2 finite observations yield an
    all-NaN struct (never an exception).

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({"x": list(range(1, 11))})
    >>> df.select(ps.bootstrap_ci("x", statistic="median", seed=42))
    """
    if statistic not in ("mean", "median", "std"):
        raise ValueError(f"statistic must be 'mean', 'median', or 'std', got {statistic!r}")

    if isinstance(x, str):
        x = pl.col(x)
    x = x.cast(pl.Float64)

    x_clean = x.filter(x.is_finite())

    seed_lit = pl.lit(seed, dtype=pl.UInt64) if seed is not None else pl.lit(None, dtype=pl.UInt64)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_bootstrap_ci",
        args=[
            x_clean,
            pl.lit(statistic, dtype=pl.String),
            pl.lit(n_bootstrap, dtype=pl.UInt32),
            pl.lit(conf_level, dtype=pl.Float64),
            seed_lit,
        ],
        returns_scalar=True,
    )
