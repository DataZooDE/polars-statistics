"""Distributional tests (normality, etc.) as Polars expressions."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import polars as pl
from polars.plugins import register_plugin_function

LIB = Path(__file__).parent.parent


def shapiro_wilk(
    x: Union[pl.Expr, str],
) -> pl.Expr:
    """
    Perform Shapiro-Wilk test for normality.

    Tests the null hypothesis that the data was drawn from a normal distribution.

    Parameters
    ----------
    x : pl.Expr or str
        Sample expression or column name.

    Returns
    -------
    pl.Expr
        Expression returning struct{statistic: f64, p_value: f64}

    Notes
    -----
    The Shapiro-Wilk test is appropriate for sample sizes between 3 and 5000.
    For larger samples, the test may be overly sensitive.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({
    ...     "values": [1.0, 2.0, 3.0, 4.0, 5.0],
    ... })
    >>>
    >>> df.select(ps.shapiro_wilk("values"))
    """
    if isinstance(x, str):
        x = pl.col(x)

    x_clean = x.filter(x.is_finite())

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_shapiro_wilk",
        args=[x_clean],
        returns_scalar=True,
    )


def dagostino(
    x: Union[pl.Expr, str],
) -> pl.Expr:
    """
    Perform D'Agostino K-squared test for normality.

    Tests the null hypothesis that the data was drawn from a normal distribution
    by testing skewness and kurtosis.

    Parameters
    ----------
    x : pl.Expr or str
        Sample expression or column name.

    Returns
    -------
    pl.Expr
        Expression returning struct{statistic: f64, p_value: f64}

    Notes
    -----
    This test requires at least 20 observations.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({
    ...     "values": list(range(100)),
    ... })
    >>>
    >>> df.select(ps.dagostino("values"))
    """
    if isinstance(x, str):
        x = pl.col(x)

    x_clean = x.filter(x.is_finite())

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_dagostino",
        args=[x_clean],
        returns_scalar=True,
    )
