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


def jarque_bera(
    x: Union[pl.Expr, str],
) -> pl.Expr:
    """
    Perform the Jarque-Bera test for normality.

    An asymptotic chi-squared(2) test on sample skewness and excess kurtosis
    (population moments), matching R `tseries::jarque.bera.test` (also
    `moments::jarque.test`).

    Parameters
    ----------
    x : pl.Expr or str
        Sample expression or column name.

    Returns
    -------
    pl.Expr
        Expression returning struct{statistic: f64, p_value: f64, skewness: f64,
        kurtosis: f64, n: u32}.

    Notes
    -----
    Requires at least 3 finite, non-constant values. Fewer than 3 values, a
    non-finite input, or constant data yield NaN statistic/p_value/skewness/kurtosis
    and n == 0 (never an exception).

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({
    ...     "values": [2.1, 3.4, 1.9, 5.6, 4.4, 3.3, 2.8, 9.7, 3.0, 4.1],
    ... })
    >>>
    >>> df.select(ps.jarque_bera("values"))
    """
    if isinstance(x, str):
        x = pl.col(x)

    x_clean = x.filter(x.is_finite())

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_jarque_bera",
        args=[x_clean],
        returns_scalar=True,
    )
