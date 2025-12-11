"""Parametric statistical tests as Polars expressions."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Union

import polars as pl
from polars.plugins import register_plugin_function

LIB = Path(__file__).parent.parent


def ttest_ind(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    equal_var: bool = False,
    mu: float = 0.0,
    conf_level: float = 0.95,
) -> pl.Expr:
    """
    Perform independent samples t-test.

    This function works with group_by and over operations, computing
    the t-test for each group independently.

    Parameters
    ----------
    x : pl.Expr or str
        First sample expression or column name.
    y : pl.Expr or str
        Second sample expression or column name.
    alternative : {"two-sided", "less", "greater"}, default "two-sided"
        Alternative hypothesis direction.
    equal_var : bool, default False
        If True, use Student's t-test (assumes equal variances).
        If False, use Welch's t-test.
    mu : float, default 0.0
        The hypothesized difference in means under the null hypothesis.
    conf_level : float, default 0.95
        Confidence level for the confidence interval.

    Returns
    -------
    pl.Expr
        Expression returning struct{statistic: f64, p_value: f64}

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({
    ...     "group": ["A", "A", "A", "B", "B", "B"],
    ...     "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ...     "y": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
    ... })
    >>>
    >>> # Simple t-test
    >>> df.select(ps.ttest_ind("x", "y"))
    >>>
    >>> # T-test per group
    >>> df.group_by("group").agg(ps.ttest_ind("x", "y").alias("ttest"))
    """
    if isinstance(x, str):
        x = pl.col(x)
    if isinstance(y, str):
        y = pl.col(y)

    # Filter out nulls and non-finite values
    x_clean = x.filter(x.is_finite())
    y_clean = y.filter(y.is_finite())

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_ttest_ind",
        args=[
            x_clean,
            y_clean,
            pl.lit(alternative, dtype=pl.String),
            pl.lit(equal_var, dtype=pl.Boolean),
            pl.lit(mu, dtype=pl.Float64),
            pl.lit(conf_level, dtype=pl.Float64),
        ],
        returns_scalar=True,
    )


def ttest_paired(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    mu: float = 0.0,
    conf_level: float = 0.95,
) -> pl.Expr:
    """
    Perform paired samples t-test.

    Parameters
    ----------
    x : pl.Expr or str
        First sample (before treatment).
    y : pl.Expr or str
        Second sample (after treatment).
    alternative : {"two-sided", "less", "greater"}, default "two-sided"
        Alternative hypothesis direction.
    mu : float, default 0.0
        The hypothesized difference in means under the null hypothesis.
    conf_level : float, default 0.95
        Confidence level for the confidence interval.

    Returns
    -------
    pl.Expr
        Expression returning struct{statistic: f64, p_value: f64}

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({
    ...     "before": [1.0, 2.0, 3.0, 4.0, 5.0],
    ...     "after": [1.5, 2.8, 3.2, 4.5, 5.1],
    ... })
    >>>
    >>> df.select(ps.ttest_paired("before", "after"))
    """
    if isinstance(x, str):
        x = pl.col(x)
    if isinstance(y, str):
        y = pl.col(y)

    # Both must be finite for paired test
    x_clean = x.filter(x.is_finite() & y.is_finite())
    y_clean = y.filter(x.is_finite() & y.is_finite())

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_ttest_paired",
        args=[
            x_clean,
            y_clean,
            pl.lit(alternative, dtype=pl.String),
            pl.lit(mu, dtype=pl.Float64),
            pl.lit(conf_level, dtype=pl.Float64),
        ],
        returns_scalar=True,
    )


def brown_forsythe(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
) -> pl.Expr:
    """
    Perform Brown-Forsythe test for equality of variances.

    This is a robust test for homogeneity of variances that uses
    deviations from the median instead of the mean.

    Parameters
    ----------
    x : pl.Expr or str
        First sample expression or column name.
    y : pl.Expr or str
        Second sample expression or column name.

    Returns
    -------
    pl.Expr
        Expression returning struct{statistic: f64, p_value: f64}
    """
    if isinstance(x, str):
        x = pl.col(x)
    if isinstance(y, str):
        y = pl.col(y)

    x_clean = x.filter(x.is_finite())
    y_clean = y.filter(y.is_finite())

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_brown_forsythe",
        args=[x_clean, y_clean],
        returns_scalar=True,
    )


def yuen_test(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    trim: float = 0.2,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    conf_level: float = 0.95,
) -> pl.Expr:
    """
    Perform Yuen's test for trimmed means.

    A robust alternative to the t-test that compares trimmed means,
    making it less sensitive to outliers and violations of normality.

    Parameters
    ----------
    x : pl.Expr or str
        First sample expression or column name.
    y : pl.Expr or str
        Second sample expression or column name.
    trim : float, default 0.2
        Proportion to trim from each end (0 to 0.5).
    alternative : {"two-sided", "less", "greater"}, default "two-sided"
        Alternative hypothesis direction.
    conf_level : float, default 0.95
        Confidence level for the confidence interval.

    Returns
    -------
    pl.Expr
        Expression returning struct{statistic: f64, p_value: f64}

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({
    ...     "x": [1.0, 2.0, 3.0, 4.0, 100.0],  # outlier
    ...     "y": [1.5, 2.5, 3.5, 4.5, 5.5],
    ... })
    >>>
    >>> # Robust comparison using trimmed means
    >>> df.select(ps.yuen_test("x", "y", trim=0.2))
    """
    if isinstance(x, str):
        x = pl.col(x)
    if isinstance(y, str):
        y = pl.col(y)

    x_clean = x.filter(x.is_finite())
    y_clean = y.filter(y.is_finite())

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_yuen_test",
        args=[
            x_clean,
            y_clean,
            pl.lit(trim, dtype=pl.Float64),
            pl.lit(alternative, dtype=pl.String),
            pl.lit(conf_level, dtype=pl.Float64),
        ],
        returns_scalar=True,
    )
