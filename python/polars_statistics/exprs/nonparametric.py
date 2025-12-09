"""Non-parametric statistical tests as Polars expressions."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import polars as pl
from polars.plugins import register_plugin_function

LIB = Path(__file__).parent.parent


def mann_whitney_u(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
) -> pl.Expr:
    """
    Perform Mann-Whitney U test (Wilcoxon rank-sum test).

    A non-parametric test for testing whether two independent samples
    were drawn from the same distribution. Uses two-sided test.

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

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({
    ...     "x": [1.0, 2.0, 3.0, 4.0, 5.0],
    ...     "y": [2.0, 4.0, 6.0, 8.0, 10.0],
    ... })
    >>>
    >>> df.select(ps.mann_whitney_u("x", "y"))
    """
    if isinstance(x, str):
        x = pl.col(x)
    if isinstance(y, str):
        y = pl.col(y)

    x_clean = x.filter(x.is_finite())
    y_clean = y.filter(y.is_finite())

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_mann_whitney_u",
        args=[x_clean, y_clean],
        returns_scalar=True,
    )


def wilcoxon_signed_rank(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
) -> pl.Expr:
    """
    Perform Wilcoxon signed-rank test.

    A non-parametric test for paired samples, testing whether
    paired differences come from a symmetric distribution around zero.
    Uses two-sided test.

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

    x_clean = x.filter(x.is_finite() & y.is_finite())
    y_clean = y.filter(x.is_finite() & y.is_finite())

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_wilcoxon_signed_rank",
        args=[x_clean, y_clean],
        returns_scalar=True,
    )


def kruskal_wallis(
    *groups: Union[pl.Expr, str],
) -> pl.Expr:
    """
    Perform Kruskal-Wallis H test.

    A non-parametric test for testing whether samples from two or more
    groups originate from the same distribution.

    Parameters
    ----------
    *groups : pl.Expr or str
        Two or more sample expressions or column names.

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
    ...     "group1": [1.0, 2.0, 3.0],
    ...     "group2": [4.0, 5.0, 6.0],
    ...     "group3": [7.0, 8.0, 9.0],
    ... })
    >>>
    >>> df.select(ps.kruskal_wallis("group1", "group2", "group3"))
    """
    cleaned_groups = []
    for g in groups:
        if isinstance(g, str):
            g = pl.col(g)
        cleaned_groups.append(g.filter(g.is_finite()))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_kruskal_wallis",
        args=cleaned_groups,
        returns_scalar=True,
    )


def brunner_munzel(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    alternative: str = "two-sided",
) -> pl.Expr:
    """
    Perform Brunner-Munzel test for stochastic equality.

    A robust non-parametric test for comparing two independent samples
    that doesn't assume equal variances or shape of distributions.
    Tests whether P(X < Y) = 0.5.

    Parameters
    ----------
    x : pl.Expr or str
        First sample expression or column name.
    y : pl.Expr or str
        Second sample expression or column name.
    alternative : {"two-sided", "less", "greater"}, default "two-sided"
        Alternative hypothesis direction.

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
    ...     "x": [1.0, 2.0, 3.0, 4.0, 5.0],
    ...     "y": [2.0, 4.0, 6.0, 8.0, 10.0],
    ... })
    >>>
    >>> df.select(ps.brunner_munzel("x", "y"))
    """
    if isinstance(x, str):
        x = pl.col(x)
    if isinstance(y, str):
        y = pl.col(y)

    x_clean = x.filter(x.is_finite())
    y_clean = y.filter(y.is_finite())

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_brunner_munzel",
        args=[
            x_clean,
            y_clean,
            pl.lit(alternative, dtype=pl.String),
        ],
        returns_scalar=True,
    )
