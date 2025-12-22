"""Correlation test expressions."""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional, Union

import polars as pl
from polars.plugins import register_plugin_function

LIB = Path(__file__).parent.parent


def pearson(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    conf_level: Optional[float] = 0.95,
) -> pl.Expr:
    """
    Compute Pearson correlation coefficient with hypothesis test.

    Parameters
    ----------
    x : pl.Expr or str
        First variable expression or column name.
    y : pl.Expr or str
        Second variable expression or column name.
    conf_level : float, optional, default 0.95
        Confidence level for the confidence interval.

    Returns
    -------
    pl.Expr
        Expression returning struct{estimate, statistic, p_value,
        ci_lower, ci_upper, n}.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({
    ...     "x": [1.0, 2.0, 3.0, 4.0, 5.0],
    ...     "y": [1.2, 2.1, 2.9, 4.2, 5.1],
    ... })
    >>>
    >>> df.select(ps.pearson("x", "y"))
    """
    if isinstance(x, str):
        x = pl.col(x)
    if isinstance(y, str):
        y = pl.col(y)

    # Both must be finite for correlation
    x_clean = x.filter(x.is_finite() & y.is_finite())
    y_clean = y.filter(x.is_finite() & y.is_finite())

    conf_lit = pl.lit(conf_level, dtype=pl.Float64) if conf_level is not None else pl.lit(None, dtype=pl.Float64)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_pearson",
        args=[x_clean, y_clean, conf_lit],
        returns_scalar=True,
    )


def spearman(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    conf_level: Optional[float] = 0.95,
) -> pl.Expr:
    """
    Compute Spearman rank correlation coefficient with hypothesis test.

    Parameters
    ----------
    x : pl.Expr or str
        First variable expression or column name.
    y : pl.Expr or str
        Second variable expression or column name.
    conf_level : float, optional, default 0.95
        Confidence level for the confidence interval.

    Returns
    -------
    pl.Expr
        Expression returning struct{estimate, statistic, p_value,
        ci_lower, ci_upper, n}.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({
    ...     "x": [1.0, 2.0, 3.0, 4.0, 5.0],
    ...     "y": [1.2, 2.1, 2.9, 4.2, 5.1],
    ... })
    >>>
    >>> df.select(ps.spearman("x", "y"))
    """
    if isinstance(x, str):
        x = pl.col(x)
    if isinstance(y, str):
        y = pl.col(y)

    x_clean = x.filter(x.is_finite() & y.is_finite())
    y_clean = y.filter(x.is_finite() & y.is_finite())

    conf_lit = pl.lit(conf_level, dtype=pl.Float64) if conf_level is not None else pl.lit(None, dtype=pl.Float64)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_spearman",
        args=[x_clean, y_clean, conf_lit],
        returns_scalar=True,
    )


def kendall(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    variant: Literal["a", "b", "c"] = "b",
) -> pl.Expr:
    """
    Compute Kendall's tau correlation coefficient with hypothesis test.

    Parameters
    ----------
    x : pl.Expr or str
        First variable expression or column name.
    y : pl.Expr or str
        Second variable expression or column name.
    variant : {"a", "b", "c"}, default "b"
        Which variant of Kendall's tau to compute:
        - "a": Tau-a (unadjusted for ties)
        - "b": Tau-b (adjusted for ties, most common)
        - "c": Tau-c (Stuart's tau-c, for rectangular tables)

    Returns
    -------
    pl.Expr
        Expression returning struct{estimate, statistic, p_value,
        ci_lower, ci_upper, n}.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({
    ...     "x": [1.0, 2.0, 3.0, 4.0, 5.0],
    ...     "y": [1.2, 2.1, 2.9, 4.2, 5.1],
    ... })
    >>>
    >>> df.select(ps.kendall("x", "y", variant="b"))
    """
    if isinstance(x, str):
        x = pl.col(x)
    if isinstance(y, str):
        y = pl.col(y)

    x_clean = x.filter(x.is_finite() & y.is_finite())
    y_clean = y.filter(x.is_finite() & y.is_finite())

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_kendall",
        args=[
            x_clean,
            y_clean,
            pl.lit(variant, dtype=pl.String),
        ],
        returns_scalar=True,
    )


def distance_cor(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    n_permutations: int = 999,
    seed: Optional[int] = None,
) -> pl.Expr:
    """
    Compute distance correlation with permutation test.

    Distance correlation measures both linear and nonlinear association
    between two random variables. It is zero if and only if the variables
    are independent.

    Parameters
    ----------
    x : pl.Expr or str
        First variable expression or column name.
    y : pl.Expr or str
        Second variable expression or column name.
    n_permutations : int, default 999
        Number of permutations for the significance test.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pl.Expr
        Expression returning struct{estimate, statistic, p_value, n}.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({
    ...     "x": [1.0, 2.0, 3.0, 4.0, 5.0],
    ...     "y": [1.0, 4.0, 9.0, 16.0, 25.0],  # nonlinear relationship
    ... })
    >>>
    >>> df.select(ps.distance_cor("x", "y"))
    """
    if isinstance(x, str):
        x = pl.col(x)
    if isinstance(y, str):
        y = pl.col(y)

    x_clean = x.filter(x.is_finite() & y.is_finite())
    y_clean = y.filter(x.is_finite() & y.is_finite())

    seed_lit = pl.lit(seed, dtype=pl.UInt64) if seed is not None else pl.lit(None, dtype=pl.UInt64)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_distance_cor",
        args=[
            x_clean,
            y_clean,
            pl.lit(n_permutations, dtype=pl.UInt32),
            seed_lit,
        ],
        returns_scalar=True,
    )


def partial_cor(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    covariates: List[Union[pl.Expr, str]],
) -> pl.Expr:
    """
    Compute partial correlation controlling for covariates.

    Measures the association between x and y after removing the linear
    effects of the covariates from both variables.

    Parameters
    ----------
    x : pl.Expr or str
        First variable expression or column name.
    y : pl.Expr or str
        Second variable expression or column name.
    covariates : list of pl.Expr or str
        List of covariate expressions or column names to control for.

    Returns
    -------
    pl.Expr
        Expression returning struct{estimate, statistic, p_value, n}.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({
    ...     "x": [1.0, 2.0, 3.0, 4.0, 5.0],
    ...     "y": [1.2, 2.1, 2.9, 4.2, 5.1],
    ...     "z": [0.5, 1.0, 1.5, 2.0, 2.5],
    ... })
    >>>
    >>> # Correlation between x and y, controlling for z
    >>> df.select(ps.partial_cor("x", "y", ["z"]))
    """
    if isinstance(x, str):
        x = pl.col(x)
    if isinstance(y, str):
        y = pl.col(y)

    # Convert string covariates to expressions
    cov_exprs = []
    for cov in covariates:
        if isinstance(cov, str):
            cov_exprs.append(pl.col(cov))
        else:
            cov_exprs.append(cov)

    # Build the args list
    args = [
        x.filter(x.is_finite() & y.is_finite()),
        y.filter(x.is_finite() & y.is_finite()),
        pl.lit(len(cov_exprs), dtype=pl.UInt32),
    ]
    args.extend([cov.filter(x.is_finite() & y.is_finite()) for cov in cov_exprs])

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_partial_cor",
        args=args,
        returns_scalar=True,
    )


def semi_partial_cor(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    covariates: List[Union[pl.Expr, str]],
) -> pl.Expr:
    """
    Compute semi-partial (part) correlation.

    Measures the association between x and y after removing the linear
    effects of the covariates from y only (keeping x unchanged).

    Parameters
    ----------
    x : pl.Expr or str
        First variable expression or column name.
    y : pl.Expr or str
        Second variable expression or column name.
    covariates : list of pl.Expr or str
        List of covariate expressions or column names to control for (in y).

    Returns
    -------
    pl.Expr
        Expression returning struct{estimate, statistic, p_value, n}.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({
    ...     "x": [1.0, 2.0, 3.0, 4.0, 5.0],
    ...     "y": [1.2, 2.1, 2.9, 4.2, 5.1],
    ...     "z": [0.5, 1.0, 1.5, 2.0, 2.5],
    ... })
    >>>
    >>> # Semi-partial correlation: control for z in y only
    >>> df.select(ps.semi_partial_cor("x", "y", ["z"]))
    """
    if isinstance(x, str):
        x = pl.col(x)
    if isinstance(y, str):
        y = pl.col(y)

    cov_exprs = []
    for cov in covariates:
        if isinstance(cov, str):
            cov_exprs.append(pl.col(cov))
        else:
            cov_exprs.append(cov)

    args = [
        x.filter(x.is_finite() & y.is_finite()),
        y.filter(x.is_finite() & y.is_finite()),
        pl.lit(len(cov_exprs), dtype=pl.UInt32),
    ]
    args.extend([cov.filter(x.is_finite() & y.is_finite()) for cov in cov_exprs])

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_semi_partial_cor",
        args=args,
        returns_scalar=True,
    )


def icc(
    values: Union[pl.Expr, str],
    icc_type: Literal["icc1", "icc2", "icc3", "icc2k", "icc3k"] = "icc1",
    conf_level: float = 0.95,
) -> pl.Expr:
    """
    Compute Intraclass Correlation Coefficient (ICC).

    Note: This is a placeholder that requires proper matrix input handling.
    ICC requires data organized as subjects x raters matrix.

    Parameters
    ----------
    values : pl.Expr or str
        Values expression or column name.
    icc_type : {"icc1", "icc2", "icc3", "icc2k", "icc3k"}, default "icc1"
        Type of ICC to compute:
        - "icc1": One-way random effects, single rater
        - "icc2": Two-way random effects, single rater
        - "icc3": Two-way mixed effects, single rater
        - "icc2k": Two-way random effects, average of k raters
        - "icc3k": Two-way mixed effects, average of k raters
    conf_level : float, default 0.95
        Confidence level for the confidence interval.

    Returns
    -------
    pl.Expr
        Expression returning struct{estimate, statistic, p_value,
        ci_lower, ci_upper, n}.
    """
    if isinstance(values, str):
        values = pl.col(values)

    values_clean = values.filter(values.is_finite())

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_icc",
        args=[
            values_clean,
            pl.lit(icc_type, dtype=pl.String),
            pl.lit(conf_level, dtype=pl.Float64),
        ],
        returns_scalar=True,
    )
