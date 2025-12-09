"""Regression model expressions for Polars.

These expressions allow fitting regression models within group_by and over operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import polars as pl
from polars.plugins import register_plugin_function

LIB = Path(__file__).parent.parent


# ============================================================================
# Linear Regression Expressions
# ============================================================================


def ols(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr:
    """Ordinary Least Squares regression as a Polars expression.

    Works with group_by and over operations to fit OLS per group.

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    *x : pl.Expr or str
        Feature variables (one or more).
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing: intercept, coefficients, r_squared, adj_r_squared,
        mse, rmse, f_statistic, f_pvalue, aic, bic, n_observations.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({
    ...     "group": ["A"] * 50 + ["B"] * 50,
    ...     "y": [...],
    ...     "x1": [...],
    ...     "x2": [...],
    ... })
    >>>
    >>> # OLS per group
    >>> df.group_by("group").agg(
    ...     ps.ols("y", "x1", "x2").alias("model")
    ... )
    >>>
    >>> # Access results
    >>> result.with_columns(
    ...     pl.col("model").struct.field("r_squared"),
    ...     pl.col("model").struct.field("coefficients"),
    ... )
    """
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_ols",
        args=[
            y.cast(pl.Float64),
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def ridge(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 1.0,
    with_intercept: bool = True,
) -> pl.Expr:
    """Ridge regression (L2 regularization) as a Polars expression.

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    *x : pl.Expr or str
        Feature variables (one or more).
    lambda_ : float, default 1.0
        Regularization strength.
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing regression results.
    """
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_ridge",
        args=[
            y.cast(pl.Float64),
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def elastic_net(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 1.0,
    alpha: float = 0.5,
    with_intercept: bool = True,
) -> pl.Expr:
    """Elastic Net regression (L1 + L2 regularization) as a Polars expression.

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    *x : pl.Expr or str
        Feature variables (one or more).
    lambda_ : float, default 1.0
        Total regularization strength.
    alpha : float, default 0.5
        L1 ratio (0 = Ridge, 1 = Lasso).
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing regression results.
    """
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_elastic_net",
        args=[
            y.cast(pl.Float64),
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(alpha, dtype=pl.Float64),
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def wls(
    y: Union[pl.Expr, str],
    weights: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr:
    """Weighted Least Squares regression as a Polars expression.

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    weights : pl.Expr or str
        Observation weights.
    *x : pl.Expr or str
        Feature variables (one or more).
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing regression results.
    """
    if isinstance(y, str):
        y = pl.col(y)
    if isinstance(weights, str):
        weights = pl.col(weights)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_wls",
        args=[
            y.cast(pl.Float64),
            weights.cast(pl.Float64),
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def rls(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    forgetting_factor: float = 0.99,
    with_intercept: bool = True,
) -> pl.Expr:
    """Recursive Least Squares regression as a Polars expression.

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    *x : pl.Expr or str
        Feature variables (one or more).
    forgetting_factor : float, default 0.99
        Forgetting factor (0 < lambda <= 1).
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing regression results.
    """
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_rls",
        args=[
            y.cast(pl.Float64),
            pl.lit(forgetting_factor, dtype=pl.Float64),
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def bls(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    with_intercept: bool = True,
) -> pl.Expr:
    """Bounded Least Squares regression as a Polars expression.

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    *x : pl.Expr or str
        Feature variables (one or more).
    lower_bound : float, optional
        Lower bound for coefficients.
    upper_bound : float, optional
        Upper bound for coefficients.
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing regression results.

    Notes
    -----
    For non-negative least squares (NNLS), use lower_bound=0.
    """
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    lb = pl.lit(lower_bound, dtype=pl.Float64) if lower_bound is not None else pl.lit(None, dtype=pl.Float64)
    ub = pl.lit(upper_bound, dtype=pl.Float64) if upper_bound is not None else pl.lit(None, dtype=pl.Float64)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_bls",
        args=[
            y.cast(pl.Float64),
            lb,
            ub,
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def nnls(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr:
    """Non-negative Least Squares regression as a Polars expression.

    Shorthand for bls(..., lower_bound=0).

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    *x : pl.Expr or str
        Feature variables (one or more).
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing regression results.
    """
    return bls(y, *x, lower_bound=0.0, with_intercept=with_intercept)


# ============================================================================
# GLM Expressions
# ============================================================================


def logistic(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr:
    """Logistic regression as a Polars expression.

    Parameters
    ----------
    y : pl.Expr or str
        Binary target variable (0/1).
    *x : pl.Expr or str
        Feature variables (one or more).
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing: intercept, coefficients, deviance, null_deviance,
        aic, bic, n_observations.
    """
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_logistic",
        args=[
            y.cast(pl.Float64),
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def poisson(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr:
    """Poisson regression as a Polars expression.

    Parameters
    ----------
    y : pl.Expr or str
        Count target variable.
    *x : pl.Expr or str
        Feature variables (one or more).
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing GLM results.
    """
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_poisson",
        args=[
            y.cast(pl.Float64),
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def negative_binomial(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    theta: float | None = None,
    with_intercept: bool = True,
) -> pl.Expr:
    """Negative Binomial regression as a Polars expression.

    Parameters
    ----------
    y : pl.Expr or str
        Overdispersed count target variable.
    *x : pl.Expr or str
        Feature variables (one or more).
    theta : float, optional
        Dispersion parameter. If None, estimated from data.
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing GLM results.
    """
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    theta_lit = pl.lit(theta, dtype=pl.Float64) if theta is not None else pl.lit(None, dtype=pl.Float64)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_negative_binomial",
        args=[
            y.cast(pl.Float64),
            theta_lit,
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def tweedie(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    var_power: float = 1.5,
    with_intercept: bool = True,
) -> pl.Expr:
    """Tweedie GLM as a Polars expression.

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    *x : pl.Expr or str
        Feature variables (one or more).
    var_power : float, default 1.5
        Variance power (0=Gaussian, 1=Poisson, 2=Gamma, 3=Inverse Gaussian).
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing GLM results.
    """
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_tweedie",
        args=[
            y.cast(pl.Float64),
            pl.lit(var_power, dtype=pl.Float64),
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def probit(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr:
    """Probit regression as a Polars expression.

    Parameters
    ----------
    y : pl.Expr or str
        Binary target variable (0/1).
    *x : pl.Expr or str
        Feature variables (one or more).
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing GLM results.
    """
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_probit",
        args=[
            y.cast(pl.Float64),
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def cloglog(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr:
    """Complementary log-log regression as a Polars expression.

    Parameters
    ----------
    y : pl.Expr or str
        Binary target variable (0/1).
    *x : pl.Expr or str
        Feature variables (one or more).
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing GLM results.
    """
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_cloglog",
        args=[
            y.cast(pl.Float64),
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )
