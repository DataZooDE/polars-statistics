"""Regression model expressions for Polars.

These expressions allow fitting regression models within group_by and over operations.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Literal, Sequence, Union

import polars as pl
from polars.plugins import register_plugin_function

LIB = Path(__file__).parent.parent


def _resolve_intercept(
    add_intercept: bool | None,
    with_intercept: bool | None,
    default: bool = True,
) -> bool:
    """Resolve intercept parameter from either name, with deprecation warning.

    Parameters ``add_intercept`` (preferred) and ``with_intercept`` (deprecated)
    are mutually exclusive.
    """
    if add_intercept is not None and with_intercept is not None:
        raise ValueError(
            "Cannot specify both 'add_intercept' and 'with_intercept'. "
            "Use 'add_intercept' (with_intercept is deprecated)."
        )
    if with_intercept is not None:
        warnings.warn(
            "Parameter 'with_intercept' is deprecated and will be removed in a "
            "future release. Use 'add_intercept' instead.",
            FutureWarning,
            stacklevel=3,
        )
        return with_intercept
    if add_intercept is not None:
        return add_intercept
    return default


def _check_alias_tolerance(tol: float) -> float:
    """Validate the R-`lm`-style rank-deficiency (aliasing) tolerance.

    Must be finite and non-negative; raises ``ValueError`` otherwise.
    """
    if not math.isfinite(tol) or tol < 0:
        raise ValueError(f"alias_tolerance must be finite and >= 0, got {tol}")
    return tol


# ============================================================================
# Linear Regression Expressions
# ============================================================================


def ols(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
    solve_method: str | None = None,
    alias_tolerance: float = 1e-7,
) -> pl.Expr:
    """Ordinary Least Squares regression as a Polars expression.

    Works with group_by and over operations to fit OLS per group.

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    *x : pl.Expr or str
        Feature variables (one or more).
    add_intercept : bool, default True
        Whether to include an intercept term.
    solve_method : str or None, default None
        Solver for the linear system: "qr" (default), "svd", or "cholesky".
    alias_tolerance : float, default 1e-7
        Relative tolerance of R `lm`'s rank-deficiency test; a column within
        it of the span of the intercept and earlier columns gets a NaN
        coefficient. Default matches R `lm` / anofox's `LM_ALIAS_TOLERANCE`.

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
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
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
            pl.lit(add_intercept, dtype=pl.Boolean),
            pl.lit(solve_method, dtype=pl.String),
            *x_exprs,
        ],
        kwargs={"alias_tolerance": _check_alias_tolerance(alias_tolerance)},
        returns_scalar=True,
    )


def ridge(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 1.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
    solve_method: str | None = None,
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
    add_intercept : bool, default True
        Whether to include an intercept term.
    solve_method : str or None, default None
        Solver for the linear system: "qr" (default), "svd", or "cholesky".

    Returns
    -------
    pl.Expr
        Struct containing regression results.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
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
            pl.lit(add_intercept, dtype=pl.Boolean),
            pl.lit(solve_method, dtype=pl.String),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def elastic_net(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 1.0,
    alpha: float = 0.5,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
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
    add_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing regression results.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
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
            pl.lit(add_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def wls(
    y: Union[pl.Expr, str],
    weights: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
    solve_method: str | None = None,
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
    add_intercept : bool, default True
        Whether to include an intercept term.
    solve_method : str or None, default None
        Solver for the linear system: "qr" (default), "svd", or "cholesky".

    Returns
    -------
    pl.Expr
        Struct containing regression results.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
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
            pl.lit(add_intercept, dtype=pl.Boolean),
            pl.lit(solve_method, dtype=pl.String),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def rls(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    forgetting_factor: float = 0.99,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
    initial_p_diagonal: float = 1e6,
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
    add_intercept : bool, default True
        Whether to include an intercept term.
    initial_p_diagonal : float, default 1e6
        Diagonal delta of the initial covariance P0 = delta*I. Large values
        express a vague prior (approaches batch OLS when forgetting_factor=1);
        small values shrink early estimates toward zero and are numerically
        more stable. Must be positive and finite; an invalid value yields a
        NaN struct (no exception).

    Returns
    -------
    pl.Expr
        Struct containing regression results.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
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
            pl.lit(add_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        kwargs={"initial_p_diagonal": initial_p_diagonal},
        returns_scalar=True,
    )


def expanding_ols(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Expanding window OLS regression as a Polars expression.

    Equivalent to rls(..., forgetting_factor=1.0), i.e. recursive least squares
    with no forgetting -- all past observations are weighted equally.

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    *x : pl.Expr or str
        Feature variables (one or more).
    add_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing regression results.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return rls(y, *x, forgetting_factor=1.0, add_intercept=add_intercept)


def bls(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
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
    add_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing regression results.

    Notes
    -----
    For non-negative least squares (NNLS), use lower_bound=0.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
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
            pl.lit(add_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def nnls(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Non-negative Least Squares regression as a Polars expression.

    Shorthand for bls(..., lower_bound=0).

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    *x : pl.Expr or str
        Feature variables (one or more).
    add_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing regression results.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return bls(y, *x, lower_bound=0.0, add_intercept=add_intercept)


def lasso(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 1.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Lasso regression (L1 regularization) as a Polars expression.

    Shorthand for elastic_net(..., alpha=1.0).

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    *x : pl.Expr or str
        Feature variables (one or more).
    lambda_ : float, default 1.0
        Regularization strength.
    add_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing regression results.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return elastic_net(y, *x, lambda_=lambda_, alpha=1.0, add_intercept=add_intercept)


# ============================================================================
# Robust Regression Expressions
# ============================================================================


def huber(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    epsilon: float = 1.35,
    alpha: float = 0.0001,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> pl.Expr:
    """Huber M-estimator regression (robust to outliers).

    Parameters
    ----------
    y : pl.Expr or str
        Response variable.
    *x : pl.Expr or str
        Feature variables (one or more).
    epsilon : float, default 1.35
        Huber threshold; must be > 1.0. Smaller values down-weight more rows.
    alpha : float, default 0.0001
        L2 ridge penalty applied during the weighted least-squares update.
    add_intercept : bool, default True
    max_iter : int, default 100
    tol : float, default 1e-5

    Returns
    -------
    pl.Expr
        Struct matching the linear-regression output schema (intercept,
        coefficients, r_squared, ..., n_observations).
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_huber",
        args=[
            y.cast(pl.Float64),
            pl.lit(epsilon, dtype=pl.Float64),
            pl.lit(alpha, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            pl.lit(max_iter, dtype=pl.UInt32),
            pl.lit(tol, dtype=pl.Float64),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def quantile(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    tau: float = 0.5,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Quantile regression as a Polars expression.

    Estimates conditional quantiles using Iteratively Reweighted Least Squares (IRLS).
    More robust to outliers than OLS.

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    *x : pl.Expr or str
        Feature variables (one or more).
    tau : float, default 0.5
        The quantile to estimate (must be between 0 and 1).
        0.5 corresponds to median regression.
    add_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing: intercept, coefficients, tau, pseudo_r_squared,
        check_loss, n_observations.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> # Median regression (tau=0.5)
    >>> df.group_by("group").agg(
    ...     ps.quantile("y", "x1", "x2", tau=0.5).alias("model")
    ... )
    >>>
    >>> # Lower quartile regression
    >>> df.select(ps.quantile("y", "x1", tau=0.25))
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_quantile",
        args=[
            y.cast(pl.Float64),
            pl.lit(tau, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def pls(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    n_components: int = 2,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
    tol: float = 1e-6,
    scale: bool = True,
) -> pl.Expr:
    """Partial Least Squares regression as a Polars expression.

    PLS projects ``X`` onto ``n_components`` latent variables maximizing
    covariance with ``y``, then fits a linear model in that space. Useful
    when features are highly collinear.

    Parameters
    ----------
    y : pl.Expr or str
        Response variable.
    *x : pl.Expr or str
        Feature variables.
    n_components : int, default 2
        Number of latent components.
    add_intercept : bool, default True
    tol : float, default 1e-6
    scale : bool, default True
        Whether to scale ``X`` columns to unit variance before fitting.

    Returns
    -------
    pl.Expr
        Struct matching the linear-regression output schema.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_pls",
        args=[
            y.cast(pl.Float64),
            pl.lit(n_components, dtype=pl.UInt32),
            pl.lit(add_intercept, dtype=pl.Boolean),
            pl.lit(tol, dtype=pl.Float64),
            pl.lit(scale, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def isotonic(
    y: Union[pl.Expr, str],
    x: Union[pl.Expr, str],
    increasing: bool = True,
) -> pl.Expr:
    """Isotonic (monotonic) regression as a Polars expression.

    Fits a non-decreasing (or non-increasing) step function to the data
    using the Pool Adjacent Violators Algorithm (PAVA).

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    x : pl.Expr or str
        Single feature variable.
    increasing : bool, default True
        Whether to fit an increasing (True) or decreasing (False) function.

    Returns
    -------
    pl.Expr
        Struct containing: r_squared, increasing, fitted_values, n_observations.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> # Fit monotonically increasing function
    >>> df.select(ps.isotonic("y", "x", increasing=True))
    >>>
    >>> # Per-group isotonic regression
    >>> df.group_by("group").agg(
    ...     ps.isotonic("y", "x").alias("model")
    ... )
    """
    if isinstance(y, str):
        y = pl.col(y)
    if isinstance(x, str):
        x = pl.col(x)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_isotonic",
        args=[
            y.cast(pl.Float64),
            x.cast(pl.Float64),
            pl.lit(increasing, dtype=pl.Boolean),
        ],
        returns_scalar=True,
    )


# ============================================================================
# Diagnostics Expressions
# ============================================================================


def condition_number(
    *x: Union[pl.Expr, str],
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Compute condition number diagnostics for a design matrix.

    The condition number measures how sensitive the regression is to numerical
    errors. A high condition number indicates multicollinearity, which can make
    coefficients unreliable.

    Parameters
    ----------
    *x : pl.Expr or str
        Feature variables (one or more).
    add_intercept : bool, default True
        Whether to include an intercept column in the analysis.

    Returns
    -------
    pl.Expr
        Struct containing:
        - condition_number: κ(X), ratio of max to min singular values
        - condition_number_xtx: κ(X'X) = κ(X)², useful for some diagnostics
        - singular_values: List of singular values (descending)
        - condition_indices: List of max(σ)/σ_j for each singular value
        - severity: Classification ("WellConditioned", "Moderate", "High", "Severe")
        - warning: Warning message if condition number is problematic

    Interpretation
    --------------
    - κ < 30: Well-conditioned, numerically stable
    - 30 ≤ κ < 100: Moderate collinearity, some instability possible
    - 100 ≤ κ < 1000: High collinearity, numerical instability likely
    - κ ≥ 1000: Severe collinearity, coefficients may be unreliable

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({
    ...     "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
    ...     "x2": [2.0, 4.0, 6.0, 8.0, 10.0],  # Collinear with x1
    ...     "x3": [1.1, 2.3, 2.9, 4.1, 5.2],
    ... })
    >>>
    >>> # Check for multicollinearity
    >>> df.select(ps.condition_number("x1", "x2", "x3").alias("diagnostics"))
    >>>
    >>> # Access specific fields
    >>> result.with_columns(
    ...     pl.col("diagnostics").struct.field("severity"),
    ...     pl.col("diagnostics").struct.field("condition_number"),
    ... )
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_condition_number",
        args=[
            pl.lit(add_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def check_binary_separation(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
) -> pl.Expr:
    """Check for quasi-separation in binary response data.

    Quasi-separation occurs when a predictor (or combination of predictors)
    perfectly or nearly perfectly separates the response categories. This causes
    coefficient estimates to diverge to infinity in logistic/probit regression.

    Parameters
    ----------
    y : pl.Expr or str
        Binary target variable (0/1).
    *x : pl.Expr or str
        Feature variables (one or more).

    Returns
    -------
    pl.Expr
        Struct containing:
        - has_separation: Boolean indicating if separation was detected
        - separated_predictors: List of predictor indices showing separation (0-based)
        - separation_types: List of separation types ("Complete", "Quasi", "MonotonicResponse")
        - warning: Warning message with details and recommendations

    Separation Types
    ----------------
    - Complete: A predictor perfectly divides the classes (all y=0 on one side,
      all y=1 on the other)
    - Quasi: Nearly perfect separation with only 1-2 observations crossing the boundary
    - MonotonicResponse: For categorical-like predictors where each level has all
      observations in the same class

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> # Data with perfect separation
    >>> df = pl.DataFrame({
    ...     "y": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    ...     "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ... })
    >>>
    >>> # Check for separation before fitting logistic regression
    >>> result = df.select(ps.check_binary_separation("y", "x").alias("sep"))
    >>> print(result["sep"][0]["has_separation"])  # True
    >>> print(result["sep"][0]["separation_types"])  # ["Complete"]

    Notes
    -----
    Run this diagnostic before fitting logistic, probit, or cloglog models.
    If separation is detected:
    1. Consider using regularization (lambda_ parameter)
    2. Remove or combine problematic predictors
    3. Collect more data with overlap in predictor values
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
        function_name="pl_check_binary_separation",
        args=[
            y.cast(pl.Float64),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def check_count_sparsity(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
) -> pl.Expr:
    """Check for sparsity-induced separation in count data.

    For Poisson/Negative Binomial regression, extreme sparsity (all-zero segments)
    can cause coefficient divergence similar to separation in logistic regression.
    This occurs when some predictor values (especially binary indicators) always
    have zero responses.

    Parameters
    ----------
    y : pl.Expr or str
        Count target variable (non-negative integers).
    *x : pl.Expr or str
        Feature variables (one or more).

    Returns
    -------
    pl.Expr
        Struct containing:
        - has_separation: Boolean indicating if sparsity issues detected
        - separated_predictors: List of predictor indices with issues (0-based)
        - separation_types: List of separation types (typically "MonotonicResponse")
        - warning: Warning message with details and recommendations

    Notes
    -----
    This check is most relevant when:
    - Data has >90% zero responses
    - Predictors include binary indicators (e.g., changepoint variables)
    - A binary indicator has all-zero responses when active

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> # Sparse data with problematic indicator
    >>> df = pl.DataFrame({
    ...     "y": [0, 1, 2, 0, 0, 0, 0, 0, 0, 0] * 10,  # 95% zeros after position 30
    ...     "indicator": [0]*50 + [1]*50,  # Binary indicator, all y=0 when indicator=1
    ... })
    >>>
    >>> # Check for sparsity issues before fitting Poisson regression
    >>> result = df.select(ps.check_count_sparsity("y", "indicator").alias("sparse"))
    >>> if result["sparse"][0]["has_separation"]:
    ...     print("Warning:", result["sparse"][0]["warning"])

    Recommendations
    ---------------
    If sparsity issues are detected:
    1. Use regularization (lambda_ parameter)
    2. Remove sparse indicator features
    3. Aggregate sparse categories
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
        function_name="pl_check_count_sparsity",
        args=[
            y.cast(pl.Float64),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def vif(*x: Union[pl.Expr, str]) -> pl.Expr:
    """Variance Inflation Factor (VIF) for multicollinearity detection.

    VIF measures how much the variance of a coefficient estimate is inflated
    by multicollinearity. For predictor j:

        VIF_j = 1 / (1 - R²_j)

    where R²_j is the R² from regressing x_j on all other predictors.

    Parameters
    ----------
    *x : pl.Expr or str
        Feature variables (two or more required). The intercept is NOT
        included — VIF is defined per predictor only.

    Returns
    -------
    pl.Expr
        Struct containing:
        - terms: List[str] with one entry per feature column (``x1``, ``x2``, ...)
        - vif: List[float] with one VIF value per feature
        - n_observations: number of rows used

    Interpretation
    --------------
    - VIF = 1: no collinearity with the other predictors
    - VIF > 5: moderate multicollinearity
    - VIF > 10: severe multicollinearity (estimates may be unreliable)

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>> df = pl.DataFrame({"x1": [...], "x2": [...], "x3": [...]})
    >>> df.select(ps.vif("x1", "x2", "x3").alias("vif"))
    """
    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_vif",
        args=x_exprs,
        returns_scalar=True,
    )


def leverage(
    *x: Union[pl.Expr, str],
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Leverage (hat matrix diagonal) for each observation.

    Leverage h_ii measures the influence of observation i on its own fitted
    value. It is the i-th diagonal element of H = X (X'X)^(-1) X'. Properties:

    - 0 ≤ h_ii ≤ 1
    - Σ h_ii = p (number of parameters)
    - Points with h_ii > 2p/n are typically considered high-leverage.

    Parameters
    ----------
    *x : pl.Expr or str
        Feature variables.
    add_intercept : bool, default True
        Whether to include an intercept column in the design matrix.

    Returns
    -------
    pl.Expr
        Struct containing:
        - leverage: List[float] with one value per input row
        - n_observations: number of rows used

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>> df = pl.DataFrame({"x1": [...], "x2": [...]})
    >>> df.select(ps.leverage("x1", "x2").alias("h"))
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_leverage",
        args=[
            pl.lit(add_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def cooks_distance(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Cook's distance for each observation under an OLS fit.

    Cook's distance combines residual size and leverage to identify
    observations that disproportionately influence the fitted regression:

        D_i = (e_i² / (p · MSE)) · (h_ii / (1 - h_ii)²)

    Observations with D_i > 4/n (or D_i > 1) are commonly flagged as
    influential.

    Parameters
    ----------
    y : pl.Expr or str
        Response variable.
    *x : pl.Expr or str
        Feature variables.
    add_intercept : bool, default True
        Whether to include an intercept column when fitting the OLS model
        and computing leverage.

    Returns
    -------
    pl.Expr
        Struct containing:
        - cooks_d: List[float] with one Cook's distance value per input row
        - n_observations: number of rows used

    Notes
    -----
    Internally fits an OLS model to obtain residuals and MSE; the leverage
    component reuses the same design matrix. The fit is discarded after the
    diagnostic is computed.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_cooks_distance",
        args=[
            y.cast(pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def _residual_diag_args(y, x, add_intercept):
    if isinstance(y, str):
        y = pl.col(y)
    x_exprs = [(pl.col(xi) if isinstance(xi, str) else xi).cast(pl.Float64) for xi in x]
    return [y.cast(pl.Float64), pl.lit(add_intercept, dtype=pl.Boolean), *x_exprs]


# ============================================================================
# Influence diagnostics (issue #27 batch 3)
# ============================================================================


def dffits(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """DFFITS per observation from an internal OLS fit.

    Measures the scaled change in fitted value if observation ``i`` were
    dropped. |DFFITS_i| > 2 * sqrt(p / n) is the common influence cutoff.

    Returns a struct with ``dffits`` (List[float]) and ``n_observations``.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_dffits",
        args=_residual_diag_args(y, x, add_intercept),
        returns_scalar=True,
    )


def _influence_args(y, x, threshold, add_intercept):
    if isinstance(y, str):
        y = pl.col(y)
    x_exprs = [(pl.col(xi) if isinstance(xi, str) else xi).cast(pl.Float64) for xi in x]
    return [
        y.cast(pl.Float64),
        pl.lit(add_intercept, dtype=pl.Boolean),
        pl.lit(threshold, dtype=pl.Float64),
        *x_exprs,
    ]


def influential_cooks(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    threshold: float | None = None,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Boolean mask of observations with Cook's distance above threshold.

    Default threshold is ``4 / n`` (Belsley-Kuh-Welsch). Returns a struct
    with ``is_influential`` (List[bool]), ``n_influential`` and
    ``n_observations``.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_influential_cooks",
        args=_influence_args(y, x, threshold, add_intercept),
        returns_scalar=True,
    )


def influential_dffits(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    threshold: float | None = None,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Boolean mask of observations with |DFFITS| above threshold.

    Default threshold is ``2 * sqrt(p / n)``. Returns a struct with
    ``is_influential`` (List[bool]), ``n_influential`` and ``n_observations``.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_influential_dffits",
        args=_influence_args(y, x, threshold, add_intercept),
        returns_scalar=True,
    )


def high_leverage_points(
    *x: Union[pl.Expr, str],
    threshold: float | None = None,
    add_intercept: bool = True,
) -> pl.Expr:
    """Boolean mask of high-leverage observations.

    Default threshold is ``2 * p / n``. Returns a struct with
    ``is_influential`` (List[bool]), ``n_influential`` and ``n_observations``.

    Note: takes only feature columns (no y).
    """
    x_exprs = [(pl.col(xi) if isinstance(xi, str) else xi).cast(pl.Float64) for xi in x]
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_high_leverage_points",
        args=[
            pl.lit(add_intercept, dtype=pl.Boolean),
            pl.lit(threshold, dtype=pl.Float64),
            *x_exprs,
        ],
        returns_scalar=True,
    )


# ============================================================================
# Multicollinearity + dispersion (issue #27 batch 4)
# ============================================================================


def high_vif_predictors(
    *x: Union[pl.Expr, str],
    threshold: float = 10.0,
) -> pl.Expr:
    """Boolean mask of features whose VIF exceeds ``threshold``.

    Internally computes the variance inflation factor for each feature and
    flags those above the threshold. The intercept is *not* included.

    Parameters
    ----------
    *x : pl.Expr or str
        Feature variables (two or more for a meaningful VIF).
    threshold : float, default 10.0
        VIF cutoff above which a predictor is flagged. Common values: 5
        (moderate collinearity) or 10 (severe).

    Returns
    -------
    pl.Expr
        Struct containing:

        - ``is_high``: List[bool] — one boolean per input column
        - ``n_high``: number of flagged columns
        - ``n_features``: total number of input feature columns
    """
    x_exprs = [(pl.col(xi) if isinstance(xi, str) else xi).cast(pl.Float64) for xi in x]
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_high_vif_predictors",
        args=[
            pl.lit(threshold, dtype=pl.Float64),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def generalized_vif(
    *x: Union[pl.Expr, str],
    group_sizes: list[int],
) -> pl.Expr:
    """Generalized VIF (GVIF) for grouped predictors (e.g., categorical dummies).

    For a predictor that spans multiple columns (such as one-hot encoded
    categorical levels), GVIF measures multicollinearity for the *group*
    rather than the individual columns.

    Parameters
    ----------
    *x : pl.Expr or str
        All feature columns concatenated. The order must match ``group_sizes``.
    group_sizes : list of int
        Number of columns belonging to each predictor group. Must sum to the
        total number of feature columns. Example: ``[1, 1, 3]`` for two
        single-column numeric predictors followed by a 3-level categorical.

    Returns
    -------
    pl.Expr
        Struct containing:

        - ``gvif``: List[float] — one GVIF value per group
        - ``n_groups``: number of groups

    Notes
    -----
    For single-column groups GVIF coincides with the regular VIF.
    """
    if not group_sizes:
        raise ValueError("group_sizes must be non-empty")
    sizes_str = ",".join(str(int(s)) for s in group_sizes)
    x_exprs = [(pl.col(xi) if isinstance(xi, str) else xi).cast(pl.Float64) for xi in x]
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_generalized_vif",
        args=[
            pl.lit(sizes_str, dtype=pl.String),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def pearson_chi_squared_logistic(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Pearson chi-squared goodness-of-fit statistic from a logistic regression fit.

    Computes ``Σ pearson_residual²`` together with the residual degrees of
    freedom (``n - p``). For a well-specified model X²/df_resid should be
    close to 1.

    Returns a struct with ``chi_squared`` (Float64), ``df_resid`` (UInt32)
    and ``n_observations`` (UInt32).
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_pearson_chi_squared_logistic",
        args=_glm_residual_args(y, x, lambda_, add_intercept),
        returns_scalar=True,
    )


def pearson_chi_squared_poisson(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Pearson chi-squared goodness-of-fit statistic from a Poisson regression fit.

    Computes ``Σ pearson_residual²`` together with the residual degrees of
    freedom (``n - p``). For a well-specified model X²/df_resid should be
    close to 1.

    Returns a struct with ``chi_squared`` (Float64), ``df_resid`` (UInt32)
    and ``n_observations`` (UInt32).
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_pearson_chi_squared_poisson",
        args=_glm_residual_args(y, x, lambda_, add_intercept),
        returns_scalar=True,
    )


def standardized_residuals(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Standardized residuals (r_i / sqrt(MSE)) from an internal OLS fit.

    Returns a struct with ``residuals`` (List[float]) and ``n_observations``.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_standardized_residuals",
        args=_residual_diag_args(y, x, add_intercept),
        returns_scalar=True,
    )


def studentized_residuals(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Internally studentized residuals — r_i / sqrt(MSE · (1 - h_ii)) — from an
    internal OLS fit.

    Returns a struct with ``residuals`` (List[float]) and ``n_observations``.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_studentized_residuals",
        args=_residual_diag_args(y, x, add_intercept),
        returns_scalar=True,
    )


def externally_studentized_residuals(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Externally studentized residuals (leave-one-out) from an internal OLS fit.

    Follow a t-distribution with ``n - p - 1`` degrees of freedom under the
    null hypothesis that observation ``i`` is not an outlier.

    Returns a struct with ``residuals`` (List[float]) and ``n_observations``.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_externally_studentized_residuals",
        args=_residual_diag_args(y, x, add_intercept),
        returns_scalar=True,
    )


def residual_outliers(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    threshold: float = 2.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Boolean outlier mask from studentized residuals.

    Flags observations whose internally-studentized residual exceeds the
    given threshold in absolute value (default 2.0 — roughly the 95% CI on
    standardized residuals).

    Returns a struct with ``is_outlier`` (List[bool]), ``n_outliers`` and
    ``n_observations``.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    if isinstance(y, str):
        y = pl.col(y)
    x_exprs = [(pl.col(xi) if isinstance(xi, str) else xi).cast(pl.Float64) for xi in x]
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_residual_outliers",
        args=[
            y.cast(pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            pl.lit(threshold, dtype=pl.Float64),
            *x_exprs,
        ],
        returns_scalar=True,
    )


# ============================================================================
# GLM residual diagnostics (issue #27 batch 2)
# ============================================================================


def _glm_residual_args(y, x, lambda_, add_intercept):
    if isinstance(y, str):
        y = pl.col(y)
    x_exprs = [(pl.col(xi) if isinstance(xi, str) else xi).cast(pl.Float64) for xi in x]
    return [
        y.cast(pl.Float64),
        pl.lit(lambda_, dtype=pl.Float64),
        pl.lit(add_intercept, dtype=pl.Boolean),
        *x_exprs,
    ]


def logistic_pearson_residuals(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Pearson residuals from an internal logistic regression fit.

    Returns a struct with ``residuals`` (List[float]) and ``n_observations``.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_logistic_pearson_residuals",
        args=_glm_residual_args(y, x, lambda_, add_intercept),
        returns_scalar=True,
    )


def logistic_deviance_residuals(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Deviance residuals from an internal logistic regression fit."""
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_logistic_deviance_residuals",
        args=_glm_residual_args(y, x, lambda_, add_intercept),
        returns_scalar=True,
    )


def logistic_working_residuals(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Working (IRLS adjusted-dependent-variable) residuals from a logistic fit."""
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_logistic_working_residuals",
        args=_glm_residual_args(y, x, lambda_, add_intercept),
        returns_scalar=True,
    )


def poisson_pearson_residuals(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Pearson residuals from an internal Poisson regression fit."""
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_poisson_pearson_residuals",
        args=_glm_residual_args(y, x, lambda_, add_intercept),
        returns_scalar=True,
    )


def poisson_deviance_residuals(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Deviance residuals from an internal Poisson regression fit."""
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_poisson_deviance_residuals",
        args=_glm_residual_args(y, x, lambda_, add_intercept),
        returns_scalar=True,
    )


def poisson_working_residuals(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Working residuals from an internal Poisson regression fit."""
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_poisson_working_residuals",
        args=_glm_residual_args(y, x, lambda_, add_intercept),
        returns_scalar=True,
    )


def gamma_dispersion_deviance(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Dispersion estimate (deviance method) from an internal Gamma GLM fit.

    Returns a struct with ``dispersion`` (Float64): φ̂ = D / (n − p).
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_gamma_dispersion_deviance",
        args=_glm_residual_args(y, x, lambda_, add_intercept),
        returns_scalar=True,
    )


def gamma_dispersion_pearson(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Dispersion estimate (Pearson method) from an internal Gamma GLM fit.

    Returns a struct with ``dispersion`` (Float64): φ̂ = X² / (n − p).
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_gamma_dispersion_pearson",
        args=_glm_residual_args(y, x, lambda_, add_intercept),
        returns_scalar=True,
    )


def gamma_pearson_chi_squared(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Pearson chi-squared goodness-of-fit from an internal Gamma GLM fit.

    Returns a struct with ``chi_squared`` (Float64), ``df_resid`` (UInt32),
    and ``n_observations`` (UInt32).
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_gamma_pearson_chi_squared",
        args=_glm_residual_args(y, x, lambda_, add_intercept),
        returns_scalar=True,
    )


def gamma_standardized_pearson_residuals(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Standardized Pearson residuals from an internal Gamma GLM fit.

    Returns a struct with ``residuals`` (List[float]) and ``n_observations``.
    Standardization: r_P / sqrt(φ · (1 − h_ii)).
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_gamma_standardized_pearson_residuals",
        args=_glm_residual_args(y, x, lambda_, add_intercept),
        returns_scalar=True,
    )


def gamma_standardized_deviance_residuals(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Standardized deviance residuals from an internal Gamma GLM fit.

    Returns a struct with ``residuals`` (List[float]) and ``n_observations``.
    Standardization: r_D / sqrt(φ · (1 − h_ii)).
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_gamma_standardized_deviance_residuals",
        args=_glm_residual_args(y, x, lambda_, add_intercept),
        returns_scalar=True,
    )


# ============================================================================
# GLM Expressions
# ============================================================================


def logistic(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Logistic regression as a Polars expression.

    Parameters
    ----------
    y : pl.Expr or str
        Binary target variable (0/1).
    *x : pl.Expr or str
        Feature variables (one or more).
    lambda_ : float, default 0.0
        L2 (Ridge) regularization strength.
    add_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing: intercept, coefficients, deviance, null_deviance,
        aic, bic, n_observations.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
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
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def logistic_regression(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    penalty: str = "l2",
    C: float = 1.0,
    threshold: float = 0.5,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> pl.Expr:
    """Sklearn-style logistic regression as a Polars expression.

    Distinct from :func:`logistic`, which wraps the lower-level
    ``BinomialRegressor``. This expression uses the sklearn convention:
    inverse-strength regularization parameter ``C`` and an explicit
    ``penalty`` choice.

    Parameters
    ----------
    y : pl.Expr or str
        Binary target variable (0/1).
    *x : pl.Expr or str
        Feature variables.
    penalty : {"l2", "none"}, default "l2"
    C : float, default 1.0
        Inverse of regularization strength (lambda = 1/C). Ignored when
        ``penalty="none"``.
    threshold : float, default 0.5
    add_intercept : bool, default True
    max_iter : int, default 100
    tol : float, default 1e-8

    Returns
    -------
    pl.Expr
        Struct matching the GLM output schema.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_logistic_regression",
        args=[
            y.cast(pl.Float64),
            pl.lit(C, dtype=pl.Float64),
            pl.lit(penalty, dtype=pl.String),
            pl.lit(threshold, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            pl.lit(max_iter, dtype=pl.UInt32),
            pl.lit(tol, dtype=pl.Float64),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def poisson(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Poisson regression as a Polars expression.

    Parameters
    ----------
    y : pl.Expr or str
        Count target variable.
    *x : pl.Expr or str
        Feature variables (one or more).
    lambda_ : float, default 0.0
        L2 (Ridge) regularization strength.
    add_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing GLM results.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
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
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def negative_binomial(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    theta: float | None = None,
    lambda_: float = 0.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
    estimate_dispersion: bool = False,
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
    lambda_ : float, default 0.0
        L2 (Ridge) regularization strength.
    add_intercept : bool, default True
        Whether to include an intercept term.
    estimate_dispersion : bool, default False
        Whether to scale the coefficient covariance by the Pearson chi2/df
        dispersion estimate, matching R ``summary.glm`` for
        ``family = negative.binomial(theta)``. When False (the default),
        dispersion is fixed at 1.0 (``MASS::glm.nb`` convention, since 0.8.0 —
        previously ``max(1, chi2/df)``). Coefficients are unaffected; only
        standard errors (and therefore intervals/p-values) scale by
        ``sqrt(dispersion)``.

    Returns
    -------
    pl.Expr
        Struct containing GLM results.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
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
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        kwargs={"estimate_dispersion": estimate_dispersion},
        returns_scalar=True,
    )


def tweedie(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    var_power: float = 1.5,
    lambda_: float = 0.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
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
    lambda_ : float, default 0.0
        L2 (Ridge) regularization strength.
    add_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing GLM results.

    Notes
    -----
    ``aic``/``bic`` are NaN unless ``var_power`` is one of 0, 1, 2, or 3 — other
    powers have no closed-form Tweedie density (matches R's ``tweedie`` package),
    so the log-likelihood (and therefore AIC/BIC) cannot be computed. The default
    ``var_power=1.5`` therefore reports NaN ``aic``/``bic``. Coefficients and
    standard errors are unaffected and remain finite.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
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
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def probit(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Probit regression as a Polars expression.

    Parameters
    ----------
    y : pl.Expr or str
        Binary target variable (0/1).
    *x : pl.Expr or str
        Feature variables (one or more).
    lambda_ : float, default 0.0
        L2 (Ridge) regularization strength.
    add_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing GLM results.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
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
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def cloglog(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Complementary log-log regression as a Polars expression.

    Parameters
    ----------
    y : pl.Expr or str
        Binary target variable (0/1).
    *x : pl.Expr or str
        Feature variables (one or more).
    lambda_ : float, default 0.0
        L2 (Ridge) regularization strength.
    add_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing GLM results.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
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
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


# ============================================================================
# Augment (per-observation diagnostics) — issue #66, UPST-09
# ============================================================================

_AUGMENT_MODELS = (
    "ols",
    "wls",
    "logistic",
    "probit",
    "cloglog",
    "poisson",
    "negative_binomial",
    "tweedie",
    "gamma",
)
_AUGMENT_RESIDUAL_TYPES = ("response", "pearson", "deviance")


def augment(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    model: str = "ols",
    weights: Union[pl.Expr, str, None] = None,
    residual_type: str | None = None,
    add_intercept: bool = True,
    theta: float | None = None,
    var_power: float = 1.5,
    link_power: float | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> pl.Expr:
    """Per-observation regression diagnostics, R ``broom::augment`` style.

    One combined, row-aligned expression returning, for every input row, the
    fitted mean, a residual, standardised and studentised residuals
    (``rstandard``/``rstudent``), the leverage (``hatvalues``), Cook's
    distance and DFFITS — for any of the nine supported models.

    Parameters
    ----------
    y : pl.Expr or str
        Response variable.
    *x : pl.Expr or str
        Feature variables (at least one).
    model : str, default "ols"
        One of ``"ols"``, ``"wls"``, ``"logistic"``, ``"probit"``,
        ``"cloglog"``, ``"poisson"``, ``"negative_binomial"``, ``"tweedie"``,
        or ``"gamma"``.
    weights : pl.Expr or str, optional
        Prior weights. Required (and only valid) with ``model="wls"``; when
        omitted for WLS the fit's own weights are used.
    residual_type : str, optional
        ``"response"``, ``"pearson"``, or ``"deviance"``. Defaults to
        ``"response"`` for ``ols``/``wls`` and ``"deviance"`` for every GLM
        (matching R's ``broom::augment``).
    add_intercept : bool, default True
        Whether to include an intercept term.
    theta : float, optional
        Fixed negative-binomial dispersion (``model="negative_binomial"``
        only); ``None`` estimates it from the data.
    var_power : float, default 1.5
        Tweedie variance power (``model="tweedie"`` only): 0 Gaussian, 1
        Poisson, 2 Gamma, 3 inverse Gaussian.
    link_power : float, optional
        Tweedie link power (``model="tweedie"`` only). ``None`` (default)
        uses the canonical link (``1 - var_power``), matching ``ps.tweedie()``
        and ``Tweedie(link_power=None)`` — the same model the user fits with
        the same arguments. Pass an explicit value (e.g. ``0.0`` for the log
        link) to override it.
    max_iter : int, default 100
        Maximum IRLS iterations (GLMs only).
    tol : float, default 1e-8
        IRLS convergence tolerance (GLMs only).

    Returns
    -------
    pl.Expr
        Struct with ``fitted``, ``residual``, ``std_residual``,
        ``stud_residual``, ``leverage``, ``cooks_d``, ``dffits`` — one row per
        input row. Rows with a null or non-finite ``y``/``x``/``weights``
        entry are ``NaN`` in every field; the remaining rows equal a fit
        without them. Works with ``.over(group)`` (per-row) or
        ``group_by(...).agg(...)`` (list of structs per group).

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>> df = pl.DataFrame({"y": [...], "x1": [...], "x2": [...]})
    >>> df.select(ps.augment("y", "x1", "x2").alias("a")).unnest("a")
    """
    model_l = model.lower()
    if model_l not in _AUGMENT_MODELS:
        raise ValueError(f"ps.augment: unknown model '{model}' (expected one of {_AUGMENT_MODELS})")
    if residual_type is not None and residual_type.lower() not in _AUGMENT_RESIDUAL_TYPES:
        raise ValueError(
            f"ps.augment: unknown residual_type '{residual_type}' "
            f"(expected one of {_AUGMENT_RESIDUAL_TYPES})"
        )
    if weights is not None and model_l != "wls":
        raise ValueError("ps.augment: weights is only valid with model='wls'")
    if model_l == "wls" and weights is None:
        raise ValueError("ps.augment: model='wls' requires weights")
    if not x:
        raise ValueError("ps.augment: at least one feature column (*x) is required")
    if theta is not None and model_l != "negative_binomial":
        raise ValueError("ps.augment: theta is only valid with model='negative_binomial'")
    if theta is not None and not (math.isfinite(theta) and theta > 0):
        raise ValueError("ps.augment: theta must be a finite positive number")
    if model_l == "tweedie" and var_power <= 0:
        raise ValueError("ps.augment: var_power must be > 0")
    if link_power is not None and model_l != "tweedie":
        raise ValueError("ps.augment: link_power is only valid with model='tweedie'")
    if link_power is not None and not math.isfinite(link_power):
        raise ValueError("ps.augment: link_power must be finite")
    if max_iter < 1:
        raise ValueError("ps.augment: max_iter must be >= 1")
    if tol <= 0:
        raise ValueError("ps.augment: tol must be > 0")

    resolved_residual_type = (
        residual_type.lower()
        if residual_type is not None
        else ("response" if model_l in ("ols", "wls") else "deviance")
    )

    if isinstance(y, str):
        y = pl.col(y)
    x_exprs = [(pl.col(xi) if isinstance(xi, str) else xi).cast(pl.Float64) for xi in x]

    if weights is not None:
        weights_expr = (pl.col(weights) if isinstance(weights, str) else weights).cast(pl.Float64)
    else:
        weights_expr = pl.lit(None, dtype=pl.Float64)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_augment",
        args=[y.cast(pl.Float64), weights_expr, *x_exprs],
        kwargs={
            "model": model_l,
            "residual_type": resolved_residual_type,
            "with_intercept": add_intercept,
            "theta": theta,
            "var_power": var_power,
            "link_power": link_power,
            "max_iter": max_iter,
            "tol": tol,
        },
        returns_scalar=False,
    )


# ============================================================================
# Penalized GLM Expressions
# ============================================================================

_PRIOR_KINDS = ("flat", "normal", "laplace")


def _normalize_priors(
    priors: Sequence[Union[None, str, tuple, dict]] | None,
) -> list[dict]:
    """Normalize a priors spec into a list of ``{"kind", "loc", "scale"}`` dicts.

    Accepts, per entry: ``None`` or ``"flat"`` (flat prior), a tuple/list
    ``(kind, loc, scale)`` or ``(kind,)``, or a dict with ``"kind"`` and
    optional ``"loc"`` (default ``0.0``) / ``"scale"`` (default ``inf``).
    An optional leading entry applies to the intercept.
    """
    if priors is None:
        return []
    out: list[dict] = []
    for entry in priors:
        if entry is None:
            out.append({"kind": "flat", "loc": 0.0, "scale": float("inf")})
            continue
        if isinstance(entry, str):
            kind, loc, scale = entry, 0.0, float("inf")
        elif isinstance(entry, dict):
            if "kind" not in entry:
                raise ValueError("prior dict entries require a 'kind' key")
            kind = entry["kind"]
            loc = float(entry.get("loc", 0.0))
            scale = float(entry.get("scale", float("inf")))
        elif isinstance(entry, (tuple, list)):
            if len(entry) == 1:
                kind, loc, scale = entry[0], 0.0, float("inf")
            elif len(entry) == 2:
                kind, loc, scale = entry[0], float(entry[1]), float("inf")
            elif len(entry) == 3:
                kind, loc, scale = entry[0], float(entry[1]), float(entry[2])
            else:
                raise ValueError(
                    f"prior tuple entries must have 1-3 elements, got {len(entry)}"
                )
        else:
            raise ValueError(f"unsupported prior entry type: {type(entry)!r}")

        kind_lc = str(kind).lower()
        if kind_lc not in _PRIOR_KINDS:
            raise ValueError(
                f"unknown prior kind {kind!r} (expected one of {_PRIOR_KINDS})"
            )
        if kind_lc in ("normal", "laplace"):
            if not math.isfinite(loc):
                raise ValueError(f"prior loc must be finite for kind={kind_lc!r}, got {loc}")
            if math.isnan(scale) or scale <= 0.0:
                raise ValueError(
                    f"prior scale must be > 0 (or +inf for a flat prior) for "
                    f"kind={kind_lc!r}, got {scale}"
                )
        out.append({"kind": kind_lc, "loc": loc, "scale": scale})
    return out


def _normalize_offset(offset: Union[pl.Expr, str, None]) -> pl.Expr:
    """Normalize a penalized-GLM ``offset=`` argument to a `Float64` expression.

    ``None`` becomes a literal ``0.0`` (no offset); a string is treated as a
    column name; anything else (an expression) is cast directly. Shared by
    `penalized_glm`, `penalized_glm_summary`, and `penalized_glm_predict`
    (IN-01).
    """
    if offset is None:
        return pl.lit(0.0, dtype=pl.Float64)
    if isinstance(offset, str):
        return pl.col(offset).cast(pl.Float64)
    return offset.cast(pl.Float64)


_PENALIZED_GLM_FAMILIES = (
    "poisson",
    "binomial",
    "logistic",
    "negative_binomial",
    "gamma",
    "tweedie",
)
_PENALIZED_GLM_LINKS = {
    "poisson": ("log", "identity", "sqrt"),
    "binomial": ("logit", "probit", "cloglog"),
    "logistic": ("logit",),
    "negative_binomial": ("log",),
    "gamma": ("log",),
    "tweedie": ("log",),
}


def _validate_penalized_glm_family(
    family_lc: str,
    link: str | None,
    theta: float | None,
    power: float | None,
    threshold: float,
    estimate_dispersion: bool,
) -> None:
    """Mirror ``parse_penalized_glm_family`` (Rust) so invalid configuration
    raises ``ValueError`` when the expression is built, not at collect time."""
    if family_lc not in _PENALIZED_GLM_FAMILIES:
        raise ValueError(
            f"unknown family {family_lc!r} (expected one of {_PENALIZED_GLM_FAMILIES})"
        )
    if estimate_dispersion and family_lc != "poisson":
        raise ValueError(
            f'estimate_dispersion=True is only supported for family="poisson" '
            f"(got {family_lc!r})"
        )
    if theta is not None and family_lc != "negative_binomial":
        raise ValueError('theta is only valid for family="negative_binomial"')
    if theta is not None and not (math.isfinite(theta) and theta > 0.0):
        raise ValueError(f"theta must be finite and > 0, got {theta}")
    if power is not None and family_lc != "tweedie":
        raise ValueError('power is only valid for family="tweedie"')
    if family_lc == "tweedie":
        p = 1.5 if power is None else power
        if not (1.0 <= p <= 2.0):
            raise ValueError(f'power must be in [1, 2] for family="tweedie", got {p}')
    if link is not None:
        allowed = _PENALIZED_GLM_LINKS[family_lc]
        if link not in allowed:
            raise ValueError(
                f"invalid link {link!r} for family={family_lc!r} (expected one of {allowed})"
            )
    if family_lc == "logistic" and not (0.0 < threshold < 1.0):
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")


def _penalized_glm_kwargs(
    family: str,
    link: str | None,
    theta: float | None,
    power: float | None,
    threshold: float,
    lambda_: float,
    estimate_dispersion: bool,
    priors: Sequence[Union[None, str, tuple, dict]] | None,
    vcov_type: str,
    add_intercept: bool,
    max_iter: int,
    tol: float,
    conf_level: float = 0.95,
    prefix: str = "penalized_glm",
    predict_type: str = "response",
    null_policy: str = "drop",
) -> dict:
    """Validate and assemble the kwargs dict shared by every penalized GLM
    expression (fit here; summary/predict reuse the same shape in Plan 04)."""
    family_lc = str(family).lower()
    if lambda_ < 0.0:
        raise ValueError(f"lambda_ must be >= 0, got {lambda_}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if tol <= 0.0:
        raise ValueError(f"tol must be > 0, got {tol}")
    link_lc = link.lower() if link is not None else None
    _validate_penalized_glm_family(family_lc, link_lc, theta, power, threshold, estimate_dispersion)
    vcov_type_lc = str(vcov_type).lower()
    if vcov_type_lc not in ("laplace", "sandwich", "naive"):
        raise ValueError(
            f"unknown vcov_type {vcov_type!r} (expected one of 'laplace', 'sandwich', 'naive')"
        )
    return {
        "family": family_lc,
        "link": link_lc,
        "theta": float(theta) if theta is not None else None,
        "power": float(power) if power is not None else None,
        "threshold": float(threshold),
        "lambda": float(lambda_),
        "estimate_dispersion": bool(estimate_dispersion),
        "priors": _normalize_priors(priors),
        "vcov_type": vcov_type_lc,
        "with_intercept": bool(add_intercept),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "conf_level": float(conf_level),
        "prefix": str(prefix),
        "predict_type": str(predict_type),
        "null_policy": str(null_policy),
    }


def penalized_glm(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    family: Literal["poisson", "binomial", "logistic", "negative_binomial", "gamma", "tweedie"] = (
        "poisson"
    ),
    link: str | None = None,
    theta: float | None = None,
    power: float | None = None,
    threshold: float = 0.5,
    lambda_: float = 0.0,
    estimate_dispersion: bool = False,
    priors: Sequence[Union[None, str, tuple, dict]] | None = None,
    offset: Union[pl.Expr, str, None] = None,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> pl.Expr:
    """Family-generic penalized GLM as a Polars expression.

    One IRLS engine shared by Poisson, Binomial, Logistic, Negative Binomial,
    Gamma and Tweedie, with optional explicit per-coefficient priors and a
    selectable covariance at the mode. Works with ``group_by`` and ``.over``.

    Parameters
    ----------
    y : pl.Expr or str
        Response variable.
    *x : pl.Expr or str
        Feature variables (one or more).
    family : {"poisson", "binomial", "logistic", "negative_binomial", "gamma", "tweedie"}
        Response family:

        - ``"poisson"``: links ``"log"`` (default), ``"identity"``, ``"sqrt"``.
        - ``"binomial"``: links ``"logit"`` (default), ``"probit"``, ``"cloglog"``.
        - ``"logistic"``: binomial/logit plus training ``accuracy``.
        - ``"negative_binomial"``: log link; ``theta`` fixed or ML-estimated.
        - ``"gamma"``: log link.
        - ``"tweedie"``: log link; variance ``power`` in ``[1, 2]``.
    link : str, optional
        Link function; defaults to the family's canonical link. Invalid for
        the family raises ``ValueError`` when the expression is built.
    theta : float, optional
        Fixed Negative Binomial dispersion; ``None`` estimates it by maximum
        likelihood (``MASS::glm.nb``). Only valid for
        ``family="negative_binomial"``.
    power : float, optional
        Tweedie variance power in ``[1, 2]``; default ``1.5``. Only valid for
        ``family="tweedie"``.
    threshold : float, default 0.5
        Classification threshold for ``family="logistic"``, in ``(0, 1)``.
    lambda_ : float, default 0.0
        Uniform L2 ridge penalty on non-intercept coefficients, composed
        additively with ``priors``.
    estimate_dispersion : bool, default False
        Poisson only: scale the coefficient covariance by the Pearson
        chi2/df dispersion estimate (R's ``quasipoisson``). Every other
        family fixes its own dispersion rule and rejects this option with
        ``ValueError`` (Poisson itself otherwise uses dispersion 1, matching
        R's ``glm(family = poisson)``).
    priors : sequence, optional
        Per-coefficient priors, positionally aligned with ``x``. Each entry
        is ``None``/``"flat"`` (flat), a ``(kind, loc, scale)`` tuple/list
        (or ``(kind,)``), or a dict ``{"kind", "loc", "scale"}`` — ``kind``
        is one of ``"flat"``, ``"normal"``, ``"laplace"``. An optional
        leading entry applies to the intercept (``p`` or ``p + 1`` entries).
    offset : pl.Expr, str, or None
        Offset added to the linear predictor (on the link scale, e.g.
        ``log(exposure)``). Defaults to a constant ``0.0`` column.
    add_intercept : bool, default True
        Whether to include an intercept term.
    max_iter : int, default 100
        Maximum IRLS iterations.
    tol : float, default 1e-8
        Convergence tolerance.

    Returns
    -------
    pl.Expr
        Struct containing: intercept, coefficients, deviance, null_deviance,
        pseudo_r_squared, log_likelihood, aic, bic, dispersion, theta,
        accuracy, has_separation, iterations, converged, n_observations.

    Notes
    -----
    Rows with a null / non-finite value in ``y``, any feature, or the offset
    are dropped before fitting. Validated against R ``glm`` / ``MASS::glm.nb``
    / ``negative.binomial(theta)`` on the `anofox-regression` 0.5.22 R
    reference fixture.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df.group_by("group").agg(
    ...     ps.penalized_glm("y", "x1", "x2", family="poisson").alias("model")
    ... )
    >>>
    >>> # Gaussian priors (ridge-like, posterior mode)
    >>> df.select(
    ...     ps.penalized_glm(
    ...         "y", "x1", "x2", priors=[None, ("normal", 0.0, 1.0), ("normal", 0.0, 1.0)]
    ...     ).alias("model")
    ... )
    """
    kwargs = _penalized_glm_kwargs(
        family=family,
        link=link,
        theta=theta,
        power=power,
        threshold=threshold,
        lambda_=lambda_,
        estimate_dispersion=estimate_dispersion,
        priors=priors,
        vcov_type="laplace",
        add_intercept=add_intercept,
        max_iter=max_iter,
        tol=tol,
    )

    if isinstance(y, str):
        y = pl.col(y)
    offset_expr = _normalize_offset(offset)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_penalized_glm",
        args=[y.cast(pl.Float64), offset_expr, *x_exprs],
        kwargs=kwargs,
        returns_scalar=True,
    )


# ============================================================================
# Survival (AFT) Expressions
# ============================================================================

_AFT_DISTRIBUTION_ALIASES = (
    "weibull",
    "lognormal",
    "log_normal",
    "log-normal",
    "loglogistic",
    "log_logistic",
    "log-logistic",
    "exponential",
    "exp",
)


def _validate_aft_distribution(distribution: str) -> None:
    """Mirror ``parse_aft_distribution`` (Rust, via upstream's
    ``AftDistribution::from_name``) so an unknown distribution raises
    ``ValueError`` when the expression is built, not at collect time."""
    if str(distribution).lower() not in _AFT_DISTRIBUTION_ALIASES:
        raise ValueError(
            f"unknown AFT distribution {distribution!r} (expected one of "
            f'"weibull", "lognormal" (or "log_normal"/"log-normal"), '
            f'"loglogistic" (or "log_logistic"/"log-logistic"), '
            f'"exponential" (or "exp"))'
        )


def _aft_kwargs(
    distribution: str,
    priors: Sequence[Union[None, str, tuple, dict]] | None,
    vcov_type: str,
    add_intercept: bool,
    max_iter: int,
    tol: float,
    conf_level: float = 0.95,
    prefix: str = "aft",
    p: float = 0.5,
    t: float | None = None,
    null_policy: str = "drop",
) -> dict:
    """Validate and assemble the kwargs dict shared by every AFT expression
    (fit here; summary/predict reuse the same shape in Plan 05 Task 3)."""
    distribution_lc = str(distribution).lower()
    _validate_aft_distribution(distribution_lc)
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if tol <= 0.0:
        raise ValueError(f"tol must be > 0, got {tol}")
    vcov_type_lc = str(vcov_type).lower()
    if vcov_type_lc not in ("laplace", "sandwich", "naive"):
        raise ValueError(
            f"unknown vcov_type {vcov_type!r} (expected one of 'laplace', 'sandwich', 'naive')"
        )
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")
    return {
        "distribution": distribution_lc,
        "priors": _normalize_priors(priors),
        "vcov_type": vcov_type_lc,
        "with_intercept": bool(add_intercept),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "conf_level": float(conf_level),
        "prefix": str(prefix),
        "p": float(p),
        "t": float(t) if t is not None else float("nan"),
        "null_policy": str(null_policy),
    }


def aft(
    time: Union[pl.Expr, str],
    event: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    distribution: Literal["weibull", "exponential", "lognormal", "loglogistic"] = "weibull",
    priors: Sequence[Union[None, str, tuple, dict]] | None = None,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> pl.Expr:
    """Accelerated failure time (AFT) survival regression as a Polars expression.

    Models ``log T = x'beta + sigma * W`` for a fixed error distribution ``W``
    (the parametrisation of R's ``survival::survreg``), with right censoring.
    Works with ``group_by`` and ``.over``.

    Parameters
    ----------
    time : pl.Expr or str
        Event or censoring time. Must be strictly positive; a non-positive or
        non-finite time fails the whole fit (returned as all-NaN).
    event : pl.Expr or str
        Right-censoring indicator: ``True``/``1`` for an observed event,
        ``False``/``0`` for a right-censored observation. Accepts a Boolean
        column directly, or a numeric column containing only ``0``/``1``
        (any other value, e.g. R's ``1``/``2`` coding, fails the whole fit
        rather than being silently misread). A null time or a null event
        drops that row from the fit.
    *x : pl.Expr or str
        Feature variables (one or more).
    distribution : {"weibull", "exponential", "lognormal", "loglogistic"}
        Error distribution for ``T`` (see upstream's ``AftDistribution``).
        Aliases ``"log_normal"``/``"log-normal"`` and
        ``"log_logistic"``/``"log-logistic"`` are also accepted; invalid for
        every distribution raises ``ValueError`` when the expression is
        built.
    priors : sequence, optional
        Per-coefficient priors, positionally aligned with ``x``. Same format
        as :func:`penalized_glm`'s ``priors``.
    add_intercept : bool, default True
        Whether to include an intercept term.
    max_iter : int, default 100
        Maximum Newton iterations.
    tol : float, default 1e-9
        Convergence tolerance on the log-likelihood.

    Returns
    -------
    pl.Expr
        Struct containing: intercept, coefficients, scale, log_likelihood,
        null_log_likelihood, aic, bic, iterations, converged, n_observations,
        n_events, n_censored.

    Notes
    -----
    Rows with a null/non-finite value in ``time``, the event indicator, or
    any feature are dropped before fitting. Validated against R
    ``survival::survreg`` on the `anofox-regression` 0.5.22 R reference
    fixture (all four distributions).

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df.group_by("group").agg(
    ...     ps.aft("time", "event", "x1", "x2", distribution="weibull").alias("model")
    ... )
    """
    kwargs = _aft_kwargs(
        distribution=distribution,
        priors=priors,
        vcov_type="laplace",
        add_intercept=add_intercept,
        max_iter=max_iter,
        tol=tol,
    )

    if isinstance(time, str):
        time = pl.col(time)
    if isinstance(event, str):
        event = pl.col(event)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_aft",
        args=[time.cast(pl.Float64), event, *x_exprs],
        kwargs=kwargs,
        returns_scalar=True,
    )


def aft_summary(
    time: Union[pl.Expr, str],
    event: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    distribution: Literal["weibull", "exponential", "lognormal", "loglogistic"] = "weibull",
    priors: Sequence[Union[None, str, tuple, dict]] | None = None,
    vcov_type: Literal["laplace", "sandwich", "naive"] = "laplace",
    conf_level: float = 0.95,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> pl.Expr:
    """AFT coefficient summary in tidy format, with confidence intervals.

    Rows are ``intercept`` (when ``add_intercept=True``), ``x1..xp``, then
    ``log_scale`` (omitted for ``distribution="exponential"``, whose scale is
    fixed at 1 and not estimated).

    Parameters
    ----------
    time, event, x, distribution, priors, add_intercept, max_iter, tol
        See :func:`aft`.
    vcov_type : {"laplace", "sandwich", "naive"}, default "laplace"
        Covariance at the mode used for the standard errors / CIs below.
    conf_level : float, default 0.95
        Confidence level for ``ci_lower``/``ci_upper``, in ``(0, 1)``.

    Returns
    -------
    pl.Expr
        List of structs with columns ``term``, ``estimate``, ``std_error``,
        ``statistic``, ``p_value``, ``ci_lower``, ``ci_upper``. The
        ``intercept`` row is read directly from upstream's ``AftInference``
        intercept fields, never recomputed locally; the ``log_scale`` row's
        statistic/p_value/CI are computed from upstream's
        ``log_scale_std_error`` using the identical Wald formula.

    Examples
    --------
    >>> df.select(
    ...     ps.aft_summary("time", "event", "x1", "x2", distribution="weibull").alias("s")
    ... ).explode("s").unnest("s")
    """
    if not (0.0 < conf_level < 1.0):
        raise ValueError(f"conf_level must be in (0, 1), got {conf_level}")

    kwargs = _aft_kwargs(
        distribution=distribution,
        priors=priors,
        vcov_type=vcov_type,
        add_intercept=add_intercept,
        max_iter=max_iter,
        tol=tol,
        conf_level=conf_level,
    )

    if isinstance(time, str):
        time = pl.col(time)
    if isinstance(event, str):
        event = pl.col(event)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_aft_summary",
        args=[time.cast(pl.Float64), event, *x_exprs],
        kwargs=kwargs,
        returns_scalar=True,
    )


def _aft_predict(
    function_name: str,
    time: Union[pl.Expr, str],
    event: Union[pl.Expr, str],
    x: tuple,
    distribution: str,
    priors: Sequence[Union[None, str, tuple, dict]] | None,
    add_intercept: bool,
    max_iter: int,
    tol: float,
    null_policy: str,
    name: str | None,
    p: float = 0.5,
    t: float | None = None,
) -> pl.Expr:
    """Shared implementation for the four `aft_predict_*` builders."""
    if null_policy not in ("drop", "drop_y_zero_x"):
        raise ValueError(
            f'null_policy must be "drop" or "drop_y_zero_x", got {null_policy!r}'
        )
    prefix = name if name is not None else function_name.removeprefix("pl_")
    kwargs = _aft_kwargs(
        distribution=distribution,
        priors=priors,
        vcov_type="laplace",
        add_intercept=add_intercept,
        max_iter=max_iter,
        tol=tol,
        prefix=prefix,
        p=p,
        t=t,
        null_policy=null_policy,
    )

    if isinstance(time, str):
        time = pl.col(time)
    if isinstance(event, str):
        event = pl.col(event)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name=function_name,
        args=[time.cast(pl.Float64), event, *x_exprs],
        kwargs=kwargs,
        returns_scalar=False,
    )


def aft_predict_quantile(
    time: Union[pl.Expr, str],
    event: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    p: float = 0.5,
    distribution: Literal["weibull", "exponential", "lognormal", "loglogistic"] = "weibull",
    priors: Sequence[Union[None, str, tuple, dict]] | None = None,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-9,
    null_policy: Literal["drop", "drop_y_zero_x"] = "drop",
    name: str | None = None,
) -> pl.Expr:
    """AFT predicted `p`-quantile of the survival time.

    Refits per group/`.over()` partition with the same options used for
    :func:`aft`, then predicts on every row.

    Parameters
    ----------
    time, event, x, distribution, priors, add_intercept, max_iter, tol
        See :func:`aft`.
    p : float, default 0.5
        Quantile probability in ``[0, 1]``.
    null_policy : {"drop", "drop_y_zero_x"}, default "drop"
        - ``"drop"``: rows with a null/non-finite value in ``time``, the
          event indicator, or any feature get a NaN prediction.
        - ``"drop_y_zero_x"``: rows with a null time/event are excluded from
          the fit but still predicted (null features zero-filled).
    name : str, optional
        Custom prefix for output column names. Defaults to
        ``"aft_predict_quantile"``.

    Returns
    -------
    pl.Expr
        Struct with ``{name}_prediction``/``{name}_lower``/``{name}_upper``
        per row. ``lower``/``upper`` are always NaN — the upstream engine has
        no interval-prediction method.

    Examples
    --------
    >>> df.with_columns(
    ...     ps.aft_predict_quantile("time", "event", "x1", "x2", p=0.9)
    ...         .over("group").alias("pred")
    ... ).unnest("pred")
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")
    return _aft_predict(
        "pl_aft_predict_quantile",
        time,
        event,
        x,
        distribution,
        priors,
        add_intercept,
        max_iter,
        tol,
        null_policy,
        name,
        p=p,
    )


def aft_predict_median(
    time: Union[pl.Expr, str],
    event: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    distribution: Literal["weibull", "exponential", "lognormal", "loglogistic"] = "weibull",
    priors: Sequence[Union[None, str, tuple, dict]] | None = None,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-9,
    null_policy: Literal["drop", "drop_y_zero_x"] = "drop",
    name: str | None = None,
) -> pl.Expr:
    """AFT predicted median survival time (``aft_predict_quantile`` at ``p=0.5``).

    See :func:`aft_predict_quantile` for parameter semantics (other than
    ``p``, which is fixed at ``0.5``).
    """
    return _aft_predict(
        "pl_aft_predict_median",
        time,
        event,
        x,
        distribution,
        priors,
        add_intercept,
        max_iter,
        tol,
        null_policy,
        name,
    )


def aft_predict_survival(
    time: Union[pl.Expr, str],
    event: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    t: float,
    distribution: Literal["weibull", "exponential", "lognormal", "loglogistic"] = "weibull",
    priors: Sequence[Union[None, str, tuple, dict]] | None = None,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-9,
    null_policy: Literal["drop", "drop_y_zero_x"] = "drop",
    name: str | None = None,
) -> pl.Expr:
    """AFT predicted survival probability ``S(t | x) = P(T > t | x)``.

    Parameters
    ----------
    time, event, x, distribution, priors, add_intercept, max_iter, tol, null_policy, name
        See :func:`aft_predict_quantile`.
    t : float
        Time at which to evaluate the survival function. Must be finite
        and ``> 0``; keyword-only and required.
    """
    if t is None or not math.isfinite(t) or t <= 0.0:
        raise ValueError(f"t must be finite and > 0, got {t}")
    return _aft_predict(
        "pl_aft_predict_survival",
        time,
        event,
        x,
        distribution,
        priors,
        add_intercept,
        max_iter,
        tol,
        null_policy,
        name,
        t=t,
    )


def aft_predict_cdf(
    time: Union[pl.Expr, str],
    event: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    t: float,
    distribution: Literal["weibull", "exponential", "lognormal", "loglogistic"] = "weibull",
    priors: Sequence[Union[None, str, tuple, dict]] | None = None,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-9,
    null_policy: Literal["drop", "drop_y_zero_x"] = "drop",
    name: str | None = None,
) -> pl.Expr:
    """AFT predicted failure probability ``F(t | x) = 1 - S(t | x)``.

    Parameters
    ----------
    time, event, x, distribution, priors, add_intercept, max_iter, tol, null_policy, name
        See :func:`aft_predict_quantile`.
    t : float
        Time at which to evaluate the CDF. Must be finite and ``> 0``;
        keyword-only and required.
    """
    if t is None or not math.isfinite(t) or t <= 0.0:
        raise ValueError(f"t must be finite and > 0, got {t}")
    return _aft_predict(
        "pl_aft_predict_cdf",
        time,
        event,
        x,
        distribution,
        priors,
        add_intercept,
        max_iter,
        tol,
        null_policy,
        name,
        t=t,
    )


# ============================================================================
# ALM Expressions
# ============================================================================


def alm(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    distribution: str = "normal",
    link: str | None = None,
    loss: str = "likelihood",
    role_trim: float | None = None,
    extra_parameter: float | None = None,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Augmented Linear Model (ALM) as a Polars expression.

    A flexible regression model supporting 25 distributions, configurable
    link functions and loss criteria. Mirrors the `ALM` model class.

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    *x : pl.Expr or str
        Feature variables (one or more).
    distribution : str, default "normal"
        Distribution family. All 25 variants supported:
        - Continuous: "normal", "laplace", "student_t", "logistic",
          "asymmetric_laplace", "generalised_normal", "s"
        - Positive: "lognormal", "loglaplace", "logs", "loggeneralisednormal",
          "gamma", "inverse_gaussian", "exponential", "folded_normal",
          "rectified_normal"
        - Bounded (0,1): "beta", "logit_normal"
        - Count: "poisson", "negative_binomial", "binomial", "geometric"
        - Ordinal: "cumulative_logistic", "cumulative_normal"
        - Transformed: "boxcox_normal"
    link : str, optional
        Link function. If None, uses the canonical link for the distribution.
        Options: "identity", "log", "logit", "probit", "inverse", "sqrt", "cloglog".
    loss : str, default "likelihood"
        Loss function. Options: "likelihood", "mse", "mae", "ham", "role".
    role_trim : float, optional
        Trim fraction for the "role" loss (default 0.05).
    extra_parameter : float, optional
        Extra parameter for certain distributions (e.g., degrees of freedom for Student-t).
    add_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing: intercept, coefficients, aic, bic, n_observations.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> # Laplace regression per group
    >>> df.group_by("group").agg(
    ...     ps.alm("y", "x1", "x2", distribution="laplace").alias("model")
    ... )
    >>>
    >>> # Gamma regression for positive data
    >>> df.group_by("group").agg(
    ...     ps.alm("y", "x1", distribution="gamma").alias("model")
    ... )
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_alm",
        args=[
            y.cast(pl.Float64),
            pl.lit(distribution, dtype=pl.String),
            pl.lit(link, dtype=pl.String),
            pl.lit(loss, dtype=pl.String),
            pl.lit(role_trim, dtype=pl.Float64),
            pl.lit(extra_parameter, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


# ============================================================================
# Summary Expressions (Tidy Coefficient Output)
# ============================================================================


def ols_summary(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
    solve_method: str | None = None,
    hc_type: str | None = None,
) -> pl.Expr:
    """OLS coefficient summary in tidy format (like R's broom::tidy).

    Returns a List[Struct] with one row per coefficient containing:
    term, estimate, std_error, statistic, p_value.

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    *x : pl.Expr or str
        Feature variables (one or more).
    add_intercept : bool, default True
        Whether to include an intercept term.
    solve_method : str or None, default None
        Solver for the linear system: "qr" (default), "svd", or "cholesky".
    hc_type : str or None, default None
        Heteroskedasticity-consistent standard errors variant:
        "hc0", "hc1", "hc2", or "hc3". If None, uses homoskedastic SEs.

    Returns
    -------
    pl.Expr
        List of structs containing coefficient statistics.

    Examples
    --------
    >>> df.group_by("group").agg(
    ...     ps.ols_summary("y", "x1", "x2").alias("coef")
    ... ).explode("coef").unnest("coef")
    >>>
    >>> # With robust standard errors
    >>> df.group_by("group").agg(
    ...     ps.ols_summary("y", "x1", "x2", hc_type="hc3").alias("coef")
    ... ).explode("coef").unnest("coef")
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_ols_summary",
        args=[
            y.cast(pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            pl.lit(solve_method, dtype=pl.String),
            pl.lit(hc_type, dtype=pl.String),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def ridge_summary(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 1.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
    solve_method: str | None = None,
) -> pl.Expr:
    """Ridge regression coefficient summary in tidy format.

    Notes
    -----
    For ``lambda_ > 0``, ``std_error`` is the ridge sandwich standard error
    (``sigma^2 * A @ X'X @ A'``, matching R), but ``statistic``/``p_value`` are
    NaN because a classical t-test on shrunken/biased coefficients is not valid
    (anofox-regression 0.5.14 #37). ``lambda_ = 0`` delegates to OLS inference,
    so ``statistic``/``p_value`` are finite in that case.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_ridge_summary",
        args=[
            y.cast(pl.Float64),
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            pl.lit(solve_method, dtype=pl.String),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def elastic_net_summary(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 1.0,
    alpha: float = 0.5,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Elastic Net regression coefficient summary in tidy format."""
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_elastic_net_summary",
        args=[
            y.cast(pl.Float64),
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(alpha, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def lasso_summary(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 1.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Lasso regression coefficient summary in tidy format.

    Shorthand for elastic_net_summary(..., alpha=1.0).
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return elastic_net_summary(
        y, *x, lambda_=lambda_, alpha=1.0, add_intercept=add_intercept
    )


def wls_summary(
    y: Union[pl.Expr, str],
    weights: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
    solve_method: str | None = None,
) -> pl.Expr:
    """Weighted Least Squares coefficient summary in tidy format."""
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
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
        function_name="pl_wls_summary",
        args=[
            y.cast(pl.Float64),
            weights.cast(pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            pl.lit(solve_method, dtype=pl.String),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def rls_summary(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    forgetting_factor: float = 0.99,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
    initial_p_diagonal: float = 1e6,
) -> pl.Expr:
    """Recursive Least Squares coefficient summary in tidy format.

    Parameters
    ----------
    initial_p_diagonal : float, default 1e6
        See :func:`rls`.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_rls_summary",
        args=[
            y.cast(pl.Float64),
            pl.lit(forgetting_factor, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        kwargs={"initial_p_diagonal": initial_p_diagonal},
        returns_scalar=True,
    )


def bls_summary(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Bounded Least Squares coefficient summary in tidy format."""
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
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
        function_name="pl_bls_summary",
        args=[
            y.cast(pl.Float64),
            lb,
            ub,
            pl.lit(add_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


# ============================================================================
# GLM Summary Expressions
# ============================================================================


def logistic_summary(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Logistic regression coefficient summary in tidy format."""
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_logistic_summary",
        args=[
            y.cast(pl.Float64),
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def poisson_summary(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Poisson regression coefficient summary in tidy format."""
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_poisson_summary",
        args=[
            y.cast(pl.Float64),
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def negative_binomial_summary(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    theta: float | None = None,
    lambda_: float = 0.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
    estimate_dispersion: bool = False,
) -> pl.Expr:
    """Negative Binomial regression coefficient summary in tidy format.

    Parameters
    ----------
    estimate_dispersion : bool, default False
        Whether to scale the coefficient covariance by the Pearson chi2/df
        dispersion estimate, matching R ``summary.glm`` for
        ``family = negative.binomial(theta)``. When False (the default),
        dispersion is fixed at 1.0 (``MASS::glm.nb`` convention). Estimates
        are unaffected; only ``std_error``/``statistic``/``p_value`` scale.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
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
        function_name="pl_negative_binomial_summary",
        args=[
            y.cast(pl.Float64),
            theta_lit,
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        kwargs={"estimate_dispersion": estimate_dispersion},
        returns_scalar=True,
    )


def tweedie_summary(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    var_power: float = 1.5,
    lambda_: float = 0.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Tweedie GLM coefficient summary in tidy format.

    Notes
    -----
    Model-level ``aic``/``bic`` are NaN unless ``var_power`` is one of 0, 1, 2,
    or 3 — other powers have no closed-form Tweedie density, so the
    log-likelihood (and therefore AIC/BIC) cannot be computed. Coefficient
    estimates and standard errors are unaffected.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_tweedie_summary",
        args=[
            y.cast(pl.Float64),
            pl.lit(var_power, dtype=pl.Float64),
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def probit_summary(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Probit regression coefficient summary in tidy format."""
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_probit_summary",
        args=[
            y.cast(pl.Float64),
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def cloglog_summary(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Complementary log-log regression coefficient summary in tidy format."""
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_cloglog_summary",
        args=[
            y.cast(pl.Float64),
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def penalized_glm_summary(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    family: Literal["poisson", "binomial", "logistic", "negative_binomial", "gamma", "tweedie"] = (
        "poisson"
    ),
    link: str | None = None,
    theta: float | None = None,
    power: float | None = None,
    threshold: float = 0.5,
    lambda_: float = 0.0,
    estimate_dispersion: bool = False,
    priors: Sequence[Union[None, str, tuple, dict]] | None = None,
    vcov_type: Literal["laplace", "sandwich", "naive"] = "laplace",
    conf_level: float = 0.95,
    offset: Union[pl.Expr, str, None] = None,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> pl.Expr:
    """Penalized GLM coefficient summary in tidy format, with confidence intervals.

    Unlike every other ``*_summary`` expression (which share the 5-field
    ``{term, estimate, std_error, statistic, p_value}`` schema), this returns a
    widened 7-field schema that additionally carries ``ci_lower``/``ci_upper``
    (CONTEXT resolution 3) — the shared ``summary_output_dtype`` and every
    other ``*_summary`` expression are unchanged.

    Parameters
    ----------
    y, x, family, link, theta, power, threshold, lambda_, estimate_dispersion,
    priors, offset, add_intercept, max_iter, tol
        See :func:`penalized_glm`.
    vcov_type : {"laplace", "sandwich", "naive"}, default "laplace"
        Covariance at the mode used for the standard errors / CIs below.
    conf_level : float, default 0.95
        Confidence level for ``ci_lower``/``ci_upper``, in ``(0, 1)``.

    Returns
    -------
    pl.Expr
        List of structs with columns ``term``, ``estimate``, ``std_error``,
        ``statistic`` (Wald z, normal reference — R's ``summary.glm`` uses a
        t-reference for estimated-dispersion families: Gamma, Tweedie,
        quasipoisson), ``p_value``, ``ci_lower``, ``ci_upper``. The intercept
        row (when ``add_intercept=True``) is read directly from upstream's
        ``GlmInferenceResult`` intercept fields, never recomputed locally.

    Examples
    --------
    >>> df.select(
    ...     ps.penalized_glm_summary("y", "x1", "x2", family="poisson").alias("s")
    ... ).explode("s").unnest("s")
    """
    if not (0.0 < conf_level < 1.0):
        raise ValueError(f"conf_level must be in (0, 1), got {conf_level}")

    kwargs = _penalized_glm_kwargs(
        family=family,
        link=link,
        theta=theta,
        power=power,
        threshold=threshold,
        lambda_=lambda_,
        estimate_dispersion=estimate_dispersion,
        priors=priors,
        vcov_type=vcov_type,
        add_intercept=add_intercept,
        max_iter=max_iter,
        tol=tol,
        conf_level=conf_level,
    )

    if isinstance(y, str):
        y = pl.col(y)
    offset_expr = _normalize_offset(offset)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_penalized_glm_summary",
        args=[y.cast(pl.Float64), offset_expr, *x_exprs],
        kwargs=kwargs,
        returns_scalar=True,
    )


def alm_summary(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    distribution: str = "normal",
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Augmented Linear Model coefficient summary in tidy format."""
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_alm_summary",
        args=[
            y.cast(pl.Float64),
            pl.lit(distribution, dtype=pl.String),
            pl.lit(add_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


# ============================================================================
# Prediction Expressions
# ============================================================================


def ols_predict(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    add_intercept: bool = True,
    solve_method: str | None = None,
    interval: str | None = None,
    level: float = 0.95,
    null_policy: str = "drop",
    name: str | None = None,
) -> pl.Expr:
    """OLS predictions with optional confidence/prediction intervals.

    Returns per-row predictions. Works with both group_by and over operations.

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    *x : pl.Expr or str
        Feature variables (one or more).
    add_intercept : bool, default True
        Whether to include an intercept term in the model.
    solve_method : str or None, default None
        Solver for the linear system: "qr" (default), "svd", or "cholesky".
    interval : str or None, default None
        Type of interval to compute:
        - None: No intervals (lower/upper will be NaN)
        - "confidence": Confidence interval for mean response
        - "prediction": Prediction interval for new observation (wider)
    level : float, default 0.95
        Confidence level for intervals (e.g., 0.95 for 95% intervals).
    null_policy : str, default "drop"
        How to handle missing values:
        - "drop": Drop rows with any nulls for fitting, mask predictions with NaN
        - "drop_y_zero_x": Drop rows with null targets, zero fill null features.
          Model is fit on non-null target rows, predictions made on all rows.
    name : str or None, default None
        Custom prefix for output column names. If None, uses "ols".
        Output columns will be: {name}_prediction, {name}_lower, {name}_upper.

    Returns
    -------
    pl.Expr
        Struct containing {name}_prediction, {name}_lower, {name}_upper per row.
        Default names: ols_prediction, ols_lower, ols_upper.
        In `.over()` context, gives per-row predictions.
        In `group_by` context, returns list of structs per group.

    Examples
    --------
    >>> # Per-row predictions with .over()
    >>> df.with_columns(
    ...     ps.ols_predict("y", "x1", "x2").over("group").alias("pred")
    ... ).unnest("pred")

    >>> # With prediction intervals
    >>> df.with_columns(
    ...     ps.ols_predict("y", "x1", "x2", interval="prediction", level=0.95)
    ...         .over("group").alias("pred")
    ... ).unnest("pred")

    >>> # Handle missing data - fit on complete cases, predict on all
    >>> df.with_columns(
    ...     ps.ols_predict("y", "x1", "x2", null_policy="drop_y_zero_x")
    ...         .over("group").alias("pred")
    ... ).unnest("pred")

    >>> # Custom name for multiple models
    >>> df.with_columns(
    ...     ps.ols_predict("y", "x1", name="model1").over("group"),
    ...     ps.ols_predict("y", "x2", name="model2").over("group"),
    ... )
    """
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    # Handle interval parameter - pass as string or null
    interval_lit = pl.lit(interval, dtype=pl.String) if interval else pl.lit(None, dtype=pl.String)

    # Determine prefix for output column names
    prefix = name if name is not None else "ols"

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_ols_predict",
        args=[
            y.cast(pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            pl.lit(solve_method, dtype=pl.String),
            interval_lit,
            pl.lit(level, dtype=pl.Float64),
            pl.lit(null_policy, dtype=pl.String),
            *x_exprs,
        ],
        kwargs={"prefix": prefix},
        returns_scalar=False,
    )


def ridge_predict(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 1.0,
    add_intercept: bool = True,
    solve_method: str | None = None,
    interval: str | None = None,
    level: float = 0.95,
    null_policy: str = "drop",
    name: str | None = None,
) -> pl.Expr:
    """Ridge regression predictions with optional intervals.

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    *x : pl.Expr or str
        Feature variables.
    lambda_ : float, default 1.0
        Regularization strength.
    add_intercept : bool, default True
        Whether to include an intercept term.
    solve_method : str or None, default None
        Solver for the linear system: "qr" (default), "svd", or "cholesky".
    interval : str or None, default None
        "confidence" or "prediction" intervals.
    level : float, default 0.95
        Confidence level for intervals.
    null_policy : str, default "drop"
        "drop" or "drop_y_zero_x".
    name : str or None, default None
        Custom prefix for output column names. If None, uses "ridge".

    Returns
    -------
    pl.Expr
        Struct containing {name}_prediction, {name}_lower, {name}_upper per row.
    """
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    interval_lit = pl.lit(interval, dtype=pl.String) if interval else pl.lit(None, dtype=pl.String)
    prefix = name if name is not None else "ridge"

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_ridge_predict",
        args=[
            y.cast(pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            pl.lit(solve_method, dtype=pl.String),
            interval_lit,
            pl.lit(level, dtype=pl.Float64),
            pl.lit(null_policy, dtype=pl.String),
            pl.lit(lambda_, dtype=pl.Float64),
            *x_exprs,
        ],
        kwargs={"prefix": prefix},
        returns_scalar=False,
    )


def elastic_net_predict(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 1.0,
    alpha: float = 0.5,
    add_intercept: bool = True,
    interval: str | None = None,
    level: float = 0.95,
    null_policy: str = "drop",
    name: str | None = None,
) -> pl.Expr:
    """Elastic Net regression predictions with optional intervals.

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    *x : pl.Expr or str
        Feature variables.
    lambda_ : float, default 1.0
        Total regularization strength.
    alpha : float, default 0.5
        L1 ratio (0 = Ridge, 1 = Lasso).
    add_intercept : bool, default True
        Whether to include an intercept term.
    interval : str or None, default None
        "confidence" or "prediction" intervals.
    level : float, default 0.95
        Confidence level for intervals.
    null_policy : str, default "drop"
        "drop" or "drop_y_zero_x".
    name : str or None, default None
        Custom prefix for output column names. If None, uses "elastic_net".

    Returns
    -------
    pl.Expr
        Struct containing {name}_prediction, {name}_lower, {name}_upper per row.
    """
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    interval_lit = pl.lit(interval, dtype=pl.String) if interval else pl.lit(None, dtype=pl.String)
    prefix = name if name is not None else "elastic_net"

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_elastic_net_predict",
        args=[
            y.cast(pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            interval_lit,
            pl.lit(level, dtype=pl.Float64),
            pl.lit(null_policy, dtype=pl.String),
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(alpha, dtype=pl.Float64),
            *x_exprs,
        ],
        kwargs={"prefix": prefix},
        returns_scalar=False,
    )


def lasso_predict(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 1.0,
    add_intercept: bool = True,
    interval: str | None = None,
    level: float = 0.95,
    null_policy: str = "drop",
    name: str | None = None,
) -> pl.Expr:
    """Lasso regression predictions with optional intervals.

    Shorthand for elastic_net_predict(..., alpha=1.0).

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    *x : pl.Expr or str
        Feature variables.
    lambda_ : float, default 1.0
        Regularization strength.
    add_intercept : bool, default True
        Whether to include an intercept term.
    interval : str or None, default None
        "confidence" or "prediction" intervals.
    level : float, default 0.95
        Confidence level for intervals.
    null_policy : str, default "drop"
        "drop" or "drop_y_zero_x".
    name : str or None, default None
        Custom prefix for output column names. If None, uses "lasso".

    Returns
    -------
    pl.Expr
        Struct containing {name}_prediction, {name}_lower, {name}_upper per row.
    """
    return elastic_net_predict(
        y, *x,
        lambda_=lambda_,
        alpha=1.0,
        add_intercept=add_intercept,
        interval=interval,
        level=level,
        null_policy=null_policy,
        name=name if name is not None else "lasso",
    )


def wls_predict(
    y: Union[pl.Expr, str],
    weights: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    add_intercept: bool = True,
    solve_method: str | None = None,
    interval: str | None = None,
    level: float = 0.95,
    null_policy: str = "drop",
    name: str | None = None,
) -> pl.Expr:
    """Weighted Least Squares predictions with optional intervals.

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    weights : pl.Expr or str
        Observation weights.
    *x : pl.Expr or str
        Feature variables.
    add_intercept : bool, default True
        Whether to include an intercept term.
    solve_method : str or None, default None
        Solver for the linear system: "qr" (default), "svd", or "cholesky".
    interval : str or None, default None
        "confidence" or "prediction" intervals.
    level : float, default 0.95
        Confidence level for intervals.
    null_policy : str, default "drop"
        "drop" or "drop_y_zero_x".
    name : str or None, default None
        Custom prefix for output column names. If None, uses "wls".

    Returns
    -------
    pl.Expr
        Struct containing {name}_prediction, {name}_lower, {name}_upper per row.
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

    interval_lit = pl.lit(interval, dtype=pl.String) if interval else pl.lit(None, dtype=pl.String)
    prefix = name if name is not None else "wls"

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_wls_predict",
        args=[
            y.cast(pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            pl.lit(solve_method, dtype=pl.String),
            interval_lit,
            pl.lit(level, dtype=pl.Float64),
            pl.lit(null_policy, dtype=pl.String),
            weights.cast(pl.Float64),
            *x_exprs,
        ],
        kwargs={"prefix": prefix},
        returns_scalar=False,
    )


def rls_predict(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    forgetting_factor: float = 0.99,
    add_intercept: bool = True,
    interval: str | None = None,
    level: float = 0.95,
    null_policy: str = "drop",
    name: str | None = None,
    initial_p_diagonal: float = 1e6,
) -> pl.Expr:
    """Recursive Least Squares predictions with optional intervals.

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    *x : pl.Expr or str
        Feature variables.
    forgetting_factor : float, default 0.99
        Forgetting factor (0 < lambda <= 1).
    add_intercept : bool, default True
        Whether to include an intercept term.
    interval : str or None, default None
        "confidence" or "prediction" intervals.
    level : float, default 0.95
        Confidence level for intervals.
    null_policy : str, default "drop"
        "drop" or "drop_y_zero_x".
    name : str or None, default None
        Custom prefix for output column names. If None, uses "rls".
    initial_p_diagonal : float, default 1e6
        See :func:`rls`.

    Returns
    -------
    pl.Expr
        Struct containing {name}_prediction, {name}_lower, {name}_upper per row.
    """
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    interval_lit = pl.lit(interval, dtype=pl.String) if interval else pl.lit(None, dtype=pl.String)
    prefix = name if name is not None else "rls"

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_rls_predict",
        args=[
            y.cast(pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            interval_lit,
            pl.lit(level, dtype=pl.Float64),
            pl.lit(null_policy, dtype=pl.String),
            pl.lit(forgetting_factor, dtype=pl.Float64),
            *x_exprs,
        ],
        kwargs={"prefix": prefix, "initial_p_diagonal": initial_p_diagonal},
        returns_scalar=False,
    )


def bls_predict(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    add_intercept: bool = True,
    interval: str | None = None,
    level: float = 0.95,
    null_policy: str = "drop",
    name: str | None = None,
) -> pl.Expr:
    """Bounded Least Squares predictions with optional intervals.

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    *x : pl.Expr or str
        Feature variables.
    lower_bound : float or None
        Lower bound for coefficients.
    upper_bound : float or None
        Upper bound for coefficients.
    add_intercept : bool, default True
        Whether to include an intercept term.
    interval : str or None, default None
        "confidence" or "prediction" intervals.
    level : float, default 0.95
        Confidence level for intervals.
    null_policy : str, default "drop"
        "drop" or "drop_y_zero_x".
    name : str or None, default None
        Custom prefix for output column names. If None, uses "bls".

    Returns
    -------
    pl.Expr
        Struct containing {name}_prediction, {name}_lower, {name}_upper per row.
    """
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    interval_lit = pl.lit(interval, dtype=pl.String) if interval else pl.lit(None, dtype=pl.String)
    lb = pl.lit(lower_bound, dtype=pl.Float64) if lower_bound is not None else pl.lit(None, dtype=pl.Float64)
    ub = pl.lit(upper_bound, dtype=pl.Float64) if upper_bound is not None else pl.lit(None, dtype=pl.Float64)
    prefix = name if name is not None else "bls"

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_bls_predict",
        args=[
            y.cast(pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            interval_lit,
            pl.lit(level, dtype=pl.Float64),
            pl.lit(null_policy, dtype=pl.String),
            lb,
            ub,
            *x_exprs,
        ],
        kwargs={"prefix": prefix},
        returns_scalar=False,
    )


def nnls_predict(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    add_intercept: bool = True,
    interval: str | None = None,
    level: float = 0.95,
    null_policy: str = "drop",
    name: str | None = None,
) -> pl.Expr:
    """Non-negative Least Squares predictions.

    Shorthand for bls_predict(..., lower_bound=0).
    """
    return bls_predict(
        y, *x,
        lower_bound=0.0,
        add_intercept=add_intercept,
        interval=interval,
        level=level,
        null_policy=null_policy,
        name=name if name is not None else "nnls",
    )


# ============================================================================
# GLM Prediction Expressions
# ============================================================================


def logistic_predict(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool = True,
    interval: str | None = None,
    level: float = 0.95,
    null_policy: str = "drop",
    name: str | None = None,
) -> pl.Expr:
    """Logistic regression predictions with optional intervals.

    Parameters
    ----------
    y : pl.Expr or str
        Binary target variable (0/1).
    *x : pl.Expr or str
        Feature variables.
    lambda_ : float, default 0.0
        L2 (Ridge) regularization strength.
    add_intercept : bool, default True
        Whether to include an intercept term.
    interval : str or None, default None
        "confidence" or "prediction" intervals.
    level : float, default 0.95
        Confidence level for intervals.
    null_policy : str, default "drop"
        "drop" or "drop_y_zero_x".
    name : str or None, default None
        Custom prefix for output column names. If None, uses "logistic".

    Returns
    -------
    pl.Expr
        Struct containing {name}_prediction, {name}_lower, {name}_upper per row.
        Predictions are probabilities in [0, 1].
    """
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    interval_lit = pl.lit(interval, dtype=pl.String) if interval else pl.lit(None, dtype=pl.String)
    prefix = name if name is not None else "logistic"

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_logistic_predict",
        args=[
            y.cast(pl.Float64),
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            interval_lit,
            pl.lit(level, dtype=pl.Float64),
            pl.lit(null_policy, dtype=pl.String),
            *x_exprs,
        ],
        kwargs={"prefix": prefix},
        returns_scalar=False,
    )


def poisson_predict(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool = True,
    interval: str | None = None,
    level: float = 0.95,
    null_policy: str = "drop",
    name: str | None = None,
) -> pl.Expr:
    """Poisson regression predictions with optional intervals.

    Parameters
    ----------
    y : pl.Expr or str
        Count target variable.
    *x : pl.Expr or str
        Feature variables.
    lambda_ : float, default 0.0
        L2 (Ridge) regularization strength.
    add_intercept : bool, default True
        Whether to include an intercept term.
    interval : str or None, default None
        "confidence" or "prediction" intervals.
    level : float, default 0.95
        Confidence level for intervals.
    null_policy : str, default "drop"
        "drop" or "drop_y_zero_x".
    name : str or None, default None
        Custom prefix for output column names. If None, uses "poisson".

    Returns
    -------
    pl.Expr
        Struct containing {name}_prediction, {name}_lower, {name}_upper per row.
        Predictions are expected counts.
    """
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    interval_lit = pl.lit(interval, dtype=pl.String) if interval else pl.lit(None, dtype=pl.String)
    prefix = name if name is not None else "poisson"

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_poisson_predict",
        args=[
            y.cast(pl.Float64),
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            interval_lit,
            pl.lit(level, dtype=pl.Float64),
            pl.lit(null_policy, dtype=pl.String),
            *x_exprs,
        ],
        kwargs={"prefix": prefix},
        returns_scalar=False,
    )


def negative_binomial_predict(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool = True,
    interval: str | None = None,
    level: float = 0.95,
    null_policy: str = "drop",
    name: str | None = None,
    estimate_dispersion: bool = False,
) -> pl.Expr:
    """Negative Binomial regression predictions with optional intervals.

    Parameters
    ----------
    y : pl.Expr or str
        Overdispersed count target variable.
    *x : pl.Expr or str
        Feature variables.
    lambda_ : float, default 0.0
        L2 (Ridge) regularization strength.
    add_intercept : bool, default True
        Whether to include an intercept term.
    interval : str or None, default None
        "confidence" or "prediction" intervals.
    level : float, default 0.95
        Confidence level for intervals.
    null_policy : str, default "drop"
        "drop" or "drop_y_zero_x".
    name : str or None, default None
        Custom prefix for output column names. If None, uses "negative_binomial".
    estimate_dispersion : bool, default False
        Whether to scale interval width by the Pearson chi2/df dispersion
        estimate, matching R ``summary.glm`` for
        ``family = negative.binomial(theta)``. The point prediction is
        unaffected; only interval width changes.

    Returns
    -------
    pl.Expr
        Struct containing {name}_prediction, {name}_lower, {name}_upper per row.
    """
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    interval_lit = pl.lit(interval, dtype=pl.String) if interval else pl.lit(None, dtype=pl.String)
    prefix = name if name is not None else "negative_binomial"

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_negative_binomial_predict",
        args=[
            y.cast(pl.Float64),
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            interval_lit,
            pl.lit(level, dtype=pl.Float64),
            pl.lit(null_policy, dtype=pl.String),
            *x_exprs,
        ],
        kwargs={"prefix": prefix, "estimate_dispersion": estimate_dispersion},
        returns_scalar=False,
    )


def tweedie_predict(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    var_power: float = 1.5,
    lambda_: float = 0.0,
    add_intercept: bool = True,
    interval: str | None = None,
    level: float = 0.95,
    null_policy: str = "drop",
    name: str | None = None,
) -> pl.Expr:
    """Tweedie GLM predictions with optional intervals.

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    *x : pl.Expr or str
        Feature variables.
    var_power : float, default 1.5
        Variance power (0=Gaussian, 1=Poisson, 2=Gamma, 3=Inverse Gaussian).
    lambda_ : float, default 0.0
        L2 (Ridge) regularization strength.
    add_intercept : bool, default True
        Whether to include an intercept term.
    interval : str or None, default None
        "confidence" or "prediction" intervals.
    level : float, default 0.95
        Confidence level for intervals.
    null_policy : str, default "drop"
        "drop" or "drop_y_zero_x".
    name : str or None, default None
        Custom prefix for output column names. If None, uses "tweedie".

    Returns
    -------
    pl.Expr
        Struct containing {name}_prediction, {name}_lower, {name}_upper per row.
    """
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    interval_lit = pl.lit(interval, dtype=pl.String) if interval else pl.lit(None, dtype=pl.String)
    prefix = name if name is not None else "tweedie"

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_tweedie_predict",
        args=[
            y.cast(pl.Float64),
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            interval_lit,
            pl.lit(level, dtype=pl.Float64),
            pl.lit(null_policy, dtype=pl.String),
            pl.lit(var_power, dtype=pl.Float64),
            *x_exprs,
        ],
        kwargs={"prefix": prefix},
        returns_scalar=False,
    )


def probit_predict(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool = True,
    interval: str | None = None,
    level: float = 0.95,
    null_policy: str = "drop",
    name: str | None = None,
) -> pl.Expr:
    """Probit regression predictions with optional intervals.

    Parameters
    ----------
    y : pl.Expr or str
        Binary target variable (0/1).
    *x : pl.Expr or str
        Feature variables.
    lambda_ : float, default 0.0
        L2 (Ridge) regularization strength.
    add_intercept : bool, default True
        Whether to include an intercept term.
    interval : str or None, default None
        "confidence" or "prediction" intervals.
    level : float, default 0.95
        Confidence level for intervals.
    null_policy : str, default "drop"
        "drop" or "drop_y_zero_x".
    name : str or None, default None
        Custom prefix for output column names. If None, uses "probit".

    Returns
    -------
    pl.Expr
        Struct containing {name}_prediction, {name}_lower, {name}_upper per row.
        Predictions are probabilities in [0, 1].
    """
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    interval_lit = pl.lit(interval, dtype=pl.String) if interval else pl.lit(None, dtype=pl.String)
    prefix = name if name is not None else "probit"

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_probit_predict",
        args=[
            y.cast(pl.Float64),
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            interval_lit,
            pl.lit(level, dtype=pl.Float64),
            pl.lit(null_policy, dtype=pl.String),
            *x_exprs,
        ],
        kwargs={"prefix": prefix},
        returns_scalar=False,
    )


def cloglog_predict(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    add_intercept: bool = True,
    interval: str | None = None,
    level: float = 0.95,
    null_policy: str = "drop",
    name: str | None = None,
) -> pl.Expr:
    """Complementary log-log regression predictions with optional intervals.

    Parameters
    ----------
    y : pl.Expr or str
        Binary target variable (0/1).
    *x : pl.Expr or str
        Feature variables.
    lambda_ : float, default 0.0
        L2 (Ridge) regularization strength.
    add_intercept : bool, default True
        Whether to include an intercept term.
    interval : str or None, default None
        "confidence" or "prediction" intervals.
    level : float, default 0.95
        Confidence level for intervals.
    null_policy : str, default "drop"
        "drop" or "drop_y_zero_x".
    name : str or None, default None
        Custom prefix for output column names. If None, uses "cloglog".

    Returns
    -------
    pl.Expr
        Struct containing {name}_prediction, {name}_lower, {name}_upper per row.
        Predictions are probabilities in [0, 1].
    """
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    interval_lit = pl.lit(interval, dtype=pl.String) if interval else pl.lit(None, dtype=pl.String)
    prefix = name if name is not None else "cloglog"

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_cloglog_predict",
        args=[
            y.cast(pl.Float64),
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            interval_lit,
            pl.lit(level, dtype=pl.Float64),
            pl.lit(null_policy, dtype=pl.String),
            *x_exprs,
        ],
        kwargs={"prefix": prefix},
        returns_scalar=False,
    )


def penalized_glm_predict(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    family: Literal["poisson", "binomial", "logistic", "negative_binomial", "gamma", "tweedie"] = (
        "poisson"
    ),
    link: str | None = None,
    theta: float | None = None,
    power: float | None = None,
    threshold: float = 0.5,
    lambda_: float = 0.0,
    estimate_dispersion: bool = False,
    priors: Sequence[Union[None, str, tuple, dict]] | None = None,
    offset: Union[pl.Expr, str, None] = None,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-8,
    predict_type: Literal["response", "link"] = "response",
    null_policy: Literal["drop", "drop_y_zero_x"] = "drop",
    name: str | None = None,
) -> pl.Expr:
    """Family-generic penalized GLM predictions (response or link scale).

    Refits per group/`.over()` partition with the same options used for
    :func:`penalized_glm`, then predicts on every row.

    Parameters
    ----------
    y, x, family, link, theta, power, threshold, lambda_, estimate_dispersion,
    priors, offset, add_intercept, max_iter, tol
        See :func:`penalized_glm`.
    predict_type : {"response", "link"}, default "response"
        Prediction scale: ``"response"`` applies the family's inverse link to
        the linear predictor plus offset; ``"link"`` returns the linear
        predictor plus offset directly (no inverse link).
    null_policy : {"drop", "drop_y_zero_x"}, default "drop"
        - ``"drop"``: rows with a null/non-finite value in ``y``, any
          feature, or the offset get a NaN prediction.
        - ``"drop_y_zero_x"``: rows with a null target are excluded from the
          fit but still predicted (null features zero-filled). A null offset
          still yields a NaN prediction under either policy — the engine has
          no way to guess a missing offset.
    name : str, optional
        Custom prefix for output column names. Defaults to ``"penalized_glm"``.

    Returns
    -------
    pl.Expr
        Struct with ``{name}_prediction``/``{name}_lower``/``{name}_upper``
        per row. ``lower``/``upper`` are always NaN: `FittedPenalizedGlm` has
        no interval-prediction method (COVERAGE OPT-OUT).

    Examples
    --------
    >>> df.with_columns(
    ...     ps.penalized_glm_predict("y", "x1", "x2", offset="exposure")
    ...         .over("group").alias("pred")
    ... ).unnest("pred")
    """
    if predict_type not in ("response", "link"):
        raise ValueError(f'predict_type must be "response" or "link", got {predict_type!r}')
    if null_policy not in ("drop", "drop_y_zero_x"):
        raise ValueError(
            f'null_policy must be "drop" or "drop_y_zero_x", got {null_policy!r}'
        )

    prefix = name if name is not None else "penalized_glm"
    kwargs = _penalized_glm_kwargs(
        family=family,
        link=link,
        theta=theta,
        power=power,
        threshold=threshold,
        lambda_=lambda_,
        estimate_dispersion=estimate_dispersion,
        priors=priors,
        vcov_type="laplace",
        add_intercept=add_intercept,
        max_iter=max_iter,
        tol=tol,
        prefix=prefix,
        predict_type=predict_type,
        null_policy=null_policy,
    )

    if isinstance(y, str):
        y = pl.col(y)
    offset_expr = _normalize_offset(offset)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_penalized_glm_predict",
        args=[y.cast(pl.Float64), offset_expr, *x_exprs],
        kwargs=kwargs,
        returns_scalar=False,
    )


def alm_predict(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    distribution: str = "normal",
    add_intercept: bool = True,
    interval: str | None = None,
    level: float = 0.95,
    null_policy: str = "drop",
    name: str | None = None,
) -> pl.Expr:
    """Augmented Linear Model (ALM) predictions with optional intervals.

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    *x : pl.Expr or str
        Feature variables.
    distribution : str, default "normal"
        Distribution family. Options include:
        - Continuous: "normal", "laplace", "student_t", "logistic"
        - Positive: "lognormal", "loglaplace", "gamma", "inverse_gaussian", "exponential"
        - Bounded (0,1): "beta"
        - Count: "poisson", "negative_binomial", "binomial", "geometric"
    add_intercept : bool, default True
        Whether to include an intercept term.
    interval : str or None, default None
        "confidence" or "prediction" intervals.
    level : float, default 0.95
        Confidence level for intervals.
    null_policy : str, default "drop"
        "drop" or "drop_y_zero_x".
    name : str or None, default None
        Custom prefix for output column names. If None, uses "alm".

    Returns
    -------
    pl.Expr
        Struct containing {name}_prediction, {name}_lower, {name}_upper per row.
    """
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    interval_lit = pl.lit(interval, dtype=pl.String) if interval else pl.lit(None, dtype=pl.String)
    prefix = name if name is not None else "alm"

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_alm_predict",
        args=[
            y.cast(pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            interval_lit,
            pl.lit(level, dtype=pl.Float64),
            pl.lit(null_policy, dtype=pl.String),
            pl.lit(distribution, dtype=pl.String),
            *x_exprs,
        ],
        kwargs={"prefix": prefix},
        returns_scalar=False,
    )


# ============================================================================
# Formula-Based Regression Expressions
# ============================================================================


def _parse_formula(formula: str):
    """Parse a formula and return response variable and feature expressions."""
    from polars_statistics.formula import FormulaParser, expand_terms_to_expressions

    parser = FormulaParser()
    parsed = parser.parse(formula)
    x_exprs, term_names = expand_terms_to_expressions(parsed.terms)
    return parsed.response, x_exprs, term_names


def ols_formula(
    formula: str,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
    solve_method: str | None = None,
) -> pl.Expr:
    """OLS regression using R-style formula syntax.

    Supports polynomial terms, interactions, and transforms that work
    per-group with group_by and over operations.

    Parameters
    ----------
    formula : str
        R-style formula like "y ~ x1 + x2 + x1:x2" or "y ~ poly(x, 2)".
        Supported syntax:
        - Main effects: "y ~ x1 + x2"
        - Interaction only: "y ~ x1:x2"
        - Full crossing: "y ~ x1 * x2" (expands to x1 + x2 + x1:x2)
        - Polynomial: "y ~ poly(x, 2)" (centered) or "y ~ poly(x, 2, raw=True)"
        - Transform: "y ~ I(x^2)"
    add_intercept : bool, default True
        Whether to include an intercept term.
    solve_method : str or None, default None
        Solver for the linear system: "qr" (default), "svd", or "cholesky".

    Returns
    -------
    pl.Expr
        Struct containing regression results.

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
    >>> # Main effects + interaction per group
    >>> df.group_by("group").agg(
    ...     ps.ols_formula("y ~ x1 * x2").alias("model")
    ... )
    >>>
    >>> # Polynomial regression per group (centered within each group)
    >>> df.group_by("group").agg(
    ...     ps.ols_formula("y ~ poly(x1, 2)").alias("model")
    ... )
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    response, x_exprs, _ = _parse_formula(formula)
    return ols(response, *x_exprs, add_intercept=add_intercept, solve_method=solve_method)


def ridge_formula(
    formula: str,
    lambda_: float = 1.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
    solve_method: str | None = None,
) -> pl.Expr:
    """Ridge regression using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    lambda_ : float, default 1.0
        Regularization strength.
    add_intercept : bool, default True
        Whether to include an intercept term.
    solve_method : str or None, default None
        Solver for the linear system: "qr" (default), "svd", or "cholesky".

    Returns
    -------
    pl.Expr
        Struct containing regression results.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    response, x_exprs, _ = _parse_formula(formula)
    return ridge(response, *x_exprs, lambda_=lambda_, add_intercept=add_intercept, solve_method=solve_method)


def elastic_net_formula(
    formula: str,
    lambda_: float = 1.0,
    alpha: float = 0.5,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Elastic Net regression using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    lambda_ : float, default 1.0
        Total regularization strength.
    alpha : float, default 0.5
        L1 ratio (0 = Ridge, 1 = Lasso).
    add_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing regression results.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    response, x_exprs, _ = _parse_formula(formula)
    return elastic_net(
        response, *x_exprs, lambda_=lambda_, alpha=alpha, add_intercept=add_intercept
    )


def lasso_formula(
    formula: str,
    lambda_: float = 1.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Lasso regression (L1 regularization) using R-style formula syntax.

    Shorthand for elastic_net_formula(..., alpha=1.0).

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    lambda_ : float, default 1.0
        Regularization strength.
    add_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing regression results.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return elastic_net_formula(
        formula, lambda_=lambda_, alpha=1.0, add_intercept=add_intercept
    )


def wls_formula(
    formula: str,
    weights: Union[pl.Expr, str],
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
    solve_method: str | None = None,
) -> pl.Expr:
    """Weighted Least Squares regression using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    weights : pl.Expr or str
        Observation weights column.
    add_intercept : bool, default True
        Whether to include an intercept term.
    solve_method : str or None, default None
        Solver for the linear system: "qr" (default), "svd", or "cholesky".

    Returns
    -------
    pl.Expr
        Struct containing regression results.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    response, x_exprs, _ = _parse_formula(formula)
    return wls(response, weights, *x_exprs, add_intercept=add_intercept, solve_method=solve_method)


def rls_formula(
    formula: str,
    forgetting_factor: float = 0.99,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
    initial_p_diagonal: float = 1e6,
) -> pl.Expr:
    """Recursive Least Squares regression using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    forgetting_factor : float, default 0.99
        Forgetting factor (0 < lambda <= 1).
    add_intercept : bool, default True
        Whether to include an intercept term.
    initial_p_diagonal : float, default 1e6
        See :func:`rls`.

    Returns
    -------
    pl.Expr
        Struct containing regression results.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    response, x_exprs, _ = _parse_formula(formula)
    return rls(
        response,
        *x_exprs,
        forgetting_factor=forgetting_factor,
        add_intercept=add_intercept,
        initial_p_diagonal=initial_p_diagonal,
    )


def bls_formula(
    formula: str,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Bounded Least Squares regression using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    lower_bound : float or None
        Lower bound for coefficients.
    upper_bound : float or None
        Upper bound for coefficients.
    add_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing regression results.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    response, x_exprs, _ = _parse_formula(formula)
    return bls(
        response,
        *x_exprs,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        add_intercept=add_intercept,
    )


def nnls_formula(
    formula: str,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Non-negative Least Squares regression using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    add_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing regression results.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return bls_formula(formula, lower_bound=0.0, add_intercept=add_intercept)


def logistic_formula(
    formula: str,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Logistic regression using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    add_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing GLM results.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    response, x_exprs, _ = _parse_formula(formula)
    return logistic(response, *x_exprs, add_intercept=add_intercept)


def poisson_formula(
    formula: str,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Poisson regression using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    add_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing GLM results.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    response, x_exprs, _ = _parse_formula(formula)
    return poisson(response, *x_exprs, add_intercept=add_intercept)


def negative_binomial_formula(
    formula: str,
    theta: float | None = None,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
    estimate_dispersion: bool = False,
) -> pl.Expr:
    """Negative Binomial regression using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    theta : float or None
        Dispersion parameter. If None, estimated from data.
    add_intercept : bool, default True
        Whether to include an intercept term.
    estimate_dispersion : bool, default False
        See :func:`negative_binomial`.

    Returns
    -------
    pl.Expr
        Struct containing GLM results.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    response, x_exprs, _ = _parse_formula(formula)
    return negative_binomial(
        response,
        *x_exprs,
        theta=theta,
        add_intercept=add_intercept,
        estimate_dispersion=estimate_dispersion,
    )


def tweedie_formula(
    formula: str,
    var_power: float = 1.5,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Tweedie GLM using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    var_power : float, default 1.5
        Variance power.
    add_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing GLM results.

    Notes
    -----
    ``aic``/``bic`` are NaN unless ``var_power`` is one of 0, 1, 2, or 3 — other
    powers have no closed-form Tweedie density, so the log-likelihood (and
    therefore AIC/BIC) cannot be computed. Coefficients and standard errors are
    unaffected.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    response, x_exprs, _ = _parse_formula(formula)
    return tweedie(response, *x_exprs, var_power=var_power, add_intercept=add_intercept)


def probit_formula(
    formula: str,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Probit regression using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    add_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing GLM results.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    response, x_exprs, _ = _parse_formula(formula)
    return probit(response, *x_exprs, add_intercept=add_intercept)


def cloglog_formula(
    formula: str,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Complementary log-log regression using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    add_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing GLM results.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    response, x_exprs, _ = _parse_formula(formula)
    return cloglog(response, *x_exprs, add_intercept=add_intercept)


def penalized_glm_formula(
    formula: str,
    family: Literal["poisson", "binomial", "logistic", "negative_binomial", "gamma", "tweedie"] = (
        "poisson"
    ),
    link: str | None = None,
    theta: float | None = None,
    power: float | None = None,
    threshold: float = 0.5,
    lambda_: float = 0.0,
    estimate_dispersion: bool = False,
    priors: Sequence[Union[None, str, tuple, dict]] | None = None,
    offset: Union[pl.Expr, str, None] = None,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> pl.Expr:
    """Family-generic penalized GLM using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see :func:`ols_formula` for syntax).
    family, link, theta, power, threshold, lambda_, estimate_dispersion, priors,
    add_intercept, max_iter, tol
        See :func:`penalized_glm`.
    offset : pl.Expr, str, or None
        Offset column or expression, added to the linear predictor on the
        link scale. R's formula ``offset()`` term is not parsed here — pass
        the offset as a column/expression directly.

    Returns
    -------
    pl.Expr
        Struct containing penalized GLM results (see :func:`penalized_glm`).
    """
    response, x_exprs, _ = _parse_formula(formula)
    return penalized_glm(
        response,
        *x_exprs,
        family=family,
        link=link,
        theta=theta,
        power=power,
        threshold=threshold,
        lambda_=lambda_,
        estimate_dispersion=estimate_dispersion,
        priors=priors,
        offset=offset,
        add_intercept=add_intercept,
        max_iter=max_iter,
        tol=tol,
    )


def aft_formula(
    formula: str,
    event: Union[pl.Expr, str],
    distribution: Literal["weibull", "exponential", "lognormal", "loglogistic"] = "weibull",
    priors: Sequence[Union[None, str, tuple, dict]] | None = None,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> pl.Expr:
    """AFT survival regression using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula whose left-hand side is the ``time`` column (see
        :func:`ols_formula` for syntax).
    event : pl.Expr or str
        Right-censoring indicator (see :func:`aft`). Not part of the formula
        syntax — passed separately.
    distribution, priors, add_intercept, max_iter, tol
        See :func:`aft`.

    Returns
    -------
    pl.Expr
        Struct containing AFT results (see :func:`aft`).
    """
    response, x_exprs, _ = _parse_formula(formula)
    return aft(
        response,
        event,
        *x_exprs,
        distribution=distribution,
        priors=priors,
        add_intercept=add_intercept,
        max_iter=max_iter,
        tol=tol,
    )


def alm_formula(
    formula: str,
    distribution: str = "normal",
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Augmented Linear Model using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    distribution : str, default "normal"
        Distribution family.
    add_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing ALM results.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    response, x_exprs, _ = _parse_formula(formula)
    return alm(response, *x_exprs, distribution=distribution, add_intercept=add_intercept)


# ============================================================================
# Formula-Based Summary Expressions
# ============================================================================


def ols_formula_summary(
    formula: str,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
    solve_method: str | None = None,
    hc_type: str | None = None,
) -> pl.Expr:
    """OLS coefficient summary using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    add_intercept : bool, default True
        Whether to include an intercept term.
    solve_method : str or None, default None
        Solver for the linear system: "qr" (default), "svd", or "cholesky".
    hc_type : str or None, default None
        Heteroskedasticity-consistent standard errors variant:
        "hc0", "hc1", "hc2", or "hc3". If None, uses homoskedastic SEs.

    Returns
    -------
    pl.Expr
        List of structs containing coefficient statistics.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    response, x_exprs, _ = _parse_formula(formula)
    return ols_summary(response, *x_exprs, add_intercept=add_intercept, solve_method=solve_method, hc_type=hc_type)


def ridge_formula_summary(
    formula: str,
    lambda_: float = 1.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
    solve_method: str | None = None,
) -> pl.Expr:
    """Ridge coefficient summary using R-style formula syntax."""
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    response, x_exprs, _ = _parse_formula(formula)
    return ridge_summary(response, *x_exprs, lambda_=lambda_, add_intercept=add_intercept, solve_method=solve_method)


def elastic_net_formula_summary(
    formula: str,
    lambda_: float = 1.0,
    alpha: float = 0.5,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Elastic Net coefficient summary using R-style formula syntax."""
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    response, x_exprs, _ = _parse_formula(formula)
    return elastic_net_summary(
        response, *x_exprs, lambda_=lambda_, alpha=alpha, add_intercept=add_intercept
    )


def lasso_formula_summary(
    formula: str,
    lambda_: float = 1.0,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Lasso regression coefficient summary using R-style formula syntax.

    Shorthand for elastic_net_formula_summary(..., alpha=1.0).
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    return elastic_net_formula_summary(
        formula, lambda_=lambda_, alpha=1.0, add_intercept=add_intercept
    )


def logistic_formula_summary(
    formula: str,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Logistic regression coefficient summary using R-style formula syntax."""
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    response, x_exprs, _ = _parse_formula(formula)
    return logistic_summary(response, *x_exprs, add_intercept=add_intercept)


def poisson_formula_summary(
    formula: str,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Poisson regression coefficient summary using R-style formula syntax."""
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    response, x_exprs, _ = _parse_formula(formula)
    return poisson_summary(response, *x_exprs, add_intercept=add_intercept)


def penalized_glm_formula_summary(
    formula: str,
    family: Literal["poisson", "binomial", "logistic", "negative_binomial", "gamma", "tweedie"] = (
        "poisson"
    ),
    link: str | None = None,
    theta: float | None = None,
    power: float | None = None,
    threshold: float = 0.5,
    lambda_: float = 0.0,
    estimate_dispersion: bool = False,
    priors: Sequence[Union[None, str, tuple, dict]] | None = None,
    vcov_type: Literal["laplace", "sandwich", "naive"] = "laplace",
    conf_level: float = 0.95,
    offset: Union[pl.Expr, str, None] = None,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> pl.Expr:
    """Penalized GLM coefficient summary using R-style formula syntax.

    See :func:`penalized_glm_summary` for parameter semantics.
    """
    response, x_exprs, _ = _parse_formula(formula)
    return penalized_glm_summary(
        response,
        *x_exprs,
        family=family,
        link=link,
        theta=theta,
        power=power,
        threshold=threshold,
        lambda_=lambda_,
        estimate_dispersion=estimate_dispersion,
        priors=priors,
        vcov_type=vcov_type,
        conf_level=conf_level,
        offset=offset,
        add_intercept=add_intercept,
        max_iter=max_iter,
        tol=tol,
    )


def aft_formula_summary(
    formula: str,
    event: Union[pl.Expr, str],
    distribution: Literal["weibull", "exponential", "lognormal", "loglogistic"] = "weibull",
    priors: Sequence[Union[None, str, tuple, dict]] | None = None,
    vcov_type: Literal["laplace", "sandwich", "naive"] = "laplace",
    conf_level: float = 0.95,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> pl.Expr:
    """AFT coefficient summary using R-style formula syntax.

    See :func:`aft_summary` for parameter semantics; the formula's left-hand
    side is the ``time`` column and ``event`` is passed separately.
    """
    response, x_exprs, _ = _parse_formula(formula)
    return aft_summary(
        response,
        event,
        *x_exprs,
        distribution=distribution,
        priors=priors,
        vcov_type=vcov_type,
        conf_level=conf_level,
        add_intercept=add_intercept,
        max_iter=max_iter,
        tol=tol,
    )


def alm_formula_summary(
    formula: str,
    distribution: str = "normal",
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """ALM coefficient summary using R-style formula syntax."""
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    response, x_exprs, _ = _parse_formula(formula)
    return alm_summary(response, *x_exprs, distribution=distribution, add_intercept=add_intercept)


# ============================================================================
# Formula-Based Prediction Expressions
# ============================================================================


def ols_formula_predict(
    formula: str,
    add_intercept: bool = True,
    solve_method: str | None = None,
    interval: str | None = None,
    level: float = 0.95,
    null_policy: str = "drop",
    name: str | None = None,
) -> pl.Expr:
    """OLS predictions using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    add_intercept : bool, default True
        Whether to include an intercept term.
    solve_method : str or None, default None
        Solver for the linear system: "qr" (default), "svd", or "cholesky".
    interval : str or None, default None
        "confidence" or "prediction" intervals.
    level : float, default 0.95
        Confidence level for intervals.
    null_policy : str, default "drop"
        "drop" or "drop_y_zero_x".
    name : str or None, default None
        Custom prefix for output column names. Defaults to "ols".

    Returns
    -------
    pl.Expr
        Struct containing {prefix}_prediction, {prefix}_lower, {prefix}_upper per row.

    Examples
    --------
    >>> # Per-row predictions with formula
    >>> df.with_columns(
    ...     ps.ols_formula_predict("y ~ x1 * x2", interval="prediction")
    ...         .over("group").alias("pred")
    ... ).unnest("pred")
    """
    response, x_exprs, _ = _parse_formula(formula)
    return ols_predict(
        response,
        *x_exprs,
        add_intercept=add_intercept,
        solve_method=solve_method,
        interval=interval,
        level=level,
        null_policy=null_policy,
        name=name,
    )


def ridge_formula_predict(
    formula: str,
    lambda_: float = 1.0,
    add_intercept: bool = True,
    solve_method: str | None = None,
    interval: str | None = None,
    level: float = 0.95,
    null_policy: str = "drop",
    name: str | None = None,
) -> pl.Expr:
    """Ridge predictions using R-style formula syntax."""
    response, x_exprs, _ = _parse_formula(formula)
    return ridge_predict(
        response,
        *x_exprs,
        lambda_=lambda_,
        add_intercept=add_intercept,
        solve_method=solve_method,
        interval=interval,
        level=level,
        null_policy=null_policy,
        name=name,
    )


def elastic_net_formula_predict(
    formula: str,
    lambda_: float = 1.0,
    alpha: float = 0.5,
    add_intercept: bool = True,
    interval: str | None = None,
    level: float = 0.95,
    null_policy: str = "drop",
    name: str | None = None,
) -> pl.Expr:
    """Elastic Net predictions using R-style formula syntax."""
    response, x_exprs, _ = _parse_formula(formula)
    return elastic_net_predict(
        response,
        *x_exprs,
        lambda_=lambda_,
        alpha=alpha,
        add_intercept=add_intercept,
        interval=interval,
        level=level,
        null_policy=null_policy,
        name=name,
    )


def lasso_formula_predict(
    formula: str,
    lambda_: float = 1.0,
    add_intercept: bool = True,
    interval: str | None = None,
    level: float = 0.95,
    null_policy: str = "drop",
    name: str | None = None,
) -> pl.Expr:
    """Lasso regression predictions using R-style formula syntax.

    Shorthand for elastic_net_formula_predict(..., alpha=1.0).
    """
    return elastic_net_formula_predict(
        formula,
        lambda_=lambda_,
        alpha=1.0,
        add_intercept=add_intercept,
        interval=interval,
        level=level,
        null_policy=null_policy,
        name=name if name is not None else "lasso",
    )


def logistic_formula_predict(
    formula: str,
    add_intercept: bool = True,
    interval: str | None = None,
    level: float = 0.95,
    null_policy: str = "drop",
    name: str | None = None,
) -> pl.Expr:
    """Logistic regression predictions using R-style formula syntax."""
    response, x_exprs, _ = _parse_formula(formula)
    return logistic_predict(
        response,
        *x_exprs,
        add_intercept=add_intercept,
        interval=interval,
        level=level,
        null_policy=null_policy,
        name=name,
    )


def poisson_formula_predict(
    formula: str,
    add_intercept: bool = True,
    interval: str | None = None,
    level: float = 0.95,
    null_policy: str = "drop",
    name: str | None = None,
) -> pl.Expr:
    """Poisson regression predictions using R-style formula syntax."""
    response, x_exprs, _ = _parse_formula(formula)
    return poisson_predict(
        response,
        *x_exprs,
        add_intercept=add_intercept,
        interval=interval,
        level=level,
        null_policy=null_policy,
        name=name,
    )


def penalized_glm_formula_predict(
    formula: str,
    family: Literal["poisson", "binomial", "logistic", "negative_binomial", "gamma", "tweedie"] = (
        "poisson"
    ),
    link: str | None = None,
    theta: float | None = None,
    power: float | None = None,
    threshold: float = 0.5,
    lambda_: float = 0.0,
    estimate_dispersion: bool = False,
    priors: Sequence[Union[None, str, tuple, dict]] | None = None,
    offset: Union[pl.Expr, str, None] = None,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-8,
    predict_type: Literal["response", "link"] = "response",
    null_policy: Literal["drop", "drop_y_zero_x"] = "drop",
    name: str | None = None,
) -> pl.Expr:
    """Penalized GLM predictions using R-style formula syntax.

    See :func:`penalized_glm_predict` for parameter semantics.
    """
    response, x_exprs, _ = _parse_formula(formula)
    return penalized_glm_predict(
        response,
        *x_exprs,
        family=family,
        link=link,
        theta=theta,
        power=power,
        threshold=threshold,
        lambda_=lambda_,
        estimate_dispersion=estimate_dispersion,
        priors=priors,
        offset=offset,
        add_intercept=add_intercept,
        max_iter=max_iter,
        tol=tol,
        predict_type=predict_type,
        null_policy=null_policy,
        name=name,
    )


def alm_formula_predict(
    formula: str,
    distribution: str = "normal",
    add_intercept: bool = True,
    interval: str | None = None,
    level: float = 0.95,
    null_policy: str = "drop",
    name: str | None = None,
) -> pl.Expr:
    """ALM predictions using R-style formula syntax."""
    response, x_exprs, _ = _parse_formula(formula)
    return alm_predict(
        response,
        *x_exprs,
        distribution=distribution,
        add_intercept=add_intercept,
        interval=interval,
        level=level,
        null_policy=null_policy,
        name=name,
    )


def aft_formula_predict_quantile(
    formula: str,
    event: Union[pl.Expr, str],
    p: float = 0.5,
    distribution: Literal["weibull", "exponential", "lognormal", "loglogistic"] = "weibull",
    priors: Sequence[Union[None, str, tuple, dict]] | None = None,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-9,
    null_policy: Literal["drop", "drop_y_zero_x"] = "drop",
    name: str | None = None,
) -> pl.Expr:
    """AFT predicted `p`-quantile of the survival time, using R-style formula syntax.

    See :func:`aft_predict_quantile` for parameter semantics; the formula's
    left-hand side is the ``time`` column and ``event`` is passed separately
    (see :func:`aft_formula`).
    """
    response, x_exprs, _ = _parse_formula(formula)
    return aft_predict_quantile(
        response,
        event,
        *x_exprs,
        p=p,
        distribution=distribution,
        priors=priors,
        add_intercept=add_intercept,
        max_iter=max_iter,
        tol=tol,
        null_policy=null_policy,
        name=name,
    )


def aft_formula_predict_median(
    formula: str,
    event: Union[pl.Expr, str],
    distribution: Literal["weibull", "exponential", "lognormal", "loglogistic"] = "weibull",
    priors: Sequence[Union[None, str, tuple, dict]] | None = None,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-9,
    null_policy: Literal["drop", "drop_y_zero_x"] = "drop",
    name: str | None = None,
) -> pl.Expr:
    """AFT predicted median survival time, using R-style formula syntax
    (``aft_formula_predict_quantile`` at ``p=0.5``).

    See :func:`aft_predict_median` for parameter semantics; the formula's
    left-hand side is the ``time`` column and ``event`` is passed separately
    (see :func:`aft_formula`).
    """
    response, x_exprs, _ = _parse_formula(formula)
    return aft_predict_median(
        response,
        event,
        *x_exprs,
        distribution=distribution,
        priors=priors,
        add_intercept=add_intercept,
        max_iter=max_iter,
        tol=tol,
        null_policy=null_policy,
        name=name,
    )


def aft_formula_predict_survival(
    formula: str,
    event: Union[pl.Expr, str],
    *,
    t: float,
    distribution: Literal["weibull", "exponential", "lognormal", "loglogistic"] = "weibull",
    priors: Sequence[Union[None, str, tuple, dict]] | None = None,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-9,
    null_policy: Literal["drop", "drop_y_zero_x"] = "drop",
    name: str | None = None,
) -> pl.Expr:
    """AFT predicted survival probability ``S(t | x) = P(T > t | x)``, using
    R-style formula syntax.

    See :func:`aft_predict_survival` for parameter semantics; the formula's
    left-hand side is the ``time`` column and ``event`` is passed separately
    (see :func:`aft_formula`). ``t`` is keyword-only and required.
    """
    response, x_exprs, _ = _parse_formula(formula)
    return aft_predict_survival(
        response,
        event,
        *x_exprs,
        t=t,
        distribution=distribution,
        priors=priors,
        add_intercept=add_intercept,
        max_iter=max_iter,
        tol=tol,
        null_policy=null_policy,
        name=name,
    )


def aft_formula_predict_cdf(
    formula: str,
    event: Union[pl.Expr, str],
    *,
    t: float,
    distribution: Literal["weibull", "exponential", "lognormal", "loglogistic"] = "weibull",
    priors: Sequence[Union[None, str, tuple, dict]] | None = None,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-9,
    null_policy: Literal["drop", "drop_y_zero_x"] = "drop",
    name: str | None = None,
) -> pl.Expr:
    """AFT predicted failure probability ``F(t | x) = 1 - S(t | x)``, using
    R-style formula syntax.

    See :func:`aft_predict_cdf` for parameter semantics; the formula's
    left-hand side is the ``time`` column and ``event`` is passed separately
    (see :func:`aft_formula`). ``t`` is keyword-only and required.
    """
    response, x_exprs, _ = _parse_formula(formula)
    return aft_predict_cdf(
        response,
        event,
        *x_exprs,
        t=t,
        distribution=distribution,
        priors=priors,
        add_intercept=add_intercept,
        max_iter=max_iter,
        tol=tol,
        null_policy=null_policy,
        name=name,
    )


# ============================================================================
# AID (Automatic Identification of Demand) Expression
# ============================================================================


def aid(
    y: Union[pl.Expr, str],
    intermittent_threshold: float = 0.3,
    detect_anomalies: bool = True,
) -> pl.Expr:
    """Automatic Identification of Demand (AID) classifier.

    Classifies demand patterns as regular or intermittent and selects the
    best-fitting distribution. Based on the aid function from the greybox R package.

    Parameters
    ----------
    y : pl.Expr or str
        Demand time series.
    intermittent_threshold : float, default 0.3
        Proportion of zeros above which demand is classified as intermittent.
    detect_anomalies : bool, default True
        Whether to detect anomalies (stockouts, lifecycle events, outliers).

    Returns
    -------
    pl.Expr
        Struct containing: demand_type, is_intermittent, is_fractional,
        distribution, mean, variance, zero_proportion, n_observations,
        has_stockouts, is_new_product, is_obsolete_product, stockout_count,
        new_product_count, obsolete_product_count, high_outlier_count, low_outlier_count.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({
    ...     "sku": ["A"] * 100 + ["B"] * 100,
    ...     "demand": [...]
    ... })
    >>>
    >>> # Classify demand per SKU
    >>> df.group_by("sku").agg(
    ...     ps.aid("demand").alias("classification")
    ... )
    >>>
    >>> # Access results
    >>> result.with_columns(
    ...     pl.col("classification").struct.field("demand_type"),
    ...     pl.col("classification").struct.field("distribution"),
    ... )
    """
    if isinstance(y, str):
        y = pl.col(y)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_aid",
        args=[
            y.cast(pl.Float64),
            pl.lit(intermittent_threshold, dtype=pl.Float64),
            pl.lit(detect_anomalies, dtype=pl.Boolean),
        ],
        returns_scalar=True,
    )


def aid_anomalies(
    y: Union[pl.Expr, str],
    intermittent_threshold: float = 0.3,
) -> pl.Expr:
    """AID anomaly detection - returns per-observation anomaly flags.

    Analyzes demand time series and returns per-row boolean flags indicating
    which observations are flagged as each anomaly type. Use with `.over()`
    to add anomaly columns back to the original DataFrame.

    Parameters
    ----------
    y : pl.Expr or str
        Demand time series.
    intermittent_threshold : float, default 0.3
        Proportion of zeros above which demand is classified as intermittent.

    Returns
    -------
    pl.Expr
        Struct with per-row boolean flags for each anomaly type:
        stockout, new_product, obsolete_product, high_outlier, low_outlier.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({
    ...     "sku": ["A"] * 20 + ["B"] * 20,
    ...     "demand": [...]
    ... })
    >>>
    >>> # Add anomaly flags per SKU using .over()
    >>> result = df.with_columns(
    ...     ps.aid_anomalies("demand").over("sku").alias("anomalies")
    ... )
    >>>
    >>> # Unnest to get individual columns
    >>> result = result.unnest("anomalies")
    >>> # Now has columns: sku, demand, stockout, new_product, ...
    """
    if isinstance(y, str):
        y = pl.col(y)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_aid_anomalies",
        args=[
            y.cast(pl.Float64),
            pl.lit(intermittent_threshold, dtype=pl.Float64),
        ],
        returns_scalar=False,
    )


# ============================================================================
# Dynamic Linear Model (LmDynamic) Expression
# ============================================================================


def lm_dynamic(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    ic: str = "aicc",
    distribution: str = "normal",
    lowess_span: float = 0.3,
    max_models: int = 64,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Dynamic Linear Model regression.

    A time-varying parameter model that combines multiple candidate regression
    models using pointwise information criteria weighting. Based on the
    lmDynamic function from the greybox R package.

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    *x : pl.Expr or str
        Feature variables (one or more).
    ic : str, default "aicc"
        Information criterion for model weighting.
        Options: "aic", "aicc", "bic"
    distribution : str, default "normal"
        Error distribution family (same options as ALM).
    lowess_span : float, default 0.3
        LOWESS smoothing span (0.05 to 1.0). Set to 0.0 to disable smoothing.
    max_models : int, default 64
        Maximum number of candidate models to consider.
    add_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing: intercept, coefficients (time-averaged), r_squared,
        adj_r_squared, mse, rmse, n_observations.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({
    ...     "group": ["A"] * 100 + ["B"] * 100,
    ...     "y": [...],
    ...     "x1": [...],
    ...     "x2": [...],
    ... })
    >>>
    >>> # Dynamic regression per group
    >>> df.group_by("group").agg(
    ...     ps.lm_dynamic("y", "x1", "x2", ic="aicc").alias("model")
    ... )
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    if isinstance(y, str):
        y = pl.col(y)

    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_lm_dynamic",
        args=[
            y.cast(pl.Float64),
            pl.lit(ic, dtype=pl.String),
            pl.lit(distribution, dtype=pl.String),
            pl.lit(lowess_span, dtype=pl.Float64),
            pl.lit(max_models, dtype=pl.UInt32),
            pl.lit(add_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


# ============================================================================
# Summary / Predict completeness (issue #18)
# ============================================================================


def quantile_summary(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    tau: float = 0.5,
    add_intercept: bool | None = None,
    with_intercept: bool | None = None,
) -> pl.Expr:
    """Quantile regression coefficient summary in tidy format.

    Quantile regression has no analytic standard errors, so the std_error,
    statistic, and p_value columns are NaN.
    """
    add_intercept = _resolve_intercept(add_intercept, with_intercept)
    if isinstance(y, str):
        y = pl.col(y)
    x_exprs = [(pl.col(xi) if isinstance(xi, str) else xi).cast(pl.Float64) for xi in x]
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_quantile_summary",
        args=[
            y.cast(pl.Float64),
            pl.lit(tau, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def quantile_predict(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    tau: float = 0.5,
    add_intercept: bool = True,
    null_policy: str = "drop",
    name: str | None = None,
) -> pl.Expr:
    """Quantile-regression per-row predictions.

    Lower/upper interval columns are NaN — quantile regression intervals
    require bootstrap, which is not yet wired through.
    """
    if isinstance(y, str):
        y = pl.col(y)
    x_exprs = [(pl.col(xi) if isinstance(xi, str) else xi).cast(pl.Float64) for xi in x]
    prefix = name if name is not None else "quantile"
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_quantile_predict",
        args=[
            y.cast(pl.Float64),
            pl.lit(tau, dtype=pl.Float64),
            pl.lit(add_intercept, dtype=pl.Boolean),
            pl.lit(null_policy, dtype=pl.String),
            *x_exprs,
        ],
        kwargs={"prefix": prefix},
        returns_scalar=False,
    )


def isotonic_predict(
    y: Union[pl.Expr, str],
    x: Union[pl.Expr, str],
    increasing: bool = True,
    null_policy: str = "drop",
    name: str | None = None,
) -> pl.Expr:
    """Isotonic-regression per-row predictions on a single feature.

    Lower/upper interval columns are NaN — isotonic regression produces a
    step function with no parametric uncertainty estimate.
    """
    if isinstance(y, str):
        y = pl.col(y)
    if isinstance(x, str):
        x = pl.col(x)
    prefix = name if name is not None else "isotonic"
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_isotonic_predict",
        args=[
            y.cast(pl.Float64),
            x.cast(pl.Float64),
            pl.lit(increasing, dtype=pl.Boolean),
            pl.lit(null_policy, dtype=pl.String),
        ],
        kwargs={"prefix": prefix},
        returns_scalar=False,
    )


def lm_dynamic_predict(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    ic: str = "aicc",
    distribution: str = "normal",
    lowess_span: float = 0.3,
    max_models: int = 64,
    add_intercept: bool = True,
    null_policy: str = "drop",
    name: str | None = None,
) -> pl.Expr:
    """Dynamic Linear Model per-row predictions using time-averaged coefs.

    Lower/upper interval columns are NaN — no analytic intervals for the
    time-varying coefficient mixture.
    """
    if isinstance(y, str):
        y = pl.col(y)
    x_exprs = [(pl.col(xi) if isinstance(xi, str) else xi).cast(pl.Float64) for xi in x]
    prefix = name if name is not None else "lm_dynamic"
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_lm_dynamic_predict",
        args=[
            y.cast(pl.Float64),
            pl.lit(ic, dtype=pl.String),
            pl.lit(distribution, dtype=pl.String),
            pl.lit(lowess_span, dtype=pl.Float64),
            pl.lit(max_models, dtype=pl.UInt32),
            pl.lit(add_intercept, dtype=pl.Boolean),
            pl.lit(null_policy, dtype=pl.String),
            *x_exprs,
        ],
        kwargs={"prefix": prefix},
        returns_scalar=False,
    )
