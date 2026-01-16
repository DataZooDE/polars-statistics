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
# Robust Regression Expressions
# ============================================================================


def quantile(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    tau: float = 0.5,
    with_intercept: bool = True,
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
    with_intercept : bool, default True
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
            pl.lit(with_intercept, dtype=pl.Boolean),
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
    with_intercept: bool = True,
) -> pl.Expr:
    """Compute condition number diagnostics for a design matrix.

    The condition number measures how sensitive the regression is to numerical
    errors. A high condition number indicates multicollinearity, which can make
    coefficients unreliable.

    Parameters
    ----------
    *x : pl.Expr or str
        Feature variables (one or more).
    with_intercept : bool, default True
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
    x_exprs = []
    for xi in x:
        if isinstance(xi, str):
            xi = pl.col(xi)
        x_exprs.append(xi.cast(pl.Float64))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_condition_number",
        args=[
            pl.lit(with_intercept, dtype=pl.Boolean),
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


# ============================================================================
# GLM Expressions
# ============================================================================


def logistic(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    with_intercept: bool = True,
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
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def poisson(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    with_intercept: bool = True,
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
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def negative_binomial(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    theta: float | None = None,
    lambda_: float = 0.0,
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
    lambda_ : float, default 0.0
        L2 (Ridge) regularization strength.
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
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def tweedie(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    var_power: float = 1.5,
    lambda_: float = 0.0,
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
    lambda_ : float, default 0.0
        L2 (Ridge) regularization strength.
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
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def probit(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    with_intercept: bool = True,
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
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def cloglog(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    with_intercept: bool = True,
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
            pl.lit(lambda_, dtype=pl.Float64),
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


# ============================================================================
# ALM Expressions
# ============================================================================


def alm(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    distribution: str = "normal",
    with_intercept: bool = True,
) -> pl.Expr:
    """Augmented Linear Model (ALM) as a Polars expression.

    A flexible regression model supporting 24+ distributions.

    Parameters
    ----------
    y : pl.Expr or str
        Target variable.
    *x : pl.Expr or str
        Feature variables (one or more).
    distribution : str, default "normal"
        Distribution family. Options include:
        - Continuous: "normal", "laplace", "student_t", "logistic"
        - Positive: "lognormal", "loglaplace", "gamma", "inverse_gaussian", "exponential"
        - Bounded (0,1): "beta"
        - Count: "poisson", "negative_binomial", "binomial", "geometric"
    with_intercept : bool, default True
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
            pl.lit(with_intercept, dtype=pl.Boolean),
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
    with_intercept: bool = True,
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
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        List of structs containing coefficient statistics.

    Examples
    --------
    >>> df.group_by("group").agg(
    ...     ps.ols_summary("y", "x1", "x2").alias("coef")
    ... ).explode("coef").unnest("coef")
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
        function_name="pl_ols_summary",
        args=[
            y.cast(pl.Float64),
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def ridge_summary(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 1.0,
    with_intercept: bool = True,
) -> pl.Expr:
    """Ridge regression coefficient summary in tidy format."""
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
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def elastic_net_summary(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 1.0,
    alpha: float = 0.5,
    with_intercept: bool = True,
) -> pl.Expr:
    """Elastic Net regression coefficient summary in tidy format."""
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
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def wls_summary(
    y: Union[pl.Expr, str],
    weights: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr:
    """Weighted Least Squares coefficient summary in tidy format."""
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
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def rls_summary(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    forgetting_factor: float = 0.99,
    with_intercept: bool = True,
) -> pl.Expr:
    """Recursive Least Squares coefficient summary in tidy format."""
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
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def bls_summary(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    with_intercept: bool = True,
) -> pl.Expr:
    """Bounded Least Squares coefficient summary in tidy format."""
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
            pl.lit(with_intercept, dtype=pl.Boolean),
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
    with_intercept: bool = True,
) -> pl.Expr:
    """Logistic regression coefficient summary in tidy format."""
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
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def poisson_summary(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    with_intercept: bool = True,
) -> pl.Expr:
    """Poisson regression coefficient summary in tidy format."""
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
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def negative_binomial_summary(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    theta: float | None = None,
    lambda_: float = 0.0,
    with_intercept: bool = True,
) -> pl.Expr:
    """Negative Binomial regression coefficient summary in tidy format."""
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
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def tweedie_summary(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    var_power: float = 1.5,
    lambda_: float = 0.0,
    with_intercept: bool = True,
) -> pl.Expr:
    """Tweedie GLM coefficient summary in tidy format."""
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
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def probit_summary(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    with_intercept: bool = True,
) -> pl.Expr:
    """Probit regression coefficient summary in tidy format."""
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
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def cloglog_summary(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    lambda_: float = 0.0,
    with_intercept: bool = True,
) -> pl.Expr:
    """Complementary log-log regression coefficient summary in tidy format."""
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
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )


def alm_summary(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    distribution: str = "normal",
    with_intercept: bool = True,
) -> pl.Expr:
    """Augmented Linear Model coefficient summary in tidy format."""
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
            pl.lit(with_intercept, dtype=pl.Boolean),
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


def wls_predict(
    y: Union[pl.Expr, str],
    weights: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    add_intercept: bool = True,
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
        kwargs={"prefix": prefix},
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
        kwargs={"prefix": prefix},
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
    with_intercept: bool = True,
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
    with_intercept : bool, default True
        Whether to include an intercept term.

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
    response, x_exprs, _ = _parse_formula(formula)
    return ols(response, *x_exprs, with_intercept=with_intercept)


def ridge_formula(
    formula: str,
    lambda_: float = 1.0,
    with_intercept: bool = True,
) -> pl.Expr:
    """Ridge regression using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    lambda_ : float, default 1.0
        Regularization strength.
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing regression results.
    """
    response, x_exprs, _ = _parse_formula(formula)
    return ridge(response, *x_exprs, lambda_=lambda_, with_intercept=with_intercept)


def elastic_net_formula(
    formula: str,
    lambda_: float = 1.0,
    alpha: float = 0.5,
    with_intercept: bool = True,
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
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing regression results.
    """
    response, x_exprs, _ = _parse_formula(formula)
    return elastic_net(
        response, *x_exprs, lambda_=lambda_, alpha=alpha, with_intercept=with_intercept
    )


def wls_formula(
    formula: str,
    weights: Union[pl.Expr, str],
    with_intercept: bool = True,
) -> pl.Expr:
    """Weighted Least Squares regression using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    weights : pl.Expr or str
        Observation weights column.
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing regression results.
    """
    response, x_exprs, _ = _parse_formula(formula)
    return wls(response, weights, *x_exprs, with_intercept=with_intercept)


def rls_formula(
    formula: str,
    forgetting_factor: float = 0.99,
    with_intercept: bool = True,
) -> pl.Expr:
    """Recursive Least Squares regression using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    forgetting_factor : float, default 0.99
        Forgetting factor (0 < lambda <= 1).
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing regression results.
    """
    response, x_exprs, _ = _parse_formula(formula)
    return rls(
        response, *x_exprs, forgetting_factor=forgetting_factor, with_intercept=with_intercept
    )


def bls_formula(
    formula: str,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    with_intercept: bool = True,
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
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing regression results.
    """
    response, x_exprs, _ = _parse_formula(formula)
    return bls(
        response,
        *x_exprs,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        with_intercept=with_intercept,
    )


def nnls_formula(
    formula: str,
    with_intercept: bool = True,
) -> pl.Expr:
    """Non-negative Least Squares regression using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing regression results.
    """
    return bls_formula(formula, lower_bound=0.0, with_intercept=with_intercept)


def logistic_formula(
    formula: str,
    with_intercept: bool = True,
) -> pl.Expr:
    """Logistic regression using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing GLM results.
    """
    response, x_exprs, _ = _parse_formula(formula)
    return logistic(response, *x_exprs, with_intercept=with_intercept)


def poisson_formula(
    formula: str,
    with_intercept: bool = True,
) -> pl.Expr:
    """Poisson regression using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing GLM results.
    """
    response, x_exprs, _ = _parse_formula(formula)
    return poisson(response, *x_exprs, with_intercept=with_intercept)


def negative_binomial_formula(
    formula: str,
    theta: float | None = None,
    with_intercept: bool = True,
) -> pl.Expr:
    """Negative Binomial regression using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    theta : float or None
        Dispersion parameter. If None, estimated from data.
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing GLM results.
    """
    response, x_exprs, _ = _parse_formula(formula)
    return negative_binomial(response, *x_exprs, theta=theta, with_intercept=with_intercept)


def tweedie_formula(
    formula: str,
    var_power: float = 1.5,
    with_intercept: bool = True,
) -> pl.Expr:
    """Tweedie GLM using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    var_power : float, default 1.5
        Variance power.
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing GLM results.
    """
    response, x_exprs, _ = _parse_formula(formula)
    return tweedie(response, *x_exprs, var_power=var_power, with_intercept=with_intercept)


def probit_formula(
    formula: str,
    with_intercept: bool = True,
) -> pl.Expr:
    """Probit regression using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing GLM results.
    """
    response, x_exprs, _ = _parse_formula(formula)
    return probit(response, *x_exprs, with_intercept=with_intercept)


def cloglog_formula(
    formula: str,
    with_intercept: bool = True,
) -> pl.Expr:
    """Complementary log-log regression using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing GLM results.
    """
    response, x_exprs, _ = _parse_formula(formula)
    return cloglog(response, *x_exprs, with_intercept=with_intercept)


def alm_formula(
    formula: str,
    distribution: str = "normal",
    with_intercept: bool = True,
) -> pl.Expr:
    """Augmented Linear Model using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    distribution : str, default "normal"
        Distribution family.
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        Struct containing ALM results.
    """
    response, x_exprs, _ = _parse_formula(formula)
    return alm(response, *x_exprs, distribution=distribution, with_intercept=with_intercept)


# ============================================================================
# Formula-Based Summary Expressions
# ============================================================================


def ols_formula_summary(
    formula: str,
    with_intercept: bool = True,
) -> pl.Expr:
    """OLS coefficient summary using R-style formula syntax.

    Parameters
    ----------
    formula : str
        R-style formula (see ols_formula for syntax).
    with_intercept : bool, default True
        Whether to include an intercept term.

    Returns
    -------
    pl.Expr
        List of structs containing coefficient statistics.
    """
    response, x_exprs, _ = _parse_formula(formula)
    return ols_summary(response, *x_exprs, with_intercept=with_intercept)


def ridge_formula_summary(
    formula: str,
    lambda_: float = 1.0,
    with_intercept: bool = True,
) -> pl.Expr:
    """Ridge coefficient summary using R-style formula syntax."""
    response, x_exprs, _ = _parse_formula(formula)
    return ridge_summary(response, *x_exprs, lambda_=lambda_, with_intercept=with_intercept)


def elastic_net_formula_summary(
    formula: str,
    lambda_: float = 1.0,
    alpha: float = 0.5,
    with_intercept: bool = True,
) -> pl.Expr:
    """Elastic Net coefficient summary using R-style formula syntax."""
    response, x_exprs, _ = _parse_formula(formula)
    return elastic_net_summary(
        response, *x_exprs, lambda_=lambda_, alpha=alpha, with_intercept=with_intercept
    )


def logistic_formula_summary(
    formula: str,
    with_intercept: bool = True,
) -> pl.Expr:
    """Logistic regression coefficient summary using R-style formula syntax."""
    response, x_exprs, _ = _parse_formula(formula)
    return logistic_summary(response, *x_exprs, with_intercept=with_intercept)


def poisson_formula_summary(
    formula: str,
    with_intercept: bool = True,
) -> pl.Expr:
    """Poisson regression coefficient summary using R-style formula syntax."""
    response, x_exprs, _ = _parse_formula(formula)
    return poisson_summary(response, *x_exprs, with_intercept=with_intercept)


def alm_formula_summary(
    formula: str,
    distribution: str = "normal",
    with_intercept: bool = True,
) -> pl.Expr:
    """ALM coefficient summary using R-style formula syntax."""
    response, x_exprs, _ = _parse_formula(formula)
    return alm_summary(response, *x_exprs, distribution=distribution, with_intercept=with_intercept)


# ============================================================================
# Formula-Based Prediction Expressions
# ============================================================================


def ols_formula_predict(
    formula: str,
    add_intercept: bool = True,
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
        interval=interval,
        level=level,
        null_policy=null_policy,
        name=name,
    )


def ridge_formula_predict(
    formula: str,
    lambda_: float = 1.0,
    add_intercept: bool = True,
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
    with_intercept: bool = True,
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
    with_intercept : bool, default True
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
            pl.lit(with_intercept, dtype=pl.Boolean),
            *x_exprs,
        ],
        returns_scalar=True,
    )
