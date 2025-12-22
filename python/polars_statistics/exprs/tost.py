"""TOST (Two One-Sided Tests) equivalence testing expressions."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import polars as pl
from polars.plugins import register_plugin_function

LIB = Path(__file__).parent.parent


def tost_t_test_one_sample(
    x: Union[pl.Expr, str],
    mu: float = 0.0,
    bounds_type: Literal["symmetric", "raw", "cohen_d"] = "symmetric",
    delta: float = 0.5,
    lower: float = -0.5,
    upper: float = 0.5,
    alpha: float = 0.05,
) -> pl.Expr:
    """
    One-sample TOST equivalence test.

    Tests whether the mean of x is practically equivalent to mu
    using Two One-Sided Tests procedure.

    Parameters
    ----------
    x : pl.Expr or str
        Sample expression or column name.
    mu : float, default 0.0
        Hypothesized population mean.
    bounds_type : {"symmetric", "raw", "cohen_d"}, default "symmetric"
        Type of equivalence bounds:
        - "symmetric": Uses delta for symmetric bounds (-delta, +delta)
        - "raw": Uses lower and upper as raw bounds
        - "cohen_d": Uses delta as Cohen's d effect size
    delta : float, default 0.5
        Equivalence bound for symmetric/cohen_d bounds.
    lower : float, default -0.5
        Lower equivalence bound for raw bounds.
    upper : float, default 0.5
        Upper equivalence bound for raw bounds.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    pl.Expr
        Expression returning struct with estimate, ci_lower, ci_upper,
        bound_lower, bound_upper, tost_p_value, equivalent, alpha, n.
    """
    if isinstance(x, str):
        x = pl.col(x)

    x_clean = x.filter(x.is_finite())

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_tost_t_test_one_sample",
        args=[
            x_clean,
            pl.lit(mu, dtype=pl.Float64),
            pl.lit(bounds_type, dtype=pl.String),
            pl.lit(delta, dtype=pl.Float64),
            pl.lit(lower, dtype=pl.Float64),
            pl.lit(upper, dtype=pl.Float64),
            pl.lit(alpha, dtype=pl.Float64),
        ],
        returns_scalar=True,
    )


def tost_t_test_two_sample(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    bounds_type: Literal["symmetric", "raw", "cohen_d"] = "symmetric",
    delta: float = 0.5,
    lower: float = -0.5,
    upper: float = 0.5,
    alpha: float = 0.05,
    pooled: bool = False,
) -> pl.Expr:
    """
    Two-sample TOST equivalence test.

    Tests whether the difference in means between x and y is practically
    equivalent to zero using Two One-Sided Tests procedure.

    Parameters
    ----------
    x : pl.Expr or str
        First sample expression or column name.
    y : pl.Expr or str
        Second sample expression or column name.
    bounds_type : {"symmetric", "raw", "cohen_d"}, default "symmetric"
        Type of equivalence bounds.
    delta : float, default 0.5
        Equivalence bound for symmetric/cohen_d bounds.
    lower : float, default -0.5
        Lower equivalence bound for raw bounds.
    upper : float, default 0.5
        Upper equivalence bound for raw bounds.
    alpha : float, default 0.05
        Significance level.
    pooled : bool, default False
        If True, use pooled variance (assumes equal variances).

    Returns
    -------
    pl.Expr
        Expression returning struct with equivalence test results.
    """
    if isinstance(x, str):
        x = pl.col(x)
    if isinstance(y, str):
        y = pl.col(y)

    x_clean = x.filter(x.is_finite())
    y_clean = y.filter(y.is_finite())

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_tost_t_test_two_sample",
        args=[
            x_clean,
            y_clean,
            pl.lit(bounds_type, dtype=pl.String),
            pl.lit(delta, dtype=pl.Float64),
            pl.lit(lower, dtype=pl.Float64),
            pl.lit(upper, dtype=pl.Float64),
            pl.lit(alpha, dtype=pl.Float64),
            pl.lit(pooled, dtype=pl.Boolean),
        ],
        returns_scalar=True,
    )


def tost_t_test_paired(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    bounds_type: Literal["symmetric", "raw", "cohen_d"] = "symmetric",
    delta: float = 0.5,
    lower: float = -0.5,
    upper: float = 0.5,
    alpha: float = 0.05,
) -> pl.Expr:
    """
    Paired-samples TOST equivalence test.

    Tests whether the mean difference between paired observations is
    practically equivalent to zero.

    Parameters
    ----------
    x : pl.Expr or str
        First sample (e.g., before treatment).
    y : pl.Expr or str
        Second sample (e.g., after treatment).
    bounds_type : {"symmetric", "raw", "cohen_d"}, default "symmetric"
        Type of equivalence bounds.
    delta : float, default 0.5
        Equivalence bound for symmetric/cohen_d bounds.
    lower : float, default -0.5
        Lower equivalence bound for raw bounds.
    upper : float, default 0.5
        Upper equivalence bound for raw bounds.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    pl.Expr
        Expression returning struct with equivalence test results.
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
        function_name="pl_tost_t_test_paired",
        args=[
            x_clean,
            y_clean,
            pl.lit(bounds_type, dtype=pl.String),
            pl.lit(delta, dtype=pl.Float64),
            pl.lit(lower, dtype=pl.Float64),
            pl.lit(upper, dtype=pl.Float64),
            pl.lit(alpha, dtype=pl.Float64),
        ],
        returns_scalar=True,
    )


def tost_correlation(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    method: Literal["pearson", "spearman"] = "pearson",
    rho_null: float = 0.0,
    bounds_type: Literal["symmetric", "raw"] = "symmetric",
    delta: float = 0.3,
    lower: float = -0.3,
    upper: float = 0.3,
    alpha: float = 0.05,
) -> pl.Expr:
    """
    Correlation TOST equivalence test.

    Tests whether a correlation coefficient is practically equivalent to
    rho_null (typically 0) using Fisher's z-transformation.

    Parameters
    ----------
    x : pl.Expr or str
        First variable.
    y : pl.Expr or str
        Second variable.
    method : {"pearson", "spearman"}, default "pearson"
        Correlation method to use.
    rho_null : float, default 0.0
        Null value for correlation (usually 0).
    bounds_type : {"symmetric", "raw"}, default "symmetric"
        Type of equivalence bounds in correlation scale.
    delta : float, default 0.3
        Equivalence bound for symmetric bounds.
    lower : float, default -0.3
        Lower equivalence bound for raw bounds.
    upper : float, default 0.3
        Upper equivalence bound for raw bounds.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    pl.Expr
        Expression returning struct with equivalence test results.
    """
    if isinstance(x, str):
        x = pl.col(x)
    if isinstance(y, str):
        y = pl.col(y)

    # Both must be finite for correlation
    x_clean = x.filter(x.is_finite() & y.is_finite())
    y_clean = y.filter(x.is_finite() & y.is_finite())

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_tost_correlation",
        args=[
            x_clean,
            y_clean,
            pl.lit(method, dtype=pl.String),
            pl.lit(rho_null, dtype=pl.Float64),
            pl.lit(bounds_type, dtype=pl.String),
            pl.lit(delta, dtype=pl.Float64),
            pl.lit(lower, dtype=pl.Float64),
            pl.lit(upper, dtype=pl.Float64),
            pl.lit(alpha, dtype=pl.Float64),
        ],
        returns_scalar=True,
    )


def tost_prop_one(
    successes: int,
    n: int,
    p0: float = 0.5,
    bounds_type: Literal["symmetric", "raw"] = "symmetric",
    delta: float = 0.1,
    lower: float = -0.1,
    upper: float = 0.1,
    alpha: float = 0.05,
) -> pl.Expr:
    """
    One-proportion TOST equivalence test.

    Tests whether an observed proportion is practically equivalent to p0.

    Parameters
    ----------
    successes : int
        Number of successes.
    n : int
        Total number of trials.
    p0 : float, default 0.5
        Hypothesized proportion under null.
    bounds_type : {"symmetric", "raw"}, default "symmetric"
        Type of equivalence bounds.
    delta : float, default 0.1
        Equivalence bound for symmetric bounds.
    lower : float, default -0.1
        Lower equivalence bound for raw bounds.
    upper : float, default 0.1
        Upper equivalence bound for raw bounds.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    pl.Expr
        Expression returning struct with equivalence test results.
    """
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_tost_prop_one",
        args=[
            pl.lit(successes, dtype=pl.UInt32),
            pl.lit(n, dtype=pl.UInt32),
            pl.lit(p0, dtype=pl.Float64),
            pl.lit(bounds_type, dtype=pl.String),
            pl.lit(delta, dtype=pl.Float64),
            pl.lit(lower, dtype=pl.Float64),
            pl.lit(upper, dtype=pl.Float64),
            pl.lit(alpha, dtype=pl.Float64),
        ],
        returns_scalar=True,
    )


def tost_prop_two(
    successes1: int,
    n1: int,
    successes2: int,
    n2: int,
    bounds_type: Literal["symmetric", "raw"] = "symmetric",
    delta: float = 0.1,
    lower: float = -0.1,
    upper: float = 0.1,
    alpha: float = 0.05,
) -> pl.Expr:
    """
    Two-proportion TOST equivalence test.

    Tests whether the difference between two proportions is practically
    equivalent to zero.

    Parameters
    ----------
    successes1 : int
        Number of successes in first sample.
    n1 : int
        Total number of trials in first sample.
    successes2 : int
        Number of successes in second sample.
    n2 : int
        Total number of trials in second sample.
    bounds_type : {"symmetric", "raw"}, default "symmetric"
        Type of equivalence bounds.
    delta : float, default 0.1
        Equivalence bound for symmetric bounds.
    lower : float, default -0.1
        Lower equivalence bound for raw bounds.
    upper : float, default 0.1
        Upper equivalence bound for raw bounds.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    pl.Expr
        Expression returning struct with equivalence test results.
    """
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_tost_prop_two",
        args=[
            pl.lit(successes1, dtype=pl.UInt32),
            pl.lit(n1, dtype=pl.UInt32),
            pl.lit(successes2, dtype=pl.UInt32),
            pl.lit(n2, dtype=pl.UInt32),
            pl.lit(bounds_type, dtype=pl.String),
            pl.lit(delta, dtype=pl.Float64),
            pl.lit(lower, dtype=pl.Float64),
            pl.lit(upper, dtype=pl.Float64),
            pl.lit(alpha, dtype=pl.Float64),
        ],
        returns_scalar=True,
    )


def tost_wilcoxon_paired(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    bounds_type: Literal["symmetric", "raw"] = "symmetric",
    delta: float = 0.5,
    lower: float = -0.5,
    upper: float = 0.5,
    alpha: float = 0.05,
) -> pl.Expr:
    """
    Wilcoxon paired-samples TOST equivalence test.

    Non-parametric equivalence test for paired samples using the
    Wilcoxon signed-rank test.

    Parameters
    ----------
    x : pl.Expr or str
        First sample.
    y : pl.Expr or str
        Second sample.
    bounds_type : {"symmetric", "raw"}, default "symmetric"
        Type of equivalence bounds.
    delta : float, default 0.5
        Equivalence bound for symmetric bounds.
    lower : float, default -0.5
        Lower equivalence bound for raw bounds.
    upper : float, default 0.5
        Upper equivalence bound for raw bounds.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    pl.Expr
        Expression returning struct with equivalence test results.
    """
    if isinstance(x, str):
        x = pl.col(x)
    if isinstance(y, str):
        y = pl.col(y)

    x_clean = x.filter(x.is_finite() & y.is_finite())
    y_clean = y.filter(x.is_finite() & y.is_finite())

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_tost_wilcoxon_paired",
        args=[
            x_clean,
            y_clean,
            pl.lit(bounds_type, dtype=pl.String),
            pl.lit(delta, dtype=pl.Float64),
            pl.lit(lower, dtype=pl.Float64),
            pl.lit(upper, dtype=pl.Float64),
            pl.lit(alpha, dtype=pl.Float64),
        ],
        returns_scalar=True,
    )


def tost_wilcoxon_two_sample(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    bounds_type: Literal["symmetric", "raw"] = "symmetric",
    delta: float = 0.5,
    lower: float = -0.5,
    upper: float = 0.5,
    alpha: float = 0.05,
) -> pl.Expr:
    """
    Wilcoxon two-sample TOST equivalence test.

    Non-parametric equivalence test for two independent samples using the
    Mann-Whitney U test.

    Parameters
    ----------
    x : pl.Expr or str
        First sample.
    y : pl.Expr or str
        Second sample.
    bounds_type : {"symmetric", "raw"}, default "symmetric"
        Type of equivalence bounds.
    delta : float, default 0.5
        Equivalence bound for symmetric bounds.
    lower : float, default -0.5
        Lower equivalence bound for raw bounds.
    upper : float, default 0.5
        Upper equivalence bound for raw bounds.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    pl.Expr
        Expression returning struct with equivalence test results.
    """
    if isinstance(x, str):
        x = pl.col(x)
    if isinstance(y, str):
        y = pl.col(y)

    x_clean = x.filter(x.is_finite())
    y_clean = y.filter(y.is_finite())

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_tost_wilcoxon_two_sample",
        args=[
            x_clean,
            y_clean,
            pl.lit(bounds_type, dtype=pl.String),
            pl.lit(delta, dtype=pl.Float64),
            pl.lit(lower, dtype=pl.Float64),
            pl.lit(upper, dtype=pl.Float64),
            pl.lit(alpha, dtype=pl.Float64),
        ],
        returns_scalar=True,
    )


def tost_bootstrap(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    bounds_type: Literal["symmetric", "raw"] = "symmetric",
    delta: float = 0.5,
    lower: float = -0.5,
    upper: float = 0.5,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
) -> pl.Expr:
    """
    Bootstrap TOST equivalence test.

    Uses bootstrap resampling to construct confidence intervals for
    equivalence testing when distributional assumptions are violated.

    Parameters
    ----------
    x : pl.Expr or str
        First sample.
    y : pl.Expr or str
        Second sample.
    bounds_type : {"symmetric", "raw"}, default "symmetric"
        Type of equivalence bounds.
    delta : float, default 0.5
        Equivalence bound for symmetric bounds.
    lower : float, default -0.5
        Lower equivalence bound for raw bounds.
    upper : float, default 0.5
        Upper equivalence bound for raw bounds.
    alpha : float, default 0.05
        Significance level.
    n_bootstrap : int, default 1000
        Number of bootstrap resamples.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pl.Expr
        Expression returning struct with equivalence test results.
    """
    if isinstance(x, str):
        x = pl.col(x)
    if isinstance(y, str):
        y = pl.col(y)

    x_clean = x.filter(x.is_finite())
    y_clean = y.filter(y.is_finite())

    seed_lit = pl.lit(seed, dtype=pl.UInt64) if seed is not None else pl.lit(None, dtype=pl.UInt64)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_tost_bootstrap",
        args=[
            x_clean,
            y_clean,
            pl.lit(bounds_type, dtype=pl.String),
            pl.lit(delta, dtype=pl.Float64),
            pl.lit(lower, dtype=pl.Float64),
            pl.lit(upper, dtype=pl.Float64),
            pl.lit(alpha, dtype=pl.Float64),
            pl.lit(n_bootstrap, dtype=pl.UInt32),
            seed_lit,
        ],
        returns_scalar=True,
    )


def tost_yuen(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    trim: float = 0.2,
    bounds_type: Literal["symmetric", "raw"] = "symmetric",
    delta: float = 0.5,
    lower: float = -0.5,
    upper: float = 0.5,
    alpha: float = 0.05,
) -> pl.Expr:
    """
    Yuen TOST equivalence test using trimmed means.

    Robust equivalence test that uses trimmed means, making it less
    sensitive to outliers and violations of normality.

    Parameters
    ----------
    x : pl.Expr or str
        First sample.
    y : pl.Expr or str
        Second sample.
    trim : float, default 0.2
        Proportion to trim from each end (0 to 0.5).
    bounds_type : {"symmetric", "raw"}, default "symmetric"
        Type of equivalence bounds.
    delta : float, default 0.5
        Equivalence bound for symmetric bounds.
    lower : float, default -0.5
        Lower equivalence bound for raw bounds.
    upper : float, default 0.5
        Upper equivalence bound for raw bounds.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    pl.Expr
        Expression returning struct with equivalence test results.
    """
    if isinstance(x, str):
        x = pl.col(x)
    if isinstance(y, str):
        y = pl.col(y)

    x_clean = x.filter(x.is_finite())
    y_clean = y.filter(y.is_finite())

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_tost_yuen",
        args=[
            x_clean,
            y_clean,
            pl.lit(trim, dtype=pl.Float64),
            pl.lit(bounds_type, dtype=pl.String),
            pl.lit(delta, dtype=pl.Float64),
            pl.lit(lower, dtype=pl.Float64),
            pl.lit(upper, dtype=pl.Float64),
            pl.lit(alpha, dtype=pl.Float64),
        ],
        returns_scalar=True,
    )
