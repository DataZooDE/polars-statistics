"""Non-parametric statistical tests as Polars expressions."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl
from polars.plugins import register_plugin_function

LIB = Path(__file__).parent.parent


def mann_whitney_u(
    x: pl.Expr | str,
    y: pl.Expr | str,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    continuity_correction: bool = True,
    exact: bool = False,
    conf_level: float | None = None,
    mu: float | None = None,
) -> pl.Expr:
    """
    Perform Mann-Whitney U test (Wilcoxon rank-sum test).

    A non-parametric test for testing whether two independent samples
    were drawn from the same distribution.

    Parameters
    ----------
    x : pl.Expr or str
        First sample expression or column name.
    y : pl.Expr or str
        Second sample expression or column name.
    alternative : {"two-sided", "less", "greater"}, default "two-sided"
        Alternative hypothesis direction.
    continuity_correction : bool, default True
        Whether to apply continuity correction for the normal approximation.
    exact : bool, default False
        Whether to use exact method (only for small samples).
    conf_level : float, optional
        Confidence level for the Hodges-Lehmann estimator confidence interval.
    mu : float, optional
        Hypothesized location shift under null hypothesis.

    Returns
    -------
    pl.Expr
        Expression returning struct{statistic: f64, p_value: f64, estimate: f64,
        ci_lower: f64, ci_upper: f64}. `estimate` is the Hodges-Lehmann estimate of
        the location shift (difference in medians); `estimate`/`ci_lower`/`ci_upper`
        are NaN unless `conf_level` is given, matching R's `wilcox.test`, which only
        reports `estimate`/`conf.int` when `conf.int = TRUE`. A one-sided
        `alternative` ("less"/"greater") may give an infinite bound. Values match
        R `wilcox.test(..., conf.int = TRUE)`.

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
    >>> df.select(ps.mann_whitney_u("x", "y", conf_level=0.95))
    """
    if isinstance(x, str):
        x = pl.col(x)
    if isinstance(y, str):
        y = pl.col(y)

    x_clean = x.filter(x.is_finite())
    y_clean = y.filter(y.is_finite())

    conf_level_expr = pl.lit(conf_level, dtype=pl.Float64) if conf_level is not None else pl.lit(None, dtype=pl.Float64)
    mu_expr = pl.lit(mu, dtype=pl.Float64) if mu is not None else pl.lit(None, dtype=pl.Float64)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_mann_whitney_u",
        args=[
            x_clean,
            y_clean,
            pl.lit(alternative, dtype=pl.String),
            pl.lit(continuity_correction, dtype=pl.Boolean),
            pl.lit(exact, dtype=pl.Boolean),
            conf_level_expr,
            mu_expr,
        ],
        returns_scalar=True,
    )


def wilcoxon_signed_rank(
    x: pl.Expr | str,
    y: pl.Expr | str,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    continuity_correction: bool = True,
    exact: bool = False,
    conf_level: float | None = None,
    mu: float | None = None,
) -> pl.Expr:
    """
    Perform Wilcoxon signed-rank test.

    A non-parametric test for paired samples, testing whether
    paired differences come from a symmetric distribution around zero.

    Parameters
    ----------
    x : pl.Expr or str
        First sample expression or column name.
    y : pl.Expr or str
        Second sample expression or column name.
    alternative : {"two-sided", "less", "greater"}, default "two-sided"
        Alternative hypothesis direction.
    continuity_correction : bool, default True
        Whether to apply continuity correction for the normal approximation.
    exact : bool, default False
        Whether to use exact method (only for small samples).
    conf_level : float, optional
        Confidence level for the pseudomedian confidence interval.
    mu : float, optional
        Hypothesized location under null hypothesis.

    Returns
    -------
    pl.Expr
        Expression returning struct{statistic: f64, p_value: f64, estimate: f64,
        ci_lower: f64, ci_upper: f64}. `estimate` is the Hodges-Lehmann pseudo-median
        of x - y; `estimate`/`ci_lower`/`ci_upper` are NaN unless `conf_level` is
        given, matching R's `wilcox.test`, which only reports `estimate`/`conf.int`
        when `conf.int = TRUE`. A one-sided `alternative` ("less"/"greater") may
        give an infinite bound. Values match R `wilcox.test(..., conf.int = TRUE)`.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({
    ...     "x": [1.0, 2.0, 3.0, 4.0, 5.0],
    ...     "y": [1.2, 2.1, 2.9, 4.3, 4.8],
    ... })
    >>>
    >>> df.select(ps.wilcoxon_signed_rank("x", "y"))
    >>> df.select(ps.wilcoxon_signed_rank("x", "y", conf_level=0.95))
    """
    if isinstance(x, str):
        x = pl.col(x)
    if isinstance(y, str):
        y = pl.col(y)

    x_clean = x.filter(x.is_finite() & y.is_finite())
    y_clean = y.filter(x.is_finite() & y.is_finite())

    conf_level_expr = pl.lit(conf_level, dtype=pl.Float64) if conf_level is not None else pl.lit(None, dtype=pl.Float64)
    mu_expr = pl.lit(mu, dtype=pl.Float64) if mu is not None else pl.lit(None, dtype=pl.Float64)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_wilcoxon_signed_rank",
        args=[
            x_clean,
            y_clean,
            pl.lit(alternative, dtype=pl.String),
            pl.lit(continuity_correction, dtype=pl.Boolean),
            pl.lit(exact, dtype=pl.Boolean),
            conf_level_expr,
            mu_expr,
        ],
        returns_scalar=True,
    )


def kruskal_wallis(
    *groups: pl.Expr | str,
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
    x: pl.Expr | str,
    y: pl.Expr | str,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    alpha: float | None = None,
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
    alpha : float, optional
        Significance level for the confidence interval.

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

    alpha_expr = pl.lit(alpha, dtype=pl.Float64) if alpha is not None else pl.lit(None, dtype=pl.Float64)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_brunner_munzel",
        args=[
            x_clean,
            y_clean,
            pl.lit(alternative, dtype=pl.String),
            alpha_expr,
        ],
        returns_scalar=True,
    )


def rank_biserial(
    x: pl.Expr | str,
    y: pl.Expr | str,
    mu: float | None = None,
) -> pl.Expr:
    """
    Compute the rank-biserial correlation effect size for the Mann-Whitney U test.

    Parameters
    ----------
    x : pl.Expr or str
        First sample expression or column name.
    y : pl.Expr or str
        Second sample expression or column name.
    mu : float, optional
        Location shift applied to ``y`` under the null hypothesis, exactly as
        in :func:`mann_whitney_u`.

    Returns
    -------
    pl.Expr
        Expression returning struct{estimate: f64}.

    Notes
    -----
    ``r = 1 - 2 * U1 / (n1 * n2)``, where ``U1`` is the Mann-Whitney U
    statistic of ``x`` (:attr:`MannWhitneyU.statistic`, or
    ``ps.mann_whitney_u(...).struct.field("statistic")``). ``r > 0`` when
    ``y`` tends to be larger than ``x``; R's ``effectsize::rank_biserial``
    reports the opposite sign. Non-finite values are dropped by this builder
    (the :class:`MannWhitneyU` class instead ranks +/-Inf like R). This
    expression is O(n1 * n2) time with O(1) memory, the same complexity as
    the Mann-Whitney U statistic itself.

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
    >>> df.select(ps.rank_biserial("x", "y"))
    """
    if isinstance(x, str):
        x = pl.col(x)
    if isinstance(y, str):
        y = pl.col(y)

    x_clean = x.filter(x.is_finite())
    y_clean = y.filter(y.is_finite())

    mu_expr = pl.lit(mu, dtype=pl.Float64) if mu is not None else pl.lit(None, dtype=pl.Float64)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_rank_biserial",
        args=[x_clean, y_clean, mu_expr],
        returns_scalar=True,
    )


def matched_pairs_rank_biserial(
    x: pl.Expr | str,
    y: pl.Expr | str,
    mu: float | None = None,
) -> pl.Expr:
    """
    Compute the matched-pairs rank-biserial correlation effect size for the
    Wilcoxon signed-rank test.

    Parameters
    ----------
    x : pl.Expr or str
        First sample (before) expression or column name.
    y : pl.Expr or str
        Second sample (after) expression or column name.
    mu : float, optional
        Hypothesized location under the null hypothesis, exactly as in
        :func:`wilcoxon_signed_rank`.

    Returns
    -------
    pl.Expr
        Expression returning struct{estimate: f64}.

    Notes
    -----
    ``r = (T- - T+) / (T+ + T-)``, the sums of signed ranks of
    ``|x - y - mu|`` for positive / negative differences (zero differences
    dropped). ``r > 0`` when ``y`` tends to be larger than ``x``; R's
    ``effectsize::rank_biserial(x, y, paired = TRUE)`` reports the opposite
    sign. Returns NaN when every difference is zero or lengths are unequal.
    Both columns must be finite in the same rows (jointly filtered).

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
    >>> df.select(ps.matched_pairs_rank_biserial("before", "after"))
    """
    if isinstance(x, str):
        x = pl.col(x)
    if isinstance(y, str):
        y = pl.col(y)

    x_clean = x.filter(x.is_finite() & y.is_finite())
    y_clean = y.filter(x.is_finite() & y.is_finite())

    mu_expr = pl.lit(mu, dtype=pl.Float64) if mu is not None else pl.lit(None, dtype=pl.Float64)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_matched_pairs_rank_biserial",
        args=[x_clean, y_clean, mu_expr],
        returns_scalar=True,
    )


def rank_biserial_from_u(
    u1: float,
    n1: int,
    n2: int,
) -> pl.Expr:
    """
    Compute the rank-biserial correlation directly from a Mann-Whitney U statistic.

    Parameters
    ----------
    u1 : float
        The Mann-Whitney U statistic of the first sample (R's ``W``):
        :attr:`MannWhitneyU.statistic`, or
        ``ps.mann_whitney_u(...).struct.field("statistic")``.
    n1 : int
        Size of the first sample.
    n2 : int
        Size of the second sample.

    Returns
    -------
    pl.Expr
        Expression returning struct{estimate: f64}.

    Notes
    -----
    ``r = 1 - 2 * u1 / (n1 * n2)``; returns NaN when ``n1 * n2 == 0``. Same
    sign convention as :func:`rank_biserial`: positive when the second
    sample tends to be larger (R's ``effectsize::rank_biserial`` uses the
    opposite sign).

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>> from polars_statistics import MannWhitneyU
    >>> import numpy as np
    >>>
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    >>> u1 = MannWhitneyU().fit(x, y).statistic
    >>> pl.select(ps.rank_biserial_from_u(u1, 5, 5))
    """
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_rank_biserial_from_u",
        args=[
            pl.lit(u1, dtype=pl.Float64),
            pl.lit(n1, dtype=pl.UInt32),
            pl.lit(n2, dtype=pl.UInt32),
        ],
        returns_scalar=True,
    )
