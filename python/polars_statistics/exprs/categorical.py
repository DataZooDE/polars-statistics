"""Categorical statistical test expressions."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import polars as pl
from polars.plugins import register_plugin_function

LIB = Path(__file__).parent.parent


def binom_test(
    successes: int,
    n: int,
    p0: float = 0.5,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    conf_level: float = 0.95,
) -> pl.Expr:
    """
    Exact binomial test.

    Tests whether the proportion of successes differs from a hypothesized
    proportion p0.

    Parameters
    ----------
    successes : int
        Number of successes.
    n : int
        Total number of trials.
    p0 : float, default 0.5
        Hypothesized probability of success.
    alternative : {"two-sided", "less", "greater"}, default "two-sided"
        Alternative hypothesis direction.
    conf_level : float, default 0.95
        Confidence level of the reported interval (default 0.95 = previous
        behaviour). The Clopper-Pearson interval is two-sided for
        ``"two-sided"`` and one-sided (``[0, U]`` / ``[L, 1]``) for
        ``"less"`` / ``"greater"``, matching R's ``binom.test``.

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
    >>> # Test if coin is fair: 7 heads out of 10 flips
    >>> ps.binom_test(7, 10, p0=0.5)
    """
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_binom_test",
        args=[
            pl.lit(successes, dtype=pl.UInt32),
            pl.lit(n, dtype=pl.UInt32),
            pl.lit(p0, dtype=pl.Float64),
            pl.lit(alternative, dtype=pl.String),
            pl.lit(conf_level, dtype=pl.Float64),
        ],
        returns_scalar=True,
    )


def prop_test_one(
    successes: int,
    n: int,
    p0: float = 0.5,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    conf_level: float = 0.95,
) -> pl.Expr:
    """
    One-sample proportion test.

    Normal approximation test for a single proportion.

    Parameters
    ----------
    successes : int
        Number of successes.
    n : int
        Total number of trials.
    p0 : float, default 0.5
        Hypothesized probability of success.
    alternative : {"two-sided", "less", "greater"}, default "two-sided"
        Alternative hypothesis direction.
    conf_level : float, default 0.95
        Confidence level of the reported interval (default 0.95 = previous
        behaviour). The Wilson score interval is two-sided for
        ``"two-sided"`` and one-sided for ``"less"`` / ``"greater"``,
        matching R's ``prop.test(correct = FALSE)``.

    Returns
    -------
    pl.Expr
        Expression returning struct{estimate, statistic, p_value,
        ci_lower, ci_upper, n}.
    """
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_prop_test_one",
        args=[
            pl.lit(successes, dtype=pl.UInt32),
            pl.lit(n, dtype=pl.UInt32),
            pl.lit(p0, dtype=pl.Float64),
            pl.lit(alternative, dtype=pl.String),
            pl.lit(conf_level, dtype=pl.Float64),
        ],
        returns_scalar=True,
    )


def prop_test_two(
    successes1: int,
    n1: int,
    successes2: int,
    n2: int,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    correction: bool = False,
    conf_level: float = 0.95,
) -> pl.Expr:
    """
    Two-sample proportion test.

    Tests whether two proportions are equal.

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
    alternative : {"two-sided", "less", "greater"}, default "two-sided"
        Alternative hypothesis direction.
    correction : bool, default False
        Apply Yates' continuity correction.
    conf_level : float, default 0.95
        Confidence level of the reported interval (default 0.95 = previous
        behaviour). The interval for ``p1 - p2`` is two-sided for
        ``"two-sided"`` and one-sided for ``"less"`` / ``"greater"``,
        matching R's ``prop.test``.

    Returns
    -------
    pl.Expr
        Expression returning struct{estimate, statistic, p_value,
        ci_lower, ci_upper, n}.
    """
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_prop_test_two",
        args=[
            pl.lit(successes1, dtype=pl.UInt32),
            pl.lit(n1, dtype=pl.UInt32),
            pl.lit(successes2, dtype=pl.UInt32),
            pl.lit(n2, dtype=pl.UInt32),
            pl.lit(alternative, dtype=pl.String),
            pl.lit(correction, dtype=pl.Boolean),
            pl.lit(conf_level, dtype=pl.Float64),
        ],
        returns_scalar=True,
    )


def chisq_test(
    data: Union[pl.Expr, str],
    n_rows: int = 2,
    n_cols: int = 2,
    correction: bool = False,
) -> pl.Expr:
    """
    Chi-square test for independence in contingency table.

    Parameters
    ----------
    data : pl.Expr or str
        Flattened contingency table data (row-major order).
    n_rows : int, default 2
        Number of rows in the contingency table.
    n_cols : int, default 2
        Number of columns in the contingency table.
    correction : bool, default False
        Apply Yates' continuity correction (only for 2x2 tables).

    Returns
    -------
    pl.Expr
        Expression returning struct{statistic, p_value, df, n}.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> # 2x2 contingency table: [[10, 20], [30, 40]]
    >>> df = pl.DataFrame({"counts": [10, 20, 30, 40]})
    >>> df.select(ps.chisq_test("counts", n_rows=2, n_cols=2))
    """
    if isinstance(data, str):
        data = pl.col(data)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_chisq_test",
        args=[
            data.cast(pl.UInt32),
            pl.lit(n_rows, dtype=pl.UInt32),
            pl.lit(n_cols, dtype=pl.UInt32),
            pl.lit(correction, dtype=pl.Boolean),
        ],
        returns_scalar=True,
    )


def chisq_goodness_of_fit(
    observed: Union[pl.Expr, str],
    expected: Optional[Union[pl.Expr, str]] = None,
) -> pl.Expr:
    """
    Chi-square goodness-of-fit test.

    Tests whether observed frequencies match expected frequencies.

    Parameters
    ----------
    observed : pl.Expr or str
        Observed frequency counts.
    expected : pl.Expr or str, optional
        Expected frequency counts. If None, assumes uniform distribution.

    Returns
    -------
    pl.Expr
        Expression returning struct{statistic, p_value, df, n}.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> # Test if die is fair: observed counts for each face
    >>> df = pl.DataFrame({"observed": [18, 22, 16, 19, 24, 21]})
    >>> df.select(ps.chisq_goodness_of_fit("observed"))
    """
    if isinstance(observed, str):
        observed = pl.col(observed)

    if expected is None:
        return register_plugin_function(
            plugin_path=LIB,
            function_name="pl_chisq_goodness_of_fit",
            args=[
                observed.cast(pl.UInt32),
                pl.lit(False, dtype=pl.Boolean),
            ],
            returns_scalar=True,
        )
    else:
        if isinstance(expected, str):
            expected = pl.col(expected)
        return register_plugin_function(
            plugin_path=LIB,
            function_name="pl_chisq_goodness_of_fit",
            args=[
                observed.cast(pl.UInt32),
                pl.lit(True, dtype=pl.Boolean),
                expected.cast(pl.Float64),
            ],
            returns_scalar=True,
        )


def g_test(
    data: Union[pl.Expr, str],
    n_rows: int = 2,
    n_cols: int = 2,
) -> pl.Expr:
    """
    G-test (likelihood ratio test) for independence.

    An alternative to chi-square test using log-likelihood ratio.

    Parameters
    ----------
    data : pl.Expr or str
        Flattened contingency table data (row-major order).
    n_rows : int, default 2
        Number of rows in the contingency table.
    n_cols : int, default 2
        Number of columns in the contingency table.

    Returns
    -------
    pl.Expr
        Expression returning struct{statistic, p_value, df, n}.
    """
    if isinstance(data, str):
        data = pl.col(data)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_g_test",
        args=[
            data.cast(pl.UInt32),
            pl.lit(n_rows, dtype=pl.UInt32),
            pl.lit(n_cols, dtype=pl.UInt32),
        ],
        returns_scalar=True,
    )


def fisher_exact(
    a: int,
    b: int,
    c: int,
    d: int,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    conf_level: float = 0.95,
) -> pl.Expr:
    """
    Fisher's exact test for 2x2 contingency tables.

    Computes exact p-value for the test of independence.

    Parameters
    ----------
    a, b, c, d : int
        Cell counts in the 2x2 table:
        [[a, b],
         [c, d]]
    alternative : {"two-sided", "less", "greater"}, default "two-sided"
        Alternative hypothesis direction.
    conf_level : float, default 0.95
        Confidence level of the reported interval (default 0.95 = previous
        behaviour).

    Returns
    -------
    pl.Expr
        Expression returning struct{statistic (sample odds ratio ad/bc),
        p_value, ci_lower, ci_upper}. The interval is the two-sided Woolf
        (logit) interval at ``conf_level``. For R's conditional
        maximum-likelihood odds ratio and exact conditional interval (full
        ``fisher.test`` semantics, including one-sided intervals), use
        :func:`fisher_exact_conditional`.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> # Test association in 2x2 table
    >>> ps.fisher_exact(10, 2, 3, 15)
    """
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_fisher_exact",
        args=[
            pl.lit(a, dtype=pl.UInt32),
            pl.lit(b, dtype=pl.UInt32),
            pl.lit(c, dtype=pl.UInt32),
            pl.lit(d, dtype=pl.UInt32),
            pl.lit(alternative, dtype=pl.String),
            pl.lit(conf_level, dtype=pl.Float64),
        ],
        returns_scalar=True,
    )


def fisher_exact_conditional(
    a: int,
    b: int,
    c: int,
    d: int,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    conf_level: float = 0.95,
) -> pl.Expr:
    """
    Fisher's exact test for 2x2 tables with full R `fisher.test` semantics.

    Unlike :func:`fisher_exact` (which reports the *sample* odds ratio
    ``ad/bc`` and a Woolf interval), this reports the conditional
    maximum-likelihood estimate of the odds ratio and the exact conditional
    confidence interval, exactly as R's ``fisher.test``.

    Parameters
    ----------
    a, b, c, d : int
        Cell counts in the 2x2 table:
        [[a, b],
         [c, d]]
    alternative : {"two-sided", "less", "greater"}, default "two-sided"
        Alternative hypothesis direction. One-sided alternatives give
        one-sided intervals: ``"greater"`` gives ``ci_upper = +inf``,
        ``"less"`` gives ``ci_lower = 0.0``.
    conf_level : float, default 0.95
        Confidence level of the reported interval (default 0.95 = previous
        behaviour).

    Returns
    -------
    pl.Expr
        Expression returning struct{statistic (conditional MLE odds ratio),
        p_value (identical to :func:`fisher_exact`), ci_lower, ci_upper
        (exact conditional interval)}.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> # R: fisher.test(matrix(c(18, 9, 7, 16), 2))
    >>> ps.fisher_exact_conditional(18, 7, 9, 16)
    """
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_fisher_exact_conditional",
        args=[
            pl.lit(a, dtype=pl.UInt32),
            pl.lit(b, dtype=pl.UInt32),
            pl.lit(c, dtype=pl.UInt32),
            pl.lit(d, dtype=pl.UInt32),
            pl.lit(alternative, dtype=pl.String),
            pl.lit(conf_level, dtype=pl.Float64),
        ],
        returns_scalar=True,
    )


def mcnemar_test(
    a: int,
    b: int,
    c: int,
    d: int,
    correction: bool = False,
) -> pl.Expr:
    """
    McNemar's test for paired proportions.

    Tests for significant changes in paired binary data.

    Parameters
    ----------
    a, b, c, d : int
        Cell counts in the 2x2 table:
        [[a, b],
         [c, d]]
        where b and c are the discordant pairs.
    correction : bool, default False
        Apply Edwards' continuity correction.

    Returns
    -------
    pl.Expr
        Expression returning struct{statistic, p_value, df, n}.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> # Before/after treatment: 45 yes->yes, 15 yes->no, 5 no->yes, 35 no->no
    >>> ps.mcnemar_test(45, 15, 5, 35)
    """
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_mcnemar_test",
        args=[
            pl.lit(a, dtype=pl.UInt32),
            pl.lit(b, dtype=pl.UInt32),
            pl.lit(c, dtype=pl.UInt32),
            pl.lit(d, dtype=pl.UInt32),
            pl.lit(correction, dtype=pl.Boolean),
        ],
        returns_scalar=True,
    )


def mcnemar_exact(
    a: int,
    b: int,
    c: int,
    d: int,
) -> pl.Expr:
    """
    McNemar's exact test for paired proportions.

    Uses exact binomial distribution for small sample sizes.

    Parameters
    ----------
    a, b, c, d : int
        Cell counts in the 2x2 table.

    Returns
    -------
    pl.Expr
        Expression returning struct{statistic, p_value}.
    """
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_mcnemar_exact",
        args=[
            pl.lit(a, dtype=pl.UInt32),
            pl.lit(b, dtype=pl.UInt32),
            pl.lit(c, dtype=pl.UInt32),
            pl.lit(d, dtype=pl.UInt32),
        ],
        returns_scalar=True,
    )


def cohen_kappa(
    data: Union[pl.Expr, str],
    n_categories: int = 2,
    weighted: bool = False,
) -> pl.Expr:
    """
    Cohen's Kappa for inter-rater agreement.

    Measures the level of agreement between two raters above chance.

    Parameters
    ----------
    data : pl.Expr or str
        Flattened confusion matrix data (row-major order).
    n_categories : int, default 2
        Number of rating categories.
    weighted : bool, default False
        If True, compute weighted kappa using linear weights.

    Returns
    -------
    pl.Expr
        Expression returning struct{estimate (kappa), statistic (se), p_value}.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> # Confusion matrix: [[20, 5], [3, 22]]
    >>> df = pl.DataFrame({"counts": [20, 5, 3, 22]})
    >>> df.select(ps.cohen_kappa("counts", n_categories=2))
    """
    if isinstance(data, str):
        data = pl.col(data)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_cohen_kappa",
        args=[
            data.cast(pl.UInt32),
            pl.lit(n_categories, dtype=pl.UInt32),
            pl.lit(weighted, dtype=pl.Boolean),
        ],
        returns_scalar=True,
    )


def cramers_v(
    data: Union[pl.Expr, str],
    n_rows: int = 2,
    n_cols: int = 2,
) -> pl.Expr:
    """
    Cramer's V for association strength.

    Measures the strength of association between two categorical variables.
    Values range from 0 (no association) to 1 (perfect association).

    Parameters
    ----------
    data : pl.Expr or str
        Flattened contingency table data (row-major order).
    n_rows : int, default 2
        Number of rows in the contingency table.
    n_cols : int, default 2
        Number of columns in the contingency table.

    Returns
    -------
    pl.Expr
        Expression returning struct{estimate (V), statistic (se), p_value}.
    """
    if isinstance(data, str):
        data = pl.col(data)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_cramers_v",
        args=[
            data.cast(pl.UInt32),
            pl.lit(n_rows, dtype=pl.UInt32),
            pl.lit(n_cols, dtype=pl.UInt32),
        ],
        returns_scalar=True,
    )


def phi_coefficient(
    a: int,
    b: int,
    c: int,
    d: int,
) -> pl.Expr:
    """
    Phi coefficient for 2x2 tables.

    Measures the correlation between two binary variables.
    Equivalent to Pearson correlation for dichotomous data.

    Parameters
    ----------
    a, b, c, d : int
        Cell counts in the 2x2 table.

    Returns
    -------
    pl.Expr
        Expression returning struct{estimate (phi), statistic (se), p_value}.
    """
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_phi_coefficient",
        args=[
            pl.lit(a, dtype=pl.UInt32),
            pl.lit(b, dtype=pl.UInt32),
            pl.lit(c, dtype=pl.UInt32),
            pl.lit(d, dtype=pl.UInt32),
        ],
        returns_scalar=True,
    )


def contingency_coef(
    data: Union[pl.Expr, str],
    n_rows: int = 2,
    n_cols: int = 2,
) -> pl.Expr:
    """
    Contingency coefficient (Pearson's C).

    Measures association in contingency tables. Related to chi-square,
    with values ranging from 0 to a maximum that depends on table size.

    Parameters
    ----------
    data : pl.Expr or str
        Flattened contingency table data (row-major order).
    n_rows : int, default 2
        Number of rows in the contingency table.
    n_cols : int, default 2
        Number of columns in the contingency table.

    Returns
    -------
    pl.Expr
        Expression returning struct{estimate, statistic (se), p_value}.
    """
    if isinstance(data, str):
        data = pl.col(data)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_contingency_coef",
        args=[
            data.cast(pl.UInt32),
            pl.lit(n_rows, dtype=pl.UInt32),
            pl.lit(n_cols, dtype=pl.UInt32),
        ],
        returns_scalar=True,
    )
