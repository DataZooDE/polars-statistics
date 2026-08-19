"""Parametric statistical tests as Polars expressions."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Union

import polars as pl
from polars.plugins import register_plugin_function

LIB = Path(__file__).parent.parent


def one_way_anova(
    *groups: Union[pl.Expr, str],
    kind: Literal["fisher", "welch"] = "fisher",
) -> pl.Expr:
    """
    Perform one-way ANOVA on two or more independent groups.

    Use this on the full DataFrame with ``df.select(ps.one_way_anova(...))``.
    Do not use inside ``group_by(...).agg(...)`` unless each group contains all
    ANOVA groups — the expression already aggregates across the named group columns.

    Parameters
    ----------
    *groups : pl.Expr or str
        Two or more column expressions or names, each holding one group's values.
    kind : {"fisher", "welch"}, default "fisher"
        ANOVA variant. ``"welch"`` does not assume equal variances; its output
        returns NaN for ss_between, ss_within, ms_between, ms_within, eta_squared.

    Returns
    -------
    pl.Expr
        Expression returning struct with fields:
        statistic (f64), df_between (f64), df_within (f64), p_value (f64),
        ss_between (f64), ss_within (f64), ms_between (f64), ms_within (f64),
        eta_squared (f64), n_groups (u32).

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({
    ...     "control": [1.0, 2.0, 3.0],
    ...     "treatment_a": [4.0, 5.0, 6.0],
    ...     "treatment_b": [7.0, 8.0, 9.0],
    ... })
    >>>
    >>> df.select(ps.one_way_anova("control", "treatment_a", "treatment_b"))
    >>> df.select(ps.one_way_anova("control", "treatment_a", "treatment_b", kind="welch"))
    """
    cleaned: list[pl.Expr] = []
    for g in groups:
        if isinstance(g, str):
            g = pl.col(g)
        cleaned.append(g.filter(g.is_finite()))

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_one_way_anova",
        args=[*cleaned, pl.lit(kind, dtype=pl.String)],
        returns_scalar=True,
    )


def two_way_anova(
    value: Union[pl.Expr, str],
    factor_a: Union[pl.Expr, str],
    factor_b: Union[pl.Expr, str],
) -> pl.Expr:
    """
    Perform two-way ANOVA with two categorical factors.

    String factor columns are automatically label-encoded via Categorical casting;
    you do not need to pre-encode them.

    Parameters
    ----------
    value : pl.Expr or str
        Column of dependent-variable values (f64).
    factor_a : pl.Expr or str
        Column of factor A labels (string or integer). Encoded automatically.
    factor_b : pl.Expr or str
        Column of factor B labels (string or integer). Encoded automatically.

    Returns
    -------
    pl.Expr
        Expression returning struct with flattened AnovaTableRow fields:
        a_ss, a_df, a_ms, a_f, a_p_value (factor A main effect),
        b_ss, b_df, b_ms, b_f, b_p_value (factor B main effect),
        ab_ss, ab_df, ab_ms, ab_f, ab_p_value (A×B interaction),
        residual_ss, residual_df, residual_ms, grand_mean (f64), n (u32).

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({
    ...     "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    ...     "method": ["A", "A", "A", "A", "B", "B", "B", "B"],
    ...     "gender": ["M", "F", "M", "F", "M", "F", "M", "F"],
    ... })
    >>>
    >>> df.select(ps.two_way_anova("score", "method", "gender"))
    """
    if isinstance(value, str):
        value = pl.col(value)
    if isinstance(factor_a, str):
        factor_a = pl.col(factor_a)
    if isinstance(factor_b, str):
        factor_b = pl.col(factor_b)

    # Drop rows where the value is non-finite OR either factor is null, so all three
    # columns stay row-aligned (the Rust side uses into_no_null_iter(), which would
    # otherwise silently shorten a column that still contained nulls — CR-02).
    mask = value.is_finite() & factor_a.is_not_null() & factor_b.is_not_null()
    value_clean = value.filter(mask)

    # Encode each factor as 0-indexed UInt32 codes AFTER filtering, so the dense
    # codes are contiguous starting at 0 over the surviving rows. Ranking before the
    # filter can leave a gap (e.g. codes [1, 2]) if a whole factor level is dropped,
    # which the crate would misread as an extra phantom level (CR-01). rank("dense")
    # is also per-column independent, avoiding the shared Categorical-catalog shift
    # that plagues to_physical() inside a group_by (Pitfall 2).
    factor_a_clean = (
        factor_a.filter(mask).rank(method="dense").cast(pl.UInt32)
        - pl.lit(1, dtype=pl.UInt32)
    )
    factor_b_clean = (
        factor_b.filter(mask).rank(method="dense").cast(pl.UInt32)
        - pl.lit(1, dtype=pl.UInt32)
    )

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_two_way_anova",
        args=[value_clean, factor_a_clean, factor_b_clean],
        returns_scalar=True,
    )


def repeated_measures_anova(
    value: Union[pl.Expr, str],
    subject: Union[pl.Expr, str],
    condition: Union[pl.Expr, str],
    compute_sphericity: bool = True,
) -> pl.Expr:
    """
    Perform repeated-measures ANOVA on long-format balanced data.

    Requires a balanced design: every subject must have exactly the same set of
    conditions. Unbalanced input returns an all-NaN struct without panicking.

    Subject and condition columns are automatically label-encoded.

    Parameters
    ----------
    value : pl.Expr or str
        Column of observed values (f64).
    subject : pl.Expr or str
        Column identifying each subject (string or integer).
    condition : pl.Expr or str
        Column identifying each condition/time-point (string or integer).
    compute_sphericity : bool, default True
        Whether to compute Mauchly's test and Greenhouse-Geisser / Huynh-Feldt
        corrections. Requires k >= 3 conditions; returns NaN for these fields if
        sphericity cannot be computed.

    Returns
    -------
    pl.Expr
        Expression returning struct with fields:
        ws_f, ws_df, ws_ss, ws_ms, ws_p_value (within-subjects effect),
        error_df, error_ss, error_ms (error term),
        mauchly_w, mauchly_p_value (sphericity test; NaN if not computed),
        gg_epsilon, gg_p_value (Greenhouse-Geisser correction; NaN if not computed),
        hf_epsilon, hf_p_value (Huynh-Feldt correction; NaN if not computed),
        grand_mean (f64).

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({
    ...     "score": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0],
    ...     "subject": [1, 1, 1, 2, 2, 2, 3, 3, 3],
    ...     "condition": [1, 2, 3, 1, 2, 3, 1, 2, 3],
    ... })
    >>>
    >>> df.select(ps.repeated_measures_anova("score", "subject", "condition"))
    """
    if isinstance(value, str):
        value = pl.col(value)
    if isinstance(subject, str):
        subject = pl.col(subject)
    if isinstance(condition, str):
        condition = pl.col(condition)

    # Drop rows where the value is non-finite OR subject/condition is null, so all
    # three columns stay row-aligned (the Rust side uses into_no_null_iter() — an
    # unmasked null would silently shorten a column and misalign the arrays, CR-02).
    mask = value.is_finite() & subject.is_not_null() & condition.is_not_null()
    value_clean = value.filter(mask)

    # Encode subject and condition as 0-indexed dense UInt32 codes AFTER filtering.
    # rank("dense") is per-column independent, so it starts at 0 regardless of the
    # global Polars Categorical catalog. The previous cast(Categorical).to_physical()
    # shared that catalog: inside a group_by(...).agg(...) an earlier group could fill
    # code slots 0..k so this group's first level got code k+1 instead of 0, breaking
    # the balance check and yielding an all-NaN result (CR-03). Ranking after the
    # filter also keeps the codes contiguous (mirrors two_way_anova, CR-01).
    subject_clean = (
        subject.filter(mask).rank(method="dense").cast(pl.UInt32)
        - pl.lit(1, dtype=pl.UInt32)
    )
    condition_clean = (
        condition.filter(mask).rank(method="dense").cast(pl.UInt32)
        - pl.lit(1, dtype=pl.UInt32)
    )

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_repeated_measures_anova",
        args=[
            value_clean,
            subject_clean,
            condition_clean,
            pl.lit(compute_sphericity, dtype=pl.Boolean),
        ],
        returns_scalar=True,
    )


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
