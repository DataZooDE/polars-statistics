"""Post-hoc multiple-comparison tests wrapping anofox-statistics 0.4.7's `posthoc` module.

- ``p_adjust``: R's ``p.adjust`` (holm, hochberg, hommel, bonferroni, BH / fdr, BY, none).
- ``ptukey`` / ``qtukey``: R's studentized-range distribution (Task 2).
- ``tukey_hsd`` / ``pairwise_t_test`` / ``dunn_test``: long-format pairwise comparisons
  matching R's ``TukeyHSD(aov(...))`` / ``pairwise.t.test`` / ``dunn.test`` (Task 3).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function

LIB = Path(__file__).parent.parent

_P_ADJUST_METHODS = ("holm", "hochberg", "hommel", "bonferroni", "bh", "fdr", "by", "none")


def _check_p_adjust_method(method: str) -> str:
    """Validate a p-value adjustment method name (case-insensitive); return it unchanged."""
    if method.lower() not in _P_ADJUST_METHODS:
        raise ValueError(
            f"method must be one of {_P_ADJUST_METHODS!r} (case-insensitive), got {method!r}"
        )
    return method


def _as_float_expr(v: pl.Expr | str | int | float) -> pl.Expr:
    """Coerce a column name, expression, or scalar number to a Float64 expression."""
    if isinstance(v, str):
        v = pl.col(v)
    elif isinstance(v, (int, float)):
        v = pl.lit(float(v))
    return v.cast(pl.Float64)


def p_adjust(p: pl.Expr | str, method: str = "holm") -> pl.Expr:
    """
    Adjust p-values for multiple comparisons, matching R's ``p.adjust(p, method)``.

    Parameters
    ----------
    p : pl.Expr or str
        Column of raw p-values, or a column name.
    method : str, default "holm"
        One of ``"holm"``, ``"hochberg"``, ``"hommel"``, ``"bonferroni"``,
        ``"BH"`` (alias ``"fdr"``), ``"BY"``, ``"none"`` (case-insensitive).

    Returns
    -------
    pl.Expr
        Float64 expression, same length as ``p``.

    Notes
    -----
    Null and NaN entries are kept (null stays null, NaN stays NaN) and are
    excluded from the comparison count ``n``, matching R's ``NA`` handling.

    This is a whole-column operation, not an elementwise one: it adjusts the
    entire family of p-values together. Use ``.over(group_col)`` to adjust
    several independent families within one DataFrame; inside
    ``group_by().agg()`` it returns one adjusted list per group.

    Raises
    ------
    ValueError
        If ``method`` is not one of the allowed names.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({"p": [0.01, 0.02, 0.03, 0.04, 0.05]})
    >>> df.select(ps.p_adjust("p", method="holm"))
    """
    method = _check_p_adjust_method(method)

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_p_adjust",
        args=[_as_float_expr(p), pl.lit(method, dtype=pl.String)],
        returns_scalar=False,
    )


def ptukey(
    q: pl.Expr | str | float,
    nmeans: pl.Expr | str | float,
    df: pl.Expr | str | float,
    nranges: pl.Expr | str | float = 1,
    lower_tail: bool = True,
) -> pl.Expr:
    """
    Studentized range distribution CDF, matching R's ``ptukey(q, nmeans, df, nranges)``.

    Used by Tukey's HSD to convert the studentized range statistic to a
    p-value. Agrees with R to within 1e-10.

    Parameters
    ----------
    q : pl.Expr, str, or float
        Studentized range quantile, column name, expression, or scalar.
    nmeans : pl.Expr, str, or float
        Number of means in the range.
    df : pl.Expr, str, or float
        Degrees of freedom.
    nranges : pl.Expr, str, or float, default 1
        Number of ranges.
    lower_tail : bool, default True
        If True, return P(Q <= q); if False, return P(Q > q).

    Returns
    -------
    pl.Expr
        Float64 expression. Each argument may be a scalar (broadcasts
        against the others) or a column (all columns must share the same
        length).

    Notes
    -----
    Returns NaN (never an exception) for NaN arguments or invalid
    parameters (``df < 2``, ``nranges < 1``, ``nmeans < 2``).

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({"q": [2.0, 3.0, 4.0]})
    >>> df.select(ps.ptukey("q", nmeans=3, df=10))
    """
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_ptukey",
        args=[
            _as_float_expr(q),
            _as_float_expr(nmeans),
            _as_float_expr(df),
            _as_float_expr(nranges),
            pl.lit(lower_tail, dtype=pl.Boolean),
        ],
        returns_scalar=False,
        is_elementwise=True,
    )


def qtukey(
    p: pl.Expr | str | float,
    nmeans: pl.Expr | str | float,
    df: pl.Expr | str | float,
    nranges: pl.Expr | str | float = 1,
    lower_tail: bool = True,
) -> pl.Expr:
    """
    Studentized range distribution quantile function, matching R's
    ``qtukey(p, nmeans, df, nranges)``.

    The inverse of :func:`ptukey`; used by Tukey's HSD to compute the
    critical value for a given confidence level. Agrees with R to within
    1e-8.

    Parameters
    ----------
    p : pl.Expr, str, or float
        Probability in [0, 1], column name, expression, or scalar.
    nmeans : pl.Expr, str, or float
        Number of means in the range.
    df : pl.Expr, str, or float
        Degrees of freedom.
    nranges : pl.Expr, str, or float, default 1
        Number of ranges.
    lower_tail : bool, default True
        If True, ``p`` is P(Q <= q); if False, ``p`` is P(Q > q).

    Returns
    -------
    pl.Expr
        Float64 expression. Each argument may be a scalar (broadcasts
        against the others) or a column (all columns must share the same
        length).

    Notes
    -----
    Returns NaN (never an exception) for NaN arguments, invalid parameters
    (``df < 2``, ``nranges < 1``, ``nmeans < 2``), or ``p`` outside [0, 1].

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({"p": [0.9, 0.95, 0.99]})
    >>> df.select(ps.qtukey("p", nmeans=3, df=10))
    """
    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_qtukey",
        args=[
            _as_float_expr(p),
            _as_float_expr(nmeans),
            _as_float_expr(df),
            _as_float_expr(nranges),
            pl.lit(lower_tail, dtype=pl.Boolean),
        ],
        returns_scalar=False,
        is_elementwise=True,
    )
