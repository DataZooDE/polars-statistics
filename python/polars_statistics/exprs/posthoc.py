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
