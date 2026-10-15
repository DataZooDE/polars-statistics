"""Empirical-Bayes shrinkage of per-group estimates, wrapping anofox-regression 0.5.17's
`eb_shrink` (DerSimonian-Laird random-effects pooling, issue #58).
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function

LIB = Path(__file__).parent.parent

_METHODS = ("dl", "dersimonian_laird", "dersimonian-laird", "none")


def _check_method(method: str) -> str:
    """Validate an eb_shrink tau-method name (case-insensitive); return it unchanged."""
    if method.lower() not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS!r} (case-insensitive), got {method!r}")
    return method


def _check_tau_squared(tau_squared: float | None) -> None:
    """Validate a fixed tau_squared override: None, or finite and non-negative."""
    if tau_squared is None:
        return
    if not math.isfinite(tau_squared) or tau_squared < 0.0:
        raise ValueError(f"tau_squared must be finite and non-negative, got {tau_squared!r}")


def eb_shrink(
    estimate: pl.Expr | str,
    std_error: pl.Expr | str,
    method: str = "dl",
    tau_squared: float | None = None,
) -> pl.Expr:
    """
    Empirical-Bayes shrinkage of per-group estimates toward their precision-weighted mean,
    matching ``metafor::rma(yi, sei = se, method = "DL")`` / ``blup()``.

    The model is the random-effects meta-analysis one::

        theta_g ~ N(mu, tau^2)        (between-group variation)
        est_g   ~ N(theta_g, se_g^2)  (within-group sampling error)

    ``tau^2`` is estimated by the DerSimonian-Laird moment estimator (or fixed, or zero
    for complete pooling), after which each group's posterior mean is the
    precision-weighted blend of its own estimate and the pooled mean. A group measured
    precisely (small ``se_g``) barely moves; a group measured poorly is pulled most of
    the way to ``mu``.

    Parameters
    ----------
    estimate : pl.Expr or str
        Per-group estimate column, or a column name.
    std_error : pl.Expr or str
        Per-group standard error column, or a column name. Same length as
        ``estimate``.
    method : {"dl", "dersimonian_laird", "dersimonian-laird", "none"}, default "dl"
        Estimator for ``tau^2`` (case-insensitive). ``"dl"`` is the
        DerSimonian-Laird moment estimator; ``"none"`` pools completely
        (``tau^2 = 0``, every ``shrunken`` equals ``mu``). Ignored when
        ``tau_squared`` is given.
    tau_squared : float, optional
        A fixed ``tau^2`` instead of estimating one via ``method``. Must be
        finite and non-negative.

    Returns
    -------
    pl.Expr
        Expression returning one struct row per input row:
        ``{shrunken, shrunken_se, weight, mu, mu_se, tau_squared, i_squared, q,
        n_groups}``. ``mu``/``mu_se``/``tau_squared``/``i_squared``/``q``/``n_groups``
        repeat on every row (whole-sample scalars); ``shrunken``/``shrunken_se``/
        ``weight`` are per-row. ``weight`` is the shrinkage weight in ``[0, 1]``
        (``tau^2 / (tau^2 + se^2)``: 1 means untouched, 0 means fully pooled).
        ``n_groups`` is ``UInt32``, the rest ``Float64``.

    Notes
    -----
    Rows with a null/non-finite ``estimate`` or a non-positive/non-finite
    ``std_error`` are excluded from ``mu``/``tau^2``/``n_groups`` but stay
    aligned with NaN ``shrunken``/``shrunken_se``/``weight``. Fewer than two
    usable rows give all-NaN rows with ``n_groups`` 0 — never an exception.

    ``i_squared`` is always Higgins-Thompson ``max(0, (Q - df) / Q)``. This
    equals metafor's ``I2 / 100`` for the DL and fixed-effect fits, but not
    when ``tau_squared`` is supplied directly (metafor then derives ``I^2``
    from the supplied ``tau^2``).

    Use ``.over(group_col)`` to shrink several independent families within one
    DataFrame; inside ``group_by().agg()`` it returns one struct list per
    group.

    Raises
    ------
    ValueError
        If ``method`` is not one of the allowed names, or ``tau_squared`` is
        negative or non-finite.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>>
    >>> df = pl.DataFrame({"yi": [0.10, 0.30, 0.35, 0.65, 1.00], "sei": [0.30, 0.10, 0.50, 0.20, 0.40]})
    >>> df.with_columns(ps.eb_shrink("yi", "sei").alias("eb")).unnest("eb")
    """
    method = _check_method(method)
    _check_tau_squared(tau_squared)

    estimate_expr = pl.col(estimate) if isinstance(estimate, str) else estimate
    std_error_expr = pl.col(std_error) if isinstance(std_error, str) else std_error

    return register_plugin_function(
        plugin_path=LIB,
        function_name="pl_eb_shrink",
        args=[
            estimate_expr.cast(pl.Float64),
            std_error_expr.cast(pl.Float64),
            pl.lit(method, dtype=pl.String),
            pl.lit(tau_squared, dtype=pl.Float64),
        ],
        returns_scalar=False,
    )
