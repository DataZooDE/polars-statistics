"""Result-ergonomics helpers for the expression surface (ERGO-01, ERGO-03).

Expression results such as ``df.select(ps.ols("y", "x1", "x2"))`` or
``df.group_by(...).agg(ps.ttest_ind("x", "y"))`` come back as **Polars Struct
columns**, not Python objects.  You therefore cannot hang ``.to_dict()`` or a
``.summary()`` off them directly.  These helpers give the Struct-column surface
the same one-call ergonomics the fitted-model classes have:

* :func:`unnest` flattens a result Struct column into flat DataFrame columns.
* :func:`struct_to_dict` converts a result Struct column into a plain dict
  (single row) or list of dicts (multiple rows) without manual
  ``.struct.field(...)`` extraction.

Examples
--------
>>> import polars as pl
>>> import polars_statistics as ps
>>> df = pl.DataFrame({"y": [1.0, 2, 3, 4], "x": [2.0, 4, 6, 9]})
>>> res = df.select(ps.ttest_ind("y", "x").alias("test"))
>>> ps.struct_to_dict(res, "test")            # doctest: +SKIP
{'statistic': -1.7, 'p_value': 0.15}
>>> ps.unnest(res, "test").columns            # doctest: +SKIP
['statistic', 'p_value']
"""

from __future__ import annotations

from typing import Any

import polars as pl

__all__ = ["unnest", "struct_to_dict"]


def _resolve_struct_column(df: pl.DataFrame, column: str | None) -> str:
    """Return the name of the Struct column to operate on.

    If ``column`` is given, validate it is a Struct.  If it is ``None``, and the
    DataFrame has exactly one Struct column, use that; otherwise raise a helpful
    error.
    """
    if column is not None:
        if column not in df.columns:
            raise ValueError(
                f"Column {column!r} not found. Available columns: {df.columns}"
            )
        if not isinstance(df.schema[column], pl.Struct):
            raise TypeError(
                f"Column {column!r} is not a Struct column "
                f"(got {df.schema[column]}). "
                "unnest/struct_to_dict operate on statistical result Structs."
            )
        return column

    struct_cols = [name for name, dt in df.schema.items() if isinstance(dt, pl.Struct)]
    if len(struct_cols) == 1:
        return struct_cols[0]
    if not struct_cols:
        raise ValueError(
            "No Struct column found to unnest. Pass `column=` explicitly."
        )
    raise ValueError(
        f"Multiple Struct columns found ({struct_cols}); pass `column=` to pick one."
    )


def unnest(
    df: pl.DataFrame,
    column: str | None = None,
    *,
    prefix: str | None = None,
    keep_others: bool = True,
) -> pl.DataFrame:
    """Flatten a statistical result Struct column into flat DataFrame columns.

    A thin, well-behaved wrapper over :meth:`polars.DataFrame.unnest` that
    resolves the target column, optionally prefixes the produced fields to avoid
    collisions with existing columns, and preserves the other columns.

    Parameters
    ----------
    df
        DataFrame containing a result Struct column (e.g. the output of
        ``df.select(ps.ols(...))`` or ``df.group_by(...).agg(ps.ttest_ind(...))``).
    column
        Name of the Struct column to flatten. If ``None`` and the DataFrame has
        exactly one Struct column, it is used automatically.
    prefix
        If given, every produced field is renamed ``f"{prefix}{field}"``. Handy
        to disambiguate two result structs unnested into the same frame.
    keep_others
        If ``True`` (default) non-struct columns are preserved. If ``False`` the
        result contains only the flattened struct fields.

    Returns
    -------
    polars.DataFrame
        DataFrame with the struct fields promoted to top-level columns.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>> df = pl.DataFrame({"y": [1.0, 2, 3, 4], "x": [2.0, 4, 6, 9]})
    >>> out = ps.unnest(df.select(ps.ttest_ind("y", "x").alias("t")))
    >>> sorted(out.columns)
    ['p_value', 'statistic']
    """
    name = _resolve_struct_column(df, column)

    target = df if keep_others else df.select(name)

    if prefix:
        dtype = target.schema[name]
        assert isinstance(dtype, pl.Struct)
        fields = [f.name for f in dtype.fields]
        rename_expr = pl.col(name).struct.rename_fields(
            [f"{prefix}{f}" for f in fields]
        )
        target = target.with_columns(rename_expr)

    return target.unnest(name)


def struct_to_dict(
    df: pl.DataFrame,
    column: str | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Convert a result Struct column into a plain Python dict (or list of dicts).

    Removes the need to call ``.struct.field(...)`` for each field. When the
    column has exactly one row (the common case for a whole-frame test such as
    ``df.select(ps.ttest_ind(...))``) a single ``dict`` is returned; when it has
    multiple rows (e.g. a grouped aggregation) a ``list[dict]`` — one per row —
    is returned.

    Parameters
    ----------
    df
        DataFrame containing a result Struct column.
    column
        Name of the Struct column. If ``None`` and there is exactly one Struct
        column, it is used automatically.

    Returns
    -------
    dict or list of dict
        The struct field values keyed by field name.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_statistics as ps
    >>> df = pl.DataFrame({"y": [1.0, 2, 3, 4], "x": [2.0, 4, 6, 9]})
    >>> d = ps.struct_to_dict(df.select(ps.ttest_ind("y", "x").alias("t")))
    >>> sorted(d)
    ['p_value', 'statistic']
    """
    name = _resolve_struct_column(df, column)
    rows = df.select(name).unnest(name).to_dicts()
    if len(rows) == 1:
        return rows[0]
    return rows
