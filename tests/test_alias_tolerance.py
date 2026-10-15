"""R-reference tests for the `alias_tolerance` kwarg (anofox-regression 0.5.21
#64, R 4.6.1).

`anofox-regression`'s `OlsRegressorBuilder`/`WlsRegressorBuilder::alias_tolerance`
(default `anofox_regression::utils::LM_ALIAS_TOLERANCE = 1e-7`, R `lm.fit`'s
`tol`) is exposed on `OLS`/`WLS` (PyModel constructor kwarg) and on the
`ols`/`wls`/`ols_summary`/`wls_summary`/`ols_predict`/`wls_predict` expressions
(and their formula wrappers). Moving the tolerance moves R's rank-deficiency
("aliasing") decision boundary for near-collinear designs: a column within
`alias_tolerance` (relative to its own norm) of the span of the intercept and
the earlier kept columns gets a NaN coefficient.

R references are copied verbatim (with provenance) from the `anofox-regression`
0.5.22 crate's own test fixtures, read from the local cargo registry cache:
`~/.cargo/registry/src/*/anofox-regression-0.5.22/tests/issue64_lm_aliasing.rs`
(`near`, `near_direction`, `near_collinear_below_tolerance_is_aliased`,
`near_collinear_above_tolerance_is_kept`), generator script
`tests/r_scripts/issue64_lm_aliasing.R`.

See `.planning/phases/16.3-upstream-api-wave-2/16.3-07-PLAN.md`.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import polars_statistics as ps
import pytest
from polars_statistics import OLS, WLS

# ----------------------------------------------------------------------------
# Issue #64 data generator (N=60), reproduced from
# anofox-regression-0.5.22/tests/issue64_lm_aliasing.rs: `x1`, `x2`, `y`,
# `near_direction`, `near(ratio)`.
# ----------------------------------------------------------------------------
_N = 60
_T = np.arange(_N, dtype=float)


def _x1() -> np.ndarray:
    return _T / 10.0


def _x2() -> np.ndarray:
    return np.sin(_T)


def _y() -> np.ndarray:
    return 2.0 + 1.5 * _T / 10.0 - 0.8 * np.sin(_T) + 0.3 * np.cos(7.0 * _T)


def _w() -> np.ndarray:
    return 1.0 + (np.arange(_N) % 5) / 2.0


def _near_direction() -> np.ndarray:
    """Unit vector: residual of sin(13t) regressed on [1, x1, x2] (OLS), via
    `np.linalg.lstsq`, normalised to unit norm -- matches the Rust generator's
    `near_direction()` exactly (both use the same OLS-with-intercept residual)."""
    e0 = np.sin(13.0 * _T)
    design = np.column_stack([np.ones(_N), _x1(), _x2()])
    coef, _, _, _ = np.linalg.lstsq(design, e0, rcond=None)
    residual = e0 - design @ coef
    return residual / np.linalg.norm(residual)


def _near(ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (x1, x2, x3) where x3 is `ratio`-near-collinear with [x1, x2]."""
    x1, x2 = _x1(), _x2()
    base = x1 + x2
    nb = np.linalg.norm(base)
    e = _near_direction()
    x3 = base + ratio * nb * e
    return x1, x2, x3


def _rel_close(got: float, want: float, rel: float) -> None:
    scale = max(abs(want), 1.0)
    assert abs(got - want) <= rel * scale, f"got {got!r}, want {want!r} (rel={rel})"


# ----------------------------------------------------------------------------
# OLS: PyModel + `ps.ols` expression.
# ----------------------------------------------------------------------------


def test_ols_pymodel_alias_tolerance_matches_r():
    """A tighter alias_tolerance (1e-6) aliases a column R's default (1e-7)
    keeps at ratio=2e-7, and the reduced fit matches R's "below" reference
    (ratio=0.5e-7, default tol) exactly -- the tolerance, not the data, moved."""
    x1, x2, x3 = _near(2e-7)
    y = _y()
    X = np.column_stack([x1, x2, x3])

    model = OLS(alias_tolerance=1e-6).fit(X, y)

    assert math.isnan(model.coefficients[2])
    assert not math.isnan(model.coefficients[0])
    assert not math.isnan(model.coefficients[1])

    _rel_close(model.intercept, 2.023318557879104, 1e-9)
    _rel_close(model.coefficients[0], 1.490290680634032, 1e-9)
    _rel_close(model.coefficients[1], -0.772169023108856, 1e-9)
    _rel_close(model.std_errors[0], 0.0161213612630192, 1e-9)
    _rel_close(model.std_errors[1], 0.0396174535357292, 1e-9)
    _rel_close(model.rmse, 0.216257642550628, 1e-9)
    _rel_close(model.r_squared, 0.993640840379574, 1e-9)
    _rel_close(model.adj_r_squared, 0.993417711971839, 1e-9)


def test_ols_default_keeps_near_collinear():
    """Default alias_tolerance (1e-7) keeps the ratio=2e-7 column -- matches
    `near_collinear_above_tolerance_is_kept`."""
    x1, x2, x3 = _near(2e-7)
    y = _y()
    X = np.column_stack([x1, x2, x3])

    model = OLS().fit(X, y)

    assert not any(math.isnan(c) for c in model.coefficients)
    expect = [30169.12232610578940, 30166.85986640204646, -30167.63203542516567]
    for got, want in zip(model.coefficients, expect):
        _rel_close(got, want, 1e-4)
    _rel_close(model.rmse, 0.217068502408162, 1e-8)
    assert model.n_parameters == 4


def test_ols_smaller_tolerance_keeps_column():
    """A smaller alias_tolerance (1e-8) keeps a column the default (1e-7)
    aliases at ratio=0.5e-7 -- matches `near_collinear_below_tolerance_is_aliased`
    data with a tighter threshold, so the column survives this time."""
    x1, x2, x3 = _near(0.5e-7)
    y = _y()
    X = np.column_stack([x1, x2, x3])

    model = OLS(alias_tolerance=1e-8).fit(X, y)

    assert not any(math.isnan(c) for c in model.coefficients)
    assert model.r_squared >= 0.993640840379574 - 1e-12


def test_ols_expression_alias_tolerance():
    """`ps.ols(..., alias_tolerance=...)` matches the PyModel and the R
    reference; omitting the kwarg equals passing 1e-7 exactly."""
    x1, x2, x3 = _near(2e-7)
    y = _y()
    df = pl.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})

    row = df.select(
        ps.ols("y", "x1", "x2", "x3", alias_tolerance=1e-6).alias("m")
    )["m"][0]
    coefs = row["coefficients"]
    assert math.isnan(coefs[2])

    model = OLS(alias_tolerance=1e-6).fit(np.column_stack([x1, x2, x3]), y)
    _rel_close(row["intercept"], model.intercept, 1e-12)
    _rel_close(coefs[0], model.coefficients[0], 1e-12)
    _rel_close(coefs[1], model.coefficients[1], 1e-12)
    _rel_close(row["r_squared"], model.r_squared, 1e-12)
    _rel_close(row["adj_r_squared"], model.adj_r_squared, 1e-12)
    _rel_close(row["rmse"], model.rmse, 1e-12)

    _rel_close(row["intercept"], 2.023318557879104, 1e-9)
    _rel_close(coefs[0], 1.490290680634032, 1e-9)
    _rel_close(coefs[1], -0.772169023108856, 1e-9)
    _rel_close(row["rmse"], 0.216257642550628, 1e-9)

    # Omitting alias_tolerance equals passing 1e-7 exactly.
    row_default = df.select(ps.ols("y", "x1", "x2", "x3").alias("m"))["m"][0]
    row_explicit = df.select(
        ps.ols("y", "x1", "x2", "x3", alias_tolerance=1e-7).alias("m")
    )["m"][0]
    assert row_default["intercept"] == row_explicit["intercept"]
    for a, b in zip(row_default["coefficients"], row_explicit["coefficients"]):
        if math.isnan(a) or math.isnan(b):
            assert math.isnan(a) and math.isnan(b)
        else:
            assert a == b


def test_alias_tolerance_validation():
    """An invalid alias_tolerance (negative or NaN) raises ValueError at
    build time on both the PyModel and the expression."""
    with pytest.raises(ValueError):
        OLS(alias_tolerance=-1.0)
    with pytest.raises(ValueError):
        OLS(alias_tolerance=float("nan"))

    with pytest.raises(ValueError):
        ps.ols("y", "x1", "x2", alias_tolerance=-1.0)
    with pytest.raises(ValueError):
        ps.ols("y", "x1", "x2", alias_tolerance=float("nan"))


# ----------------------------------------------------------------------------
# WLS + the rest of the expression family (summary / predict / formula
# wrappers), plus validation and default-equals-1e-7 across every builder.
# ----------------------------------------------------------------------------


def _values_equal(a, b) -> bool:
    """Deep equality treating NaN == NaN (unlike Python's default `==`)."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_values_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b))
    return a == b


def _assert_equal(a, b) -> None:
    assert _values_equal(a, b), f"{a!r} != {b!r}"


def test_wls_pymodel_and_expression_alias_tolerance():
    """A tighter alias_tolerance (1e-6) aliases x3 on both WLS (PyModel) and
    `ps.wls`, matching a WLS fit on [x1, x2] only; the default keeps x3."""
    x1, x2, x3 = _near(2e-7)
    y = _y()
    w = _w()
    x_full = np.column_stack([x1, x2, x3])
    x_reduced = np.column_stack([x1, x2])

    model_full = WLS(alias_tolerance=1e-6).fit(x_full, y, w)
    model_reduced = WLS().fit(x_reduced, y, w)

    assert math.isnan(model_full.coefficients[2])
    _rel_close(model_full.intercept, model_reduced.intercept, 1e-10)
    _rel_close(model_full.coefficients[0], model_reduced.coefficients[0], 1e-10)
    _rel_close(model_full.coefficients[1], model_reduced.coefficients[1], 1e-10)
    _rel_close(model_full.r_squared, model_reduced.r_squared, 1e-10)

    df = pl.DataFrame({"y": y, "w": w, "x1": x1, "x2": x2, "x3": x3})
    row = df.select(
        ps.wls("y", "w", "x1", "x2", "x3", alias_tolerance=1e-6).alias("m")
    )["m"][0]
    coefs = row["coefficients"]
    assert math.isnan(coefs[2])
    _rel_close(row["intercept"], model_full.intercept, 1e-10)
    _rel_close(coefs[0], model_full.coefficients[0], 1e-10)
    _rel_close(coefs[1], model_full.coefficients[1], 1e-10)

    # Default keeps the column.
    row_default = df.select(ps.wls("y", "w", "x1", "x2", "x3").alias("m"))["m"][0]
    assert not any(math.isnan(c) for c in row_default["coefficients"])


def test_ols_summary_alias_tolerance():
    """The aliased x3 row's estimate is NaN; the other rows equal the summary
    computed on [x1, x2] alone."""
    x1, x2, x3 = _near(2e-7)
    y = _y()
    df = pl.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})

    rows = df.select(
        ps.ols_summary("y", "x1", "x2", "x3", alias_tolerance=1e-6).alias("s")
    )["s"][0].to_list()
    by_term = {r["term"]: r for r in rows}
    assert math.isnan(by_term["x3"]["estimate"])

    reduced_rows = df.select(ps.ols_summary("y", "x1", "x2").alias("s"))["s"][
        0
    ].to_list()
    reduced_by_term = {r["term"]: r for r in reduced_rows}
    for term in ("intercept", "x1", "x2"):
        _rel_close(
            by_term[term]["estimate"], reduced_by_term[term]["estimate"], 1e-10
        )


def test_wls_summary_alias_tolerance():
    """Same as `test_ols_summary_alias_tolerance` but for `ps.wls_summary`."""
    x1, x2, x3 = _near(2e-7)
    y = _y()
    w = _w()
    df = pl.DataFrame({"y": y, "w": w, "x1": x1, "x2": x2, "x3": x3})

    rows = df.select(
        ps.wls_summary("y", "w", "x1", "x2", "x3", alias_tolerance=1e-6).alias("s")
    )["s"][0].to_list()
    by_term = {r["term"]: r for r in rows}
    assert math.isnan(by_term["x3"]["estimate"])

    reduced_rows = df.select(ps.wls_summary("y", "w", "x1", "x2").alias("s"))[
        "s"
    ][0].to_list()
    reduced_by_term = {r["term"]: r for r in reduced_rows}
    for term in ("intercept", "x1", "x2"):
        _rel_close(
            by_term[term]["estimate"], reduced_by_term[term]["estimate"], 1e-10
        )


def test_ols_predict_alias_tolerance():
    """Predictions with alias_tolerance=1e-6 (x3 aliased) equal predictions
    from [x1, x2] alone."""
    x1, x2, x3 = _near(2e-7)
    y = _y()
    df = pl.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})

    preds_full = (
        df.select(
            ps.ols_predict("y", "x1", "x2", "x3", alias_tolerance=1e-6).alias("p")
        )["p"]
        .struct.field("ols_prediction")
        .to_numpy()
    )
    preds_reduced = (
        df.select(ps.ols_predict("y", "x1", "x2").alias("p"))["p"]
        .struct.field("ols_prediction")
        .to_numpy()
    )
    np.testing.assert_allclose(preds_full, preds_reduced, rtol=1e-10, atol=1e-10)


def test_wls_predict_alias_tolerance():
    """Same as `test_ols_predict_alias_tolerance` but for `ps.wls_predict`."""
    x1, x2, x3 = _near(2e-7)
    y = _y()
    w = _w()
    df = pl.DataFrame({"y": y, "w": w, "x1": x1, "x2": x2, "x3": x3})

    preds_full = (
        df.select(
            ps.wls_predict(
                "y", "w", "x1", "x2", "x3", alias_tolerance=1e-6
            ).alias("p")
        )["p"]
        .struct.field("wls_prediction")
        .to_numpy()
    )
    preds_reduced = (
        df.select(ps.wls_predict("y", "w", "x1", "x2").alias("p"))["p"]
        .struct.field("wls_prediction")
        .to_numpy()
    )
    np.testing.assert_allclose(preds_full, preds_reduced, rtol=1e-10, atol=1e-10)


def test_formula_wrappers_alias_tolerance_match_column_form():
    """`ols_formula`/`ols_formula_summary`/`ols_formula_predict`/`wls_formula`
    with alias_tolerance equal their column-argument forms exactly."""
    x1, x2, x3 = _near(2e-7)
    y = _y()
    w = _w()
    df = pl.DataFrame({"y": y, "w": w, "x1": x1, "x2": x2, "x3": x3})

    row_formula = df.select(
        ps.ols_formula("y ~ x1 + x2 + x3", alias_tolerance=1e-6).alias("m")
    )["m"][0]
    row_column = df.select(
        ps.ols("y", "x1", "x2", "x3", alias_tolerance=1e-6).alias("m")
    )["m"][0]
    _assert_equal(row_formula, row_column)

    rows_formula_summary = df.select(
        ps.ols_formula_summary("y ~ x1 + x2 + x3", alias_tolerance=1e-6).alias("s")
    )["s"][0].to_list()
    rows_column_summary = df.select(
        ps.ols_summary("y", "x1", "x2", "x3", alias_tolerance=1e-6).alias("s")
    )["s"][0].to_list()
    _assert_equal(rows_formula_summary, rows_column_summary)

    rows_formula_predict = df.select(
        ps.ols_formula_predict("y ~ x1 + x2 + x3", alias_tolerance=1e-6).alias("p")
    )["p"].to_list()
    rows_column_predict = df.select(
        ps.ols_predict("y", "x1", "x2", "x3", alias_tolerance=1e-6).alias("p")
    )["p"].to_list()
    _assert_equal(rows_formula_predict, rows_column_predict)

    row_wls_formula = df.select(
        ps.wls_formula("y ~ x1 + x2 + x3", "w", alias_tolerance=1e-6).alias("m")
    )["m"][0]
    row_wls_column = df.select(
        ps.wls("y", "w", "x1", "x2", "x3", alias_tolerance=1e-6).alias("m")
    )["m"][0]
    _assert_equal(row_wls_formula, row_wls_column)


def test_alias_tolerance_validation_all_builders():
    """Every builder raises ValueError for a negative or NaN tolerance."""
    with pytest.raises(ValueError):
        WLS(alias_tolerance=-1.0)
    with pytest.raises(ValueError):
        WLS(alias_tolerance=float("nan"))

    builders = [
        lambda tol: ps.wls("y", "w", "x1", "x2", alias_tolerance=tol),
        lambda tol: ps.ols_summary("y", "x1", "x2", alias_tolerance=tol),
        lambda tol: ps.wls_summary("y", "w", "x1", "x2", alias_tolerance=tol),
        lambda tol: ps.ols_predict("y", "x1", "x2", alias_tolerance=tol),
        lambda tol: ps.wls_predict("y", "w", "x1", "x2", alias_tolerance=tol),
        lambda tol: ps.ols_formula("y ~ x1 + x2", alias_tolerance=tol),
        lambda tol: ps.ols_formula_summary("y ~ x1 + x2", alias_tolerance=tol),
        lambda tol: ps.ols_formula_predict("y ~ x1 + x2", alias_tolerance=tol),
        lambda tol: ps.wls_formula("y ~ x1 + x2", "w", alias_tolerance=tol),
    ]
    for build in builders:
        with pytest.raises(ValueError):
            build(-1.0)
        with pytest.raises(ValueError):
            build(float("nan"))


def test_alias_tolerance_default_equals_explicit_on_every_builder():
    """Omitting alias_tolerance equals passing 1e-7 exactly, on every builder."""
    x1, x2 = _x1(), _x2()
    y = _y()
    w = _w()
    df = pl.DataFrame({"y": y, "w": w, "x1": x1, "x2": x2})

    scalar_pairs = [
        (ps.ols("y", "x1", "x2"), ps.ols("y", "x1", "x2", alias_tolerance=1e-7)),
        (
            ps.wls("y", "w", "x1", "x2"),
            ps.wls("y", "w", "x1", "x2", alias_tolerance=1e-7),
        ),
        (ps.ols_formula("y ~ x1 + x2"), ps.ols_formula("y ~ x1 + x2", alias_tolerance=1e-7)),
        (
            ps.wls_formula("y ~ x1 + x2", "w"),
            ps.wls_formula("y ~ x1 + x2", "w", alias_tolerance=1e-7),
        ),
    ]
    for default_expr, explicit_expr in scalar_pairs:
        default_row = df.select(default_expr.alias("v"))["v"][0]
        explicit_row = df.select(explicit_expr.alias("v"))["v"][0]
        _assert_equal(default_row, explicit_row)

    list_pairs = [
        (
            ps.ols_summary("y", "x1", "x2"),
            ps.ols_summary("y", "x1", "x2", alias_tolerance=1e-7),
        ),
        (
            ps.wls_summary("y", "w", "x1", "x2"),
            ps.wls_summary("y", "w", "x1", "x2", alias_tolerance=1e-7),
        ),
        (
            ps.ols_formula_summary("y ~ x1 + x2"),
            ps.ols_formula_summary("y ~ x1 + x2", alias_tolerance=1e-7),
        ),
    ]
    for default_expr, explicit_expr in list_pairs:
        default_rows = df.select(default_expr.alias("v"))["v"][0].to_list()
        explicit_rows = df.select(explicit_expr.alias("v"))["v"][0].to_list()
        _assert_equal(default_rows, explicit_rows)

    predict_pairs = [
        (
            ps.ols_predict("y", "x1", "x2"),
            ps.ols_predict("y", "x1", "x2", alias_tolerance=1e-7),
        ),
        (
            ps.wls_predict("y", "w", "x1", "x2"),
            ps.wls_predict("y", "w", "x1", "x2", alias_tolerance=1e-7),
        ),
        (
            ps.ols_formula_predict("y ~ x1 + x2"),
            ps.ols_formula_predict("y ~ x1 + x2", alias_tolerance=1e-7),
        ),
    ]
    for default_expr, explicit_expr in predict_pairs:
        default_rows = df.select(default_expr.alias("v"))["v"].to_list()
        explicit_rows = df.select(explicit_expr.alias("v"))["v"].to_list()
        _assert_equal(default_rows, explicit_rows)
