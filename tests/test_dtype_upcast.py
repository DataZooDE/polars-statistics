"""Dtype-upcast parity tests (GH #52).

Integer (Int8-Int64, UInt8-UInt64) and Float32 inputs must give the same
results as the same data cast to Float64. Before the fix, every `*_fit`
Rust function read its inputs through Polars' strict typed accessors, so
e.g. ``ps.spearman("char_tokens", "char_types")`` on two ``UInt32`` columns
raised ``"invalid series dtype: expected `Float64`, got `u32`"``.
"""

from __future__ import annotations

import math

import polars as pl
import polars_statistics as ps
import pytest

DTYPES = [pl.Int32, pl.UInt32, pl.Int64, pl.Float32]
DTYPE_IDS = ["Int32", "UInt32", "Int64", "Float32"]

N = 20

# Small non-negative integers so every dtype (including UInt32) represents
# them exactly.
X = list(range(1, N + 1))
Y = [((i * 7) % 23) + 1 for i in range(N)]
COV = [((i * 3) % 17) + 1 for i in range(N)]
GROUP_A = [((i * 5) % 11) + 1 for i in range(N)]
GROUP_B = [((i * 11) % 13) + 1 for i in range(N)]
GROUP_C = [((i * 13) % 19) + 1 for i in range(N)]
RATER_1 = [((i * 2) % 9) + 1 for i in range(N)]
RATER_2 = [((i * 2) % 9) + 2 for i in range(N)]
RATER_3 = [((i * 2) % 9) + 3 for i in range(N)]
GROUP_KEY = [i % 2 for i in range(N)]
COUNTS_2X2 = [10, 20, 30, 40]


def _assert_same(a, b) -> None:
    """Recursively assert two decoded struct values are equal."""
    if isinstance(a, dict):
        assert isinstance(b, dict)
        assert set(a.keys()) == set(b.keys())
        for key in a:
            _assert_same(a[key], b[key])
    elif isinstance(a, list):
        assert isinstance(b, list)
        assert len(a) == len(b)
        for av, bv in zip(a, b):
            _assert_same(av, bv)
    elif isinstance(a, float):
        assert isinstance(b, float)
        if math.isnan(a) or math.isnan(b):
            assert math.isnan(a) and math.isnan(b), f"{a!r} != {b!r}"
        else:
            assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15), f"{a!r} != {b!r}"
    else:
        assert a == b


def test_issue_52_spearman_uint32_repro() -> None:
    """The literal GH #52 repro: ps.spearman on two UInt32 columns."""
    df = pl.DataFrame(
        {
            "char_tokens": X,
            "char_types": Y,
        },
        schema={"char_tokens": pl.UInt32, "char_types": pl.UInt32},
    )
    result = df.select(ps.spearman("char_tokens", "char_types").alias("r")).item()

    df_f64 = df.with_columns(
        pl.col("char_tokens").cast(pl.Float64),
        pl.col("char_types").cast(pl.Float64),
    )
    expected = df_f64.select(ps.spearman("char_tokens", "char_types").alias("r")).item()

    _assert_same(result, expected)


_PARITY_CASES = {
    "pearson": (lambda: ps.pearson("x", "y"), ["x", "y"]),
    "spearman": (lambda: ps.spearman("x", "y"), ["x", "y"]),
    "kendall": (lambda: ps.kendall("x", "y"), ["x", "y"]),
    "ttest_ind": (lambda: ps.ttest_ind("x", "y"), ["x", "y"]),
    "mann_whitney_u": (lambda: ps.mann_whitney_u("x", "y"), ["x", "y"]),
    "kruskal_wallis": (
        lambda: ps.kruskal_wallis("a", "b", "c"),
        ["a", "b", "c"],
    ),
    "one_way_anova": (
        lambda: ps.one_way_anova("a", "b", "c"),
        ["a", "b", "c"],
    ),
    "shapiro_wilk": (lambda: ps.shapiro_wilk("x"), ["x"]),
    "icc": (lambda: ps.icc("r1", "r2", "r3"), ["r1", "r2", "r3"]),
    "partial_cor": (
        lambda: ps.partial_cor("x", "y", ["cov"]),
        ["x", "y", "cov"],
    ),
    "ols": (lambda: ps.ols("y", "x1", "x2"), ["y", "x1", "x2"]),
}


def _base_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": X,
            "y": Y,
            "a": GROUP_A,
            "b": GROUP_B,
            "c": GROUP_C,
            "r1": RATER_1,
            "r2": RATER_2,
            "r3": RATER_3,
            "cov": COV,
            "x1": X,
            "x2": Y,
        }
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("case_name", sorted(_PARITY_CASES.keys()))
def test_dtype_parity(case_name: str, dtype: pl.DataType) -> None:
    build_expr, cols = _PARITY_CASES[case_name]
    df = _base_df()
    df_cast = df.with_columns([pl.col(c).cast(dtype) for c in cols])

    result = df_cast.select(build_expr().alias("r")).item()
    expected = df.select(build_expr().alias("r")).item()

    _assert_same(result, expected)


def test_partial_cor_integer_covariate_is_used() -> None:
    """An Int64 covariate must change the partial correlation versus plain
    pearson(x, y) — before this fix the uncast covariate was silently
    dropped by the Rust side."""
    df = pl.DataFrame({"x": X, "y": Y, "cov": COV})
    df_i64 = df.with_columns(pl.col("cov").cast(pl.Int64))

    partial_result = df_i64.select(
        ps.partial_cor("x", "y", ["cov"]).alias("r")
    ).item()
    partial_f64_result = df.select(ps.partial_cor("x", "y", ["cov"]).alias("r")).item()
    _assert_same(partial_result, partial_f64_result)

    pearson_result = df.select(ps.pearson("x", "y").alias("r")).item()

    assert abs(partial_result["estimate"] - pearson_result["estimate"]) > 1e-6


@pytest.mark.parametrize("dtype", [pl.Int32, pl.UInt32, pl.Int64], ids=["Int32", "UInt32", "Int64"])
def test_chisq_test_integer_dtypes(dtype: pl.DataType) -> None:
    df = pl.DataFrame({"counts": COUNTS_2X2}, schema={"counts": dtype})
    df_u32 = pl.DataFrame({"counts": COUNTS_2X2}, schema={"counts": pl.UInt32})

    result = df.select(ps.chisq_test("counts", n_rows=2, n_cols=2).alias("r")).item()
    expected = df_u32.select(ps.chisq_test("counts", n_rows=2, n_cols=2).alias("r")).item()

    _assert_same(result, expected)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_group_by_spearman_dtype_parity(dtype: pl.DataType) -> None:
    df = pl.DataFrame({"g": GROUP_KEY, "x": X, "y": Y})
    df_cast = df.with_columns(pl.col("x").cast(dtype), pl.col("y").cast(dtype))

    result = (
        df_cast.group_by("g", maintain_order=True)
        .agg(ps.spearman("x", "y").alias("r"))
        .to_dicts()
    )
    expected = (
        df.group_by("g", maintain_order=True).agg(ps.spearman("x", "y").alias("r")).to_dicts()
    )

    _assert_same(result, expected)
