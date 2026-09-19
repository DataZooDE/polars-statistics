# Migration Guide

This page documents API changes that require you to update existing code, with
old-vs-new examples side by side.

## `icc`: single-column → matrix input

**Affected version:** the `icc` contract changed when real ICC computation landed.
Earlier releases exposed a stub that accepted a single value column and a raters
column and returned all-`NaN`. The current API takes **one column per rater** —
subjects are rows — and returns values validated against R's `irr::icc()`.

### The change

Previously ICC was called with a long-format value/rater pair (and produced no real
output). It now takes a **wide** layout: one column per rater, stacked internally
into a subjects × raters matrix.

=== "Old (long format, stub — all NaN)"

    ```python
    # Long format: one value column, one rater id column.
    # This returned an all-NaN struct and should be migrated.
    df_long = pl.DataFrame({
        "score":  [1.0, 1.1, 0.9, 2.0, 2.1, 2.2, 3.0, 2.9, 3.1],
        "rater":  ["r1", "r2", "r3", "r1", "r2", "r3", "r1", "r2", "r3"],
        "subject": [1, 1, 1, 2, 2, 2, 3, 3, 3],
    })
    result = df_long.select(ps.icc("score", "rater"))   # ← no longer valid
    ```

=== "New (wide format, one column per rater)"

    ```python
    # Wide format: one column per rater, rows are subjects.
    df_wide = pl.DataFrame({
        "rater1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "rater2": [1.1, 2.1, 2.9, 4.2, 5.1],
        "rater3": [0.9, 2.2, 3.1, 3.8, 5.0],
    })
    result = df_wide.select(
        ps.icc("rater1", "rater2", "rater3", icc_type="icc3").alias("reliability")
    )
    print(ps.struct_to_dict(result))
    ```

    ```text
    {'icc': 0.9937, 'f_value': 475.89, 'df1': 4.0, 'df2': 8.0,
     'p_value': 1.53e-09, 'ci_lower': 0.9688, 'ci_upper': 0.9993,
     'n_subjects': 5, 'n_raters': 3}
    ```

### Migrating your data

If your data is long (one row per subject × rater), pivot it to wide before calling
`icc`:

```python
df_wide = df_long.pivot(values="score", index="subject", on="rater")
result = df_wide.select(
    ps.icc("r1", "r2", "r3", icc_type="icc2").alias("reliability")
)
```

The default `icc_type` is `"icc2"` (two-way random, absolute agreement). The
returned struct fields are `icc`, `f_value`, `df1`, `df2`, `p_value`, `ci_lower`,
`ci_upper`, `n_subjects`, `n_raters`. See the
[correlation reference](api/tests/correlation.md#icc) for the full ICC-type table
and interpretation guidance.

## `with_intercept` → `add_intercept`

**Deprecated in:** 0.7.0. **Still works:** yes — with a `FutureWarning`.
**Removal:** not scheduled.

Every regressor and expression that previously took a `with_intercept` keyword now
prefers a single `add_intercept` keyword. The two mean exactly the same thing; the
rename gives one consistent name across the whole API (expressions *and* classes).

The old keyword continues to work for backward compatibility but emits a
`FutureWarning`:

```text
FutureWarning: Parameter 'with_intercept' is deprecated and will be removed in a
future release. Use 'add_intercept' instead.
```

### The change

=== "Old (still works, warns)"

    ```python
    from polars_statistics import OLS

    model = OLS(with_intercept=True).fit(X, y)          # class
    df.select(ps.ols("y", "x", with_intercept=False))   # expression
    ```

=== "New (preferred)"

    ```python
    from polars_statistics import OLS

    model = OLS(add_intercept=True).fit(X, y)           # class
    df.select(ps.ols("y", "x", add_intercept=False))    # expression
    ```

### Timeline

| Version | Status of `with_intercept` |
|---------|----------------------------|
| ≤ 0.6.0 | Only supported keyword |
| **0.7.0** | Deprecated; `add_intercept` preferred; old keyword still works with `FutureWarning` |
| Future | May be removed in a future **major** release; **no removal is scheduled** |

!!! note "Passing both is an error"
    Supplying `with_intercept` *and* `add_intercept` in the same call raises an
    error — they'd otherwise conflict. Pass exactly one (prefer `add_intercept`).

### Silencing the warning during migration

If you need to suppress the `FutureWarning` while you update a large codebase:

```python
import warnings

warnings.filterwarnings(
    "ignore",
    message="Parameter 'with_intercept' is deprecated.*",
    category=FutureWarning,
)
```

Prefer fixing the call sites — a project-wide find/replace of `with_intercept=` →
`add_intercept=` is usually all it takes.

## See Also

- [Correlation & Reliability reference](api/tests/correlation.md) — full `icc` documentation
- [sklearn Migration](sklearn-migration.md) — porting scikit-learn workflows
- [API Conventions](api/conventions.md) — keyword and return conventions
