# Analysis of Variance (ANOVA)

ANOVA tests whether the means of several groups differ by more than chance. This
cookbook covers the three ANOVA expressions:

| Function | Design | Question it answers |
|----------|--------|---------------------|
| [`one_way_anova`](#one-way-anova) | One factor, independent groups | Do the group means differ? |
| [`two_way_anova`](#two-way-anova) | Two factors + interaction | Do A, B, or their interaction matter? |
| [`repeated_measures_anova`](#repeated-measures-anova) | Within-subject | Do conditions differ when each subject sees all of them? |

All three are **Polars expressions** that run on the whole DataFrame via
`df.select(...)` and return a result Struct. Rather than pull each field out with
`.struct.field(...)`, use the [`ps.struct_to_dict`](../api/outputs.md) helper to get
a plain Python dict.

!!! tip "Runnable script"
    The complete, runnable version of this page is
    [`examples/08_anova.py`](https://github.com/DataZooDE/polars-statistics/blob/main/examples/08_anova.py).

```python
import polars as pl
import polars_statistics as ps
```

## One-way ANOVA

One-way ANOVA compares the means of two or more independent groups. The expression
expects **one column per group** (wide format): pass each group's column and it
aggregates across them.

```python
df = pl.DataFrame({
    "control":  [72.0, 75.0, 68.0, 71.0, 74.0, 70.0],
    "method_a": [80.0, 83.0, 78.0, 82.0, 85.0, 81.0],
    "method_b": [77.0, 79.0, 76.0, 78.0, 80.0, 75.0],
})

res = ps.struct_to_dict(
    df.select(ps.one_way_anova("control", "method_a", "method_b").alias("aov"))
)
print(f"F({res['df_between']:.0f}, {res['df_within']:.0f}) = {res['statistic']:.3f}")
print(f"p-value:     {res['p_value']:.4g}")
print(f"eta-squared: {res['eta_squared']:.3f}")
```

```text
F(2, 15) = 27.396
p-value:     9.82e-06
eta-squared: 0.785
```

**Reading the output:** the F-statistic compares between-group variance to
within-group variance; a large F with a small `p_value` means at least one group
mean differs. `eta_squared` is the effect size — the proportion of total variance
explained by group membership (here 78.5%, a very large effect). The struct also
carries `ss_between`, `ss_within`, `ms_between`, `ms_within`, and `n_groups`.

### Welch's variant

If the groups have unequal variances, pass `kind="welch"`. Welch's ANOVA does not
assume homogeneity of variance; its sum-of-squares and `eta_squared` fields return
`NaN` because they aren't defined under that model.

```python
welch = ps.struct_to_dict(
    df.select(ps.one_way_anova("control", "method_a", "method_b", kind="welch").alias("aov"))
)
print(f"Welch F = {welch['statistic']:.3f}, p = {welch['p_value']:.4g}")
```

```text
Welch F = 21.772, p = 0.0002496
```

## Two-way ANOVA

Two-way ANOVA has two categorical factors and tests three effects: the two main
effects and their interaction. Unlike the one-way version, it takes **long-format**
data: a value column plus two factor columns. String factors are label-encoded
automatically — no pre-encoding needed.

```python
long = pl.DataFrame({
    "score": [72.0, 75.0, 74.0, 71.0,   # method A
              85.0, 88.0, 82.0, 80.0,   # method B
              77.0, 79.0, 76.0, 78.0],  # method C
    "method": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
    "gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
})

two = ps.struct_to_dict(
    long.select(ps.two_way_anova("score", "method", "gender").alias("aov"))
)
print(f"Factor A (method):   F={two['a_f']:.3f}, p={two['a_p_value']:.4g}")
print(f"Factor B (gender):   F={two['b_f']:.3f}, p={two['b_p_value']:.4g}")
print(f"Interaction (A x B): F={two['ab_f']:.3f}, p={two['ab_p_value']:.4g}")
```

```text
Factor A (method):   F=14.726, p=0.004847
Factor B (gender):   F=0.263, p=0.6263
Interaction (A x B): F=0.137, p=0.8748
```

**Reading the output:** each effect gets its own F and p-value. Here `method` has a
significant effect (p ≈ 0.005), `gender` does not (p ≈ 0.63), and there's no
interaction (p ≈ 0.87) — meaning the effect of `method` is the same for both
genders. **Always check the interaction first**: if it's significant, the main
effects can be misleading and should be interpreted with care. The struct also
carries the sum-of-squares (`a_ss`, `b_ss`, `ab_ss`, `residual_ss`), degrees of
freedom, mean squares, `grand_mean`, and `n`.

## Repeated-measures ANOVA

When the *same* subjects are measured under every condition (before/during/after a
treatment, say), the observations are correlated within subject. Repeated-measures
ANOVA accounts for that by partitioning out between-subject variance, giving more
power than a between-subjects design.

It requires a **balanced** long-format design: every subject must appear in every
condition. Subject and condition columns are label-encoded automatically. An
unbalanced input returns an all-`NaN` struct rather than raising.

```python
rm = pl.DataFrame({
    "score":     [10.0, 13.0, 12.0,   # subject 1: cond 1/2/3
                  9.0, 12.0, 11.0,    # subject 2
                  11.0, 15.0, 14.0,   # subject 3
                  8.0, 11.0, 10.0,    # subject 4
                  12.0, 16.0, 15.0],  # subject 5
    "subject":   [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
    "condition": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
})

r = ps.struct_to_dict(
    rm.select(ps.repeated_measures_anova("score", "subject", "condition").alias("aov"))
)
print(f"within-subject F({r['ws_df']:.0f}, {r['error_df']:.0f}) = {r['ws_f']:.3f}")
print(f"p-value: {r['ws_p_value']:.4g}")
print(f"Mauchly W = {r['mauchly_w']:.3f} (p = {r['mauchly_p_value']:.4g})")
print(f"Greenhouse-Geisser: eps = {r['gg_epsilon']:.3f}, p = {r['gg_p_value']:.4g}")
print(f"Huynh-Feldt:        eps = {r['hf_epsilon']:.3f}, p = {r['hf_p_value']:.4g}")
```

```text
within-subject F(2, 8) = 152.667
p-value: 4.249e-07
Mauchly W = 0.000 (p = 0)
Greenhouse-Geisser: eps = 0.500, p = 0.0002466
Huynh-Feldt:        eps = 0.500, p = 0.0002466
```

**Reading the output:** `ws_f` / `ws_p_value` are the within-subject effect of
condition. The **sphericity** section matters when you have three or more
conditions: Mauchly's test checks whether the variances of the differences between
conditions are equal. If Mauchly's `p_value` is small (as here), sphericity is
violated and you should report a **corrected** p-value — either Greenhouse-Geisser
(conservative) or Huynh-Feldt (less conservative). Both corrections still show a
highly significant condition effect here. Pass `compute_sphericity=False` to skip
the correction machinery when you don't need it.

## Choosing the right ANOVA

| Your design | Use |
|-------------|-----|
| One grouping factor, different subjects per group | `one_way_anova` |
| One factor but unequal variances | `one_way_anova(kind="welch")` |
| Two crossed factors | `two_way_anova` |
| Same subjects measured repeatedly | `repeated_measures_anova` |
| Two groups only | a [t-test](hypothesis-testing.md) is simpler and equivalent |

## See Also

- [Hypothesis Testing](hypothesis-testing.md) — t-tests and other two-group comparisons
- [Parametric Tests reference](../api/tests/parametric.md) — full field lists for each ANOVA struct
- [Output Structures](../api/outputs.md) — `struct_to_dict` and `unnest` helpers
