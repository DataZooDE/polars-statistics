#!/usr/bin/env python3
"""ANOVA Example: one-way, two-way and repeated-measures.

Demonstrates the ANOVA expressions, which run on the whole DataFrame via
``df.select(...)`` and return a result Struct. Use ``ps.struct_to_dict`` to
read the fields without manual ``.struct.field()`` extraction.

- one_way_anova            - compare means across independent groups
- two_way_anova            - two factors + their interaction
- repeated_measures_anova  - within-subject design (each subject seen in every condition)
"""

import polars as pl
import polars_statistics as ps

# =============================================================================
# 1. One-way ANOVA - do three teaching methods differ in test scores?
# =============================================================================

print("=" * 60)
print("1. One-way ANOVA (wide format: one column per group)")
print("=" * 60)

df = pl.DataFrame(
    {
        "control": [72.0, 75.0, 68.0, 71.0, 74.0, 70.0],
        "method_a": [80.0, 83.0, 78.0, 82.0, 85.0, 81.0],
        "method_b": [77.0, 79.0, 76.0, 78.0, 80.0, 75.0],
    }
)

res = ps.struct_to_dict(
    df.select(ps.one_way_anova("control", "method_a", "method_b").alias("aov"))
)
print(f"F({res['df_between']:.0f}, {res['df_within']:.0f}) = {res['statistic']:.3f}")
print(f"p-value:     {res['p_value']:.4g}")
print(f"eta-squared: {res['eta_squared']:.3f}  (proportion of variance explained)")
verdict = "reject H0: group means differ" if res["p_value"] < 0.05 else "fail to reject H0"
print(f"decision (alpha=0.05): {verdict}")
print()

# Welch's ANOVA does not assume equal variances (ss/eta fields return NaN).
welch = ps.struct_to_dict(
    df.select(ps.one_way_anova("control", "method_a", "method_b", kind="welch").alias("aov"))
)
print(f"Welch F = {welch['statistic']:.3f}, p = {welch['p_value']:.4g}")
print()


# =============================================================================
# 2. Two-way ANOVA - long format with two factors and their interaction
# =============================================================================

print("=" * 60)
print("2. Two-way ANOVA (score ~ method + gender + method:gender)")
print("=" * 60)

long = pl.DataFrame(
    {
        "score": [
            72.0, 75.0, 74.0, 71.0,  # method A, gender M/F/M/F
            85.0, 88.0, 82.0, 80.0,  # method B
            77.0, 79.0, 76.0, 78.0,  # method C
        ],
        "method": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
        "gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
    }
)

two = ps.struct_to_dict(
    long.select(ps.two_way_anova("score", "method", "gender").alias("aov"))
)
print(f"Factor A (method):    F={two['a_f']:.3f}, p={two['a_p_value']:.4g}")
print(f"Factor B (gender):    F={two['b_f']:.3f}, p={two['b_p_value']:.4g}")
print(f"Interaction (A x B):  F={two['ab_f']:.3f}, p={two['ab_p_value']:.4g}")
print(f"grand mean: {two['grand_mean']:.2f}, n={two['n']}")
print()


# =============================================================================
# 3. Repeated-measures ANOVA - within-subject design (balanced, long format)
# =============================================================================

print("=" * 60)
print("3. Repeated-measures ANOVA (each subject in every condition)")
print("=" * 60)

# 5 subjects each measured under 3 conditions (e.g. before / during / after).
rm = pl.DataFrame(
    {
        "score": [
            10.0, 13.0, 12.0,  # subject 1
            9.0, 12.0, 11.0,   # subject 2
            11.0, 15.0, 14.0,  # subject 3
            8.0, 11.0, 10.0,   # subject 4
            12.0, 16.0, 15.0,  # subject 5
        ],
        "subject": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
        "condition": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
    }
)

rmres = ps.struct_to_dict(rm.select(
    ps.repeated_measures_anova("score", "subject", "condition").alias("aov")
))
print(f"within-subject F({rmres['ws_df']:.0f}, {rmres['error_df']:.0f}) = {rmres['ws_f']:.3f}")
print(f"p-value:            {rmres['ws_p_value']:.4g}")
# Sphericity: if Mauchly's test is significant, prefer a corrected p-value.
print(f"Mauchly W = {rmres['mauchly_w']:.3f} (p = {rmres['mauchly_p_value']:.4g})")
print(f"Greenhouse-Geisser: epsilon = {rmres['gg_epsilon']:.3f}, p = {rmres['gg_p_value']:.4g}")
print(f"Huynh-Feldt:        epsilon = {rmres['hf_epsilon']:.3f}, p = {rmres['hf_p_value']:.4g}")
print()

print("Done!")
