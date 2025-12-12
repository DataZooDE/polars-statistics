#!/usr/bin/env python3
"""Statistical Tests Example

Demonstrates:
- T-tests (independent and paired)
- Mann-Whitney U (non-parametric)
- Shapiro-Wilk (normality test)
- Kruskal-Wallis (multiple groups)
- Per-group testing
"""

import polars as pl
import polars_statistics as ps

# =============================================================================
# Sample Data: A/B Test Results
# =============================================================================

print("=" * 60)
print("A/B Test Data: Website Conversion Optimization")
print("=" * 60)

# A/B test data: time spent on page (seconds)
ab_test_df = pl.DataFrame({
    "control": [45.0, 52.0, 38.0, 61.0, 55.0, 48.0, 42.0, 58.0, 44.0, 51.0, 47.0, 53.0, 49.0, 56.0, 43.0],
    "treatment": [58.0, 62.0, 55.0, 71.0, 65.0, 59.0, 54.0, 68.0, 57.0, 63.0, 61.0, 67.0, 59.0, 72.0, 56.0],
})

print("Time on Page (seconds):")
print(ab_test_df)
print()

# =============================================================================
# 1. Independent Samples T-Test (Welch's)
# =============================================================================

print("=" * 60)
print("1. Welch's T-Test (Independent Samples)")
print("=" * 60)

result = ab_test_df.select(
    ps.ttest_ind("control", "treatment").alias("ttest")
)

ttest = result["ttest"][0]
print(f"H0: Mean(control) = Mean(treatment)")
print(f"H1: Mean(control) != Mean(treatment)")
print()
print(f"Test statistic: {ttest['statistic']:.4f}")
print(f"P-value:        {ttest['p_value']:.4f}")
print()

if ttest["p_value"] < 0.05:
    print("Conclusion: Reject H0 - Treatment significantly differs from control")
else:
    print("Conclusion: Fail to reject H0 - No significant difference")
print()

# =============================================================================
# 2. One-Sided T-Test
# =============================================================================

print("=" * 60)
print("2. One-Sided T-Test (Treatment > Control)")
print("=" * 60)

result = ab_test_df.select(
    ps.ttest_ind("control", "treatment", alternative="less").alias("ttest")
)

ttest = result["ttest"][0]
print(f"H0: Mean(control) >= Mean(treatment)")
print(f"H1: Mean(control) < Mean(treatment)")
print()
print(f"Test statistic: {ttest['statistic']:.4f}")
print(f"P-value (one-sided): {ttest['p_value']:.4f}")
print()

# =============================================================================
# 3. Paired T-Test: Before/After Study
# =============================================================================

print("=" * 60)
print("3. Paired T-Test: Training Program Effect")
print("=" * 60)

# Performance scores before and after training
training_df = pl.DataFrame({
    "before": [65.0, 72.0, 58.0, 80.0, 75.0, 68.0, 62.0, 78.0, 70.0, 74.0],
    "after": [70.0, 78.0, 65.0, 85.0, 82.0, 72.0, 68.0, 84.0, 76.0, 80.0],
})

print("Performance Scores:")
print(training_df)
print()

result = training_df.select(
    ps.ttest_paired("before", "after").alias("ttest")
)

ttest = result["ttest"][0]
print(f"H0: Mean(after - before) = 0")
print(f"H1: Mean(after - before) != 0")
print()
print(f"Test statistic: {ttest['statistic']:.4f}")
print(f"P-value:        {ttest['p_value']:.4f}")
print(f"Mean difference: {training_df['after'].mean() - training_df['before'].mean():.2f}")
print()

# =============================================================================
# 4. Mann-Whitney U Test (Non-Parametric)
# =============================================================================

print("=" * 60)
print("4. Mann-Whitney U Test")
print("=" * 60)

print("Non-parametric alternative to t-test (no normality assumption)")
print()

result = ab_test_df.select(
    ps.mann_whitney_u("control", "treatment").alias("mwu")
)

mwu = result["mwu"][0]
print(f"U statistic: {mwu['statistic']:.2f}")
print(f"P-value:     {mwu['p_value']:.4f}")
print()

# =============================================================================
# 5. Normality Test (Shapiro-Wilk)
# =============================================================================

print("=" * 60)
print("5. Shapiro-Wilk Normality Test")
print("=" * 60)

# Test normality of both groups
control_normality = ab_test_df.select(
    ps.shapiro_wilk("control").alias("sw")
)
treatment_normality = ab_test_df.select(
    ps.shapiro_wilk("treatment").alias("sw")
)

sw_control = control_normality["sw"][0]
sw_treatment = treatment_normality["sw"][0]

print("H0: Data comes from a normal distribution")
print()
print(f"Control group:")
print(f"  W statistic: {sw_control['statistic']:.4f}")
print(f"  P-value:     {sw_control['p_value']:.4f}")
print()
print(f"Treatment group:")
print(f"  W statistic: {sw_treatment['statistic']:.4f}")
print(f"  P-value:     {sw_treatment['p_value']:.4f}")
print()

# =============================================================================
# 6. Kruskal-Wallis Test (Multiple Groups)
# =============================================================================

print("=" * 60)
print("6. Kruskal-Wallis Test (3+ Groups)")
print("=" * 60)

# Three different website versions
multi_version_df = pl.DataFrame({
    "version_a": [45.0, 52.0, 48.0, 51.0, 47.0, 53.0, 49.0, 50.0, 46.0, 54.0],
    "version_b": [58.0, 62.0, 59.0, 63.0, 61.0, 67.0, 59.0, 64.0, 60.0, 65.0],
    "version_c": [55.0, 58.0, 52.0, 57.0, 54.0, 60.0, 56.0, 59.0, 53.0, 58.0],
})

print("Time on Page by Version:")
print(multi_version_df)
print()

result = multi_version_df.select(
    ps.kruskal_wallis("version_a", "version_b", "version_c").alias("kw")
)

kw = result["kw"][0]
print(f"H0: All groups come from the same distribution")
print()
print(f"H statistic: {kw['statistic']:.4f}")
print(f"P-value:     {kw['p_value']:.4f}")
print()

# =============================================================================
# 7. Tests Per Group
# =============================================================================

print("=" * 60)
print("7. T-Tests Per Experiment Group")
print("=" * 60)

# Multiple experiments with control/treatment
experiments_df = pl.DataFrame({
    "experiment": ["Exp1"] * 10 + ["Exp2"] * 10 + ["Exp3"] * 10,
    "control": [
        45.0, 52.0, 48.0, 51.0, 47.0, 53.0, 49.0, 50.0, 46.0, 54.0,
        30.0, 35.0, 32.0, 38.0, 33.0, 36.0, 31.0, 37.0, 34.0, 39.0,
        60.0, 65.0, 62.0, 68.0, 64.0, 67.0, 61.0, 66.0, 63.0, 69.0,
    ],
    "treatment": [
        58.0, 62.0, 59.0, 63.0, 61.0, 67.0, 59.0, 64.0, 60.0, 65.0,
        42.0, 48.0, 45.0, 50.0, 46.0, 49.0, 43.0, 51.0, 47.0, 52.0,
        62.0, 68.0, 65.0, 70.0, 66.0, 69.0, 63.0, 71.0, 67.0, 72.0,
    ],
})

print("Experiments Data:")
print(experiments_df.group_by("experiment").agg(
    pl.col("control").mean().alias("control_mean"),
    pl.col("treatment").mean().alias("treatment_mean"),
))
print()

# T-test per experiment
per_experiment = experiments_df.group_by("experiment").agg(
    ps.ttest_ind("control", "treatment").alias("ttest")
)

# Extract results
results_table = per_experiment.with_columns(
    pl.col("ttest").struct.field("statistic").alias("t_stat"),
    pl.col("ttest").struct.field("p_value").alias("p_value"),
).drop("ttest")

print("T-Test Results by Experiment:")
print(results_table)
print()

# =============================================================================
# 8. Variance Test (Brown-Forsythe)
# =============================================================================

print("=" * 60)
print("8. Brown-Forsythe Test (Equality of Variances)")
print("=" * 60)

result = ab_test_df.select(
    ps.brown_forsythe("control", "treatment").alias("bf")
)

bf = result["bf"][0]
print(f"H0: Variances are equal")
print()
print(f"F statistic: {bf['statistic']:.4f}")
print(f"P-value:     {bf['p_value']:.4f}")
print()

if bf["p_value"] >= 0.05:
    print("Conclusion: Variances can be assumed equal")
else:
    print("Conclusion: Variances differ significantly")
print()

# =============================================================================
# 9. Robust Test (Yuen's Test for Trimmed Means)
# =============================================================================

print("=" * 60)
print("9. Yuen's Test (Robust to Outliers)")
print("=" * 60)

# Data with outliers
outlier_df = pl.DataFrame({
    "x": [45.0, 52.0, 48.0, 51.0, 200.0, 53.0, 49.0, 50.0, 46.0, 54.0],  # outlier: 200
    "y": [58.0, 62.0, 59.0, 63.0, 61.0, 67.0, 59.0, 64.0, 60.0, 65.0],
})

print("Data with outlier in x:")
print(outlier_df)
print()

# Regular t-test
regular = outlier_df.select(
    ps.ttest_ind("x", "y").alias("ttest")
)
print(f"Regular t-test p-value: {regular['ttest'][0]['p_value']:.4f}")

# Yuen's test (20% trimmed)
robust = outlier_df.select(
    ps.yuen_test("x", "y", trim=0.2).alias("yuen")
)
yuen = robust["yuen"][0]
print(f"Yuen's test p-value:    {yuen['p_value']:.4f}")
print()
print("Note: Yuen's test is more robust to the outlier")
print()

print("Done!")
