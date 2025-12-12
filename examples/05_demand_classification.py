#!/usr/bin/env python3
"""Demand Classification Example (AID)

Demonstrates:
- Automatic Identification of Demand (AID)
- Demand pattern classification (regular vs intermittent)
- Best distribution selection
- Anomaly detection (stockouts, new products, obsolete products)
- Per-SKU classification
"""

import polars as pl
import polars_statistics as ps

# =============================================================================
# Sample Data: SKU Demand Patterns
# =============================================================================

print("=" * 60)
print("Demand Classification for Inventory Management")
print("=" * 60)

# Different demand patterns for various SKUs
demand_df = pl.DataFrame({
    "sku": (
        ["SKU_A"] * 24 +  # Regular, high-volume
        ["SKU_B"] * 24 +  # Intermittent (many zeros)
        ["SKU_C"] * 24 +  # New product (zeros then sales)
        ["SKU_D"] * 24    # Obsolete product (sales then zeros)
    ),
    "period": list(range(1, 25)) * 4,
    "demand": [
        # SKU_A: Regular demand (Poisson-like)
        45, 52, 48, 51, 47, 53, 49, 50, 46, 54, 52, 48,
        50, 47, 53, 49, 51, 46, 54, 48, 52, 50, 47, 53,

        # SKU_B: Intermittent demand (many zeros)
        0, 0, 5, 0, 0, 0, 8, 0, 0, 3, 0, 0,
        0, 6, 0, 0, 0, 0, 4, 0, 0, 0, 7, 0,

        # SKU_C: New product (zeros then ramp-up)
        0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 8, 12,
        15, 18, 22, 25, 28, 30, 32, 35, 38, 40, 42, 45,

        # SKU_D: Obsolete product (sales then decline to zero)
        40, 38, 35, 32, 30, 25, 20, 15, 10, 5, 2, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
})

print("Demand Data (first rows per SKU):")
print(demand_df.group_by("sku").head(5).sort("sku", "period"))
print()

# =============================================================================
# 1. Basic Demand Classification
# =============================================================================

print("=" * 60)
print("1. Demand Classification per SKU")
print("=" * 60)

# Classify demand patterns per SKU
classification = demand_df.group_by("sku").agg(
    ps.aid("demand").alias("classification")
)

# Extract key fields
results = classification.with_columns(
    pl.col("classification").struct.field("demand_type").alias("demand_type"),
    pl.col("classification").struct.field("is_intermittent").alias("is_intermittent"),
    pl.col("classification").struct.field("distribution").alias("best_distribution"),
    pl.col("classification").struct.field("zero_proportion").alias("zero_pct"),
    pl.col("classification").struct.field("mean").alias("mean_demand"),
).drop("classification")

print("Classification Results:")
print(results)
print()

# =============================================================================
# 2. Detailed Classification Output
# =============================================================================

print("=" * 60)
print("2. Detailed Analysis per SKU")
print("=" * 60)

# Get full classification for one SKU
sku_a = demand_df.filter(pl.col("sku") == "SKU_A")
sku_a_class = sku_a.select(
    ps.aid("demand").alias("aid")
)

aid = sku_a_class["aid"][0]
print("SKU_A Analysis:")
print(f"  Demand Type:      {aid['demand_type']}")
print(f"  Is Intermittent:  {aid['is_intermittent']}")
print(f"  Is Fractional:    {aid['is_fractional']}")
print(f"  Best Distribution: {aid['distribution']}")
print(f"  Mean:             {aid['mean']:.2f}")
print(f"  Variance:         {aid['variance']:.2f}")
print(f"  Zero Proportion:  {aid['zero_proportion']:.2%}")
print(f"  N Observations:   {aid['n_observations']}")
print()

# =============================================================================
# 3. Anomaly Detection
# =============================================================================

print("=" * 60)
print("3. Anomaly Detection")
print("=" * 60)

# Check for lifecycle events per SKU
lifecycle = demand_df.group_by("sku").agg(
    ps.aid("demand", detect_anomalies=True).alias("aid")
)

lifecycle_results = lifecycle.with_columns(
    pl.col("aid").struct.field("demand_type").alias("type"),
    pl.col("aid").struct.field("has_stockouts").alias("stockouts"),
    pl.col("aid").struct.field("is_new_product").alias("is_new"),
    pl.col("aid").struct.field("is_obsolete_product").alias("is_obsolete"),
    pl.col("aid").struct.field("stockout_count").alias("stockout_n"),
    pl.col("aid").struct.field("new_product_count").alias("new_prod_n"),
    pl.col("aid").struct.field("obsolete_product_count").alias("obsolete_n"),
    pl.col("aid").struct.field("high_outlier_count").alias("high_outliers"),
    pl.col("aid").struct.field("low_outlier_count").alias("low_outliers"),
).drop("aid")

print("Lifecycle Event Detection:")
print(lifecycle_results)
print()

# =============================================================================
# 4. Per-Row Anomaly Flags
# =============================================================================

print("=" * 60)
print("4. Per-Row Anomaly Flags with .over()")
print("=" * 60)

# Add anomaly flags to each row
demand_with_flags = demand_df.with_columns(
    ps.aid_anomalies("demand").over("sku").alias("anomalies")
).unnest("anomalies")

# Show rows with anomalies
print("Rows with Anomalies (SKU_C - New Product):")
print(demand_with_flags.filter(
    (pl.col("sku") == "SKU_C") & pl.col("new_product")
).select("sku", "period", "demand", "new_product"))
print()

print("Rows with Anomalies (SKU_D - Obsolete Product):")
print(demand_with_flags.filter(
    (pl.col("sku") == "SKU_D") & pl.col("obsolete_product")
).select("sku", "period", "demand", "obsolete_product"))
print()

# =============================================================================
# 5. Intermittent Demand Analysis
# =============================================================================

print("=" * 60)
print("5. Intermittent Demand Analysis (SKU_B)")
print("=" * 60)

sku_b = demand_df.filter(pl.col("sku") == "SKU_B")
sku_b_class = sku_b.select(
    ps.aid("demand").alias("aid")
)

aid = sku_b_class["aid"][0]
print("SKU_B Analysis (Intermittent Pattern):")
print(f"  Demand Type:      {aid['demand_type']}")
print(f"  Is Intermittent:  {aid['is_intermittent']}")
print(f"  Best Distribution: {aid['distribution']}")
print(f"  Zero Proportion:  {aid['zero_proportion']:.2%}")
print()

# Count non-zero periods
non_zero = sku_b.filter(pl.col("demand") > 0)
print(f"  Total Periods:    {sku_b.height}")
print(f"  Non-Zero Periods: {non_zero.height}")
print(f"  Mean (non-zero):  {non_zero['demand'].mean():.2f}")
print()

# =============================================================================
# 6. Distribution Selection Summary
# =============================================================================

print("=" * 60)
print("6. Distribution Selection Summary")
print("=" * 60)

print("Based on demand patterns, AID selects the best distribution for forecasting:")
print()

for row in results.iter_rows(named=True):
    print(f"{row['sku']}:")
    print(f"  Pattern:      {row['demand_type']}")
    print(f"  Distribution: {row['best_distribution']}")
    print(f"  Zero %:       {row['zero_pct']:.1%}")
    print()

print("Available Distributions:")
print("  - Regular demand: normal, gamma, lognormal")
print("  - Count data: poisson, negative_binomial")
print("  - Intermittent: hurdle models, zero-inflated")
print()

# =============================================================================
# 7. Adjusting Intermittent Threshold
# =============================================================================

print("=" * 60)
print("7. Custom Intermittent Threshold")
print("=" * 60)

# Default threshold is 0.3 (30% zeros = intermittent)
# Adjust based on business needs

# With default threshold
default = demand_df.group_by("sku").agg(
    ps.aid("demand", intermittent_threshold=0.3).alias("aid")
).with_columns(
    pl.col("aid").struct.field("is_intermittent").alias("intermittent_30")
).select("sku", "intermittent_30")

# With stricter threshold (50%)
strict = demand_df.group_by("sku").agg(
    ps.aid("demand", intermittent_threshold=0.5).alias("aid")
).with_columns(
    pl.col("aid").struct.field("is_intermittent").alias("intermittent_50")
).select("sku", "intermittent_50")

comparison = default.join(strict, on="sku")
print("Intermittent Classification by Threshold:")
print(comparison)
print()

print("Done!")
