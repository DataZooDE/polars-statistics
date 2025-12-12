#!/usr/bin/env python3
"""Grouped Regression Example

Demonstrates:
- Fitting separate models per group with group_by
- Per-row predictions with .over()
- Comparing model performance across groups
- Time series regression with rolling windows
"""

import polars as pl
import polars_statistics as ps

# =============================================================================
# Sample Data: Regional Sales
# =============================================================================

# Sales data for different regions
df = pl.DataFrame({
    "region": ["North"] * 10 + ["South"] * 10 + ["East"] * 10,
    "sales": [
        # North: strong price effect
        100, 95, 110, 105, 120, 115, 130, 125, 140, 135,
        # South: weaker price effect, strong advertising
        80, 90, 85, 95, 90, 100, 95, 105, 100, 110,
        # East: moderate effects
        90, 92, 95, 98, 100, 103, 106, 108, 112, 115,
    ],
    "price": [
        10, 12, 11, 13, 12, 14, 13, 15, 14, 16,
        10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    ],
    "advertising": [
        5, 5, 6, 6, 7, 7, 8, 8, 9, 9,
        5, 8, 6, 9, 7, 10, 8, 11, 9, 12,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    ],
})

print("=" * 60)
print("Regional Sales Data")
print("=" * 60)
print(df)
print()

# =============================================================================
# 1. OLS Per Region with group_by
# =============================================================================

print("=" * 60)
print("1. OLS Regression Per Region (group_by)")
print("=" * 60)

# Fit separate OLS models for each region
regional_models = df.group_by("region").agg(
    ps.ols("sales", "price", "advertising").alias("model")
)

# Extract and display coefficients per region
for row in regional_models.iter_rows(named=True):
    region = row["region"]
    model = row["model"]
    print(f"\n{region} Region:")
    print(f"  Intercept:   {model['intercept']:.2f}")
    print(f"  Price coef:  {model['coefficients'][0]:.2f}")
    print(f"  Adv coef:    {model['coefficients'][1]:.2f}")
    print(f"  R-squared:   {model['r_squared']:.4f}")

print()

# =============================================================================
# 2. Compare Regions Side by Side
# =============================================================================

print("=" * 60)
print("2. Model Comparison Across Regions")
print("=" * 60)

comparison = regional_models.with_columns(
    pl.col("model").struct.field("intercept").alias("intercept"),
    pl.col("model").struct.field("coefficients").list.get(0).alias("price_effect"),
    pl.col("model").struct.field("coefficients").list.get(1).alias("adv_effect"),
    pl.col("model").struct.field("r_squared").alias("r_squared"),
    pl.col("model").struct.field("rmse").alias("rmse"),
).drop("model")

print(comparison)
print()

# =============================================================================
# 3. Per-Row Predictions with .over()
# =============================================================================

print("=" * 60)
print("3. Per-Row Predictions Using .over()")
print("=" * 60)

# Add predictions using region-specific models
df_with_pred = df.with_columns(
    ps.ols_predict("sales", "price", "advertising", interval="prediction")
        .over("region")
        .alias("pred")
).unnest("pred")

# Show predictions vs actual
print(df_with_pred.select("region", "sales", "ols_prediction", "ols_lower", "ols_upper"))
print()

# Calculate RMSE per region
rmse_per_region = df_with_pred.group_by("region").agg(
    ((pl.col("sales") - pl.col("ols_prediction")) ** 2).mean().sqrt().alias("rmse")
)
print("RMSE per region:")
print(rmse_per_region)
print()

# =============================================================================
# 4. Coefficient Summary Per Group
# =============================================================================

print("=" * 60)
print("4. Tidy Coefficient Summary Per Region")
print("=" * 60)

# Get coefficient tables per region
coef_by_region = df.group_by("region").agg(
    ps.ols_summary("sales", "price", "advertising").alias("coefficients")
)

# Explode to get one row per coefficient per region
tidy_coefs = coef_by_region.explode("coefficients").unnest("coefficients")
print(tidy_coefs)
print()

# =============================================================================
# 5. Multiple Model Types Per Group
# =============================================================================

print("=" * 60)
print("5. Comparing OLS vs Ridge Per Region")
print("=" * 60)

# Fit both OLS and Ridge for each region
multi_model = df.group_by("region").agg(
    ps.ols("sales", "price", "advertising").alias("ols"),
    ps.ridge("sales", "price", "advertising", lambda_=0.5).alias("ridge"),
)

# Compare R-squared
model_comparison = multi_model.with_columns(
    pl.col("ols").struct.field("r_squared").alias("ols_r2"),
    pl.col("ridge").struct.field("r_squared").alias("ridge_r2"),
).select("region", "ols_r2", "ridge_r2")

print(model_comparison)
print()

# =============================================================================
# 6. Time-Series Example: Rolling Regression
# =============================================================================

print("=" * 60)
print("6. Stock Returns: Per-Stock Regression")
print("=" * 60)

# Simulated stock return data
stock_df = pl.DataFrame({
    "ticker": ["AAPL"] * 12 + ["MSFT"] * 12,
    "month": list(range(1, 13)) * 2,
    "return": [
        0.05, 0.02, -0.01, 0.03, 0.04, -0.02, 0.06, 0.01, 0.03, 0.02, 0.04, 0.03,
        0.03, 0.01, 0.02, 0.04, 0.02, 0.01, 0.03, 0.02, 0.01, 0.03, 0.02, 0.04,
    ],
    "market_return": [
        0.04, 0.01, -0.02, 0.02, 0.03, -0.01, 0.05, 0.00, 0.02, 0.01, 0.03, 0.02,
        0.04, 0.01, -0.02, 0.02, 0.03, -0.01, 0.05, 0.00, 0.02, 0.01, 0.03, 0.02,
    ],
})

print("Stock Return Data:")
print(stock_df)
print()

# CAPM regression per stock: return ~ market_return
capm_models = stock_df.group_by("ticker").agg(
    ps.ols("return", "market_return").alias("capm")
)

for row in capm_models.iter_rows(named=True):
    ticker = row["ticker"]
    model = row["capm"]
    beta = model["coefficients"][0]
    alpha = model["intercept"]
    print(f"{ticker}: alpha={alpha:.4f}, beta={beta:.4f}, R2={model['r_squared']:.4f}")

print()
print("Done!")
