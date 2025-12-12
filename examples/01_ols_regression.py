#!/usr/bin/env python3
"""OLS Regression Example

Demonstrates:
- Basic OLS fitting
- Extracting model statistics (R-squared, coefficients)
- Making predictions with confidence intervals
- R-style formula syntax
- Tidy coefficient summaries
"""

import polars as pl
import polars_statistics as ps

# =============================================================================
# Sample Data: House Prices
# =============================================================================

# Simulated house price data
df = pl.DataFrame({
    "price": [150000, 180000, 220000, 250000, 280000, 310000, 350000, 380000, 420000, 450000],
    "sqft": [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800],
    "bedrooms": [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    "age": [20, 15, 10, 12, 8, 5, 3, 2, 1, 0],
})

print("=" * 60)
print("House Price Data")
print("=" * 60)
print(df)
print()

# =============================================================================
# 1. Basic OLS Regression
# =============================================================================

print("=" * 60)
print("1. Basic OLS Regression")
print("=" * 60)

# Fit OLS: price ~ sqft + bedrooms + age
result = df.select(
    ps.ols("price", "sqft", "bedrooms", "age").alias("model")
)

# Extract model statistics
model = result["model"][0]
print(f"Intercept: {model['intercept']:,.2f}")
print(f"Coefficients: {model['coefficients']}")
print(f"  - sqft:     ${model['coefficients'][0]:,.2f} per sqft")
print(f"  - bedrooms: ${model['coefficients'][1]:,.2f} per bedroom")
print(f"  - age:      ${model['coefficients'][2]:,.2f} per year")
print()
print(f"R-squared:     {model['r_squared']:.4f}")
print(f"Adj R-squared: {model['adj_r_squared']:.4f}")
print(f"RMSE:          ${model['rmse']:,.2f}")
print(f"F-statistic:   {model['f_statistic']:.2f}")
print(f"F p-value:     {model['f_pvalue']:.4f}")
print(f"AIC:           {model['aic']:.2f}")
print(f"BIC:           {model['bic']:.2f}")
print(f"N:             {model['n_observations']}")
print()

# =============================================================================
# 2. Predictions with Intervals
# =============================================================================

print("=" * 60)
print("2. Predictions with Confidence Intervals")
print("=" * 60)

# Add predictions with 95% prediction intervals
df_pred = df.with_columns(
    ps.ols_predict("price", "sqft", "bedrooms", "age", interval="prediction", level=0.95)
        .alias("pred")
).unnest("pred")

print(df_pred.select("price", "sqft", "bedrooms", "ols_prediction", "ols_lower", "ols_upper"))
print()

# Calculate residuals
df_resid = df_pred.with_columns(
    (pl.col("price") - pl.col("ols_prediction")).alias("residual")
)
print("Residuals summary:")
print(f"  Mean: {df_resid['residual'].mean():,.2f}")
print(f"  Std:  {df_resid['residual'].std():,.2f}")
print()

# =============================================================================
# 3. R-Style Formula Syntax
# =============================================================================

print("=" * 60)
print("3. R-Style Formula Syntax")
print("=" * 60)

# Same model using formula
result_formula = df.select(
    ps.ols_formula("price ~ sqft + bedrooms + age").alias("model")
)
print(f"R-squared (formula): {result_formula['model'][0]['r_squared']:.4f}")
print()

# With interaction term: sqft * bedrooms expands to sqft + bedrooms + sqft:bedrooms
result_interact = df.select(
    ps.ols_formula("price ~ sqft * bedrooms + age").alias("model")
)
print(f"R-squared (with interaction): {result_interact['model'][0]['r_squared']:.4f}")
print(f"Coefficients: {result_interact['model'][0]['coefficients']}")
print()

# =============================================================================
# 4. Tidy Coefficient Summary
# =============================================================================

print("=" * 60)
print("4. Tidy Coefficient Summary (like R's broom::tidy)")
print("=" * 60)

# Get coefficient table with std errors, t-stats, p-values
coef_summary = df.select(
    ps.ols_summary("price", "sqft", "bedrooms", "age").alias("coef")
).explode("coef").unnest("coef")

print(coef_summary)
print()

# =============================================================================
# 5. Simple Regression (Single Predictor)
# =============================================================================

print("=" * 60)
print("5. Simple Regression: price ~ sqft")
print("=" * 60)

simple = df.select(
    ps.ols("price", "sqft").alias("model")
)
model = simple["model"][0]
print(f"price = {model['intercept']:,.2f} + {model['coefficients'][0]:.2f} * sqft")
print(f"R-squared: {model['r_squared']:.4f}")
print()

# =============================================================================
# 6. Regression Without Intercept
# =============================================================================

print("=" * 60)
print("6. Regression Without Intercept")
print("=" * 60)

no_intercept = df.select(
    ps.ols("price", "sqft", "bedrooms", with_intercept=False).alias("model")
)
model = no_intercept["model"][0]
print(f"Intercept: {model['intercept']} (forced to 0)")
print(f"Coefficients: {model['coefficients']}")
print(f"R-squared: {model['r_squared']:.4f}")
print()

print("Done!")
