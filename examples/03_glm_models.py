#!/usr/bin/env python3
"""GLM (Generalized Linear Models) Example

Demonstrates:
- Logistic regression for binary classification
- Poisson regression for count data
- Per-group GLM fitting
- Predicted probabilities and counts
"""

import polars as pl
import polars_statistics as ps

# =============================================================================
# 1. Logistic Regression: Customer Churn Prediction
# =============================================================================

print("=" * 60)
print("1. Logistic Regression: Customer Churn")
print("=" * 60)

# Synthetic customer churn data (50 samples for stable fitting)
# Linear predictor: logit(p) = -2 + 0.05*tenure - 0.02*charges
# This creates a realistic relationship between predictors and churn
churn_df = pl.DataFrame({
    "churned": [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Low risk
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,  # Medium risk
        0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,  # Mixed
        1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,  # Higher risk
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # High risk
    ],
    "tenure_months": [
        48.0, 60.0, 36.0, 52.0, 40.0, 55.0, 44.0, 50.0, 38.0, 42.0,
        30.0, 28.0, 32.0, 26.0, 34.0, 29.0, 31.0, 22.0, 20.0, 35.0,
        24.0, 27.0, 33.0, 18.0, 25.0, 15.0, 21.0, 12.0, 28.0, 10.0,
        8.0, 6.0, 16.0, 4.0, 7.0, 14.0, 3.0, 5.0, 9.0, 2.0,
        1.0, 3.0, 2.0, 4.0, 1.0, 2.0, 3.0, 1.0, 2.0, 1.0,
    ],
    "monthly_charges": [
        45.0, 42.0, 50.0, 40.0, 48.0, 38.0, 52.0, 44.0, 55.0, 46.0,
        58.0, 60.0, 56.0, 62.0, 54.0, 64.0, 59.0, 68.0, 70.0, 53.0,
        66.0, 63.0, 51.0, 72.0, 65.0, 76.0, 69.0, 80.0, 61.0, 84.0,
        88.0, 92.0, 78.0, 96.0, 90.0, 82.0, 98.0, 94.0, 86.0, 100.0,
        105.0, 102.0, 108.0, 99.0, 110.0, 104.0, 101.0, 112.0, 106.0, 115.0,
    ],
})

print("Customer Churn Data (first 10 rows):")
print(churn_df.head(10))
print(f"Total samples: {churn_df.height}")
print()

# Fit logistic regression
logit_result = churn_df.select(
    ps.logistic("churned", "tenure_months", "monthly_charges").alias("model")
)

model = logit_result["model"][0]
print("Logistic Regression Results:")
print(f"  Intercept: {model['intercept']:.4f}")
print(f"  Coefficients:")
print(f"    - tenure_months:   {model['coefficients'][0]:.4f}")
print(f"    - monthly_charges: {model['coefficients'][1]:.4f}")
print(f"  AIC:           {model['aic']:.2f}")
print(f"  BIC:           {model['bic']:.2f}")
print(f"  N:             {model['n_observations']}")
print()

# Interpretation
print("Interpretation:")
print("  - Negative tenure coefficient: longer tenure -> lower churn probability")
print("  - Positive charges coefficient: higher charges -> higher churn probability")
print()

# =============================================================================
# 2. Predicted Probabilities
# =============================================================================

print("=" * 60)
print("2. Predicted Churn Probabilities")
print("=" * 60)

# Add predicted probabilities
churn_pred = churn_df.with_columns(
    ps.logistic_predict("churned", "tenure_months", "monthly_charges")
        .alias("pred")
).unnest("pred")

print("Sample predictions:")
print(churn_pred.select("churned", "tenure_months", "monthly_charges", "logistic_prediction").head(10))
print()

# Classify predictions (threshold = 0.5)
churn_classified = churn_pred.with_columns(
    (pl.col("logistic_prediction") > 0.5).cast(pl.Int32).alias("predicted_churn")
)

# Confusion matrix
tp = churn_classified.filter((pl.col("churned") == 1) & (pl.col("predicted_churn") == 1)).height
tn = churn_classified.filter((pl.col("churned") == 0) & (pl.col("predicted_churn") == 0)).height
fp = churn_classified.filter((pl.col("churned") == 0) & (pl.col("predicted_churn") == 1)).height
fn = churn_classified.filter((pl.col("churned") == 1) & (pl.col("predicted_churn") == 0)).height

print("Confusion Matrix:")
print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"  Accuracy: {(tp + tn) / (tp + tn + fp + fn):.2%}")
print()

# =============================================================================
# 3. Poisson Regression: Insurance Claims
# =============================================================================

print("=" * 60)
print("3. Poisson Regression: Insurance Claims")
print("=" * 60)

# Synthetic insurance claims data - simpler structure with clear age effect
claims_df = pl.DataFrame({
    "claims": [
        0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0,  # Young drivers
        1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0,  # Middle age
        2.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 3.0, 4.0, 4.0,  # Older drivers
    ],
    "age": [
        22.0, 24.0, 25.0, 23.0, 26.0, 27.0, 28.0, 24.0, 29.0, 30.0,
        35.0, 38.0, 40.0, 36.0, 42.0, 44.0, 39.0, 45.0, 41.0, 46.0,
        52.0, 55.0, 58.0, 54.0, 60.0, 56.0, 62.0, 64.0, 66.0, 68.0,
    ],
})

print("Insurance Claims Data (first 10 rows):")
print(claims_df.head(10))
print(f"Total samples: {claims_df.height}")
print()

# Fit Poisson regression (single predictor for clarity)
poisson_result = claims_df.select(
    ps.poisson("claims", "age").alias("model")
)

model = poisson_result["model"][0]
print("Poisson Regression Results:")
print(f"  Intercept: {model['intercept']:.4f}")
print(f"  Coefficients:")
print(f"    - age: {model['coefficients'][0]:.4f}")
print(f"  AIC:     {model['aic']:.2f}")
print(f"  N:       {model['n_observations']}")
print()

print("Interpretation:")
print("  - Positive age coefficient: older drivers have more claims on average")
print(f"  - Rate ratio per year: exp({model['coefficients'][0]:.4f}) = {2.718281828**model['coefficients'][0]:.4f}")
print()

# =============================================================================
# 4. Predicted Counts
# =============================================================================

print("=" * 60)
print("4. Predicted Claim Counts")
print("=" * 60)

# Add predicted counts
claims_pred = claims_df.with_columns(
    ps.poisson_predict("claims", "age")
        .alias("pred")
).unnest("pred")

print("Sample predictions:")
print(claims_pred.select("claims", "age", "poisson_prediction").head(10))
print()

# =============================================================================
# 5. Coefficient Summary for GLMs
# =============================================================================

print("=" * 60)
print("5. Tidy Coefficient Summary for Logistic Regression")
print("=" * 60)

coef_summary = churn_df.select(
    ps.logistic_summary("churned", "tenure_months", "monthly_charges").alias("coef")
).explode("coef").unnest("coef")

print(coef_summary)
print()

# =============================================================================
# 6. Formula Syntax for GLMs
# =============================================================================

print("=" * 60)
print("6. R-Style Formula Syntax for GLMs")
print("=" * 60)

# Logistic regression with formula
logit_formula = churn_df.select(
    ps.logistic_formula("churned ~ tenure_months + monthly_charges").alias("model")
)
print("Logistic with formula: churned ~ tenure_months + monthly_charges")
print(f"  AIC: {logit_formula['model'][0]['aic']:.2f}")
print()

# Poisson regression with formula
poisson_formula = claims_df.select(
    ps.poisson_formula("claims ~ age").alias("model")
)
print("Poisson with formula: claims ~ age")
print(f"  AIC: {poisson_formula['model'][0]['aic']:.2f}")
print()

print("Done!")
