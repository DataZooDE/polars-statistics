# Regression Workflow

A complete regression analysis: fit a model, inspect coefficients, generate predictions with intervals, and run diagnostics — all in Polars.

## Setup

```python
import polars as pl
import polars_statistics as ps

# Apartment pricing data
df = pl.DataFrame({
    "price":    [250, 320, 280, 450, 380, 520, 290, 410, 350, 600,
                 270, 340, 310, 480, 400, 550, 300, 430, 370, 620],
    "sqft":     [800, 1000, 850, 1400, 1200, 1600, 900, 1300, 1100, 1800,
                 820, 1050, 950, 1450, 1250, 1550, 880, 1350, 1150, 1900],
    "bedrooms": [1, 2, 1, 3, 2, 3, 1, 2, 2, 4,
                 1, 2, 2, 3, 2, 3, 1, 3, 2, 4],
    "age":      [15, 10, 20, 5, 8, 3, 18, 6, 12, 2,
                 16, 9, 14, 4, 7, 3, 19, 5, 11, 1],
})
```

## Step 1: Fit a Model

```python
result = df.select(
    ps.ols("price", "sqft", "bedrooms", "age").alias("model")
)

model = result["model"][0]
print(f"R²:     {model['r_squared']:.4f}")
print(f"Adj R²: {model['adj_r_squared']:.4f}")
print(f"RMSE:   {model['rmse']:.2f}")
print(f"AIC:    {model['aic']:.2f}")
print(f"F-stat: {model['f_statistic']:.2f} (p={model['f_pvalue']:.6f})")
print(f"Intercept:    {model['intercept']:.4f}")
print(f"Coefficients: {model['coefficients']}")
```

## Step 2: Tidy Coefficient Table

Get a publication-ready coefficient summary with standard errors and p-values:

```python
coef_table = (
    df.select(
        ps.ols_summary("price", "sqft", "bedrooms", "age").alias("coef")
    )
    .explode("coef")
    .unnest("coef")
)

print(coef_table)
# ┌───────────┬──────────┬───────────┬───────────┬─────────┐
# │ term      ┆ estimate ┆ std_error ┆ statistic ┆ p_value │
# ╞═══════════╪══════════╪═══════════╪═══════════╪═════════╡
# │ intercept ┆ ...      ┆ ...       ┆ ...       ┆ ...     │
# │ x0        ┆ ...      ┆ ...       ┆ ...       ┆ ...     │
# │ x1        ┆ ...      ┆ ...       ┆ ...       ┆ ...     │
# │ x2        ┆ ...      ┆ ...       ┆ ...       ┆ ...     │
# └───────────┴──────────┴───────────┴───────────┴─────────┘
```

### Robust Standard Errors

Use heteroskedasticity-consistent (HC) standard errors when variance isn't constant:

```python
robust_coefs = (
    df.select(
        ps.ols_summary("price", "sqft", "bedrooms", "age", hc_type="hc3").alias("coef")
    )
    .explode("coef")
    .unnest("coef")
)
# hc_type options: "hc0", "hc1", "hc2", "hc3"
```

## Step 3: Predictions with Intervals

Generate predictions for every row, with 95% prediction intervals:

```python
predictions = (
    df.with_columns(
        ps.ols_predict("price", "sqft", "bedrooms", "age",
                       interval="prediction", level=0.95).alias("pred")
    )
    .unnest("pred")
)

print(predictions.select("price", "ols_prediction", "ols_lower", "ols_upper").head(5))
# ┌───────┬────────────────┬───────────┬───────────┐
# │ price ┆ ols_prediction ┆ ols_lower ┆ ols_upper │
# ╞═══════╪════════════════╪═══════════╪═══════════╡
# │ 250   ┆ 255.3          ┆ 210.1     ┆ 300.5     │
# │ 320   ┆ 318.7          ┆ 273.5     ┆ 363.9     │
# │ ...   ┆ ...            ┆ ...       ┆ ...       │
# └───────┴────────────────┴───────────┴───────────┘
```

Confidence intervals (for the mean response) are narrower than prediction intervals:

```python
ci = (
    df.with_columns(
        ps.ols_predict("price", "sqft", "bedrooms", "age",
                       interval="confidence", level=0.95).alias("pred")
    )
    .unnest("pred")
)
```

## Step 4: Diagnostics

### Multicollinearity Check

```python
cond = df.select(
    ps.condition_number("sqft", "bedrooms", "age").alias("cond")
)

c = cond["cond"][0]
print(f"Condition number: {c['condition_number']:.1f}")
print(f"Severity: {c['severity']}")
# "WellConditioned" (< 30), "Moderate" (30-100),
# "Serious" (100-1000), "Severe" (> 1000)
```

### Regularization When Needed

If multicollinearity is a concern, compare OLS with regularized models:

```python
comparison = df.select(
    ps.ols("price", "sqft", "bedrooms", "age").alias("ols"),
    ps.ridge("price", "sqft", "bedrooms", "age", lambda_=1.0).alias("ridge"),
    ps.elastic_net("price", "sqft", "bedrooms", "age",
                   lambda_=1.0, alpha=0.5).alias("enet"),
)

for name in ["ols", "ridge", "enet"]:
    m = comparison[name][0]
    print(f"{name:12s} R²={m['r_squared']:.4f}  RMSE={m['rmse']:.2f}  "
          f"coefficients={m['coefficients']}")
```

## Formula Syntax

Use R-style formulas for interactions and polynomials:

```python
# Main effects only
result = df.select(ps.ols_formula("price ~ sqft + bedrooms + age").alias("model"))

# Interaction: sqft effect may depend on number of bedrooms
result = df.select(ps.ols_formula("price ~ sqft * bedrooms + age").alias("model"))
# Expands to: sqft + bedrooms + sqft:bedrooms + age

# Polynomial: non-linear relationship with age
result = df.select(ps.ols_formula("price ~ sqft + bedrooms + poly(age, 2)").alias("model"))
model = result["model"][0]
print(f"R² with quadratic age: {model['r_squared']:.4f}")
```

## Quantile Regression

When you care about the median (or other quantiles) rather than the mean:

```python
quantiles = df.select(
    ps.quantile("price", "sqft", "bedrooms", "age", tau=0.25).alias("q25"),
    ps.quantile("price", "sqft", "bedrooms", "age", tau=0.50).alias("q50"),
    ps.quantile("price", "sqft", "bedrooms", "age", tau=0.75).alias("q75"),
)

for name, tau in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
    m = quantiles[name][0]
    print(f"τ={tau}: intercept={m['intercept']:.2f}, coefficients={m['coefficients']}")
```

## GLM: Logistic Regression

Binary outcome — predict whether a unit sells above median price:

```python
median_price = df["price"].median()
df_binary = df.with_columns(
    (pl.col("price") > median_price).cast(pl.Float64).alias("above_median")
)

# Check for separation issues first
sep = df_binary.select(
    ps.check_binary_separation("above_median", "sqft", "bedrooms", "age").alias("sep")
)
print(f"Has separation: {sep['sep'][0]['has_separation']}")

# Fit logistic regression
logit = df_binary.select(
    ps.logistic("above_median", "sqft", "bedrooms", "age").alias("model")
)

model = logit["model"][0]
print(f"Intercept:    {model['intercept']:.4f}")
print(f"Coefficients: {model['coefficients']}")

# Coefficient summary
logit_coefs = (
    df_binary.select(
        ps.logistic_summary("above_median", "sqft", "bedrooms", "age").alias("coef")
    )
    .explode("coef")
    .unnest("coef")
)
print(logit_coefs)
```
