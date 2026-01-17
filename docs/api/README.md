# polars-statistics API Reference

Complete API documentation for polars-statistics.

## Quick Links

| Category | Description |
|----------|-------------|
| [Conventions](conventions.md) | API patterns, column references, return types |
| **Statistical Tests** | |
| [Parametric Tests](tests/parametric.md) | t-tests, Brown-Forsythe, Yuen |
| [Non-Parametric Tests](tests/nonparametric.md) | Mann-Whitney, Wilcoxon, Kruskal-Wallis, Brunner-Munzel |
| [Distributional Tests](tests/distributional.md) | Shapiro-Wilk, D'Agostino normality tests |
| [Forecast Tests](tests/forecast.md) | Diebold-Mariano, Clark-West, SPA, MCS |
| [Correlation Tests](tests/correlation.md) | Pearson, Spearman, Kendall, distance correlation, partial correlation |
| [Categorical Tests](tests/categorical.md) | Chi-square, Fisher exact, McNemar, Cohen's Kappa |
| [TOST Equivalence](tests/tost.md) | Two One-Sided Tests for equivalence |
| **Regression** | |
| [Linear Models](regression/linear.md) | OLS, Ridge, Elastic Net, WLS, RLS, BLS, NNLS, Quantile, Isotonic |
| [GLM Models](regression/glm.md) | Logistic, Poisson, Negative Binomial, Tweedie, Probit, Cloglog |
| [ALM](regression/alm.md) | Augmented Linear Model (24+ distributions) |
| [Dynamic Models](regression/dynamic.md) | LmDynamic time-varying coefficients |
| [Demand Classification](regression/aid.md) | AID demand patterns and anomaly detection |
| [Formula Syntax](regression/formula.md) | R-style formulas with interactions and polynomials |
| [Summary & Predict](regression/summary-predict.md) | Coefficient tables and prediction intervals |
| [Diagnostics](regression/diagnostics.md) | Condition number, quasi-separation detection |
| **Model Classes** | |
| [Linear Model Classes](classes/linear.md) | OLS, Ridge, ElasticNet, WLS, RLS, BLS, Quantile, Isotonic |
| [GLM Model Classes](classes/glm.md) | Logistic, Poisson, NegativeBinomial, Tweedie, Probit, Cloglog |
| [ALM Class](classes/alm.md) | Augmented Linear Model class |
| [LmDynamic Class](classes/dynamic.md) | Dynamic linear model class |
| [Aid Class](classes/aid.md) | Demand classification class |
| [Test Classes](classes/tests.md) | Statistical test classes |
| [Bootstrap Classes](classes/bootstrap.md) | Stationary and circular block bootstrap |
| **Reference** | |
| [Output Structures](outputs.md) | All return type definitions |

## Installation

```bash
pip install polars-statistics
```

## Basic Usage

All functions work as Polars expressions:

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "group": ["A"] * 50 + ["B"] * 50,
    "y": [...],
    "x1": [...],
})

# Run regression per group
result = df.group_by("group").agg(
    ps.ols("y", "x1").alias("model")
)

# Extract results
result.with_columns(
    pl.col("model").struct.field("r_squared"),
)
```

## See Also

- [README](../../README.md) - Quick start guide
- [Polars Documentation](https://docs.pola.rs/)
