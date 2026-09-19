# Model Selection Guide

polars-statistics ships more than 40 regression and test models. This page helps
you pick the right one by comparing them across the dimensions that actually drive
the decision: **what it's for**, **robustness to outliers**, **interpretability**,
**speed**, and **formula support**.

Two rules of thumb before the tables:

1. **Start simple.** OLS (or a formula-driven OLS) is the right default. Reach for a
   fancier model only when a specific assumption is violated.
2. **Match the model to the response.** Continuous → linear/robust; strictly
   positive & skewed → Gamma; counts → Poisson/NegativeBinomial; binary →
   Logistic; bounded/proportional → the appropriate GLM link.

## Linear & regularized regressors

For a continuous response, roughly Gaussian errors.

| Model | Use case | Robustness | Interpretability | Speed | Formula |
|-------|----------|:----------:|:----------------:|:-----:|:-------:|
| `OLS` | Default; unbiased, full inference | Low | High | Fast | ✅ |
| `WLS` | Known non-constant error variance | Low | High | Fast | via expr |
| `Ridge` | Correlated predictors (L2) | Low | High | Fast | via expr |
| `Lasso` | Feature selection (L1) | Low | High | Fast | via expr |
| `ElasticNet` | Correlated + selection (L1+L2) | Low | High | Fast | via expr |
| `BayesianRidge` | Auto-tuned regularization + uncertainty | Medium | High | Fast | class |
| `PLS` | Many correlated predictors, latent structure | Low | Medium | Fast | class |

**When to move off OLS:** correlated predictors inflate OLS variance → `Ridge`;
you want the model to drop features → `Lasso`/`LARS`/`ARD`; you don't want to tune a
penalty by hand → `BayesianRidge`.

## Robust & sparse regressors

When outliers or high-dimensional feature sets break the assumptions above. See the
[Robust & Sparse Regression cookbook](examples/robust-regression.md).

| Model | Use case | Robustness | Interpretability | Speed | Formula |
|-------|----------|:----------:|:----------------:|:-----:|:-------:|
| `TheilSen` | Outliers in response; unbiased slope | High | High | Medium¹ | class |
| `RANSAC` | Outliers + flag them (inlier mask) | High | High | Medium | class |
| `Huber` | Mild outliers, keep all data | Medium | High | Fast | class |
| `Quantile` | Model a quantile (e.g. median) | High | High | Medium | class |
| `ARD` | Sparse coefficients, auto-prune features | Medium | High | Medium | class |
| `LARS` | Interpretable feature-selection path | Low | High | Fast | class |
| `PassiveAggressive` | Online / streaming updates | Medium | Medium | Fast² | class |

¹ Theil-Sen's cost grows with pairs of points; `max_subpopulation` caps it.
² Per-update cost is tiny; suited to never-ending streams.

## Generalized linear models (non-Gaussian response)

Choose by the response distribution. See the
[GLM cookbook](examples/glm-models.md) and
[GLMs, Smoothers & Streaming](examples/glm-smoothers-streaming.md).

| Model | Response type | Link | Interpretability | Formula |
|-------|---------------|------|:----------------:|:-------:|
| `LogisticRegression` | Binary (0/1) | logit | High (odds ratios) | class |
| `Poisson` | Counts | log | High (rate ratios) | via expr |
| `NegativeBinomial` | Overdispersed counts | log | High | class |
| `Gamma` | Positive, right-skewed | log | High (mult. effects) | class |
| `Tweedie` | Compound Poisson-Gamma (e.g. pure premium) | power | Medium | class |
| `Probit` | Binary, normal latent | probit | Medium | class |
| `Cloglog` | Binary, asymmetric | cloglog | Medium | class |

**Picking a count model:** start with `Poisson`; if the variance far exceeds the
mean (overdispersion), switch to `NegativeBinomial`.

## Smoothers, mixed & streaming models

| Model | Use case | Interpretability | Speed | Notes |
|-------|----------|:----------------:|:-----:|-------|
| `PSpline` | Smooth non-linear trend | Medium | Fast | Auto smoothing via GCV |
| `Isotonic` | Monotone relationship | High | Fast | Order-preserving fit |
| `GLMM` | Grouped/nested data, random effects | Medium | Medium | `gaussian`/`poisson`/`binomial` |
| `MomentAccumulator` | Out-of-core / streaming least squares | High | Fast | Exact, mergeable |
| `RLS` | Recursive least squares (adaptive) | High | Fast | Forgetting factor |
| `LmDynamic` | Time-varying coefficients | Medium | Medium | Kalman-style |

## Dimension definitions

- **Robustness** — resistance to outliers and assumption violations. *Low* models
  (OLS, Ridge) can be badly distorted by a few extreme points; *High* models
  (TheilSen, RANSAC, Quantile) are designed to shrug them off.
- **Interpretability** — how directly the coefficients map to a real-world effect.
  Linear and GLM coefficients are highly interpretable (a slope, an odds ratio, a
  rate ratio); latent-variable and heavily-regularized models less so.
- **Speed** — relative fitting cost at typical sizes. *Fast* = closed-form or few
  iterations; *Medium* = iterative or combinatorial (pairs, subsampling, EM).
- **Formula support** — how you specify the model:
    - **✅** — native R-style formula builder (`ps.ols_formula("y ~ x1 + x2 + x1:x2")`).
    - **via expr** — available as a Polars expression that integrates with
      `group_by`/`over`; build the design columns yourself.
    - **class** — a Python class taking NumPy `X`/`y` arrays (sklearn-style
      `fit`/`predict`/`score`).

## The two API surfaces

Most models are reachable two ways, and which you want depends on your workflow:

| Surface | Looks like | Best when |
|---------|-----------|-----------|
| **Polars expression** | `df.group_by("region").agg(ps.ols("y", "x"))` | You want per-group models, lazy evaluation, or to stay in a DataFrame pipeline |
| **Python class** | `OLS().fit(X, y).predict(X_new)` | You want an sklearn-style object, `.predict` on new data, or diagnostics |

The expression surface shines for **fitting one model per group** across millions of
groups — see the [Group-wise Analysis cookbook](examples/group-analysis.md). The
class surface shines for the **train/predict/score** loop and is a near drop-in for
scikit-learn — see the [sklearn Migration guide](sklearn-migration.md).

## See Also

- [Robust & Sparse Regression](examples/robust-regression.md)
- [GLMs, Smoothers & Streaming](examples/glm-smoothers-streaming.md)
- [sklearn Migration](sklearn-migration.md)
- [Regression Workflow](examples/regression-workflow.md)
