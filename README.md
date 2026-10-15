# polars-statistics

[![CI](https://github.com/DataZooDE/polars-statistics/actions/workflows/ci.yml/badge.svg)](https://github.com/DataZooDE/polars-statistics/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/DataZooDE/polars-statistics/graph/badge.svg)](https://codecov.io/gh/DataZooDE/polars-statistics)
[![PyPI version](https://badge.fury.io/py/polars-statistics.svg)](https://badge.fury.io/py/polars-statistics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Claude Code Plugin](https://img.shields.io/badge/Claude_Code-plugin-8A2BE2)](#claude-code-plugin)

> **Note:** This extension is in early stage development. APIs may change and some features are experimental.

High-performance statistical hypothesis testing and regression for [Polars](https://pola.rs/) DataFrames, powered by Rust — with results **validated against R**.

Usable from **Python** (as a Polars plugin *and* an sklearn-style class API) and from **Rust** (as an rlib that other Rust crates depend on directly — see [Use from Rust](#use-from-rust)).

> The repository includes a **[Claude Code plugin](#claude-code-plugin)** with five skills that document the Python API (expressions, model classes, result fields) for the Claude Code assistant.

## Why polars-statistics?

- **Polars-native.** Everything is a Polars expression, so `group_by`, `over`, and lazy evaluation just work — fit **one model per group** across millions of groups in a single pipeline, no Python loop.
- **sklearn-compatible.** Every model class follows the familiar `fit` / `predict` / `score` contract (regressors → R², classifiers → accuracy). Porting from scikit-learn is mostly a rename — see the [sklearn migration guide](docs/sklearn-migration.md).
- **Ergonomic results.** No manual `.struct.field(...)` archaeology: call `.to_dict()`, `.summary()`, or the `ps.unnest(df)` helper and get flat, readable output.
- **Typed API.** Ships `py.typed` and full `.pyi` stubs, so your editor autocompletes every constructor, method, and getter with accurate signatures.
- **Validated against R.** Where the backing `anofox-*` crates supply references, results are checked against R (`stats::lm`, `irr::icc`, …) — with real inference (std errors, t-stats, p-values, CIs) that many libraries omit.

## Installation

```bash
pip install polars-statistics
```

## Quick Start

**Fit one model per group as a lazy Polars expression** — the thing that sets
polars-statistics apart:

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "region": ["north"] * 50 + ["south"] * 50,
    "sales":  [...],
    "spend":  [...],
    "price":  [...],
})

# One OLS per region, computed in parallel across groups.
models = df.group_by("region").agg(
    ps.ols("sales", "spend", "price").alias("model")
)

# Flatten the result struct into plain columns in one call.
ps.unnest(models, "model")   # region, intercept, coefficients, r_squared, ...
```

**Or use the sklearn-style class API** for a train / predict / score loop:

```python
from polars_statistics import OLS

model = OLS().fit(X_train, y_train)   # X: 2-D array, y: 1-D array
model.predict(X_test)                 # -> numpy array
model.score(X_test, y_test)           # -> R^2

model.to_dict()          # every statistic as a plain dict
print(model.summary())   # readable text summary
```

## Statistical Tests

Statistical tests are powered by [anofox-statistics](https://github.com/sipemu/anofox-statistics-rs), providing full API parity with R's statistical functions and validated against R implementations.

```python
# Parametric tests
ps.ttest_ind("treatment", "control", alternative="two-sided")
ps.ttest_paired("before", "after")

# Non-parametric tests
ps.mann_whitney_u("x", "y")
ps.kruskal_wallis("group1", "group2", "group3")

# Normality tests
ps.shapiro_wilk("x")

# Forecast comparison
ps.diebold_mariano("errors1", "errors2", horizon=1)

# Correlation tests
ps.pearson("x", "y")                    # Pearson correlation with CI
ps.spearman("x", "y")                   # Spearman rank correlation
ps.kendall("x", "y", variant="b")       # Kendall's tau
ps.distance_cor("x", "y")               # Distance correlation (detects nonlinear)
ps.partial_cor("x", "y", ["z1", "z2"])  # Partial correlation

# Categorical tests
ps.binom_test(successes=7, n=10, p0=0.5)  # Exact binomial test (conf_level=0.95)
ps.chisq_test("counts", n_rows=2, n_cols=2)  # Chi-square independence
ps.fisher_exact(a=10, b=2, c=3, d=15)   # Fisher's exact test (+ ci_lower/ci_upper)
ps.fisher_exact_conditional(a=10, b=2, c=3, d=15)  # R fisher.test conditional MLE odds ratio
ps.mcnemar_test(a=45, b=15, c=5, d=35)  # McNemar's test
ps.cohen_kappa("counts", n_categories=3) # Inter-rater agreement
ps.cramers_v("counts", n_rows=3, n_cols=3) # Association strength

# Effect sizes
ps.cohens_d("x", "y")                    # Cohen's d (student/welch/paired)
ps.rank_biserial("x", "y")                # Rank-biserial correlation (Mann-Whitney effect size)

# Normality
ps.jarque_bera("x")                       # Jarque-Bera normality test

# Bootstrap confidence intervals
ps.bootstrap_mean_ci("x", seed=42)        # Percentile bootstrap CI for the mean
```

All tests return a struct with `statistic` and `p_value` fields.

### Post-hoc Tests & Multiple Comparisons

After a significant omnibus test, find out *which* group pairs actually differ:

```python
import polars as pl
import polars_statistics as ps

# 1. Omnibus test (wide format: one column per group)
wide = pl.DataFrame({"a": [5.1, 5.3, 4.9], "b": [6.2, 6.5, 6.1], "c": [7.8, 8.0, 7.5]})
wide.select(ps.one_way_anova("a", "b", "c").alias("anova")).unnest("anova")

# 2. Post-hoc pairwise comparisons (long format: one values + one groups column)
long_df = pl.DataFrame({
    "value": [5.1, 5.3, 4.9, 6.2, 6.5, 6.1, 7.8, 8.0, 7.5],
    "group": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
})
pairs = long_df.select(
    ps.tukey_hsd("value", "group").alias("ph")  # or ps.pairwise_t_test / ps.dunn_test
).select(pl.col("ph").explode().struct.unnest())
# Columns: group1, group2, n1, n2, estimate, statistic, df, p_value, p_adj, ci_lower, ci_upper

ps.p_adjust("p_values", method="BH")      # Any R p.adjust method, standalone
ps.ptukey("q", nmeans=4, df=20)           # Studentized-range CDF/quantile
```

### TOST Equivalence Tests

Test for practical equivalence using Two One-Sided Tests (TOST) procedure:

```python
# t-test based equivalence
ps.tost_t_test_two_sample("x", "y", delta=0.5, alpha=0.05)
ps.tost_t_test_paired("before", "after", bounds_type="cohen_d", delta=0.3)

# Correlation equivalence (test if correlation is near zero)
ps.tost_correlation("x", "y", delta=0.3, method="pearson")

# Proportion equivalence
ps.tost_prop_two(successes1=45, n1=100, successes2=48, n2=100, delta=0.1)

# Non-parametric and robust equivalence
ps.tost_wilcoxon_paired("x", "y", delta=0.5)
ps.tost_yuen("x", "y", trim=0.2, delta=0.5)  # Trimmed means
ps.tost_bootstrap("x", "y", n_bootstrap=1000)  # Bootstrap-based
```

Returns struct with `estimate`, `ci_lower`, `ci_upper`, `tost_p_value`, `equivalent`.

## Regression Models

Regression models are powered by [anofox-regression](https://github.com/sipemu/anofox-regression), providing validated implementations against R.

### Expression API

```python
# Linear models
ps.ols("y", "x1", "x2")
ps.ridge("y", "x1", "x2", lambda_=1.0)
ps.elastic_net("y", "x1", "x2", lambda_=1.0, alpha=0.5)

# Robust regression
ps.quantile("y", "x1", "x2", tau=0.5)  # Median regression
ps.isotonic("y", "x")                   # Monotonic regression
ps.huber("y", "x1", epsilon=1.35)       # Huber M-estimator (outlier-robust)
ps.pls("y", "x1", "x2", n_components=2) # Partial Least Squares

# GLM models (with optional Ridge regularization)
ps.logistic("y", "x1", "x2", lambda_=0.1)             # Binary classification (BinomialRegressor)
ps.logistic_regression("y", "x1", "x2", penalty="l2", C=1.0)  # sklearn-style logistic
ps.poisson("y", "x1", "x2")                            # Count data

# Penalized GLM - priors, ridge, offsets, 6 families
ps.penalized_glm("claims", "age", "risk", family="poisson", offset="log_exposure")

# AFT - accelerated failure time survival regression (right censoring)
ps.aft("time", "event", "age", distribution="weibull")

# ALM - 25 distributions, loss/link/extra_parameter exposed
ps.alm("y", "x1", "x2", distribution="laplace", loss="mle")
```

### Diagnostics

```python
# Pre-fit checks
ps.condition_number("x1", "x2")              # Multicollinearity (κ + indices)
ps.vif("x1", "x2", "x3")                     # Variance inflation factor per feature
ps.generalized_vif("x1", "x2", "x3", group_sizes=[1, 2])  # GVIF for grouped predictors
ps.high_vif_predictors("x1", "x2", threshold=10.0)        # Boolean mask
ps.check_binary_separation("y", "x1")        # Quasi-separation detection
ps.check_count_sparsity("y", "x1")           # Sparse-count check

# Per-row OLS residual battery
ps.standardized_residuals("y", "x1", "x2")
ps.studentized_residuals("y", "x1", "x2")
ps.externally_studentized_residuals("y", "x1", "x2")
ps.residual_outliers("y", "x1", "x2", threshold=2.0)       # Boolean mask

# Influence / leverage
ps.leverage("x1", "x2")
ps.cooks_distance("y", "x1", "x2")
ps.dffits("y", "x1", "x2")
ps.influential_cooks("y", "x1", "x2")        # mask, default threshold 4/n
ps.influential_dffits("y", "x1", "x2")       # mask, default 2·√(p/n)
ps.high_leverage_points("x1", "x2")          # mask, default 2·p/n

# GLM residual diagnostics (logistic + Poisson)
ps.logistic_pearson_residuals("y", "x1")
ps.logistic_deviance_residuals("y", "x1")
ps.logistic_working_residuals("y", "x1")
ps.poisson_pearson_residuals("y", "x1")
ps.poisson_deviance_residuals("y", "x1")
ps.poisson_working_residuals("y", "x1")

# GLM goodness-of-fit
ps.pearson_chi_squared_logistic("y", "x1")   # Σ pearson_resid² + df_resid
ps.pearson_chi_squared_poisson("y", "x1")
```

Per-observation diagnostics (R `broom::augment`), one combined expression for
`ols`/`wls`/`logistic`/`probit`/`cloglog`/`poisson`/`negative_binomial`/`tweedie`/`gamma`
— also available as `OLS().fit(x, y).augment(x, y)`, and on `WLS` and every GLM class:

```python
import polars as pl
import polars_statistics as ps

reg = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 10.0], "x1": [1.0, 2.0, 3.0, 4.0, 5.0]})
reg.select(
    ps.augment("y", "x1", model="ols", residual_type="pearson").alias("diag")
).unnest("diag")
# -> fitted, residual, std_residual, stud_residual, leverage, cooks_d, dffits
```

### Empirical-Bayes Shrinkage

Shrink noisy per-group estimates (e.g. per-store effect sizes) toward a pooled mean,
matching R `metafor::rma(method = "DL")` / `blup()`:

```python
import polars as pl
import polars_statistics as ps

df = pl.DataFrame({
    "estimate": [0.12, -0.05, 0.30, 0.02, 0.18],
    "std_error": [0.08, 0.15, 0.05, 0.20, 0.07],
})
df.select(
    ps.eb_shrink("estimate", "std_error", method="dl").alias("eb")
).unnest("eb")
# Columns: shrunken, shrunken_se, weight, mu, mu_se, tau_squared, i_squared, q, n_groups
```

### Formula Syntax

R-style formulas with polynomial and interaction effects:

```python
# Main effects + interaction
ps.ols_formula("y ~ x1 * x2")  # Expands to: x1 + x2 + x1:x2

# Polynomial regression (centered per group)
ps.ols_formula("y ~ poly(x, 2)")

# Explicit transform
ps.ols_formula("y ~ x1 + I(x^2)")
```

### Predictions with Intervals

```python
df.with_columns(
    ps.ols_predict("y", "x1", "x2", interval="prediction", level=0.95)
        .over("group").alias("pred")
).unnest("pred")  # Columns: prediction, lower, upper
```

### Tidy Coefficient Summary

```python
df.group_by("group").agg(
    ps.ols_summary("y", "x1", "x2").alias("coef")
).explode("coef").unnest("coef")
# Columns: term, estimate, std_error, statistic, p_value
```

`*_summary` and `*_predict` are available for the full regression family,
including Quantile, Isotonic, and LmDynamic where applicable:

```python
ps.quantile_summary("y", "x1", tau=0.5)   # Tidy coefs from quantile fit
ps.quantile_predict("y", "x1", tau=0.5)   # Per-row predictions
ps.isotonic_predict("y", "x")             # Step-function predictions
ps.lm_dynamic_predict("y", "x1")          # Time-averaged predictions
```

## Model Classes

For direct model access outside Polars expressions:

```python
from polars_statistics import OLS, Ridge, Logistic, LogisticRegression, Huber, PLS, ALM

# Fit OLS with inference
model = OLS(compute_inference=True).fit(X, y)
print(model.coefficients, model.r_squared, model.p_values)

# Sklearn-style logistic with L2 penalty
lr = LogisticRegression(penalty="l2", C=1.0).fit(X, y)
lr.predict_proba(X)
lr.decision_function(X)
lr.score(X, y)

# Huber M-estimator (robust to outliers), with MASS::rlm-validated inference
hb = Huber(epsilon=1.35, compute_inference=True, confidence_level=0.95).fit(X, y)
print(hb.coefficients, hb.n_outliers, hb.scale, hb.std_errors)

# RANSAC: random sample consensus, with inlier-fit statistics
from polars_statistics import RANSAC

rs = RANSAC(compute_inference=True).fit(X, y)
print(rs.std_errors, rs.rmse)       # inlier-only inference getters
inlier_ols = rs.inlier_fit()        # fitted OLS over the inlier rows
print(inlier_ols.r_squared)

# Partial Least Squares
pls = PLS(n_components=2).fit(X, y)
print(pls.explained_variance_ratio, pls.transform(X))

# ALM with various distributions
alm = ALM.laplace().fit(X, y)  # Robust to outliers
```

Available model classes:
- **Linear / robust**: `OLS`, `Ridge`, `ElasticNet`, `WLS`, `RLS`, `BLS`, `Quantile`, `Isotonic`, `Huber`, `PLS`
- **GLMs**: `Logistic`, `LogisticRegression` (sklearn-style), `Poisson`, `NegativeBinomial`, `Tweedie`, `Probit`, `Cloglog`
- **PenalizedGLM**: penalized / prior-regularized GLM (6 families, Normal/Laplace priors, ridge, offsets)
- **AFT**: accelerated failure time survival regression (Weibull, exponential, log-normal, log-logistic; right censoring)
- **Augmented**: `ALM` (25 distributions), `LmDynamic`, `Aid`

## Test Model Classes

Statistical tests are also available as model classes with `.fit()`, `.statistic`, `.p_value`, and `.summary()`:

```python
from polars_statistics import TTestInd, ShapiroWilk, KruskalWallis
import numpy as np

# Two-sample t-test
test = TTestInd(alternative="two-sided").fit(x, y)
print(test.statistic, test.p_value)
print(test.summary())

# Normality test
test = ShapiroWilk().fit(x)
print(test.p_value)

# Multi-group comparison
test = KruskalWallis().fit(g1, g2, g3)
print(test.summary())
```

Available test classes: `TTestInd`, `TTestPaired`, `BrownForsythe`, `YuenTest`, `MannWhitneyU`, `WilcoxonSignedRank`, `KruskalWallis`, `BrunnerMunzel`, `ShapiroWilk`, `DAgostino`.

## Use from Rust

`polars-statistics` builds as both a Python extension (`cdylib`) and a Rust library (`rlib`). Other Rust crates can depend on it directly and call the same statistical and regression code that the Python plugin uses — no Python boundary, no FFI overhead.

### Cargo dependency

```toml
[dependencies]
polars = { version = "0.52", features = ["lazy", "partition_by"] }
polars-statistics = { version = "0.4", default-features = false }
```

`default-features = false` disables the `python` feature, so pyo3 / numpy are not linked.

### Calling the fit functions

Every Polars expression has a public Rust-callable counterpart named `<name>_fit` that accepts a `&[Series]` input slice matching the expression's input contract and returns a `PolarsResult<Series>` (a one-row struct with the model output).

```rust
use polars::prelude::*;
use polars_statistics::expressions::wls_fit;

fn main() -> PolarsResult<()> {
    let df = df!(
        "site"   => &["A", "A", "A", "B", "B", "B"],
        "y"      => &[1.0_f64, 3.0, 5.0, 2.0, 5.0, 8.0],
        "weight" => &[1.0_f64, 1.0, 1.0, 1.0, 1.0, 1.0],
        "x1"     => &[0.0_f64, 1.0, 2.0, 0.0, 1.0, 2.0],
    )?;

    for group in df.partition_by(["site"], true)? {
        let y  = group.column("y")?.as_materialized_series().clone();
        let w  = group.column("weight")?.as_materialized_series().clone();
        let x1 = group.column("x1")?.as_materialized_series().clone();
        let with_intercept = Series::new("with_intercept".into(), &[true]);
        let solver         = Series::new("solve_method".into(), &[None::<&str>]);

        let result = wls_fit(&[y, w, with_intercept, solver, x1])?;
        println!("{result:?}");
    }
    Ok(())
}
```

The full runnable version is in [`examples/rust_wls.rs`](examples/rust_wls.rs). Run with:

```bash
cargo run --example rust_wls --no-default-features
```

### Available fit functions

Every Polars expression has a `*_fit` Rust entry point in `polars_statistics::expressions`:

- **Regression**: `ols_fit`, `ridge_fit`, `elastic_net_fit`, `wls_fit`, `rls_fit`, `bls_fit`, `quantile_fit`, `isotonic_fit`, `huber_fit`, `pls_fit`
- **GLMs**: `logistic_fit`, `logistic_regression_fit`, `poisson_fit`, `negative_binomial_fit`, `tweedie_fit`, `probit_fit`, `cloglog_fit`, `alm_fit`
- **Diagnostics**:
  - Pre-fit / multicollinearity: `condition_number_fit`, `vif_fit`, `generalized_vif_fit`, `high_vif_predictors_fit`, `check_binary_separation_fit`, `check_count_sparsity_fit`
  - Residual battery: `standardized_residuals_fit`, `studentized_residuals_fit`, `externally_studentized_residuals_fit`, `residual_outliers_fit`
  - GLM residuals: `logistic_*_residuals_fit`, `poisson_*_residuals_fit` (`pearson` / `deviance` / `working`)
  - Goodness-of-fit: `pearson_chi_squared_logistic_fit`, `pearson_chi_squared_poisson_fit`
  - Influence / leverage: `leverage_fit`, `cooks_distance_fit`, `dffits_fit`, `influential_cooks_fit`, `influential_dffits_fit`, `high_leverage_points_fit`
- **Summaries / predictions**: `ols_summary_fit`, `ols_predict_fit`, …; plus `quantile_summary_fit`, `quantile_predict_fit`, `isotonic_predict_fit`, `lm_dynamic_predict_fit`
- **Hypothesis tests**: `ttest_ind_fit`, `ttest_paired_fit`, `mann_whitney_fit`, `wilcoxon_fit`, `kruskal_wallis_fit`, `brunner_munzel_fit`, `brown_forsythe_fit`, `yuen_fit`, `shapiro_wilk_fit`, `dagostino_fit`, `jarque_bera_fit`
- **Correlation**: `pearson_fit`, `spearman_fit`, `kendall_fit`, `distance_cor_fit`, `partial_cor_fit`, `semi_partial_cor_fit`, `icc_fit`
- **Categorical**: `binom_test_fit`, `prop_test_one_fit`, `prop_test_two_fit`, `chisq_test_fit`, `fisher_exact_fit`, `fisher_exact_conditional_fit`, `mcnemar_test_fit`, `cohen_kappa_fit`, `cramers_v_fit`, … (`binom_test_fit`/`prop_test_one_fit`/`prop_test_two_fit`/`fisher_exact_fit` accept an optional trailing `conf_level` input; old arity still works)
- **Effect sizes / bootstrap**: `cohens_d_fit`, `cohens_d_one_sample_fit`, `rank_biserial_fit`, `matched_pairs_rank_biserial_fit`, `rank_biserial_from_u_fit`, `bootstrap_mean_ci_fit`, `bootstrap_ci_fit`
- **Additive GLM-family options** (named kwargs structs, old `*_fit` signatures unchanged): `negative_binomial_fit_with`/`negative_binomial_summary_fit_with` (`NegativeBinomialKwargs`), `negative_binomial_predict_fit_with` (`NegativeBinomialPredictKwargs`), `rls_fit_with`/`rls_summary_fit_with` (`RlsKwargs`), `rls_predict_fit_with` (`RlsPredictKwargs`)
- **Penalized GLM** (`PenalizedGlmKwargs`, `PriorKwarg`):
  - `penalized_glm_fit`
  - `penalized_glm_summary_fit`
  - `penalized_glm_predict_fit`
- **AFT survival regression** (`AftKwargs`):
  - `aft_fit`
  - `aft_summary_fit`
  - `aft_predict_quantile_fit`
  - `aft_predict_median_fit`
  - `aft_predict_survival_fit`
  - `aft_predict_cdf_fit`
- **Forecast comparison / TOST / modern**: see `expressions::forecast`, `expressions::tost`, `expressions::modern`.
- **Post-hoc tests** (`src/expressions/posthoc.rs`):
  - `p_adjust_fit`
  - `ptukey_fit`
  - `qtukey_fit`
  - `tukey_hsd_fit`
  - `pairwise_t_test_fit`
  - `dunn_test_fit`
- **Empirical-Bayes shrinkage** (`src/expressions/shrinkage.rs`):
  - `eb_shrink_fit`
- **Per-observation diagnostics** (`AugmentKwargs`):
  - `augment_fit`
- **OLS/WLS alias tolerance** (named kwargs structs, old `*_fit` signatures unchanged — `LinearAliasKwargs` / `LinearAliasPredictKwargs`): `ols_fit_with`/`wls_fit_with`/`ols_summary_fit_with`/`wls_summary_fit_with`/`ols_predict_fit_with`/`wls_predict_fit_with`

The input slice layout (which input is `y`, which are scalars, which are `x` columns) is documented above each function — same contract that the Polars plugin uses.

## Examples

Each example includes a complete, runnable script that shows the expected input DataFrame
shape — so you can see exactly what columns and data types each method requires (addresses
[issue #36](https://github.com/DataZooDE/polars-statistics/issues/36)).

### Runnable examples

| Example | Description |
|---------|-------------|
| [examples/01_ols_regression.py](examples/01_ols_regression.py) | OLS regression basics: fitting, coefficients, predictions, and R-style formulas |
| [examples/02_grouped_regression.py](examples/02_grouped_regression.py) | Running regression per group with `group_by` and `over` |
| [examples/03_glm_models.py](examples/03_glm_models.py) | Generalized Linear Models: logistic (binary), Poisson (counts) |
| [examples/04_statistical_tests.py](examples/04_statistical_tests.py) | T-tests, Mann-Whitney U, Shapiro-Wilk, and other hypothesis tests |
| [examples/05_demand_classification.py](examples/05_demand_classification.py) | AID (Automatic Identification of Demand) for demand pattern classification |
| [examples/06_robust_regression.py](examples/06_robust_regression.py) | Robust & sparse regressors: TheilSen, RANSAC, BayesianRidge, ARD, LARS, PassiveAggressive |
| [examples/07_glm_smoothers_streaming.py](examples/07_glm_smoothers_streaming.py) | Gamma GLM, GLMM random effects, PSpline smoother, MomentAccumulator streaming |
| [examples/08_anova.py](examples/08_anova.py) | One-way, two-way, and repeated-measures ANOVA |
| [examples/rust_wls.rs](examples/rust_wls.rs) | Rust API example: Weighted Least Squares via `wls_fit` |

### Cookbook (docs/examples/)

| Page | Description |
|------|-------------|
| [docs/examples/ab-testing.md](docs/examples/ab-testing.md) | A/B testing |
| [docs/examples/advanced-correlation.md](docs/examples/advanced-correlation.md) | Advanced correlation |
| [docs/examples/anova.md](docs/examples/anova.md) | ANOVA: one-way, two-way, repeated-measures |
| [docs/examples/categorical-analysis.md](docs/examples/categorical-analysis.md) | Categorical analysis |
| [docs/examples/equivalence-testing.md](docs/examples/equivalence-testing.md) | Equivalence testing (TOST) |
| [docs/examples/forecast-comparison.md](docs/examples/forecast-comparison.md) | Forecast comparison |
| [docs/examples/glm-models.md](docs/examples/glm-models.md) | GLM models |
| [docs/examples/glm-smoothers-streaming.md](docs/examples/glm-smoothers-streaming.md) | Gamma, GLMM, PSpline, MomentAccumulator |
| [docs/examples/group-analysis.md](docs/examples/group-analysis.md) | Group analysis |
| [docs/examples/hypothesis-testing.md](docs/examples/hypothesis-testing.md) | Hypothesis testing |
| [docs/examples/regression-workflow.md](docs/examples/regression-workflow.md) | Regression workflow |
| [docs/examples/regularized-regression.md](docs/examples/regularized-regression.md) | Regularized regression |
| [docs/examples/robust-regression.md](docs/examples/robust-regression.md) | Robust & sparse regression |
| [docs/examples/special-models.md](docs/examples/special-models.md) | Special models |

## Documentation

- **[API Reference](docs/api/README.md)** - Complete API documentation
  - [Statistical Tests](docs/api/tests/) - Parametric, non-parametric, TOST equivalence
  - [Regression Models](docs/api/regression/) - Linear, GLM, ALM, dynamic
  - [Model Classes](docs/api/classes/) - Python classes for direct access
  - [Output Structures](docs/api/outputs.md) - Return type definitions

**Guides:**

- **[Model Selection](docs/model-selection.md)** - decision matrix for choosing among comparable models
- **[sklearn Migration](docs/sklearn-migration.md)** - map scikit-learn workflows to polars-statistics
- **[Migration Guide](docs/migration.md)** - `icc` matrix contract and `with_intercept` → `add_intercept`

For the legacy monolithic reference, see [docs/API_REFERENCE.md](docs/API_REFERENCE.md).

## Claude Code Plugin

The repository ships a [Claude Code](https://claude.com/claude-code) plugin with five skills. Each skill documents a part of the Python API (expression builders, model classes, options and result fields), so that Claude Code can write polars-statistics code from a plain-language request such as *"compare conversion rates per region with a t-test and report Cohen's d"*.

Install it in a Claude Code session:

```
/plugin marketplace add DataZooDE/polars-statistics
/plugin install polars-statistics@polars-statistics
```

| Skill | Covers |
|-------|--------|
| `polars-statistics-getting-started` | Installation, the two API styles (expressions vs. model classes), model-selection decision matrix, migration notes |
| `polars-statistics-hypothesis-tests` | Parametric and nonparametric tests, normality, correlation, categorical tests, effect sizes, bootstrap CIs, TOST equivalence, distribution comparison |
| `polars-statistics-anova` | One-way (Fisher/Welch), two-way and repeated-measures ANOVA, intraclass correlation |
| `polars-statistics-regression` | OLS, Ridge/Lasso/Elastic Net, WLS/RLS, BLS/NNLS, quantile, robust (Huber/RANSAC/Theil-Sen), PLS, ALM, GLMs, GLMM, formulas, `group_by`/`over` fitting |
| `polars-statistics-results` | Extracting struct fields, unnesting results, `to_dict()` / `summary()`, error handling, type stubs |

The skills live in [`plugins/polars-statistics/`](plugins/polars-statistics/). To use a local checkout instead of the marketplace, start Claude Code with `claude --plugin-dir ./plugins/polars-statistics`.

## Performance

Built on high-performance Rust libraries:
- **[faer](https://github.com/sarah-ek/faer-rs)**: Fast linear algebra with SIMD
- **Zero-copy**: Direct memory sharing between Python and Rust
- **Automatic parallelization**: For `group_by` operations

## Development

```bash
git clone https://github.com/DataZooDE/polars-statistics.git
cd polars-statistics
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy polars pytest
maturin develop --release
pytest
```

## License

MIT License - see [LICENSE](LICENSE) for details.
