# Forecast Comparison Tests

Tests for comparing predictive accuracy of forecasting models. Essential for model selection and evaluation in time series analysis.

> **Validation:** All forecast comparison tests are validated against R implementations (forecast package, MCS package).

## `diebold_mariano`

Diebold-Mariano test for equal predictive accuracy between two forecasts.

The standard test for comparing out-of-sample forecast accuracy. Tests whether two forecasting models have equal expected loss.

```python
ps.diebold_mariano(
    errors1: Union[pl.Expr, str],
    errors2: Union[pl.Expr, str],
    loss: str = "squared",  # "squared", "absolute"
    horizon: int = 1,       # Forecast horizon
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**Null hypothesis:** The two forecasts have equal expected loss (E[d_t] = 0 where d_t = L(e1_t) - L(e2_t)).

**Parameters:**
- `loss="squared"`: Mean squared error (MSE) - penalizes large errors more
- `loss="absolute"`: Mean absolute error (MAE) - more robust to outliers
- `horizon`: Forecast horizon h; accounts for autocorrelation in h-step ahead forecast errors

**When to use:**
- Comparing two competing forecasting models
- Model selection for time series prediction
- Evaluating whether a complex model beats a simple benchmark

**Example:**
```python
# Compare ARIMA vs neural network forecasts
df.select(ps.diebold_mariano("arima_errors", "nn_errors", horizon=1))

# Multi-step ahead comparison with MAE loss
df.select(ps.diebold_mariano("model1_errors", "model2_errors",
                              loss="absolute", horizon=4))
```

**Reference:** Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive Accuracy", Journal of Business & Economic Statistics.

---

## `permutation_t_test`

Permutation-based t-test for comparing two samples (non-parametric).

Uses permutation resampling to compute exact p-values without distributional assumptions. Valid for any sample size.

```python
ps.permutation_t_test(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    alternative: str = "two-sided",
    n_permutations: int = 999,
    seed: int | None = None,
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**Null hypothesis:** The two samples come from the same distribution.

**When to use:**
- Small samples where t-test assumptions may not hold
- Non-normal data
- When exact p-values are required

---

## `clark_west`

Clark-West test for comparing nested forecasting models.

Adjusts the Diebold-Mariano test for the case when one model nests another (e.g., comparing AR(1) vs AR(1) + X). The standard DM test is undersized for nested models.

```python
ps.clark_west(
    restricted_errors: Union[pl.Expr, str],
    unrestricted_errors: Union[pl.Expr, str],
    horizon: int = 1,
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**Null hypothesis:** The restricted (simpler) model forecasts as well as the unrestricted model.

**When to use:**
- Comparing a benchmark model to an augmented version
- Testing if additional predictors improve forecasts
- Nested model comparison (e.g., random walk vs model with fundamentals)

**Reference:** Clark, T.E. & West, K.D. (2007). "Approximately Normal Tests for Equal Predictive Accuracy in Nested Models", Journal of Econometrics.

---

## `spa_test`

Superior Predictive Ability (SPA) test by Hansen (2005).

Tests whether any model in a set significantly outperforms a benchmark, controlling for data-snooping bias when many models are compared.

```python
ps.spa_test(
    benchmark_loss: Union[pl.Expr, str],
    *model_losses: Union[pl.Expr, str],
    n_bootstrap: int = 999,
    block_length: float = 5.0,
    seed: int | None = None,
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**Null hypothesis:** No model outperforms the benchmark (max expected loss difference ≤ 0).

**Parameters:**
- `block_length`: Block length for stationary bootstrap; accounts for serial correlation

**When to use:**
- Comparing many models to a single benchmark
- Addressing data-snooping concerns in model selection
- Robust model comparison with multiple alternatives

**Reference:** Hansen, P.R. (2005). "A Test for Superior Predictive Ability", Journal of Business & Economic Statistics.

---

## `model_confidence_set`

Model Confidence Set (MCS) for identifying the set of best-performing models.

Returns a set of models that contains the best model(s) with a given confidence level. Unlike pairwise tests, MCS considers all models simultaneously.

```python
ps.model_confidence_set(
    *model_losses: Union[pl.Expr, str],
    alpha: float = 0.1,
    statistic: str = "range",  # "range" or "semi-quadratic"
    n_bootstrap: int = 999,
    block_length: float = 5.0,
    seed: int | None = None,
) -> pl.Expr
```

**Returns:** `Struct{included: List[Boolean], p_values: List[Float64]}`

**Interpretation:**
- `included[i] = True`: Model i is in the confidence set (cannot be rejected as best)
- `p_values[i]`: p-value at which model i would be eliminated

**Parameters:**
- `alpha=0.1`: Significance level for model elimination
- `statistic="range"`: Test statistic; "range" is more powerful, "semi-quadratic" more robust

**When to use:**
- Identifying all models that are statistically indistinguishable from the best
- Model selection when multiple good models exist
- Reporting uncertainty in model rankings

**Reference:** Hansen, P.R., Lunde, A. & Nason, J.M. (2011). "The Model Confidence Set", Econometrica.

---

## `mspe_adjusted`

MSPE-Adjusted SPA test for comparing nested models.

Combines the Clark-West adjustment for nested models with the SPA framework for multiple comparisons.

```python
ps.mspe_adjusted(
    benchmark_errors: Union[pl.Expr, str],
    *model_errors: Union[pl.Expr, str],
    n_bootstrap: int = 999,
    block_length: float = 5.0,
    seed: int | None = None,
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**When to use:**
- Multiple nested model comparisons
- Testing if any of several augmented models beats a simple benchmark

---

## Modern Distribution Tests

### `energy_distance`

Energy Distance test for comparing two distributions.

A powerful non-parametric test that detects differences in any moment of the distributions (mean, variance, shape). Based on the concept of statistical energy from physics.

```python
ps.energy_distance(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    n_permutations: int = 999,
    seed: int | None = None,
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**Null hypothesis:** The two samples come from the same distribution.

**Advantages:**
- Detects differences in location AND scale AND shape
- No parametric assumptions
- Works well in high dimensions

**When to use:**
- Testing if forecast error distributions differ between models
- Detecting distribution shift (covariate shift)
- General two-sample comparison

**Reference:** Székely, G.J. & Rizzo, M.L. (2013). "Energy Statistics", Wiley StatsRef.

---

### `mmd_test`

Maximum Mean Discrepancy (MMD) test with Gaussian kernel.

A kernel-based two-sample test that embeds distributions into a reproducing kernel Hilbert space. Widely used in machine learning for distribution comparison.

```python
ps.mmd_test(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    n_permutations: int = 999,
    seed: int | None = None,
) -> pl.Expr
```

**Returns:** `Struct{statistic: Float64, p_value: Float64}`

**Null hypothesis:** The two samples come from the same distribution.

**Advantages:**
- Powerful against a wide range of alternatives
- Detects subtle distributional differences
- Foundation of many generative model evaluations (GANs)

**When to use:**
- High-dimensional distribution comparison
- Detecting subtle differences in prediction distributions
- Model validation in machine learning

**Reference:** Gretton, A. et al. (2012). "A Kernel Two-Sample Test", JMLR.

---

## Choosing a Test

| Situation | Recommended Test |
|-----------|-----------------|
| Two non-nested forecasting models | `diebold_mariano` |
| Nested models (restricted vs unrestricted) | `clark_west` |
| Many models vs one benchmark | `spa_test` |
| Find all "best" models | `model_confidence_set` |
| Many nested models | `mspe_adjusted` |
| General distribution comparison | `energy_distance` or `mmd_test` |
| Non-parametric mean comparison | `permutation_t_test` |

---

## See Also

- [Parametric Tests](parametric.md) - Standard hypothesis tests
- [Non-Parametric Tests](nonparametric.md) - Distribution-free alternatives
