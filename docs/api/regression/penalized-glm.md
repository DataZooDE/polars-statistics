# Penalized GLM

One IRLS engine shared by Poisson, Binomial, Logistic, Negative Binomial, Gamma and Tweedie
regression, with optional explicit per-coefficient priors (Gaussian/Laplace), a uniform ridge
penalty (`lambda_`), offsets, and a selectable covariance at the mode. Without priors and with
`lambda_ = 0.0` this is a plain maximum-likelihood GLM matching R's `glm` / `MASS::glm.nb`; with
Gaussian priors it returns the posterior mode (as `arm::bayesglm` with fixed prior scales).
Validated against R `glm` / `MASS::glm.nb` on the `anofox-regression` 0.5.22 R reference fixture.

Works with `group_by` and `.over()`.

---

## Families and Links

| `family` | Links (`link=`) | Notes |
|----------|------------------|-------|
| `"poisson"` | `"log"` (default), `"identity"`, `"sqrt"` | `estimate_dispersion` is only valid here |
| `"binomial"` | `"logit"` (default), `"probit"`, `"cloglog"` | |
| `"logistic"` | `"logit"` only | Adds a classification `threshold` and the `accuracy` output field |
| `"negative_binomial"` | `"log"` only | `theta` fixed or ML-estimated (`None`) |
| `"gamma"` | `"log"` only | |
| `"tweedie"` | `"log"` only | `power` in `[1, 2]`, default `1.5` |

An invalid `link` for the chosen `family`, or `theta`/`power`/`threshold` supplied for the
wrong family, raises `ValueError` when the expression is **built** — before `.collect()`.

---

## Dispersion

Poisson fixes dispersion at `1.0` by default, exactly like R's `glm(family = poisson)`.
Setting `estimate_dispersion=True` scales the coefficient covariance (and therefore the
standard errors, z-statistics, p-values, and confidence intervals) by the Pearson chi2/df
dispersion estimate — R's `quasipoisson` family. `estimate_dispersion` is **Poisson-only**;
every other family fixes its own dispersion rule and rejects the option with `ValueError`.

Gamma and Tweedie always use the Pearson dispersion estimate (never fixed at `1.0`). Gamma's
log-likelihood and `aic`/`bic` are computed at deviance/n, matching R's `logLik.glm`. Tweedie's
`aic`/`bic` are `NaN` unless `power` is one of `0.0`, `1.0`, `2.0`, or `3.0` (no closed-form
density otherwise) — the default `power=1.5` therefore has `NaN` `aic`/`bic` by design.

```python
import numpy as np
import polars as pl
import polars_statistics as ps

np.random.seed(0)
n = 80
x1 = np.random.normal(0, 1, n)
mu = np.exp(0.3 + 0.5 * x1)
y = np.random.poisson(mu).astype(float)
df = pl.DataFrame({"y": y, "x1": x1})

# Fixed dispersion = 1.0 (R's glm(family = poisson) default)
default_fit = df.select(ps.penalized_glm("y", "x1", family="poisson").alias("m"))
print(default_fit["m"][0]["dispersion"])  # 1.0

# Quasipoisson: scale the covariance by the Pearson chi2/df estimate
quasi_fit = df.select(
    ps.penalized_glm("y", "x1", family="poisson", estimate_dispersion=True).alias("m")
)
print(quasi_fit["m"][0]["dispersion"])  # Pearson chi2/df, e.g. ~1.13
```

---

## Priors

`priors` accepts a sequence positionally aligned with the feature columns (`*x`). Each entry is:

- `None` or `"flat"` — flat (uninformative) prior, the default for every coefficient.
- A `(kind, loc, scale)` tuple/list (or `(kind,)`/`(kind, loc)`), or a dict
  `{"kind", "loc", "scale"}` — `kind` is one of `"flat"`, `"normal"`, `"laplace"`.

An optional **leading** entry applies to the intercept, so `priors` may have `p` entries
(one per feature) or `p + 1` entries (intercept first, then one per feature).

- `"normal"` (Gaussian): an L2 penalty with standard deviation `scale` — a ridge-like prior.
  With `lambda_ = L`, a `("normal", 0.0, 1 / sqrt(L))` prior on every non-intercept coefficient
  gives numerically identical coefficients to the uniform ridge penalty.
- `"laplace"`: an L1 penalty with scale `b` — can proximal-soft-threshold a coefficient to
  exactly `0.0`. A coefficient zeroed this way has `NaN` in `std_errors`/`ci_lower`/`ci_upper`
  (no curvature-based standard error exists for an inactive L1 coefficient).

```python
import numpy as np
import polars as pl
import polars_statistics as ps

np.random.seed(0)
n = 80
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
mu = np.exp(0.3 + 0.5 * x1 - 0.2 * x2)
y = np.random.poisson(mu).astype(float)
df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})

# Gaussian priors (ridge-like, posterior mode) on x1 and x2; flat on the intercept
result = df.select(
    ps.penalized_glm(
        "y", "x1", "x2",
        family="poisson",
        priors=[None, ("normal", 0.0, 1.0), ("normal", 0.0, 1.0)],
    ).alias("model")
)
print(result["model"][0]["coefficients"])
```

---

## Covariance types

`vcov_type` (on `penalized_glm_summary`; `PenalizedGLM.vcov` on the PyModel) selects the
covariance at the mode:

| `vcov_type` | Formula | Use case |
|-------------|---------|----------|
| `"laplace"` (default) | `(X'WX + P)^-1` | The Laplace approximation — the correct observed information for a MAP estimate |
| `"sandwich"` | sandwich estimator | Targets the frequentist sampling variance of the penalized estimator |
| `"naive"` | `(X'WX)^-1` | Ignores the penalty entirely |

All three coincide for an unpenalized fit (no priors, `lambda_ = 0.0`).

---

## Offsets

`offset` adds a known quantity to the linear predictor **on the link scale** — e.g.
`log(exposure)` for a Poisson rate model — and defaults to a constant `0.0` column. Null
deviance follows R's `glm` behaviour with an offset present or with `add_intercept=False`.
A null offset value always yields a `NaN` prediction from `penalized_glm_predict`, regardless
of `null_policy`.

```python
import numpy as np
import polars as pl
import polars_statistics as ps

np.random.seed(0)
n = 80
x1 = np.random.normal(0, 1, n)
exposure = np.random.uniform(0.5, 2.0, n)
mu = exposure * np.exp(0.3 + 0.5 * x1)
y = np.random.poisson(mu).astype(float)
df = pl.DataFrame({"y": y, "x1": x1, "exposure": exposure})

result = df.select(
    ps.penalized_glm("y", "x1", family="poisson", offset=pl.col("exposure").log()).alias("m")
)
print(result["m"][0]["intercept"])
```

---

## `penalized_glm`

```text
ps.penalized_glm(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    family: Literal["poisson", "binomial", "logistic", "negative_binomial", "gamma", "tweedie"] = "poisson",
    link: str | None = None,
    theta: float | None = None,
    power: float | None = None,
    threshold: float = 0.5,
    lambda_: float = 0.0,
    estimate_dispersion: bool = False,
    priors: Sequence[None | str | tuple | dict] | None = None,
    offset: Union[pl.Expr, str, None] = None,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> pl.Expr
```

| Parameter | Description |
|-----------|-------------|
| `family` | Response family; see [Families and Links](#families-and-links) |
| `link` | Link function; defaults to the family's canonical link |
| `theta` | Fixed Negative Binomial dispersion; `None` estimates it by ML (`family="negative_binomial"` only) |
| `power` | Tweedie variance power in `[1, 2]`; default `1.5` (`family="tweedie"` only) |
| `threshold` | Classification threshold in `(0, 1)` (`family="logistic"` only) |
| `lambda_` | Uniform L2 ridge penalty on non-intercept coefficients, composed additively with `priors` |
| `estimate_dispersion` | Poisson-only quasipoisson dispersion scaling; see [Dispersion](#dispersion) |
| `priors` | Per-coefficient priors; see [Priors](#priors) |
| `offset` | Offset column/expression on the link scale; defaults to `0.0`; see [Offsets](#offsets) |
| `add_intercept` | Whether to include an intercept term |
| `max_iter` | Maximum IRLS iterations |
| `tol` | Convergence tolerance |

**Returns:** `Struct` — see [Penalized GLM Output](../outputs.md#penalized-glm-output) for the
full 15-field schema (`intercept`, `coefficients`, `deviance`, `null_deviance`,
`pseudo_r_squared`, `log_likelihood`, `aic`, `bic`, `dispersion`, `theta`, `accuracy`,
`has_separation`, `iterations`, `converged`, `n_observations`).

Rows with a null/non-finite value in `y`, any feature, or the offset are dropped before fitting.

```python
import numpy as np
import polars as pl
import polars_statistics as ps

np.random.seed(0)
n = 80
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
mu = np.exp(0.3 + 0.5 * x1 - 0.2 * x2)
y = np.random.poisson(mu).astype(float)
df = pl.DataFrame({"group": ["A"] * n, "y": y, "x1": x1, "x2": x2})

result = df.group_by("group").agg(
    ps.penalized_glm("y", "x1", "x2", family="poisson").alias("model")
)
print(result)
```

---

## `penalized_glm_summary`

```text
ps.penalized_glm_summary(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    family: Literal[...] = "poisson",
    link: str | None = None,
    theta: float | None = None,
    power: float | None = None,
    threshold: float = 0.5,
    lambda_: float = 0.0,
    estimate_dispersion: bool = False,
    priors: Sequence[None | str | tuple | dict] | None = None,
    vcov_type: Literal["laplace", "sandwich", "naive"] = "laplace",
    conf_level: float = 0.95,
    offset: Union[pl.Expr, str, None] = None,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> pl.Expr
```

Unlike every other `*_summary` expression (which share the 5-field
`{term, estimate, std_error, statistic, p_value}` schema), this returns a widened **7-field**
schema that additionally carries `ci_lower`/`ci_upper` — the shared `summary_output_dtype` and
every other `*_summary` expression are unchanged.

**Returns:** `List[Struct]` with columns `term`, `estimate`, `std_error`, `statistic` (Wald z,
normal reference), `p_value`, `ci_lower`, `ci_upper`. The `intercept` row (when
`add_intercept=True`) is read directly from upstream's `GlmInferenceResult` intercept fields,
never recomputed locally. See [Inference Summary Output](../outputs.md#inference-summary-output).

```python
import numpy as np
import polars as pl
import polars_statistics as ps

np.random.seed(0)
n = 80
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
mu = np.exp(0.3 + 0.5 * x1 - 0.2 * x2)
y = np.random.poisson(mu).astype(float)
df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})

result = df.select(
    ps.penalized_glm_summary("y", "x1", "x2", family="poisson").alias("s")
).explode("s").unnest("s")
print(result)
```

---

## `penalized_glm_predict`

```text
ps.penalized_glm_predict(
    y: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    family: Literal[...] = "poisson",
    link: str | None = None,
    theta: float | None = None,
    power: float | None = None,
    threshold: float = 0.5,
    lambda_: float = 0.0,
    estimate_dispersion: bool = False,
    priors: Sequence[None | str | tuple | dict] | None = None,
    offset: Union[pl.Expr, str, None] = None,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-8,
    predict_type: Literal["response", "link"] = "response",
    null_policy: Literal["drop", "drop_y_zero_x"] = "drop",
    name: str | None = None,
) -> pl.Expr
```

Refits per group/`.over()` partition with the same options used for `penalized_glm`, then
predicts on every row. `predict_type="response"` applies the family's inverse link to the linear
predictor plus offset; `predict_type="link"` returns the linear predictor plus offset directly.

`lower`/`upper` are always `NaN` — `FittedPenalizedGlm` has no interval-prediction method
upstream (documented opt-out). A null `offset` value always yields a `NaN` prediction,
regardless of `null_policy`.

**Returns:** `Struct` with `{name}_prediction`/`{name}_lower`/`{name}_upper` (default
`name="penalized_glm"`). See [Prediction Output](../outputs.md#prediction-output).

```python
import numpy as np
import polars as pl
import polars_statistics as ps

np.random.seed(0)
n = 80
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
mu = np.exp(0.3 + 0.5 * x1 - 0.2 * x2)
y = np.random.poisson(mu).astype(float)
df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})

result = df.with_columns(
    ps.penalized_glm_predict("y", "x1", "x2", family="poisson").alias("pred")
).unnest("pred")
print(result.head(3))
```

---

## `penalized_glm_formula`

```text
ps.penalized_glm_formula(
    formula: str,
    family: Literal[...] = "poisson",
    link: str | None = None,
    theta: float | None = None,
    power: float | None = None,
    threshold: float = 0.5,
    lambda_: float = 0.0,
    estimate_dispersion: bool = False,
    priors: Sequence[None | str | tuple | dict] | None = None,
    offset: Union[pl.Expr, str, None] = None,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> pl.Expr
```

R-style formula wrapper delegating to `penalized_glm`. R's formula `offset()` term is not
parsed — pass `offset` as a column/expression directly.

```python
import numpy as np
import polars as pl
import polars_statistics as ps

np.random.seed(0)
n = 80
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
mu = np.exp(0.3 + 0.5 * x1 - 0.2 * x2)
y = np.random.poisson(mu).astype(float)
df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})

result = df.select(ps.penalized_glm_formula("y ~ x1 + x2", family="poisson").alias("model"))
print(result["model"][0]["intercept"])
```

---

## `penalized_glm_formula_summary`

```text
ps.penalized_glm_formula_summary(formula: str, family: Literal[...] = "poisson", ...) -> pl.Expr
```

See [`penalized_glm_summary`](#penalized_glm_summary) for parameter semantics; the formula's
left-hand side is `y`.

```python
import numpy as np
import polars as pl
import polars_statistics as ps

np.random.seed(0)
n = 80
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
mu = np.exp(0.3 + 0.5 * x1 - 0.2 * x2)
y = np.random.poisson(mu).astype(float)
df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})

result = df.select(
    ps.penalized_glm_formula_summary("y ~ x1 + x2", family="poisson").alias("s")
).explode("s").unnest("s")
print(result)
```

---

## `penalized_glm_formula_predict`

```text
ps.penalized_glm_formula_predict(formula: str, family: Literal[...] = "poisson", ...) -> pl.Expr
```

See [`penalized_glm_predict`](#penalized_glm_predict) for parameter semantics.

```python
import numpy as np
import polars as pl
import polars_statistics as ps

np.random.seed(0)
n = 80
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
mu = np.exp(0.3 + 0.5 * x1 - 0.2 * x2)
y = np.random.poisson(mu).astype(float)
df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})

result = df.with_columns(
    ps.penalized_glm_formula_predict("y ~ x1 + x2", family="poisson").alias("pred")
).unnest("pred")
print(result.head(3))
```

---

## Validated against R

Every family/link combination, offset handling, no-intercept null deviance, Gamma
log-likelihood at deviance/n, overdispersed-Poisson/quasipoisson dispersion, Negative Binomial
(estimated and fixed `theta`), Gaussian/Laplace priors across all three covariance types, the
documented ridge-`lambda_`-equals-Gaussian-prior relation, and the intercept inference rows are
validated against R `glm` / `MASS::glm.nb` on the `anofox-regression` 0.5.22 R reference fixture
(coefficients/standard errors to R's own precision; see
[R Validation](../../validation/index.md) for the full tolerance table).

---

## See Also

- [GLM](glm.md) - The non-penalized GLM expressions (`logistic`, `poisson`, `negative_binomial`, `tweedie`, `probit`, `cloglog`)
- [PenalizedGLM class](../classes/glm.md#penalizedglm) - Direct model access with the full `vcov`/`information` matrices
- [Survival Regression (AFT)](survival.md) - AFT survival regression, sharing the same priors/`vcov_type` format
- [Output Structures](../outputs.md) - Full struct field definitions
