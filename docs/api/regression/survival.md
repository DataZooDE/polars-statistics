# Survival Regression (AFT)

Accelerated failure time (AFT) survival regression with right censoring, for any of the
Weibull, log-normal, log-logistic or exponential error distributions. Validated against R
`survival::survreg` on the `anofox-regression` 0.5.22 R reference fixture (all four
distributions).

Works with `group_by` and `.over()`.

---

## Model

AFT models `log T = x'beta + sigma * W` for a fixed error distribution `W` — the same
parametrisation R's `survival::survreg` uses. An observation contributes its density when the
event was observed and its survival probability when the row is still censored.

---

## Distributions

| `distribution` | Aliases | Scale (`sigma`) |
|-----------------|---------|------------------|
| `"weibull"` | — | Estimated |
| `"lognormal"` | `"log_normal"`, `"log-normal"` | Estimated |
| `"loglogistic"` | `"log_logistic"`, `"log-logistic"` | Estimated |
| `"exponential"` | `"exp"` | Fixed at `1.0` (not estimated) |

An unknown `distribution` raises `ValueError` when the expression is **built** — before
`.collect()`.

---

## Event Coding

`event` is the right-censoring indicator, positionally after `time`:

- `True`/`1` — an observed event.
- `False`/`0` — a right-censored observation.
- Accepts a Boolean column directly, or a numeric column containing **only** `0`/`1`. Any other
  value — including R's `1`/`2` coding — fails the whole fit rather than being silently
  misread.

`time` must be strictly positive; a non-positive or non-finite `time` fails the whole fit
(returned as all-`NaN`). A null `time` or a null `event` drops that row from the fit. Rows with
a null/non-finite value in any feature are also dropped before fitting.

---

## `aft`

```text
ps.aft(
    time: Union[pl.Expr, str],
    event: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    distribution: Literal["weibull", "exponential", "lognormal", "loglogistic"] = "weibull",
    priors: Sequence[None | str | tuple | dict] | None = None,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> pl.Expr
```

| Parameter | Description |
|-----------|-------------|
| `time` | Event or censoring time; must be strictly positive |
| `event` | Right-censoring indicator; see [Event Coding](#event-coding) |
| `distribution` | Error distribution for `T`; see [Distributions](#distributions) |
| `priors` | Per-coefficient priors, positionally aligned with `x`. Same format as [`penalized_glm`'s `priors`](penalized-glm.md#priors) |
| `add_intercept` | Whether to include an intercept term |
| `max_iter` | Maximum Newton iterations |
| `tol` | Convergence tolerance on the log-likelihood |

**Returns:** `Struct` — see [AFT Output](../outputs.md#aft-output) for the full 12-field schema
(`intercept`, `coefficients`, `scale`, `log_likelihood`, `null_log_likelihood`, `aic`, `bic`,
`iterations`, `converged`, `n_observations`, `n_events`, `n_censored`).

```python
import numpy as np
import polars as pl
import polars_statistics as ps

np.random.seed(1)
n = 80
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
scale = 0.7
eta = 1.0 + 0.5 * x1 - 0.3 * x2
log_t = eta + scale * np.random.gumbel(size=n)  # extreme-value errors -> Weibull AFT
time = np.exp(log_t)
censor_time = np.random.exponential(scale=np.median(time) * 1.5, size=n)
observed_time = np.minimum(time, censor_time)
event = (time <= censor_time).astype(float)

df = pl.DataFrame(
    {"group": ["A"] * n, "time": observed_time, "event": event, "x1": x1, "x2": x2}
)

result = df.group_by("group").agg(
    ps.aft("time", "event", "x1", "x2", distribution="weibull").alias("model")
)
print(result)
```

---

## `aft_summary`

```text
ps.aft_summary(
    time: Union[pl.Expr, str],
    event: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    distribution: Literal["weibull", "exponential", "lognormal", "loglogistic"] = "weibull",
    priors: Sequence[None | str | tuple | dict] | None = None,
    vcov_type: Literal["laplace", "sandwich", "naive"] = "laplace",
    conf_level: float = 0.95,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> pl.Expr
```

Rows are `intercept` (when `add_intercept=True`), `x1..xp`, then `log_scale` (omitted for
`distribution="exponential"`, whose scale is fixed at `1` and not estimated).

**Returns:** `List[Struct]` with columns `term`, `estimate`, `std_error`, `statistic`,
`p_value`, `ci_lower`, `ci_upper` — the shared 7-field tidy summary schema, also used by
[`penalized_glm_summary`](penalized-glm.md#penalized_glm_summary). The `intercept` row is read
directly from upstream's `AftInference` intercept fields, never recomputed locally; the
`log_scale` row's `statistic`/`p_value`/CI are computed from upstream's `log_scale_std_error`
using the identical Wald formula (`z = estimate / std_error`,
`p = 2 * (1 - Phi(|z|))`). See [Inference Summary Output](../outputs.md#inference-summary-output).

```python
import numpy as np
import polars as pl
import polars_statistics as ps

np.random.seed(1)
n = 80
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
scale = 0.7
eta = 1.0 + 0.5 * x1 - 0.3 * x2
log_t = eta + scale * np.random.gumbel(size=n)
time = np.exp(log_t)
censor_time = np.random.exponential(scale=np.median(time) * 1.5, size=n)
observed_time = np.minimum(time, censor_time)
event = (time <= censor_time).astype(float)

df = pl.DataFrame({"time": observed_time, "event": event, "x1": x1, "x2": x2})

result = df.select(
    ps.aft_summary("time", "event", "x1", "x2", distribution="weibull").alias("s")
).explode("s").unnest("s")
print(result)
```

---

## `aft_predict_quantile`

```text
ps.aft_predict_quantile(
    time: Union[pl.Expr, str],
    event: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    p: float = 0.5,
    distribution: Literal["weibull", "exponential", "lognormal", "loglogistic"] = "weibull",
    priors: Sequence[None | str | tuple | dict] | None = None,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-9,
    null_policy: Literal["drop", "drop_y_zero_x"] = "drop",
    name: str | None = None,
) -> pl.Expr
```

Predicted `p`-quantile of the survival time. Refits per group/`.over()` partition with the same
options used for `aft`, then predicts on every row.

`null_policy="drop"` (default) gives a `NaN` prediction for a row with a null/non-finite
`time`, `event`, or feature. `null_policy="drop_y_zero_x"` excludes a null-`time`/`event` row
from the **fit** but still **predicts** it (unlike `penalized_glm_predict`'s offset, `event` is
fit-only data — not required predict input, since the four predict expressions only read `x`).

**Returns:** `Struct` with `{name}_prediction`/`{name}_lower`/`{name}_upper` (default
`name="aft_predict_quantile"`). `lower`/`upper` are always `NaN` — the upstream engine has no
interval-prediction method.

```python
import numpy as np
import polars as pl
import polars_statistics as ps

np.random.seed(1)
n = 80
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
scale = 0.7
eta = 1.0 + 0.5 * x1 - 0.3 * x2
log_t = eta + scale * np.random.gumbel(size=n)
time = np.exp(log_t)
censor_time = np.random.exponential(scale=np.median(time) * 1.5, size=n)
observed_time = np.minimum(time, censor_time)
event = (time <= censor_time).astype(float)

df = pl.DataFrame({"group": ["A"] * n, "time": observed_time, "event": event, "x1": x1, "x2": x2})

result = df.with_columns(
    ps.aft_predict_quantile("time", "event", "x1", "x2", p=0.9, distribution="weibull")
        .over("group").alias("pred")
).unnest("pred")
print(result.head(3))
```

---

## `aft_predict_median`

```text
ps.aft_predict_median(time, event, *x, distribution="weibull", priors=None, add_intercept=True, max_iter=100, tol=1e-9, null_policy="drop", name=None) -> pl.Expr
```

`aft_predict_quantile` at `p=0.5`. See [`aft_predict_quantile`](#aft_predict_quantile) for
parameter semantics (other than `p`, which is fixed).

```python
import numpy as np
import polars as pl
import polars_statistics as ps

np.random.seed(1)
n = 80
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
scale = 0.7
eta = 1.0 + 0.5 * x1 - 0.3 * x2
log_t = eta + scale * np.random.gumbel(size=n)
time = np.exp(log_t)
censor_time = np.random.exponential(scale=np.median(time) * 1.5, size=n)
observed_time = np.minimum(time, censor_time)
event = (time <= censor_time).astype(float)

df = pl.DataFrame({"time": observed_time, "event": event, "x1": x1, "x2": x2})

result = df.with_columns(
    ps.aft_predict_median("time", "event", "x1", "x2", distribution="weibull").alias("pred")
).unnest("pred")
print(result.head(3))
```

---

## `aft_predict_survival`

```text
ps.aft_predict_survival(
    time: Union[pl.Expr, str],
    event: Union[pl.Expr, str],
    *x: Union[pl.Expr, str],
    t: float,
    distribution: Literal["weibull", "exponential", "lognormal", "loglogistic"] = "weibull",
    priors: Sequence[None | str | tuple | dict] | None = None,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-9,
    null_policy: Literal["drop", "drop_y_zero_x"] = "drop",
    name: str | None = None,
) -> pl.Expr
```

Predicted survival probability `S(t | x) = P(T > t | x)`. `t` is required, keyword-only, and
must be finite and `> 0`. See [`aft_predict_quantile`](#aft_predict_quantile) for the other
parameters.

```python
import numpy as np
import polars as pl
import polars_statistics as ps

np.random.seed(1)
n = 80
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
scale = 0.7
eta = 1.0 + 0.5 * x1 - 0.3 * x2
log_t = eta + scale * np.random.gumbel(size=n)
time = np.exp(log_t)
censor_time = np.random.exponential(scale=np.median(time) * 1.5, size=n)
observed_time = np.minimum(time, censor_time)
event = (time <= censor_time).astype(float)

df = pl.DataFrame({"time": observed_time, "event": event, "x1": x1, "x2": x2})

result = df.with_columns(
    ps.aft_predict_survival("time", "event", "x1", "x2", t=5.0, distribution="weibull").alias("pred")
).unnest("pred")
print(result.head(3))
```

---

## `aft_predict_cdf`

```text
ps.aft_predict_cdf(time, event, *x, t: float, distribution="weibull", priors=None, add_intercept=True, max_iter=100, tol=1e-9, null_policy="drop", name=None) -> pl.Expr
```

Predicted failure probability `F(t | x) = 1 - S(t | x)`. `t` is required, keyword-only, and
must be finite and `> 0`. `S(aft_predict_quantile(p)) == 1 - p` and
`aft_predict_cdf(t) == 1 - aft_predict_survival(t)` hold exactly on the fitted model.

```python
import numpy as np
import polars as pl
import polars_statistics as ps

np.random.seed(1)
n = 80
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
scale = 0.7
eta = 1.0 + 0.5 * x1 - 0.3 * x2
log_t = eta + scale * np.random.gumbel(size=n)
time = np.exp(log_t)
censor_time = np.random.exponential(scale=np.median(time) * 1.5, size=n)
observed_time = np.minimum(time, censor_time)
event = (time <= censor_time).astype(float)

df = pl.DataFrame({"time": observed_time, "event": event, "x1": x1, "x2": x2})

result = df.with_columns(
    ps.aft_predict_cdf("time", "event", "x1", "x2", t=5.0, distribution="weibull").alias("pred")
).unnest("pred")
print(result.head(3))
```

---

## `aft_formula`

```text
ps.aft_formula(
    formula: str,
    event: Union[pl.Expr, str],
    distribution: Literal["weibull", "exponential", "lognormal", "loglogistic"] = "weibull",
    priors: Sequence[None | str | tuple | dict] | None = None,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> pl.Expr
```

R-style formula wrapper for `aft`. The formula's left-hand side is the `time` column; `event`
is **not** part of the formula syntax and is passed separately.

```python
import numpy as np
import polars as pl
import polars_statistics as ps

np.random.seed(1)
n = 80
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
scale = 0.7
eta = 1.0 + 0.5 * x1 - 0.3 * x2
log_t = eta + scale * np.random.gumbel(size=n)
time = np.exp(log_t)
censor_time = np.random.exponential(scale=np.median(time) * 1.5, size=n)
observed_time = np.minimum(time, censor_time)
event = (time <= censor_time).astype(float)

df = pl.DataFrame({"time": observed_time, "event": event, "x1": x1, "x2": x2})

result = df.select(ps.aft_formula("time ~ x1 + x2", "event", distribution="weibull").alias("model"))
print(result["model"][0]["intercept"])
```

---

## `aft_formula_summary`

```text
ps.aft_formula_summary(
    formula: str,
    event: Union[pl.Expr, str],
    distribution: Literal["weibull", "exponential", "lognormal", "loglogistic"] = "weibull",
    priors: Sequence[None | str | tuple | dict] | None = None,
    vcov_type: Literal["laplace", "sandwich", "naive"] = "laplace",
    conf_level: float = 0.95,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> pl.Expr
```

R-style formula wrapper for `aft_summary`; same `time ~ x1 + x2` / separate `event` convention
as `aft_formula`.

```python
import numpy as np
import polars as pl
import polars_statistics as ps

np.random.seed(1)
n = 80
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
scale = 0.7
eta = 1.0 + 0.5 * x1 - 0.3 * x2
log_t = eta + scale * np.random.gumbel(size=n)
time = np.exp(log_t)
censor_time = np.random.exponential(scale=np.median(time) * 1.5, size=n)
observed_time = np.minimum(time, censor_time)
event = (time <= censor_time).astype(float)

df = pl.DataFrame({"time": observed_time, "event": event, "x1": x1, "x2": x2})

result = df.select(
    ps.aft_formula_summary("time ~ x1 + x2", "event", distribution="weibull").alias("s")
).explode("s").unnest("s")
print(result)
```

---

## Validated against R `survival::survreg`

Coefficients/scale (`1e-6`), log-likelihoods (`1e-7`), and intercept/coefficient/log(scale)
standard errors (`1e-5`) are validated against R's `survival::survreg` on the
`anofox-regression` 0.5.22 R reference fixture, across all four distributions (Weibull,
log-normal, log-logistic, exponential). Predicted quantiles at `p=0.25/0.5/0.9` match `survreg`
to `1e-6`, and `S(quantile(p)) == 1 - p` holds to `1e-9`. See [R Validation](../../validation/index.md)
for the full tolerance table.

---

## See Also

- [Survival Model Classes](../classes/survival.md) - `AFT` class, with the full `vcov`/`information` matrices
- [Penalized GLM](penalized-glm.md) - Shares the same `priors`/`vcov_type` format
- [Output Structures](../outputs.md) - Full struct field definitions
