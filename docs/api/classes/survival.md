# Survival Model Classes

Direct AFT survival regression model access outside of Polars expressions.

## AFT

Accelerated failure time (AFT) survival regression with right censoring, for any of the
Weibull, log-normal, log-logistic or exponential error distributions. Models
`log T = x'beta + sigma * W` for a fixed error distribution `W` — the parametrisation of R's
`survival::survreg`. An observation contributes its density when the event was observed and
its survival probability when the row is still censored.

```text
from polars_statistics import AFT

model = AFT(
    distribution="weibull",              # "weibull", "lognormal", "loglogistic", "exponential"
    priors=None,                          # same format as PenalizedGLM's priors
    vcov_type="laplace",                  # "laplace", "sandwich", "naive"
    add_intercept=True,
    max_iter=100,
    tol=1e-9,
    compute_inference=True,
    confidence_level=0.95,
)
model.fit(x, time, event)   # event: 0/1 (bool, int, or float), any other value raises ValueError

predictions = model.predict_linear(x)        # Linear predictor (location of log T)
quantiles = model.predict_quantile(x, p=0.9) # p-quantile of the survival time
medians = model.predict_median(x)            # predict_quantile(x, p=0.5)
survival = model.predict_survival(x, t=5.0)  # S(t | x) = P(T > t | x)
cdf = model.predict_cdf(x, t=5.0)            # F(t | x) = 1 - S(t | x)
```

**`distribution`:** `"lognormal"`/`"log_normal"`/`"log-normal"` and
`"loglogistic"`/`"log_logistic"`/`"log-logistic"` aliases are also accepted. `"exponential"`
(or `"exp"`) fixes `scale = 1` (not estimated) — every other distribution estimates it.

**`event`:** right-censoring indicator passed to `fit`, positionally after `time` — `1`/`True`
for an observed event, `0`/`False` for a right-censored observation. Any other value (e.g. R's
`1`/`2` coding) raises `ValueError` rather than being silently misread.

### Fit statistics

```text
model.coefficients       # np.ndarray — feature coefficients on the log-time scale
model.intercept          # float | None
model.scale              # float — sigma (1.0, not estimated, for distribution="exponential")
model.log_likelihood     # float — on the time scale, as survreg reports it
model.null_log_likelihood  # float — intercept(+scale)-only model
model.aic                # float
model.bic                # float
model.converged          # bool
model.iterations         # int
model.n_observations     # int — rows used (non-finite/non-positive time, or null time/event, dropped)
model.n_events            # int — observed events among n_observations
model.n_censored          # int — right-censored rows among n_observations
```

### Inference

```text
model.std_errors           # np.ndarray | None — feature coefficients
model.z_values              # np.ndarray | None — Wald z-statistics
model.p_values               # np.ndarray | None — two-sided p-values
model.conf_interval_lower    # np.ndarray | None
model.conf_interval_upper    # np.ndarray | None

model.intercept_std_error    # float | None — read directly from upstream's AftInference, never recomputed
model.intercept_z_value      # float | None
model.intercept_p_value      # float | None
model.intercept_conf_interval  # tuple[float, float] | None

model.log_scale_std_error    # float | None — None when the scale is fixed (distribution="exponential")
```

Every inference getter is `None` unless `compute_inference=True`, and the intercept/`log_scale`
getters are additionally `None` without an intercept / without an estimated scale.

### Covariance

```text
model.vcov          # np.ndarray | None — full parameter covariance at the mode
model.information   # np.ndarray | None — penalised observed information at the mode
```

Parameter order for `vcov`/`information`: the intercept first (when fitted), then the feature
coefficients, then `log(scale)` (when the scale is estimated) — so for a fit with an intercept
and `p` features the matrix is `(1 + p + 1)` square.

### Example

```python
import numpy as np
from polars_statistics import AFT

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
x = np.column_stack([x1, x2])

model = AFT(distribution="weibull").fit(x, observed_time, event)
print(model.coefficients, model.intercept, model.scale)
print(model.n_events, model.n_censored)
print(model.predict_quantile(x[:3], p=0.9))
```

**Validated against R `survival::survreg`** on the `anofox-regression` 0.5.22 R reference
fixture: coefficients/scale (`1e-6`), log-likelihoods (`1e-7`), and intercept/coefficient/
log(scale) standard errors (`1e-5`), across all four distributions.

Priors and the sandwich covariance — parameter order is `[intercept, x1, x2, log_scale]`:

```python
import numpy as np
from polars_statistics import AFT

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
x = np.column_stack([x1, x2])

model = AFT(
    distribution="weibull",
    priors=[None, ("normal", 0.0, 1.0), ("normal", 0.0, 1.0)],
    vcov_type="sandwich",
).fit(x, observed_time, event)
print(model.vcov.shape, model.information.shape)
print(model.intercept_std_error, model.intercept_z_value, model.intercept_p_value)
print(model.log_scale_std_error)
```

---

## Class Summary

| Class | Parameters |
|-------|------------|
| `AFT` | `distribution`, `priors`, `vcov_type`, `add_intercept`, `max_iter`, `tol`, `compute_inference`, `confidence_level` |

---

## See Also

- [Survival Regression (AFT)](../regression/survival.md) - Expression API
- [PenalizedGLM class](glm.md#penalizedglm) - Shares the same `priors`/`vcov_type` format
- [GLM Model Classes](glm.md)
