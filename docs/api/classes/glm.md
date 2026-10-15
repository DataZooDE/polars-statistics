# GLM Model Classes

Direct GLM model access outside of Polars expressions.

All GLM classes (`Logistic`, `Poisson`, `NegativeBinomial`, `Tweedie`, `Probit`, `Cloglog`) accept a `lambda_=0.0` kwarg for L2 (ridge) penalty applied inside the IRLS update. The sklearn-style [`LogisticRegression`](#logisticregression) class exposes the same penalty via `C = 1 / lambda_` and an explicit `penalty` choice.

## Common Interface

```python
from polars_statistics import Logistic, Poisson, NegativeBinomial, Tweedie, Probit, Cloglog

model = Logistic(with_intercept=True)
model.fit(X, y)

# Properties
model.coefficients      # np.ndarray
model.intercept         # float or None
model.deviance          # float
model.null_deviance     # float
model.aic               # float
model.bic               # float
model.model_info         # dict — {"model_type": str, "family": str | None, "link": str}

# Predict
predictions = model.predict(X_new)
probs = model.predict_proba(X_new)  # For classification models
```

**Inference getters (every GLM class):** `std_errors`, `t_statistics` (Wald z, normal
reference for `Logistic`/`Probit`/`Cloglog`/`LogisticRegression`; t on the residual df for
`Poisson`/`NegativeBinomial`/`Tweedie`/`Gamma`, matching R's `summary.glm`), `p_values`,
`conf_interval_lower`, `conf_interval_upper` (feature coefficients), plus
`intercept_std_error`, `intercept_t_statistic`, `intercept_p_value`, `intercept_conf_interval`
for the intercept term — read directly from upstream's `GlmInferenceResult`, never
recomputed locally. Every getter is `None` unless `compute_inference=True` (and the
`intercept_*` getters are additionally `None` without an intercept). `confidence_level`
(default `0.95`) controls the width of every `conf_interval_*`/`intercept_conf_interval`.
R equivalent: `summary.glm`'s coefficient table / `confint.default`.

**`compute_inference`/`confidence_level` constructor kwargs:** available on `Logistic`,
`Poisson`, `Probit`, `Cloglog`, `NegativeBinomial`, `Tweedie`, `Gamma`, `LogisticRegression`,
and `PenalizedGLM` (not `GLMM`/`PSpline`, which have no such kwarg).
`compute_inference=False` skips the inference computation entirely (every getter above
returns `None`), which is faster when only `coefficients`/`predict` are needed.
Probit/Cloglog/NegativeBinomial/Tweedie/Gamma gained these two kwargs in 0.8.0 (Logistic,
Poisson, and LogisticRegression already had them).

**`.augment(x, y, residual_type=None)`:** `Logistic`, `Poisson`, `Probit`, `Cloglog`,
`LogisticRegression`, `NegativeBinomial`, `Tweedie`, and `Gamma` (not `PenalizedGLM`, `GLMM`,
or `PSpline`) expose R `broom::augment`-style per-observation diagnostics (`fitted`,
`residual`, `std_residual`, `stud_residual`, `leverage`, `cooks_d`, `dffits`, plus a trailing
`resid_type` key) as a `dict` of numpy arrays — see
[`ps.augment`](../regression/diagnostics.md#augment-per-observation-diagnostics) for the full
field semantics and R equivalence. The instance's own fitted parameters (link, `theta`,
`var_power`, etc.) are always used, so `.augment()` always describes exactly what that
instance was fit with.

**`model_info` (0.8.0):** every class on this page (`Logistic`, `Poisson`, `Probit`, `Cloglog`,
`NegativeBinomial`, `Tweedie`, `Gamma`, `LogisticRegression`, `PenalizedGLM`, `GLMM`, `PSpline`)
exposes `model_info -> dict` — `{"model_type": str, "family": str | None, "link": str}`. The
`model_type` distinguishes classes that share a family/link — e.g. `Logistic`/`Probit`/
`Cloglog` all report `family="binomial"` but `model_type` `"binomial"` with `link` `"logit"`/
`"probit"`/`"cloglog"` respectively, while `LogisticRegression` reports `model_type="logistic"`
(the separate sklearn-style solver). See [Linear Model Classes](linear.md#model_info) for the
full cross-family `model_type` table.

---

## Logistic

Logistic regression for binary classification.

```python
from polars_statistics import Logistic

model = Logistic(
    lambda_: float = 0.0,  # L2 regularization
    with_intercept: bool = True,
)
model.fit(X, y)  # y: binary (0/1)

predictions = model.predict(X_new)      # Class predictions
probabilities = model.predict_proba(X_new)  # Probability estimates
```

---

## LogisticRegression

Sklearn-style logistic regression. Distinct from [`Logistic`](#logistic): uses inverse-strength regularization `C = 1 / lambda_` and an explicit `penalty` choice.

```python
from polars_statistics import LogisticRegression

model = LogisticRegression(
    penalty: str = "l2",                # "l2" or "none"
    C: float = 1.0,                     # Inverse of regularization strength
    threshold: float = 0.5,
    with_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-8,
    compute_inference: bool = True,
    confidence_level: float = 0.95,
)
model.fit(X, y)                         # y: binary (0/1)

# Methods
classes      = model.predict(X_new)             # 0/1 predictions
probs        = model.predict_proba(X_new)       # Probability estimates
scores       = model.decision_function(X_new)   # Linear scores (log-odds)
accuracy     = model.score(X_new, y_new)        # Mean accuracy

# Properties
model.coefficients    # np.ndarray
model.intercept       # float or None
model.n_iter          # int — IRLS iterations until convergence
```

---

## Poisson

Poisson regression for count data.

```python
from polars_statistics import Poisson

model = Poisson(
    lambda_: float = 0.0,
    with_intercept: bool = True,
)
model.fit(X, y)  # y: non-negative counts
```

---

## NegativeBinomial

Negative Binomial regression for overdispersed count data.

```python
from polars_statistics import NegativeBinomial

model = NegativeBinomial(
    theta: float | None = None,  # Dispersion; None = estimate
    estimate_theta: bool = True,
    lambda_: float = 0.0,
    with_intercept: bool = True,
    estimate_dispersion: bool = False,
    compute_inference: bool = True,
    confidence_level: float = 0.95,
)
model.fit(X, y)

# Additional properties
model.theta_estimated  # Estimated dispersion parameter (theta)
model.dispersion       # float — SE-scaling dispersion factor (see below)
```

**`compute_inference`/`confidence_level` (0.8.0):** see [Common Interface](#common-interface) above.

**`estimate_dispersion`** (default `False`): controls the dispersion factor used to
scale the coefficient standard errors — **not** the negative-binomial `theta`
(overdispersion) parameter, which is always estimated/used as usual via
`theta`/`estimate_theta`. With the default `estimate_dispersion=False`,
`model.dispersion == 1.0` and `std_errors` match R `summary.glm` / `MASS::glm.nb`'s
default (dispersion fixed at 1). With `estimate_dispersion=True`, `dispersion` is
the Pearson chi2/df estimate and `std_errors` are scaled by its square root —
validated against `MASS::glm.nb` 7.3.65 (dispersion `0.8951608253323096`); the
coefficient point estimates are unaffected either way.

**Behaviour change (anofox-regression 0.5.15 #49):** before this release, standard
errors were always scaled by the Pearson dispersion when it exceeded 1 — now the
default (`estimate_dispersion=False`) always uses dispersion 1, matching R's own
default and requiring the explicit opt-in for the previous behaviour.

**Also changed (0.5.14 #42):** `log_likelihood`/`aic` now match R's `glm`/`glm.nb`
values (the full negative-binomial log-likelihood, not a Gaussian approximation);
`aic` counts `theta` as an estimated parameter only when `estimate_theta=True`.

---

## Tweedie

Tweedie GLM for flexible variance structures.

```python
from polars_statistics import Tweedie

model = Tweedie(
    var_power: float = 1.5,
    link_power: float | None = None,  # None = canonical link (1 - var_power)
    lambda_: float = 0.0,
    with_intercept: bool = True,
    compute_inference: bool = True,
    confidence_level: float = 0.95,
)
model.fit(X, y)

model.std_errors  # np.ndarray | None — Pearson-dispersion SEs
model.p_values     # np.ndarray | None
model.aic          # float | None — see note below
model.bic          # float | None
model.intercept_std_error, model.intercept_t_statistic   # float | None
model.intercept_p_value, model.intercept_conf_interval   # float | None, tuple[float, float] | None
```

**`link_power`/`compute_inference`/`confidence_level` (0.8.0):** `link_power=None` (default)
uses the canonical link `1 - var_power`; pass an explicit value (e.g. `0.0` for the log link)
to override it — this is the same `link_power` kwarg `ps.augment(model="tweedie", ...)` uses
to reproduce a specific fit. `compute_inference`/`confidence_level` — see
[Common Interface](#common-interface) above. `Tweedie` also exposes `.gamma()`,
`.gaussian()`, `.inverse_gaussian()` builder methods that return a new `Tweedie` fixed to
`var_power` `2.0`/`0.0`/`3.0` respectively, with the same `with_intercept`/`max_iter`/`tol`/
`lambda_`/`compute_inference`/`confidence_level` kwargs.

**`aic`/`bic` are `NaN` for non-closed-form `var_power`.** A closed-form Tweedie
density (hence a well-defined likelihood and `aic`/`bic`) exists only for
`var_power` in `{0.0, 1.0, 2.0, 3.0}` (Gaussian, Poisson, Gamma, Inverse Gaussian).
The default `var_power=1.5` (compound Poisson-Gamma) has **no** closed-form
density, so `aic`/`bic` are `NaN` by design — not a bug. `std_errors`/`p_values`
use the Pearson-dispersion estimate and are populated regardless of `var_power`.

**`p_values` since 0.8.0 (anofox-regression 0.5.20 #66):** every Wald `p_value`
(Tweedie and Gamma included) is a t-statistic on the residual df, exactly as R's
`summary.glm`, computed via the survival function so it never underflows to
exactly 0. Previously Tweedie/Gamma `p_values` could be `None`/`NaN`.

---

## Probit

Probit regression for binary classification.

```python
from polars_statistics import Probit

model = Probit(
    lambda_: float = 0.0,
    with_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-6,
    compute_inference: bool = True,
    confidence_level: float = 0.95,
)
model.fit(X, y)  # y: binary (0/1)
```

**`compute_inference`/`confidence_level` (0.8.0):** see
[Common Interface](#common-interface) above — `std_errors`/`t_statistics`/`p_values`/
`conf_interval_lower`/`conf_interval_upper`/`intercept_std_error`/`intercept_t_statistic`/
`intercept_p_value`/`intercept_conf_interval` are populated by default.

**Standard error precision since 0.8.0 (anofox-regression 0.5.20 #66):** the probit
link now uses `statrs`'s `erfc`/`erfc_inv` instead of an Abramowitz-Stegun
approximation, so `std_errors` (and the derived `t_statistics`/`p_values`) match
R's `summary.glm(binomial(link = "probit"))` to full precision — previously off in
the 5th digit.

---

## Cloglog

Complementary log-log regression for binary classification.

```python
from polars_statistics import Cloglog

model = Cloglog(
    lambda_: float = 0.0,
    with_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-6,
    compute_inference: bool = True,
    confidence_level: float = 0.95,
)
model.fit(X, y)  # y: binary (0/1)
```

**`compute_inference`/`confidence_level` (0.8.0):** see [Common Interface](#common-interface) above.

---

## Gamma

Gamma GLM (log link) for positive, right-skewed response data. Suitable for cost, duration,
or concentration data where variance scales with the mean squared.

```python
from polars_statistics import Gamma

model = Gamma(
    lambda_: float = 0.0,        # L2 regularization
    with_intercept: bool = True,
    compute_inference: bool = True,
    confidence_level: float = 0.95,
)
model.fit(X, y)   # y: strictly positive

# Properties
model.coefficients      # np.ndarray
model.intercept         # float or None
model.converged         # bool
model.std_errors        # np.ndarray | None — Pearson-dispersion SEs (matching R summary.glm)
model.p_values           # np.ndarray | None
model.aic               # float | None — always closed-form for Gamma (var_power is fixed at 2)
model.bic               # float | None
model.intercept_std_error, model.intercept_t_statistic   # float | None (0.8.0)
model.intercept_p_value, model.intercept_conf_interval   # float | None, tuple[float, float] | None (0.8.0)

# Predict
predictions = model.predict(X_new)      # Predicted means (exp(X @ coef + intercept))
eta = model.predict_eta(X_new)           # Linear predictor (log scale)
```

**Pearson-dispersion standard errors (anofox-regression 0.5.14 #43):** `std_errors`
now match R `summary.glm(family = Gamma)`'s Pearson chi2/df dispersion estimate
(previously a different scaling). Unlike the generic `Tweedie` class, `Gamma` has
no `var_power` parameter — its variance function is fixed at the Gamma case
(`var_power = 2`), so `aic`/`bic` are always well-defined (never `NaN`).

**Example:**
```python
import numpy as np
from polars_statistics import Gamma

X = np.column_stack([np.random.normal(0, 1, 200)])
y = np.exp(0.5 * X[:, 0] + np.random.normal(0, 0.3, 200))

model = Gamma()
model.fit(X, y)
print(f"AIC: {model.aic:.4f}, std_errors: {model.std_errors}")
```

---

## PenalizedGLM

Family-generic penalized GLM (IRLS with explicit priors) — one engine shared by Poisson,
Binomial, Logistic, Negative Binomial, Gamma and Tweedie. Without priors and with
`lambda_ = 0.0` this is a plain maximum-likelihood GLM matching R's `glm` / `MASS::glm.nb`;
with Gaussian priors it returns the posterior mode (as `arm::bayesglm` with fixed prior
scales). See [Penalized GLM](../regression/penalized-glm.md) for the full families/links,
dispersion, priors, and covariance-type semantics.

```python
from polars_statistics import PenalizedGLM

model = PenalizedGLM(
    family="poisson",             # "poisson", "binomial", "logistic", "negative_binomial", "gamma", "tweedie"
    link=None,                    # defaults to the family's canonical link
    theta=None,                   # Negative Binomial dispersion; None = estimated
    power=None,                   # Tweedie variance power in [1, 2]; default 1.5
    threshold=0.5,                # Logistic classification threshold
    lambda_=0.0,                  # Uniform L2 ridge penalty
    estimate_dispersion=False,    # Poisson-only quasipoisson scaling
    priors=None,                  # per-coefficient priors (flat / normal / laplace)
    vcov_type="laplace",          # "laplace", "sandwich", "naive"
    add_intercept=True,
    max_iter=100,
    tol=1e-8,
    compute_inference=True,
    confidence_level=0.95,
)
model.fit(x, y, offset=None)
predictions = model.predict(x_new, offset=None)   # response scale
eta = model.predict_linear(x_new)                 # linear predictor
```

### Fit statistics

```text
model.coefficients      # np.ndarray — NaN for columns dropped as constant/aliased
model.intercept         # float | None
model.family            # str — lower-case family name
model.link              # str — lower-case link name
model.deviance           # float
model.null_deviance      # float
model.pseudo_r_squared   # float — 1 - deviance / null_deviance
model.log_likelihood     # float
model.aic                # float
model.bic                # float
model.dispersion         # float — see Dispersion below
model.theta              # float | None — Negative Binomial only
model.accuracy           # float | None — Logistic only
model.iterations         # int
model.converged          # bool
model.n_observations     # int — rows with a non-finite y/x/offset are dropped
```

### Inference

```text
model.std_errors, model.z_values, model.p_values              # np.ndarray | None — feature coefficients
model.conf_interval_lower, model.conf_interval_upper            # np.ndarray | None

model.intercept_std_error, model.intercept_z_value              # float | None — read directly
model.intercept_p_value, model.intercept_conf_interval          # from upstream's GlmInferenceResult,
                                                                  # never recomputed
```

Every inference getter is `None` unless `compute_inference=True` (and the `intercept_*`
getters are additionally `None` without an intercept).

### Covariance

```text
model.vcov                # np.ndarray | None — scaled so sqrt(diag(vcov)) == std_errors (like R's vcov(glm))
model.information          # np.ndarray | None — upstream's unscaled penalized observed information
model.matrix_parameters    # list[int | None] | None — None = intercept row/column, int = feature index
```

`vcov` is scaled by `dispersion` for every family except Negative Binomial (whose `dispersion`
getter reports `theta`, not a covariance scale, so no scaling is applied there).

### Separation and fitted values

```text
model.has_separation         # bool — family="binomial"/"logistic" only; always False otherwise
model.separated_predictors   # list[int] — 0-based feature indices involved
model.separation_warning     # str | None
model.fitted_values           # np.ndarray — fitted means over the rows used
model.valid_rows               # list[int] — indices into the original rows that were used
```

### Example

```python
import numpy as np
from polars_statistics import PenalizedGLM

np.random.seed(0)
n = 80
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
mu = np.exp(0.3 + 0.5 * x1 - 0.2 * x2)
y = np.random.poisson(mu).astype(float)
X = np.column_stack([x1, x2])

model = PenalizedGLM(family="poisson", estimate_dispersion=True).fit(X, y)
print(model.coefficients, model.intercept)
print(model.dispersion, model.aic, model.bic)
print(model.matrix_parameters, model.vcov.shape)
```

**Validated against R `glm` / `MASS::glm.nb` / `negative.binomial(theta)`** on the
`anofox-regression` 0.5.22 R reference fixture.

**See Also:** [Survival (AFT)](survival.md) · [Penalized GLM expressions](../regression/penalized-glm.md)

---

## GLMM

Generalized Linear Mixed Model — extends GLM with random effects for grouped/clustered data.
`GLMM` cannot be constructed directly; use one of the six family factory methods below, each
of which returns a new (unfitted) `GLMM`.

```python
from polars_statistics import GLMM

model = GLMM.gaussian(
    with_intercept: bool = True,
    reml: bool = True,              # REML vs ML; default differs per family — see below
    max_iter: int = 100,
    tol: float = 1e-8,
    random_intercept: bool = True,
    random_slopes: list[int] | None = None,          # feature indices with a random slope
    random_slopes_per_factor: dict | None = None,    # per-factor random slopes (crossed fits)
    theta_max: float = 1000.0,       # upper bound for the variance-component search
)
model.fit(x, y, group, offset=None)              # group: 1D array of group labels (int or str)
model.fit_crossed(x, y, groups, offset=None)      # groups: list of 1D arrays, one per factor
predictions = model.predict_fixed(x, offset=None)  # fixed-effects-only prediction

# Properties
model.coefficients          # np.ndarray — fixed effects (alias: fixed_effects)
model.intercept             # float or None
model.random_effects        # dict — per-group random intercept estimates
model.n_observations        # int
model.n_groups              # int
model.model_info            # dict — {"model_type": "glmm", "family": str, "link": str}

# Wald inference
model.z_values               # np.ndarray — fixed_effects / std_errors
model.p_values                # np.ndarray — 2 * (1 - Phi(|z|))
model.conf_int(level=0.95)   # tuple[np.ndarray, np.ndarray] — (lower, upper)
```

**Family factories:** `GLMM.gaussian(...)` (`reml=True` default), `GLMM.poisson(...)`,
`GLMM.binomial(...)`, `GLMM.gamma(...)` (log link), `GLMM.tweedie(power=1.5, ...)` (log link),
`GLMM.negative_binomial(nb_theta=None, ...)` (log link; `nb_theta=None` estimates it, matching
`glmer.nb`) — all `reml=False` by default except `gaussian`. Each accepts the same
`with_intercept`/`max_iter`/`tol`/`random_intercept`/`random_slopes`/
`random_slopes_per_factor`/`theta_max` structural kwargs, validated by one shared internal
constructor (mutual exclusivity of `random_slopes`/`random_slopes_per_factor`, no-intercept-
no-slope, `theta_max` bounds — invalid combinations raise `ValueError`, never panic).

**`offset` (0.8.0):** `fit`/`fit_crossed`/`predict_fixed` all accept `offset=None` — a known
per-observation quantity added to the linear predictor on the link scale (coefficient fixed at
`1`), matching `offset()` in an `lme4` formula for every family.

**`random_slopes_per_factor` (0.8.0):** gives each crossed factor (from `fit_crossed`) its own
independent random slopes, matching `lme4`'s `(x | a) + (1 | b)` syntax. It is a list of
per-factor feature-index lists, positionally aligned with `fit_crossed`'s `groups` argument —
e.g. `random_slopes_per_factor=[[0], []]` with `groups=[factor_a, factor_b]` gives `factor_a`
a random slope on feature `0` while `factor_b` keeps only a random intercept. A single-factor
fit has no "per factor" distinction, so the factor accessors below return `None`.

**lme4-validated getters (0.8.0):** `nb_theta` (vs. the common `theta` dispersion getter —
`nb_theta` is `None` for every non-negative-binomial family), `random_cov` (the
random-effects covariance matrix), `random_effects_matrix` (BLUPs, one row per group/factor
level), `random_corr` (correlation matrix derived from `random_cov`), `n_random_effects`,
`n_factors` (`1` for a single-factor `fit`, more for `fit_crossed`), `is_reml`.

**Per-factor accessors (0.8.0, crossed fits only):** `factor_random_cov(f)`,
`factor_random_sd(f)`, `factor_random_effects_matrix(f)` — each takes a 0-based factor index
`f` and returns `None` for a single-factor fit or an out-of-range index. `factors()` lists
each factor's metadata as `list[dict]`.

**Wald inference** matches `lme4`'s standard inference (validated against `lme4`
2.0.1): `z_values == fixed_effects / std_errors`; `conf_int(level)`'s midpoint
equals `fixed_effects` and its half-width equals `z_(1-alpha/2) * std_errors`
(e.g. `1.959963984540054 * std_errors` for the default `level=0.95`). Invalid
`level` values (`0.0`, `1.0`, outside `(0, 1)`, or `NaN`) give all-`NaN` bounds.
`log_likelihood` for non-Gaussian families (`poisson`, `binomial`) now includes
the saturated-model term, matching `lme4`'s reported log-likelihood exactly.

**Validated against `lme4` 2.0.1** on `anofox-regression` 0.5.22's
`glmm_families_validation.rs` fixture: `negative_binomial`/`gamma`/`tweedie` families
(`glmer.nb`, `glmer(family=MASS::negative.binomial(theta))`, `glmer(Gamma(log))`,
`glmer(statmod::tweedie(power))`), `offset()`, and `random_slopes_per_factor` (`(x|a)+(1|b)`).

**Example — binomial family:**
```python
import numpy as np
from polars_statistics import GLMM

groups = np.repeat(np.arange(10), 20)   # 10 groups, 20 obs each
X = np.random.normal(0, 1, (200, 2))
y = (X[:, 0] + np.random.normal(0, 0.5, 200) > 0).astype(float)

model = GLMM.binomial()
model.fit(X, y, groups)
```

**Example — negative binomial with offset and crossed factors with per-factor slopes:**
```python
import numpy as np
from polars_statistics import GLMM

n = 200
X = np.random.normal(0, 1, (n, 1))
factor_a = np.repeat(np.arange(10), 20)
factor_b = np.tile(np.arange(4), 50)
offset = np.random.uniform(0.5, 2.0, n)
mu = offset * np.exp(0.3 + 0.5 * X[:, 0])
y = np.random.negative_binomial(5, 5 / (5 + mu)).astype(float)

model = GLMM.negative_binomial(random_slopes_per_factor=[[0], []])
model.fit_crossed(X, y, [factor_a, factor_b], offset=np.log(offset))
print(model.nb_theta, model.n_factors, model.factor_random_sd(0))
```

---

## PSpline

Penalized spline (P-spline) regression — fits a smooth function of one variable using
B-spline basis functions with a difference-based roughness penalty.

```python
from polars_statistics import PSpline

model = PSpline(
    n_knots: int = 20,            # Number of interior knots
    degree: int = 3,              # B-spline degree (3 = cubic)
    lambda_: float = 1.0,        # Smoothing penalty
    with_intercept: bool = True,
)
model.fit(x, y)   # x, y: 1D numpy arrays

# Properties
model.coefficients      # np.ndarray — B-spline coefficients
model.intercept         # float or None
model.knots             # np.ndarray — knot locations
model.r_squared         # float
model.n_observations    # int

# Predict
smooth = model.predict(x_new)
```

**Example:**
```python
import numpy as np
from polars_statistics import PSpline

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + np.random.normal(0, 0.1, 100)

model = PSpline(n_knots=15, lambda_=0.1)
model.fit(x, y)
smooth = model.predict(x)
```

---

## Class Summary

| Class | Parameters |
|-------|------------|
| `Logistic` | `lambda_`, `with_intercept` |
| `LogisticRegression` | `penalty`, `C`, `threshold`, `with_intercept`, `max_iter`, `tol`, `compute_inference`, `confidence_level` |
| `Poisson` | `lambda_`, `with_intercept` |
| `NegativeBinomial` | `theta`, `estimate_theta`, `lambda_`, `with_intercept` |
| `Tweedie` | `var_power`, `lambda_`, `with_intercept` |
| `Probit` | `lambda_`, `with_intercept` |
| `Cloglog` | `lambda_`, `with_intercept` |
| `Gamma` | `lambda_`, `with_intercept` |
| `PenalizedGLM` | `family`, `link`, `theta`, `power`, `threshold`, `lambda_`, `estimate_dispersion`, `priors`, `vcov_type`, `add_intercept`, `max_iter`, `tol`, `compute_inference`, `confidence_level` |
| `GLMM` | `family`, `link`, `with_intercept`, `max_iter`, `tol` |
| `PSpline` | `n_knots`, `degree`, `lambda_`, `with_intercept` |

---

## See Also

- [Linear Model Classes](linear.md)
- [ALM Class](alm.md)
- [Survival (AFT) Model Classes](survival.md)
- [Penalized GLM expressions](../regression/penalized-glm.md)
