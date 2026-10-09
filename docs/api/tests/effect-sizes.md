# Effect Sizes & Bootstrap Confidence Intervals

Standardised effect sizes for mean-difference and rank-based tests, plus percentile
bootstrap confidence intervals for the mean, median, or standard deviation.

> **Validation:** Cohen's d and the rank-biserial family are validated against R's
> `effectsize` package. Bootstrap CIs are validated against the upstream
> `anofox-statistics` reference tests.

All functions in this group return a single-row struct with an `estimate: Float64`
field (the effect-size functions) or the richer bootstrap-CI struct described under
[Bootstrap CI Output](../outputs.md#bootstrap-ci-output).

The examples below share two reference samples, matching R `effectsize` values:

```text
X  = [5.1, 4.9, 6.2, 5.8, 6.05, 5.5, 5.3, 6.1]           (n=8)
Y  = [6.5, 7.1, 6.8, 7.4, 6.0, 7.9, 6.6, 7.2, 6.9, 7.05]  (n=10)
```

`X` has fewer elements than `Y`. Whenever both are placed in one DataFrame (for the
two-sample functions), `X` is padded with trailing `None`s to `Y`'s length — the
builders' `is_finite()` filter drops the padding before computing, so the result is
identical to calling with the unpadded 8-element `X`.

## `cohens_d`

Standardised mean difference for a two-sample, Welch, or paired comparison.

```python title="Signature"
ps.cohens_d(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    kind: Literal["student", "welch", "paired"] = "student",
    mu: float = 0.0,
) -> pl.Expr
```

**Parameters:**
- `kind="student"` — pooled-SD standardiser (`(mean(x) - mean(y) - mu) / s_pooled`), matching R `effectsize::cohens_d(x, y, pooled_sd = TRUE)`.
- `kind="welch"` — average-variance standardiser that does not assume equal variances (`pooled_sd = FALSE`).
- `kind="paired"` — `d_z = mean(x - y - mu) / sd(x - y)`, matching `effectsize::cohens_d(x, y, paired = TRUE)`. `x`/`y` are filtered jointly so only rows where both are finite are used.
- `mu` — the null value of the mean difference.

**Returns:** `Struct{estimate: Float64}`

**Notes:** Returns NaN when the standardiser is zero (e.g. constant samples) or when either sample has fewer than 2 observations.

**Validated against R:** `effectsize::cohens_d` gives `-2.606655378290357` for the `kind="student"` comparison below.

**Example:**
```python
import polars as pl
import polars_statistics as ps

X = [5.1, 4.9, 6.2, 5.8, 6.05, 5.5, 5.3, 6.1, None, None]
Y = [6.5, 7.1, 6.8, 7.4, 6.0, 7.9, 6.6, 7.2, 6.9, 7.05]
df = pl.DataFrame({"X": X, "Y": Y})

print(df.select(ps.cohens_d("X", "Y").alias("d")))
# struct{estimate: -2.606655378290357...}
```

---

## `cohens_d_one_sample`

Standardised mean difference for a one-sample t-test.

```python title="Signature"
ps.cohens_d_one_sample(
    x: Union[pl.Expr, str],
    mu: float = 0.0,
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64}`

**Notes:** `d = (mean(x) - mu) / sd(x)`, matching R `effectsize::cohens_d(x, mu = mu)`. Returns NaN when `sd(x)` is zero or when `x` has fewer than 2 observations.

**Example:**
```python
import polars as pl
import polars_statistics as ps

X = [5.1, 4.9, 6.2, 5.8, 6.05, 5.5, 5.3, 6.1]
df = pl.DataFrame({"X": X})

print(df.select(ps.cohens_d_one_sample("X", mu=5.0)))
```

---

## `rank_biserial`

Rank-biserial correlation effect size for the Mann-Whitney U test.

```python title="Signature"
ps.rank_biserial(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    mu: Optional[float] = None,
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64}`

**Notes:** `r = 1 - 2 * U1 / (n1 * n2)`, where `U1` is the Mann-Whitney U statistic of `x`
(`MannWhitneyU.statistic`, or `ps.mann_whitney_u(...).struct.field("statistic")`).
`r > 0` when `y` tends to be larger than `x`. **Sign-convention note: R's
`effectsize::rank_biserial` reports the opposite sign.** Non-finite values are
dropped by this builder (the `MannWhitneyU` class instead ranks `+/-Inf` like R).
O(n1 * n2) time, O(1) memory — the same complexity as the Mann-Whitney U statistic
itself.

**Validated against R:** `0.925` for the sample below (matching `-effectsize::rank_biserial(x, y)` once the sign is flipped).

**Example:**
```python
import polars as pl
import polars_statistics as ps

X = [5.1, 4.9, 6.2, 5.8, 6.05, 5.5, 5.3, 6.1, None, None]
Y = [6.5, 7.1, 6.8, 7.4, 6.0, 7.9, 6.6, 7.2, 6.9, 7.05]
df = pl.DataFrame({"X": X, "Y": Y})

print(df.select(ps.rank_biserial("X", "Y")))
# struct{estimate: 0.925}
```

---

## `matched_pairs_rank_biserial`

Rank-biserial correlation effect size for the Wilcoxon signed-rank test.

```python title="Signature"
ps.matched_pairs_rank_biserial(
    x: Union[pl.Expr, str],
    y: Union[pl.Expr, str],
    mu: Optional[float] = None,
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64}`

**Notes:** `r = (T- - T+) / (T+ + T-)`, the sums of signed ranks of `|x - y - mu|` for
positive/negative differences (zero differences dropped). `r > 0` when `y` tends to
be larger than `x` — again the **opposite sign of R's
`effectsize::rank_biserial(x, y, paired = TRUE)`**. Returns NaN when every difference
is zero or lengths are unequal. Both columns must be finite in the same rows
(jointly filtered).

**Example:**
```python
import polars as pl
import polars_statistics as ps

X = [5.1, 4.9, 6.2, 5.8, 6.05, 5.5, 5.3, 6.1, None, None]
Y = [6.5, 7.1, 6.8, 7.4, 6.0, 7.9, 6.6, 7.2, 6.9, 7.05]
df = pl.DataFrame({"X": X, "Y": Y})

# Only the 8 jointly-finite rows are used (X's trailing Nones are dropped).
print(df.select(ps.matched_pairs_rank_biserial("X", "Y")))
```

---

## `rank_biserial_from_u`

Rank-biserial correlation computed directly from a Mann-Whitney U statistic, for
callers who already have `U1` (e.g. from a fitted `MannWhitneyU` model) and do not
want to recompute it from raw data.

```python title="Signature"
ps.rank_biserial_from_u(
    u1: float,
    n1: int,
    n2: int,
) -> pl.Expr
```

**Returns:** `Struct{estimate: Float64}`

**Notes:** `r = 1 - 2 * u1 / (n1 * n2)`; returns NaN when `n1 * n2 == 0`. Same sign
convention as `rank_biserial` (opposite of R's `effectsize::rank_biserial`).

**Example:**
```python
import polars as pl
import polars_statistics as ps
from polars_statistics import MannWhitneyU
import numpy as np

X = [5.1, 4.9, 6.2, 5.8, 6.05, 5.5, 5.3, 6.1]
Y = [6.5, 7.1, 6.8, 7.4, 6.0, 7.9, 6.6, 7.2, 6.9, 7.05]

u1 = MannWhitneyU().fit(np.array(X), np.array(Y)).statistic
print(pl.select(ps.rank_biserial_from_u(u1, len(X), len(Y))))
# struct{estimate: 0.925}
```

---

## `bootstrap_mean_ci`

Percentile bootstrap confidence interval for the mean, with IID, stationary, or
circular-block resampling.

```python title="Signature"
ps.bootstrap_mean_ci(
    x: pl.Expr | str,
    n_bootstrap: int = 1000,
    conf_level: float = 0.95,
    block_length: int = 0,
    seed: int | None = None,
) -> pl.Expr
```

**Parameters:**
- `block_length=0` — IID (nonparametric) bootstrap: resamples of size n drawn with replacement, independently per observation.
- `block_length=1` — stationary bootstrap with expected block length 1.
- `block_length>1` — circular block bootstrap with that fixed block length, for dependent (e.g. time-series) data.
- `seed` — random seed for reproducibility. Without one, each call draws fresh resamples (non-deterministic).

**Returns:** See [Bootstrap CI Output](../outputs.md#bootstrap-ci-output): `Struct{estimate, se, ci_lower, ci_upper, conf_level, n_bootstrap}`.

**Notes:** Non-finite values are dropped before resampling; for `block_length > 0` this
closes the gaps left by dropped values, changing which observations are adjacent for
block resampling. `n_bootstrap < 2`, `conf_level` outside `(0, 1)`, or fewer than 2
finite observations yield an all-NaN struct (never an exception).

**Example:**
```python
import polars as pl
import polars_statistics as ps

Y = [6.5, 7.1, 6.8, 7.4, 6.0, 7.9, 6.6, 7.2, 6.9, 7.05]
df = pl.DataFrame({"Y": Y})

print(df.select(ps.bootstrap_mean_ci("Y", seed=42)))
```

---

## `bootstrap_ci`

Percentile bootstrap confidence interval for a fixed statistic (`"mean"`, `"median"`,
or `"std"`), using the IID (nonparametric) bootstrap.

```python title="Signature"
ps.bootstrap_ci(
    x: pl.Expr | str,
    statistic: Literal["mean", "median", "std"] = "mean",
    n_bootstrap: int = 1000,
    conf_level: float = 0.95,
    seed: int | None = None,
) -> pl.Expr
```

**Returns:** See [Bootstrap CI Output](../outputs.md#bootstrap-ci-output): `Struct{estimate, se, ci_lower, ci_upper, conf_level, n_bootstrap}`.

**Why only `"mean" | "median" | "std"`?** This is a deliberate scope limit, not a
missing feature. The upstream `anofox_statistics::bootstrap_ci` is generic over a
native Rust closure, which cannot cross the Python FFI boundary — calling back into
Python from inside a Polars `#[polars_expr]` function running on a worker thread
risks a GIL-reentrancy deadlock. Only this fixed `{"mean", "median", "std"}` dispatch
is exposed; arbitrary Python callables are not accepted, and an unknown `statistic`
raises `ValueError` immediately (client-side, before reaching Rust).

**Notes:** `statistic="mean"` is bit-identical to `bootstrap_mean_ci(block_length=0)`
for the same seed (both use the same mean closure in the Rust backend). `"std"` is
the sample standard deviation (denominator `n - 1`). Non-finite values are dropped
before resampling; invalid parameters or fewer than 2 finite observations yield an
all-NaN struct (never an exception).

**Example:**
```python
import polars as pl
import polars_statistics as ps

Y = [6.5, 7.1, 6.8, 7.4, 6.0, 7.9, 6.6, 7.2, 6.9, 7.05]
df = pl.DataFrame({"Y": Y})

print(df.select(ps.bootstrap_ci("Y", statistic="median", seed=42)))
```

---

## See Also

- [Non-Parametric Tests](nonparametric.md) - `mann_whitney_u`, `wilcoxon_signed_rank` (the tests these rank-biserial effect sizes accompany)
- [Parametric Tests](parametric.md) - `ttest_ind`, `ttest_paired` (the tests Cohen's d accompanies)
- [Output Structures](../outputs.md) - Full struct-field reference
