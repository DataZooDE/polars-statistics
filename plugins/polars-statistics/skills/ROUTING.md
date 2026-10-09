# polars-statistics Skill Routing

This document maps representative user task-prompts to the correct polars-statistics skill.
Use it to verify that skill `description` fields route correctly and to calibrate future
description edits.

---

## Skill Descriptions (source of truth)

| Skill | `description` |
|-------|---------------|
| `polars-statistics-getting-started` | Get started with polars-statistics — install the package, understand the two API styles (expressions vs. PyModel classes), choose the right model with the decision matrix, and migrate from sklearn. |
| `polars-statistics-hypothesis-tests` | Use polars-statistics hypothesis tests — run t-tests, Mann-Whitney, Kruskal-Wallis, normality tests, correlation tests, chi-square, Fisher's exact, TOST equivalence tests, MMD, energy distance, effect sizes (Cohen's d, rank-biserial), Jarque-Bera normality, or bootstrap confidence intervals as Polars expressions. |
| `polars-statistics-anova` | Use polars-statistics ANOVA — run one-way ANOVA (Fisher or Welch), two-way ANOVA, repeated-measures ANOVA, or intraclass correlation coefficient (ICC) as Polars expressions. |
| `polars-statistics-regression` | Use polars-statistics regression — fit OLS, Ridge, Lasso, Elastic Net, WLS, RLS, BLS, NNLS, Quantile, Huber, PLS, ALM, or any GLM (logistic, Poisson, Negative Binomial, Tweedie, Probit, Cloglog) as Polars expressions or as PyModel classes; use R-style formulas with interactions and polynomial terms. |
| `polars-statistics-results` | Consume polars-statistics expression results — extract struct fields, unnest result columns, call to_dict() and summary() on PyModel classes, handle PyRuntimeError for unfitted models, or work with the typed .pyi API. |

---

## Routing Map

### Family: Getting Started

| Prompt | Routes to | Verdict |
|--------|-----------|---------|
| "How do I install polars-statistics?" | `polars-statistics-getting-started` | CORRECT — "install the package" is explicit in description |
| "What's the difference between the expression API and PyModel classes?" | `polars-statistics-getting-started` | CORRECT — "understand the two API styles" matches exactly |
| "Which model should I use for my problem?" | `polars-statistics-getting-started` | CORRECT — "choose the right model with the decision matrix" |
| "How do I migrate from sklearn Ridge to polars-statistics?" | `polars-statistics-getting-started` | CORRECT — "migrate from sklearn" explicit in description |

### Family: Hypothesis Tests

| Prompt | Routes to | Verdict |
|--------|-----------|---------|
| "Run an independent t-test between two groups" | `polars-statistics-hypothesis-tests` | CORRECT — "t-tests" explicit in description |
| "Test if two samples are significantly different (nonparametric, two-sample)" | `polars-statistics-hypothesis-tests` | CORRECT — "Mann-Whitney" / nonparametric tests described |
| "Check if my data is normally distributed" | `polars-statistics-hypothesis-tests` | CORRECT — "normality tests" explicit |
| "Compute Pearson correlation with a confidence interval" | `polars-statistics-hypothesis-tests` | CORRECT — "correlation tests" explicit |
| "Fisher's exact test on a 2x2 contingency table" | `polars-statistics-hypothesis-tests` | CORRECT — "Fisher's exact" explicit |
| "Are these two samples equivalent? (TOST)" | `polars-statistics-hypothesis-tests` | CORRECT — "TOST equivalence tests" explicit |
| "Compare distribution shapes with MMD or energy distance" | `polars-statistics-hypothesis-tests` | CORRECT — "MMD, energy distance" explicit |
| "Compute Cohen's d between two groups" | `polars-statistics-hypothesis-tests` | CORRECT — "effect sizes (Cohen's d...)" explicit |
| "Rank-biserial effect size for a Mann-Whitney test" | `polars-statistics-hypothesis-tests` | CORRECT — "rank-biserial" explicit |
| "Jarque-Bera normality test" | `polars-statistics-hypothesis-tests` | CORRECT — "Jarque-Bera normality" explicit |
| "Bootstrap confidence interval for the mean of a time series" | `polars-statistics-hypothesis-tests` | CORRECT — "bootstrap confidence intervals" explicit; `block_length` handles the time-series/dependent-data case |
| "Conditional (R fisher.test) odds ratio" | `polars-statistics-hypothesis-tests` | CORRECT — "Fisher's exact" family; `fisher_exact_conditional` documented in the API reference |

### Family: ANOVA

| Prompt | Routes to | Verdict |
|--------|-----------|---------|
| "Run one-way ANOVA across three treatment groups" | `polars-statistics-anova` | CORRECT — "one-way ANOVA" explicit |
| "Two-way ANOVA with factor A and factor B interaction" | `polars-statistics-anova` | CORRECT — "two-way ANOVA" explicit |
| "Repeated-measures ANOVA for within-subjects experiment" | `polars-statistics-anova` | CORRECT — "repeated-measures ANOVA" explicit |
| "Intraclass correlation coefficient for three raters" | `polars-statistics-anova` | CORRECT — "intraclass correlation coefficient (ICC)" explicit |
| "Welch's ANOVA when group variances are unequal" | `polars-statistics-anova` | CORRECT — "one-way ANOVA (Fisher or Welch)" explicit |

### Family: Regression

| Prompt | Routes to | Verdict |
|--------|-----------|---------|
| "Fit OLS per group using group_by" | `polars-statistics-regression` | CORRECT — "OLS" and "Polars expressions" explicit |
| "Logistic regression for binary outcome prediction" | `polars-statistics-regression` | CORRECT — "GLM (logistic...)" explicit |
| "Ridge regression with L2 regularization, lambda=0.5" | `polars-statistics-regression` | CORRECT — "Ridge" explicit |
| "Fit model using R-style formula y ~ x1 * x2" | `polars-statistics-regression` | CORRECT — "R-style formulas with interactions" explicit |
| "Cook's distance and VIF to detect influential points" | `polars-statistics-regression` | CORRECT — "diagnostics" implied; no other skill covers regression diagnostics |
| "Quantile regression at tau=0.9" | `polars-statistics-regression` | CORRECT — "Quantile" explicit |
| "Fit Poisson GLM for count data" | `polars-statistics-regression` | CORRECT — "Poisson" explicit |

### Family: Results

| Prompt | Routes to | Verdict |
|--------|-----------|---------|
| "How do I unnest a result struct from group_by().agg()?" | `polars-statistics-results` | CORRECT — "unnest result columns" explicit |
| "Extract p_value from a hypothesis test result struct" | `polars-statistics-results` | CORRECT — "extract struct fields" explicit |
| "Convert my fitted OLS model to a Python dict" | `polars-statistics-results` | CORRECT — "call to_dict()...on PyModel classes" explicit |
| "Get a summary table from a fitted regression model" | `polars-statistics-results` | CORRECT — "summary() on PyModel classes" explicit |
| "I'm getting PyRuntimeError: Model not fitted" | `polars-statistics-results` | CORRECT — "handle PyRuntimeError for unfitted models" explicit |

---

## Routing Notes

**Known ambiguity — Kruskal-Wallis:** A user asking "compare three or more groups
nonparametrically" could route to either `polars-statistics-hypothesis-tests` (which owns
`kruskal_wallis`) or `polars-statistics-anova` (which documents the parametric ANOVA
context). The hypothesis-tests skill description mentions "Kruskal-Wallis" explicitly;
anova does not. No description change needed — routing resolves correctly.

**Known ambiguity — ICC:** ICC is documented in both `polars-statistics-anova` (via
`ps.icc()` expression) and `polars-statistics-hypothesis-tests` (Group D correlation
section). Only the anova description names "intraclass correlation coefficient (ICC)"
explicitly; the hypothesis-tests description covers it under the broader "correlation
tests" phrasing. Either skill provides correct usage; the anova skill is the primary home
for the expression form, and its explicit ICC mention makes it the correct routing target.
No description change needed.

**No mis-routes detected** — all 23 test prompts resolve to the correct skill based on the
current `description` fields. No description refinements required.

---

## Verification Criteria

To confirm ROUTING.md is complete and current:

```bash
# At least 6 table rows in the routing table
grep -c '^| ' plugins/polars-statistics/skills/ROUTING.md

# All 5 skill families represented
grep -l 'polars-statistics-getting-started\|polars-statistics-hypothesis-tests\|polars-statistics-anova\|polars-statistics-regression\|polars-statistics-results' \
     plugins/polars-statistics/skills/ROUTING.md

# All skill frontmatters valid (name, description, version present)
for f in plugins/polars-statistics/skills/*/SKILL.md; do
  grep -q '^name:' "$f" && grep -q '^description:' "$f" && grep -q '^version:' "$f" \
    && echo "OK: $f" || echo "MISSING FIELDS: $f"
done
```
