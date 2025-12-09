"""Expression API for statistical tests and regression."""

from polars_statistics.exprs.parametric import (
    ttest_ind,
    ttest_paired,
    brown_forsythe,
    yuen_test,
)
from polars_statistics.exprs.nonparametric import (
    mann_whitney_u,
    wilcoxon_signed_rank,
    kruskal_wallis,
    brunner_munzel,
)
from polars_statistics.exprs.distributional import (
    shapiro_wilk,
    dagostino,
)
from polars_statistics.exprs.forecast import (
    diebold_mariano,
    permutation_t_test,
    clark_west,
    spa_test,
    model_confidence_set,
    mspe_adjusted,
)
from polars_statistics.exprs.modern import (
    energy_distance,
    mmd_test,
)
from polars_statistics.exprs.regression import (
    ols,
    ridge,
    elastic_net,
    wls,
    rls,
    bls,
    nnls,
    logistic,
    poisson,
    negative_binomial,
    tweedie,
    probit,
    cloglog,
    alm,
    # Summary functions
    ols_summary,
    ridge_summary,
    elastic_net_summary,
    wls_summary,
    rls_summary,
    bls_summary,
    logistic_summary,
    poisson_summary,
    negative_binomial_summary,
    tweedie_summary,
    probit_summary,
    cloglog_summary,
    alm_summary,
)

__all__ = [
    # Parametric tests
    "ttest_ind",
    "ttest_paired",
    "brown_forsythe",
    "yuen_test",
    # Non-parametric tests
    "mann_whitney_u",
    "wilcoxon_signed_rank",
    "kruskal_wallis",
    "brunner_munzel",
    # Distributional tests
    "shapiro_wilk",
    "dagostino",
    # Forecast tests
    "diebold_mariano",
    "permutation_t_test",
    "clark_west",
    "spa_test",
    "model_confidence_set",
    "mspe_adjusted",
    # Modern distribution tests
    "energy_distance",
    "mmd_test",
    # Regression expressions
    "ols",
    "ridge",
    "elastic_net",
    "wls",
    "rls",
    "bls",
    "nnls",
    "logistic",
    "poisson",
    "negative_binomial",
    "tweedie",
    "probit",
    "cloglog",
    "alm",
    # Summary expressions
    "ols_summary",
    "ridge_summary",
    "elastic_net_summary",
    "wls_summary",
    "rls_summary",
    "bls_summary",
    "logistic_summary",
    "poisson_summary",
    "negative_binomial_summary",
    "tweedie_summary",
    "probit_summary",
    "cloglog_summary",
    "alm_summary",
]
