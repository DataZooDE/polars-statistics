"""Expression API for statistical tests."""

from polars_statistics.exprs.parametric import (
    ttest_ind,
    ttest_paired,
    brown_forsythe,
)
from polars_statistics.exprs.nonparametric import (
    mann_whitney_u,
    wilcoxon_signed_rank,
    kruskal_wallis,
)
from polars_statistics.exprs.distributional import (
    shapiro_wilk,
    dagostino,
)
from polars_statistics.exprs.forecast import (
    diebold_mariano,
    permutation_t_test,
)

__all__ = [
    # Parametric tests
    "ttest_ind",
    "ttest_paired",
    "brown_forsythe",
    # Non-parametric tests
    "mann_whitney_u",
    "wilcoxon_signed_rank",
    "kruskal_wallis",
    # Distributional tests
    "shapiro_wilk",
    "dagostino",
    # Forecast tests
    "diebold_mariano",
    "permutation_t_test",
]
