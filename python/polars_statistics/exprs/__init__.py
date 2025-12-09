"""Expression API for statistical tests."""

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
]
