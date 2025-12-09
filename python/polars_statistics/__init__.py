"""polars-statistics: Statistical testing and regression for Polars DataFrames."""

from pathlib import Path

# Import Rust bindings
from polars_statistics._polars_statistics import (
    # Linear models
    OLS,
    Ridge,
    ElasticNet,
    WLS,
    # GLM models
    Logistic,
    Poisson,
    # Bootstrap
    StationaryBootstrap,
    CircularBlockBootstrap,
)

# Import expression API
from polars_statistics import exprs
from polars_statistics.exprs import (
    # Parametric tests
    ttest_ind,
    ttest_paired,
    brown_forsythe,
    # Non-parametric tests
    mann_whitney_u,
    wilcoxon_signed_rank,
    kruskal_wallis,
    # Distributional tests
    shapiro_wilk,
    dagostino,
    # Forecast tests
    diebold_mariano,
    permutation_t_test,
)

__version__ = "0.1.0"

# Library path for plugin registration
LIB = Path(__file__).parent

__all__ = [
    # Linear Models
    "OLS",
    "Ridge",
    "ElasticNet",
    "WLS",
    # GLM Models
    "Logistic",
    "Poisson",
    # Bootstrap
    "StationaryBootstrap",
    "CircularBlockBootstrap",
    # Expression API module
    "exprs",
    # Parametric test expressions
    "ttest_ind",
    "ttest_paired",
    "brown_forsythe",
    # Non-parametric test expressions
    "mann_whitney_u",
    "wilcoxon_signed_rank",
    "kruskal_wallis",
    # Distributional test expressions
    "shapiro_wilk",
    "dagostino",
    # Forecast test expressions
    "diebold_mariano",
    "permutation_t_test",
    # Library path
    "LIB",
]
