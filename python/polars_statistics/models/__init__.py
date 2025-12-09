"""High-level model wrappers with Polars DataFrame support."""

from polars_statistics._polars_statistics import (
    # Linear models
    OLS,
    Ridge,
    ElasticNet,
    WLS,
    RLS,
    BLS,
    # GLM models
    Logistic,
    Poisson,
    NegativeBinomial,
    Tweedie,
    Probit,
    Cloglog,
    # Bootstrap
    StationaryBootstrap,
    CircularBlockBootstrap,
)

__all__ = [
    # Linear models
    "OLS",
    "Ridge",
    "ElasticNet",
    "WLS",
    "RLS",
    "BLS",
    # GLM models
    "Logistic",
    "Poisson",
    "NegativeBinomial",
    "Tweedie",
    "Probit",
    "Cloglog",
    # Bootstrap
    "StationaryBootstrap",
    "CircularBlockBootstrap",
]
