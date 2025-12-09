"""High-level model wrappers with Polars DataFrame support."""

from polars_statistics._polars_statistics import (
    OLS,
    Ridge,
    ElasticNet,
    WLS,
    Logistic,
    Poisson,
    StationaryBootstrap,
    CircularBlockBootstrap,
)

__all__ = [
    # Linear models
    "OLS",
    "Ridge",
    "ElasticNet",
    "WLS",
    # GLM models
    "Logistic",
    "Poisson",
    # Bootstrap
    "StationaryBootstrap",
    "CircularBlockBootstrap",
]
