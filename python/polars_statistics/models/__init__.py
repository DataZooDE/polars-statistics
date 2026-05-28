"""High-level model wrappers with Polars DataFrame support."""

from polars_statistics._polars_statistics import (
    # Linear models
    OLS,
    Ridge,
    ElasticNet,
    WLS,
    RLS,
    BLS,
    # Robust regression
    Huber,
    # GLM models
    Logistic,
    LogisticRegression,
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
    # Robust regression
    "Huber",
    # GLM models
    "Logistic",
    "LogisticRegression",
    "Poisson",
    "NegativeBinomial",
    "Tweedie",
    "Probit",
    "Cloglog",
    # Bootstrap
    "StationaryBootstrap",
    "CircularBlockBootstrap",
]
