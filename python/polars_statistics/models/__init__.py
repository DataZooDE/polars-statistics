"""High-level model wrappers with Polars DataFrame support."""

from polars_statistics._polars_statistics import (
    # Survival
    AFT,
    BLS,
    # Linear models
    OLS,
    PLS,
    RLS,
    WLS,
    CircularBlockBootstrap,
    Cloglog,
    ElasticNet,
    # Robust regression
    Huber,
    # GLM models
    Logistic,
    LogisticRegression,
    NegativeBinomial,
    PenalizedGLM,
    Poisson,
    Probit,
    Ridge,
    # Bootstrap
    StationaryBootstrap,
    Tweedie,
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
    "PLS",
    # GLM models
    "Logistic",
    "LogisticRegression",
    "Poisson",
    "NegativeBinomial",
    "Tweedie",
    "Probit",
    "Cloglog",
    "PenalizedGLM",
    # Survival
    "AFT",
    # Bootstrap
    "StationaryBootstrap",
    "CircularBlockBootstrap",
]
