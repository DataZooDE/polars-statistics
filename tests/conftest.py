"""Pytest configuration and fixtures."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Reference-library skip guards (TEST-03)
#
# Use these fixtures in any test that compares polars-statistics output against
# scipy or statsmodels.  When the reference library is absent the test is
# skipped automatically, keeping the runtime wheel dependency-light.
# ---------------------------------------------------------------------------


@pytest.fixture
def require_scipy():
    """Return the scipy module, or skip the test if scipy is unavailable."""
    return pytest.importorskip("scipy")


@pytest.fixture
def require_statsmodels():
    """Return the statsmodels module, or skip the test if statsmodels is unavailable."""
    return pytest.importorskip("statsmodels")


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_regression_data():
    """Generate sample regression data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 3

    X = np.random.randn(n_samples, n_features)
    true_coef = np.array([1.0, 2.0, 3.0])
    y = X @ true_coef + np.random.randn(n_samples) * 0.1

    return X, y, true_coef


@pytest.fixture
def sample_classification_data():
    """Generate sample classification data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 2

    X = np.random.randn(n_samples, n_features)
    y = ((X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.5) > 0).astype(float)

    return X, y


@pytest.fixture
def sample_count_data():
    """Generate sample count data for Poisson regression."""
    np.random.seed(42)
    n_samples = 200
    n_features = 2

    X = np.random.randn(n_samples, n_features) * 0.5
    y = np.random.poisson(np.exp(X[:, 0] * 0.3 + 0.5)).astype(float)

    return X, y
