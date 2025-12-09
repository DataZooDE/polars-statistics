"""Pytest configuration and fixtures."""

import numpy as np
import pytest


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
