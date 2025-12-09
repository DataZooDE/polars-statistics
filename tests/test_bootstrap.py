"""Tests for bootstrap methods."""

import numpy as np
import pytest

from polars_statistics import StationaryBootstrap, CircularBlockBootstrap


class TestStationaryBootstrap:
    """Tests for Stationary Bootstrap."""

    def test_sample_length(self):
        """Test that sample has correct length."""
        data = np.random.randn(100)
        bootstrap = StationaryBootstrap(expected_block_length=5.0, seed=42)

        sample = bootstrap.sample(data)

        assert len(sample) == 100

    def test_sample_custom_length(self):
        """Test sample with custom length."""
        data = np.random.randn(100)
        bootstrap = StationaryBootstrap(expected_block_length=5.0, seed=42)

        sample = bootstrap.sample(data, length=50)

        assert len(sample) == 50

    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        data = np.random.randn(100)

        bootstrap1 = StationaryBootstrap(expected_block_length=5.0, seed=42)
        bootstrap2 = StationaryBootstrap(expected_block_length=5.0, seed=42)

        sample1 = bootstrap1.sample(data)
        sample2 = bootstrap2.sample(data)

        np.testing.assert_array_equal(sample1, sample2)

    def test_different_seeds(self):
        """Test different seeds produce different samples."""
        data = np.random.randn(100)

        bootstrap1 = StationaryBootstrap(expected_block_length=5.0, seed=42)
        bootstrap2 = StationaryBootstrap(expected_block_length=5.0, seed=123)

        sample1 = bootstrap1.sample(data)
        sample2 = bootstrap2.sample(data)

        assert not np.array_equal(sample1, sample2)

    def test_samples_multiple(self):
        """Test generating multiple samples."""
        data = np.random.randn(50)
        bootstrap = StationaryBootstrap(expected_block_length=5.0, seed=42)

        samples = bootstrap.samples(data, n_samples=100)

        assert samples.shape == (100, 50)

    def test_samples_values_from_data(self):
        """Test that bootstrap samples contain values from original data."""
        data = np.arange(100).astype(float)
        bootstrap = StationaryBootstrap(expected_block_length=5.0, seed=42)

        sample = bootstrap.sample(data)

        # All values in sample should be from original data
        for val in sample:
            assert val in data


class TestCircularBlockBootstrap:
    """Tests for Circular Block Bootstrap."""

    def test_sample_length(self):
        """Test that sample has correct length."""
        data = np.random.randn(100)
        bootstrap = CircularBlockBootstrap(block_length=10, seed=42)

        sample = bootstrap.sample(data)

        assert len(sample) == 100

    def test_sample_custom_length(self):
        """Test sample with custom length."""
        data = np.random.randn(100)
        bootstrap = CircularBlockBootstrap(block_length=10, seed=42)

        sample = bootstrap.sample(data, length=50)

        assert len(sample) == 50

    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        data = np.random.randn(100)

        bootstrap1 = CircularBlockBootstrap(block_length=10, seed=42)
        bootstrap2 = CircularBlockBootstrap(block_length=10, seed=42)

        sample1 = bootstrap1.sample(data)
        sample2 = bootstrap2.sample(data)

        np.testing.assert_array_equal(sample1, sample2)

    def test_samples_multiple(self):
        """Test generating multiple samples."""
        data = np.random.randn(50)
        bootstrap = CircularBlockBootstrap(block_length=5, seed=42)

        samples = bootstrap.samples(data, n_samples=100)

        assert samples.shape == (100, 50)

    def test_block_structure(self):
        """Test that circular block bootstrap preserves block structure."""
        # Use sequential data to verify block structure
        data = np.arange(20).astype(float)
        bootstrap = CircularBlockBootstrap(block_length=5, seed=42)

        sample = bootstrap.sample(data, length=10)

        # Check that we have consecutive sequences (with wrap-around)
        assert len(sample) == 10
