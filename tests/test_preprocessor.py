"""
Tests for data preprocessing utilities.
"""

import pytest
import numpy as np
from quantumml_fraud.data import DataPreprocessor


def test_preprocessor_initialization():
    """Test DataPreprocessor initialization."""
    preprocessor = DataPreprocessor(scaler_type="standard")
    assert preprocessor.scaler is not None
    assert not preprocessor.is_fitted
    
    preprocessor_robust = DataPreprocessor(scaler_type="robust")
    assert preprocessor_robust.scaler is not None


def test_preprocessor_invalid_scaler():
    """Test that invalid scaler type raises error."""
    with pytest.raises(ValueError):
        DataPreprocessor(scaler_type="invalid")


def test_fit_transform():
    """Test fit_transform method."""
    preprocessor = DataPreprocessor()
    X = np.random.randn(100, 5)
    
    X_scaled = preprocessor.fit_transform(X)
    
    assert X_scaled.shape == X.shape
    assert preprocessor.is_fitted
    
    # Check that mean is close to 0 and std close to 1
    assert np.abs(X_scaled.mean()) < 0.1
    assert np.abs(X_scaled.std() - 1.0) < 0.1


def test_transform_without_fit():
    """Test that transform fails without fitting."""
    preprocessor = DataPreprocessor()
    X = np.random.randn(100, 5)
    
    with pytest.raises(ValueError):
        preprocessor.transform(X)


def test_transform_after_fit():
    """Test transform after fitting."""
    preprocessor = DataPreprocessor()
    X_train = np.random.randn(100, 5)
    X_test = np.random.randn(50, 5)
    
    preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    assert X_test_scaled.shape == X_test.shape
