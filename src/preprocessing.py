"""Data preprocessing utilities for fraud detection."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, Optional
import config


class DataPreprocessor:
    """Handles data preprocessing for fraud detection models."""
    
    def __init__(self, use_pca: bool = False, n_components: Optional[int] = None):
        """
        Initialize the preprocessor.
        
        Args:
            use_pca: Whether to apply PCA for dimensionality reduction
            n_components: Number of PCA components (for quantum models)
        """
        self.use_pca = use_pca
        self.n_components = n_components if n_components else config.FEATURE_DIM
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components) if use_pca else None
        self.fitted = False
        
    def fit(self, X: np.ndarray) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            X: Feature matrix
            
        Returns:
            self
        """
        # Scale the features
        self.scaler.fit(X)
        
        # Fit PCA if enabled
        if self.use_pca:
            X_scaled = self.scaler.transform(X)
            self.pca.fit(X_scaled)
            print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        self.fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed feature matrix
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Scale
        X_transformed = self.scaler.transform(X)
        
        # Apply PCA if enabled
        if self.use_pca:
            X_transformed = self.pca.transform(X_transformed)
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform data in one step.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed feature matrix
        """
        return self.fit(X).transform(X)


def prepare_features_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate features and labels from DataFrame.
    
    Args:
        df: Input DataFrame with 'Class' column as target
        
    Returns:
        Tuple of (X, y) where X is feature matrix and y is label vector
    """
    # Separate features and labels
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    
    return X, y


def preprocess_for_classical(X_train: np.ndarray, 
                             X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess data for classical models (XGBoost).
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        Tuple of (X_train_processed, X_test_processed)
    """
    preprocessor = DataPreprocessor(use_pca=False)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed


def preprocess_for_quantum(X_train: np.ndarray, 
                           X_test: np.ndarray,
                           n_components: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess data for quantum models (QSVM, VQC).
    
    Applies StandardScaler and PCA to reduce dimensionality for quantum circuits.
    
    Args:
        X_train: Training features
        X_test: Test features
        n_components: Number of PCA components (default from config)
        
    Returns:
        Tuple of (X_train_processed, X_test_processed)
    """
    if n_components is None:
        n_components = config.FEATURE_DIM
    
    preprocessor = DataPreprocessor(use_pca=True, n_components=n_components)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"Reduced features from {X_train.shape[1]} to {X_train_processed.shape[1]} dimensions")
    
    return X_train_processed, X_test_processed
