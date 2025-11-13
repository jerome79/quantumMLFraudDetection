"""
Data preprocessing utilities for fraud detection.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Optional, Tuple


class DataPreprocessor:
    """
    Preprocesses credit card transaction data for machine learning models.
    
    Provides methods for scaling, normalization, and handling
    imbalanced datasets common in fraud detection.
    """
    
    def __init__(self, scaler_type: str = "standard"):
        """
        Initialize the DataPreprocessor.
        
        Args:
            scaler_type: Type of scaler to use ("standard" or "robust")
        """
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.is_fitted = False
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the scaler and transform the data.
        
        Args:
            X: Feature matrix to fit and transform
            
        Returns:
            Scaled feature matrix
        """
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        return X_scaled
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data using the fitted scaler.
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            Scaled feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_transform() first.")
        
        return self.scaler.transform(X)
    
    def handle_imbalance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = "smote"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance in the dataset.
        
        Args:
            X: Feature matrix
            y: Label vector
            method: Resampling method ("smote", "undersample", or "oversample")
            
        Returns:
            Tuple of (resampled_X, resampled_y)
        """
        if method == "smote":
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(random_state=42)
        elif method == "undersample":
            from imblearn.under_sampling import RandomUnderSampler
            sampler = RandomUnderSampler(random_state=42)
        elif method == "oversample":
            from imblearn.over_sampling import RandomOverSampler
            sampler = RandomOverSampler(random_state=42)
        else:
            raise ValueError(f"Unknown resampling method: {method}")
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled
