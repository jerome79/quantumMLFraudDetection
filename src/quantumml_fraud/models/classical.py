"""
Classical machine learning models for fraud detection.
"""

import numpy as np
from typing import Optional, Dict, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class ClassicalMLModel:
    """
    Classical machine learning model wrapper for fraud detection.
    
    Supports various classical ML algorithms including Random Forest,
    Gradient Boosting, Logistic Regression, and SVM.
    """
    
    def __init__(self, model_type: str = "random_forest", **kwargs):
        """
        Initialize the classical ML model.
        
        Args:
            model_type: Type of model ("random_forest", "gradient_boosting",
                       "logistic_regression", or "svm")
            **kwargs: Additional arguments passed to the model constructor
        """
        self.model_type = model_type
        self.model = self._create_model(model_type, **kwargs)
        self.is_trained = False
    
    def _create_model(self, model_type: str, **kwargs):
        """Create the specified model type."""
        if model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                random_state=kwargs.get('random_state', 42)
            )
        elif model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 3),
                random_state=kwargs.get('random_state', 42)
            )
        elif model_type == "logistic_regression":
            return LogisticRegression(
                max_iter=kwargs.get('max_iter', 1000),
                random_state=kwargs.get('random_state', 42)
            )
        elif model_type == "svm":
            return SVC(
                kernel=kwargs.get('kernel', 'rbf'),
                random_state=kwargs.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training feature matrix
            y_train: Training labels
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Predicted probabilities for each class
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"{self.model_type} does not support probability predictions")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores if available.
        
        Returns:
            Feature importance scores or None if not available
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None
