"""
Metrics for evaluating fraud detection models.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    average_precision_score
)


class FraudDetectionMetrics:
    """
    Comprehensive metrics for evaluating fraud detection models.
    
    Provides various metrics that are particularly relevant for
    imbalanced classification problems like fraud detection.
    """
    
    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate all available metrics for the predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for ROC-AUC)
            
        Returns:
            Dictionary containing all calculated metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Add metrics that require probability predictions
        if y_pred_proba is not None:
            try:
                # Use second column for binary classification
                if y_pred_proba.ndim == 2:
                    proba_positive = y_pred_proba[:, 1]
                else:
                    proba_positive = y_pred_proba
                
                metrics['roc_auc'] = roc_auc_score(y_true, proba_positive)
                metrics['avg_precision'] = average_precision_score(y_true, proba_positive)
            except Exception as e:
                print(f"Could not calculate probability-based metrics: {e}")
        
        return metrics
    
    @staticmethod
    def get_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Calculate confusion matrix and extract key values.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Tuple of (confusion_matrix, breakdown_dict)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        breakdown = {
            'true_negatives': int(cm[0, 0]) if cm.shape[0] > 1 else 0,
            'false_positives': int(cm[0, 1]) if cm.shape[0] > 1 else 0,
            'false_negatives': int(cm[1, 0]) if cm.shape[0] > 1 else 0,
            'true_positives': int(cm[1, 1]) if cm.shape[0] > 1 else 0,
        }
        
        return cm, breakdown
    
    @staticmethod
    def print_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[list] = None
    ) -> str:
        """
        Generate a detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names for the classes
            
        Returns:
            Classification report as string
        """
        if target_names is None:
            target_names = ['Legitimate', 'Fraud']
        
        return classification_report(y_true, y_pred, target_names=target_names)
    
    @staticmethod
    def calculate_cost_sensitive_metric(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        false_positive_cost: float = 1.0,
        false_negative_cost: float = 10.0
    ) -> float:
        """
        Calculate a cost-sensitive metric for fraud detection.
        
        In fraud detection, false negatives (missing fraud) are typically
        more costly than false positives (flagging legitimate transactions).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            false_positive_cost: Cost of a false positive
            false_negative_cost: Cost of a false negative
            
        Returns:
            Total cost based on the confusion matrix
        """
        cm, breakdown = FraudDetectionMetrics.get_confusion_matrix(y_true, y_pred)
        
        total_cost = (
            breakdown['false_positives'] * false_positive_cost +
            breakdown['false_negatives'] * false_negative_cost
        )
        
        return total_cost
