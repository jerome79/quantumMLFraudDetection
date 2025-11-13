"""Metrics and evaluation utilities for model performance."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary containing various metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Calculate ROC-AUC if we have both classes
    if len(np.unique(y_true)) > 1:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
    else:
        metrics['roc_auc'] = 0.0
    
    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = "Model"):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model for display
    """
    print(f"\n{model_name} Performance Metrics:")
    print("-" * 40)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("-" * 40)


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                          model_name: str = "Model"):
    """
    Print and display confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model for display
    """
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{model_name} Confusion Matrix:")
    print(cm)
    print(f"\nTrue Negatives:  {cm[0, 0]}")
    print(f"False Positives: {cm[0, 1]}")
    print(f"False Negatives: {cm[1, 0]}")
    print(f"True Positives:  {cm[1, 1]}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         model_name: str = "Model") -> plt.Figure:
    """
    Create a confusion matrix visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model for display
        
    Returns:
        Matplotlib figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{model_name} - Confusion Matrix')
    ax.set_xticklabels(['Non-Fraud', 'Fraud'])
    ax.set_yticklabels(['Non-Fraud', 'Fraud'])
    
    return fig


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                               model_name: str = "Model"):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model for display
    """
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_true, y_pred, 
                                target_names=['Non-Fraud', 'Fraud'],
                                zero_division=0))


def compare_models_metrics(metrics_dict: Dict[str, Dict[str, float]]) -> None:
    """
    Compare metrics across multiple models.
    
    Args:
        metrics_dict: Dictionary mapping model names to their metrics
    """
    print("\nModel Comparison:")
    print("=" * 80)
    
    # Header
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC-AUC':<12}")
    print("-" * 80)
    
    # Print each model's metrics
    for model_name, metrics in metrics_dict.items():
        print(f"{model_name:<20} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} "
              f"{metrics['f1']:<12.4f} "
              f"{metrics['roc_auc']:<12.4f}")
    
    print("=" * 80)
