"""Training script for XGBoost classifier."""

import numpy as np
import xgboost as xgb
from typing import Tuple
import pickle
import config
from data_loader import load_creditcard_data, get_train_test_split
from preprocessing import prepare_features_labels, preprocess_for_classical
from metrics_utils import (
    calculate_metrics, 
    print_metrics, 
    print_confusion_matrix,
    print_classification_report
)


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  params: dict = None) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        params: XGBoost parameters (uses config defaults if None)
        
    Returns:
        Trained XGBoost model
    """
    if params is None:
        params = config.XGBOOST_PARAMS
    
    print("Training XGBoost classifier...")
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    print("Training complete!")
    
    return model


def evaluate_xgboost(model: xgb.XGBClassifier, 
                     X_test: np.ndarray, 
                     y_test: np.ndarray) -> dict:
    """
    Evaluate XGBoost model on test data.
    
    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of metrics
    """
    print("\nEvaluating XGBoost model...")
    y_pred = model.predict(X_test)
    
    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics, "XGBoost")
    print_confusion_matrix(y_test, y_pred, "XGBoost")
    print_classification_report(y_test, y_pred, "XGBoost")
    
    return metrics


def save_model(model: xgb.XGBClassifier, filepath: str = "xgboost_model.pkl"):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        filepath: Path to save the model
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath: str = "xgboost_model.pkl") -> xgb.XGBClassifier:
    """
    Load trained model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


def main():
    """Main training pipeline for XGBoost."""
    print("=" * 60)
    print("XGBoost Fraud Detection Training")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    df = load_creditcard_data()
    
    # Split data
    print("\n2. Splitting data...")
    train_df, test_df = get_train_test_split(df)
    
    # Prepare features and labels
    print("\n3. Preparing features and labels...")
    X_train, y_train = prepare_features_labels(train_df)
    X_test, y_test = prepare_features_labels(test_df)
    
    # Preprocess
    print("\n4. Preprocessing data...")
    X_train_processed, X_test_processed = preprocess_for_classical(X_train, X_test)
    
    # Train model
    print("\n5. Training model...")
    model = train_xgboost(X_train_processed, y_train)
    
    # Evaluate model
    print("\n6. Evaluating model...")
    metrics = evaluate_xgboost(model, X_test_processed, y_test)
    
    # Save model
    print("\n7. Saving model...")
    save_model(model, "models/xgboost_model.pkl")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return model, metrics


if __name__ == "__main__":
    main()
