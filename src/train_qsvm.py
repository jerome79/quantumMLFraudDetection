"""Training script for Quantum Support Vector Machine (QSVM)."""

import numpy as np
from typing import Tuple
import pickle
from qiskit import Aer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import QuantumKernel
import config
from data_loader import load_creditcard_data, get_train_test_split, sample_balanced_dataset
from preprocessing import prepare_features_labels, preprocess_for_quantum
from metrics_utils import (
    calculate_metrics, 
    print_metrics, 
    print_confusion_matrix,
    print_classification_report
)


def create_quantum_kernel(feature_dim: int = None) -> QuantumKernel:
    """
    Create a quantum kernel for QSVM.
    
    Args:
        feature_dim: Number of features (qubits)
        
    Returns:
        QuantumKernel object
    """
    if feature_dim is None:
        feature_dim = config.FEATURE_DIM
    
    # Create feature map
    feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2)
    
    # Create quantum instance
    backend = Aer.get_backend(config.QUANTUM_BACKEND)
    quantum_instance = QuantumInstance(
        backend, 
        shots=config.QUANTUM_SHOTS,
        seed_simulator=config.RANDOM_STATE,
        seed_transpiler=config.RANDOM_STATE
    )
    
    # Create quantum kernel
    kernel = QuantumKernel(
        feature_map=feature_map,
        quantum_instance=quantum_instance
    )
    
    return kernel


def train_qsvm(X_train: np.ndarray, y_train: np.ndarray,
               feature_dim: int = None) -> QSVC:
    """
    Train a Quantum Support Vector Classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        feature_dim: Number of features (qubits)
        
    Returns:
        Trained QSVC model
    """
    if feature_dim is None:
        feature_dim = config.FEATURE_DIM
    
    print(f"Training Quantum SVM with {feature_dim} qubits...")
    print("This may take several minutes...")
    
    # Create quantum kernel
    kernel = create_quantum_kernel(feature_dim)
    
    # Create and train QSVC
    qsvc = QSVC(quantum_kernel=kernel)
    qsvc.fit(X_train, y_train)
    
    print("Training complete!")
    
    return qsvc


def evaluate_qsvm(model: QSVC, 
                  X_test: np.ndarray, 
                  y_test: np.ndarray) -> dict:
    """
    Evaluate QSVM model on test data.
    
    Args:
        model: Trained QSVM model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of metrics
    """
    print("\nEvaluating QSVM model...")
    y_pred = model.predict(X_test)
    
    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics, "QSVM")
    print_confusion_matrix(y_test, y_pred, "QSVM")
    print_classification_report(y_test, y_pred, "QSVM")
    
    return metrics


def save_model(model: QSVC, filepath: str = "qsvm_model.pkl"):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        filepath: Path to save the model
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath: str = "qsvm_model.pkl") -> QSVC:
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
    """Main training pipeline for QSVM."""
    print("=" * 60)
    print("Quantum SVM Fraud Detection Training")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    df = load_creditcard_data()
    
    # Split data
    print("\n2. Splitting data...")
    train_df, test_df = get_train_test_split(df)
    
    # Sample balanced subset for quantum training
    print("\n3. Sampling balanced dataset for quantum model...")
    train_df_balanced = sample_balanced_dataset(train_df)
    test_df_balanced = sample_balanced_dataset(test_df, sample_size=1000)
    
    # Prepare features and labels
    print("\n4. Preparing features and labels...")
    X_train, y_train = prepare_features_labels(train_df_balanced)
    X_test, y_test = prepare_features_labels(test_df_balanced)
    
    # Preprocess for quantum (with PCA)
    print("\n5. Preprocessing data for quantum model...")
    X_train_processed, X_test_processed = preprocess_for_quantum(X_train, X_test)
    
    # Train model
    print("\n6. Training quantum model...")
    model = train_qsvm(X_train_processed, y_train)
    
    # Evaluate model
    print("\n7. Evaluating model...")
    metrics = evaluate_qsvm(model, X_test_processed, y_test)
    
    # Save model
    print("\n8. Saving model...")
    save_model(model, "models/qsvm_model.pkl")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return model, metrics


if __name__ == "__main__":
    main()
