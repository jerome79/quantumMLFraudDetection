"""Training script for Variational Quantum Classifier (VQC)."""

import numpy as np
from typing import Tuple
import pickle
from qiskit import Aer
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.algorithms import VQC
from qiskit.algorithms.optimizers import COBYLA
import config
from data_loader import load_creditcard_data, get_train_test_split, sample_balanced_dataset
from preprocessing import prepare_features_labels, preprocess_for_quantum
from metrics_utils import (
    calculate_metrics, 
    print_metrics, 
    print_confusion_matrix,
    print_classification_report
)


def create_vqc(feature_dim: int = None) -> VQC:
    """
    Create a Variational Quantum Classifier.
    
    Args:
        feature_dim: Number of features (qubits)
        
    Returns:
        VQC object
    """
    if feature_dim is None:
        feature_dim = config.FEATURE_DIM
    
    # Create feature map (encoding circuit)
    feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2)
    
    # Create ansatz (variational circuit)
    ansatz = RealAmplitudes(num_qubits=feature_dim, reps=3)
    
    # Create quantum instance
    backend = Aer.get_backend(config.QUANTUM_BACKEND)
    quantum_instance = QuantumInstance(
        backend, 
        shots=config.QUANTUM_SHOTS,
        seed_simulator=config.RANDOM_STATE,
        seed_transpiler=config.RANDOM_STATE
    )
    
    # Create optimizer
    optimizer = COBYLA(maxiter=100)
    
    # Create VQC
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        quantum_instance=quantum_instance
    )
    
    return vqc


def train_vqc(X_train: np.ndarray, y_train: np.ndarray,
              feature_dim: int = None) -> VQC:
    """
    Train a Variational Quantum Classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        feature_dim: Number of features (qubits)
        
    Returns:
        Trained VQC model
    """
    if feature_dim is None:
        feature_dim = config.FEATURE_DIM
    
    print(f"Training Variational Quantum Classifier with {feature_dim} qubits...")
    print("This may take several minutes...")
    
    # Create VQC
    vqc = create_vqc(feature_dim)
    
    # Train
    vqc.fit(X_train, y_train)
    
    print("Training complete!")
    
    return vqc


def evaluate_vqc(model: VQC, 
                 X_test: np.ndarray, 
                 y_test: np.ndarray) -> dict:
    """
    Evaluate VQC model on test data.
    
    Args:
        model: Trained VQC model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of metrics
    """
    print("\nEvaluating VQC model...")
    y_pred = model.predict(X_test)
    
    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics, "VQC")
    print_confusion_matrix(y_test, y_pred, "VQC")
    print_classification_report(y_test, y_pred, "VQC")
    
    return metrics


def save_model(model: VQC, filepath: str = "vqc_model.pkl"):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        filepath: Path to save the model
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath: str = "vqc_model.pkl") -> VQC:
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
    """Main training pipeline for VQC."""
    print("=" * 60)
    print("Variational Quantum Classifier Fraud Detection Training")
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
    model = train_vqc(X_train_processed, y_train)
    
    # Evaluate model
    print("\n7. Evaluating model...")
    metrics = evaluate_vqc(model, X_test_processed, y_test)
    
    # Save model
    print("\n8. Saving model...")
    save_model(model, "models/vqc_model.pkl")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return model, metrics


if __name__ == "__main__":
    main()
