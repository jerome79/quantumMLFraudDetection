"""Script to compare performance of different models."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from data_loader import load_creditcard_data, get_train_test_split, sample_balanced_dataset
from preprocessing import prepare_features_labels, preprocess_for_classical, preprocess_for_quantum
from train_xgboost import train_xgboost, evaluate_xgboost
from train_qsvm import train_qsvm, evaluate_qsvm
from train_vqc import train_vqc, evaluate_vqc
from metrics_utils import compare_models_metrics
import config


def train_all_models(train_data_full, test_data_full, train_data_quantum, test_data_quantum):
    """
    Train all three models and return them with their metrics.
    
    Args:
        train_data_full: Full training data for classical model
        test_data_full: Full test data for classical model
        train_data_quantum: Sampled training data for quantum models
        test_data_quantum: Sampled test data for quantum models
        
    Returns:
        Dictionary of models and metrics
    """
    results = {}
    
    # Prepare data for classical model (XGBoost)
    print("\n" + "=" * 60)
    print("Training XGBoost (Classical Model)")
    print("=" * 60)
    X_train_full, y_train_full = prepare_features_labels(train_data_full)
    X_test_full, y_test_full = prepare_features_labels(test_data_full)
    X_train_classical, X_test_classical = preprocess_for_classical(X_train_full, X_test_full)
    
    xgb_model = train_xgboost(X_train_classical, y_train_full)
    xgb_metrics = evaluate_xgboost(xgb_model, X_test_classical, y_test_full)
    results['XGBoost'] = {'model': xgb_model, 'metrics': xgb_metrics}
    
    # Prepare data for quantum models
    X_train_quantum, y_train_quantum = prepare_features_labels(train_data_quantum)
    X_test_quantum, y_test_quantum = prepare_features_labels(test_data_quantum)
    X_train_q, X_test_q = preprocess_for_quantum(X_train_quantum, X_test_quantum)
    
    # Train QSVM
    print("\n" + "=" * 60)
    print("Training Quantum SVM")
    print("=" * 60)
    qsvm_model = train_qsvm(X_train_q, y_train_quantum)
    qsvm_metrics = evaluate_qsvm(qsvm_model, X_test_q, y_test_quantum)
    results['QSVM'] = {'model': qsvm_model, 'metrics': qsvm_metrics}
    
    # Train VQC
    print("\n" + "=" * 60)
    print("Training Variational Quantum Classifier")
    print("=" * 60)
    vqc_model = train_vqc(X_train_q, y_train_quantum)
    vqc_metrics = evaluate_vqc(vqc_model, X_test_q, y_test_quantum)
    results['VQC'] = {'model': vqc_model, 'metrics': vqc_metrics}
    
    return results


def plot_comparison(metrics_dict: Dict[str, Dict[str, float]], 
                   save_path: str = "model_comparison.png"):
    """
    Create visualization comparing model performance.
    
    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        save_path: Path to save the plot
    """
    models = list(metrics_dict.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    fig, axes = plt.subplots(1, len(metrics_names), figsize=(20, 4))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    for idx, metric_name in enumerate(metrics_names):
        ax = axes[idx]
        values = [metrics_dict[model][metric_name] for model in models]
        
        bars = ax.bar(models, values)
        ax.set_ylabel('Score')
        ax.set_title(metric_name.replace('_', ' ').title())
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        # Rotate x labels if needed
        ax.set_xticklabels(models, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to {save_path}")
    
    return fig


def main():
    """Main comparison pipeline."""
    print("=" * 60)
    print("Model Comparison: XGBoost vs QSVM vs VQC")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    df = load_creditcard_data()
    
    # Split data
    print("\n2. Splitting data...")
    train_df, test_df = get_train_test_split(df)
    
    # Create balanced samples for quantum models
    print("\n3. Creating balanced samples for quantum models...")
    train_df_quantum = sample_balanced_dataset(train_df, sample_size=config.SAMPLE_SIZE)
    test_df_quantum = sample_balanced_dataset(test_df, sample_size=1000)
    
    # Train all models
    print("\n4. Training all models...")
    results = train_all_models(train_df, test_df, train_df_quantum, test_df_quantum)
    
    # Extract metrics for comparison
    metrics_dict = {name: result['metrics'] for name, result in results.items()}
    
    # Print comparison
    print("\n5. Comparing models...")
    compare_models_metrics(metrics_dict)
    
    # Plot comparison
    print("\n6. Creating comparison plot...")
    plot_comparison(metrics_dict)
    
    print("\n" + "=" * 60)
    print("Comparison complete!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
