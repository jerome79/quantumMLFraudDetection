"""
Main training script for fraud detection models.

This script demonstrates how to use the quantumml_fraud package
to train and evaluate fraud detection models.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from quantumml_fraud.data import DataLoader, DataPreprocessor
from quantumml_fraud.models import ClassicalMLModel, QuantumMLModel
from quantumml_fraud.evaluation import FraudDetectionMetrics
from quantumml_fraud.utils import Config, setup_logger


def main(args):
    """Main training function."""
    
    # Setup logger
    logger = setup_logger(level=args.log_level)
    logger.info("Starting fraud detection training pipeline")
    
    # Load configuration
    config = Config(args.config) if args.config else Config()
    logger.info(f"Configuration loaded: {config.get('data.path')}")
    
    # Initialize data loader
    data_path = config.get('data.path')
    logger.info(f"Loading data from {data_path}")
    
    try:
        loader = DataLoader(data_path)
        data = loader.load_data()
        logger.info(f"Data loaded successfully: {data.shape}")
        
        # Split data
        train_data, test_data = loader.get_train_test_split(
            test_size=config.get('data.test_size', 0.2),
            random_state=config.get('data.random_state', 42)
        )
        logger.info(f"Train size: {train_data.shape}, Test size: {test_data.shape}")
        
        # Get features and labels
        X_train, y_train = loader.get_features_and_labels(train_data)
        X_test, y_test = loader.get_features_and_labels(test_data)
        
        # Preprocess data
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor(
            scaler_type=config.get('preprocessing.scaler_type', 'standard')
        )
        X_train_scaled = preprocessor.fit_transform(X_train)
        X_test_scaled = preprocessor.transform(X_test)
        
        # Handle imbalance if configured
        if config.get('preprocessing.handle_imbalance', False):
            logger.info("Handling class imbalance...")
            X_train_scaled, y_train = preprocessor.handle_imbalance(
                X_train_scaled, 
                y_train,
                method=config.get('preprocessing.imbalance_method', 'smote')
            )
        
        # Train model based on type
        if args.model_type == 'classical':
            logger.info("Training classical model...")
            model = ClassicalMLModel(
                model_type=config.get('classical_model.type', 'random_forest'),
                n_estimators=config.get('classical_model.n_estimators', 100),
                random_state=config.get('classical_model.random_state', 42)
            )
        elif args.model_type == 'quantum':
            logger.info("Training quantum model...")
            model = QuantumMLModel(
                n_qubits=config.get('quantum_model.n_qubits', 4),
                backend=config.get('quantum_model.backend', 'qiskit'),
                quantum_circuit_type=config.get('quantum_model.circuit_type', 'variational')
            )
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
        
        # Train
        model.train(X_train_scaled, y_train)
        logger.info("Model training completed")
        
        # Evaluate
        logger.info("Evaluating model...")
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = FraudDetectionMetrics.calculate_all_metrics(y_test, y_pred)
        logger.info("Evaluation metrics:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        # Print classification report
        report = FraudDetectionMetrics.print_classification_report(y_test, y_pred)
        logger.info(f"\nClassification Report:\n{report}")
        
        # Calculate confusion matrix
        cm, breakdown = FraudDetectionMetrics.get_confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info(f"Breakdown: {breakdown}")
        
        logger.info("Training pipeline completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        logger.info("Please download the dataset and place it in the data/raw directory")
        return 1
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train fraud detection models"
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['classical', 'quantum'],
        default='classical',
        help='Type of model to train'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    sys.exit(main(args))
