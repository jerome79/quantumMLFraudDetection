"""Configuration module for quantum fraud detection project."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset configuration
DATASET_PATH = RAW_DATA_DIR / "creditcard.csv"

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Quantum configuration
QUANTUM_BACKEND = "qasm_simulator"
QUANTUM_SHOTS = 1024
FEATURE_DIM = 4  # Reduced dimension for quantum models

# XGBoost configuration
XGBOOST_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "objective": "binary:logistic",
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

# Training configuration
BATCH_SIZE = 32
EPOCHS = 100

# Sampling configuration (for faster training with quantum models)
SAMPLE_SIZE = 5000  # Number of samples to use for quantum models
BALANCE_CLASSES = True  # Whether to balance fraud/non-fraud samples

# Create directories if they don't exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
