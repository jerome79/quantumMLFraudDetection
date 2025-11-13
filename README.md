# Quantum ML Fraud Detection

Quantum Machine Learning for Credit Card Fraud Detection - This repository implements a full pipeline for fraud detection on the ULB/Worldline Credit Card Fraud Detection dataset.

## Overview

This project provides a comprehensive framework for credit card fraud detection using both classical and quantum machine learning approaches. It includes data preprocessing, model training, evaluation metrics, and comparison between classical and quantum ML algorithms.

## Features

- **Data Processing**: Load and preprocess credit card transaction data
- **Classical ML Models**: Random Forest, Gradient Boosting, Logistic Regression, SVM
- **Quantum ML Models**: Variational Quantum Circuits (VQC) and Quantum Kernel methods
- **Imbalanced Data Handling**: SMOTE, oversampling, and undersampling techniques
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, and cost-sensitive metrics
- **Configurable Pipeline**: JSON-based configuration for easy experimentation

## Project Structure

```
quantumMLFraudDetection/
├── src/
│   └── quantumml_fraud/          # Main package
│       ├── __init__.py
│       ├── data/                 # Data loading and preprocessing
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   └── preprocessor.py
│       ├── models/               # ML models
│       │   ├── __init__.py
│       │   ├── classical.py      # Classical ML models
│       │   └── quantum.py        # Quantum ML models
│       ├── evaluation/           # Evaluation metrics
│       │   ├── __init__.py
│       │   └── metrics.py
│       └── utils/                # Utilities
│           ├── __init__.py
│           ├── config.py
│           └── logger.py
├── tests/                        # Unit tests
│   ├── __init__.py
│   ├── test_config.py
│   └── test_preprocessor.py
├── notebooks/                    # Jupyter notebooks
│   └── 01_fraud_detection_demo.ipynb
├── data/                         # Data directory
│   ├── raw/                      # Raw datasets
│   └── processed/                # Processed datasets
├── saved_models/                 # Saved model files
├── configs/                      # Configuration files
│   └── default_config.json
├── docs/                         # Documentation
├── train.py                      # Main training script
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── pyproject.toml               # Project configuration
└── README.md                     # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/jerome79/quantumMLFraudDetection.git
cd quantumMLFraudDetection

# Install the package and dependencies
pip install -e .

# Or install with quantum libraries
pip install -e ".[quantum]"

# For development
pip install -e ".[dev]"
```

## Dataset

Download the Credit Card Fraud Detection dataset from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the `creditcard.csv` file in the `data/raw/` directory.

## Usage

### Quick Start with Training Script

```bash
# Train a classical model (Random Forest)
python train.py --model-type classical

# Train with custom configuration
python train.py --config configs/default_config.json --model-type classical

# Train quantum model (experimental)
python train.py --model-type quantum
```

### Using the Package in Python

```python
from quantumml_fraud.data import DataLoader, DataPreprocessor
from quantumml_fraud.models import ClassicalMLModel
from quantumml_fraud.evaluation import FraudDetectionMetrics
from quantumml_fraud.utils import Config

# Load configuration
config = Config('configs/default_config.json')

# Load and preprocess data
loader = DataLoader(config.get('data.path'))
data = loader.load_data()
train_data, test_data = loader.get_train_test_split()

X_train, y_train = loader.get_features_and_labels(train_data)
X_test, y_test = loader.get_features_and_labels(test_data)

# Preprocess
preprocessor = DataPreprocessor(scaler_type='standard')
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Train model
model = ClassicalMLModel(model_type='random_forest')
model.train(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
metrics = FraudDetectionMetrics.calculate_all_metrics(y_test, y_pred)
print(metrics)
```

### Using Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/01_fraud_detection_demo.ipynb
```

## Configuration

Edit `configs/default_config.json` to customize:

- Data paths and split ratios
- Preprocessing methods
- Model hyperparameters
- Evaluation settings

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/quantumml_fraud --cov-report=html

# Run specific test file
pytest tests/test_config.py
```

## Development

### Code Style

```bash
# Format code with black
black src/ tests/

# Check code style
flake8 src/ tests/

# Type checking
mypy src/
```

## Quantum Computing

The quantum ML implementation supports:

- **Qiskit**: IBM's quantum computing framework
- **PennyLane**: Quantum machine learning library

Note: Quantum implementations are experimental and require appropriate quantum computing libraries.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Dataset: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Qiskit: https://qiskit.org/
- PennyLane: https://pennylane.ai/

## Citation

If you use this code in your research, please cite:

```bibtex
@software{quantumml_fraud,
  author = {jerome79},
  title = {Quantum Machine Learning for Credit Card Fraud Detection},
  year = {2024},
  url = {https://github.com/jerome79/quantumMLFraudDetection}
}
```

## Contact

For questions or issues, please open an issue on GitHub.
