# Quantum Machine Learning for Credit Card Fraud Detection

This repository implements a comprehensive pipeline for credit card fraud detection using both classical machine learning (XGBoost) and quantum machine learning approaches (Quantum SVM and Variational Quantum Classifier).

## 🎯 Project Overview

This project compares the performance of classical and quantum machine learning models on the ULB/Worldline Credit Card Fraud Detection dataset. The goal is to evaluate how quantum computing approaches perform against traditional methods in detecting fraudulent transactions.

## 📁 Project Structure

```
quantum-fraud-detection/
├── README.md
├── requirements.txt
├── data/
│   └── raw/
│       └── creditcard.csv          # ULB dataset (download separately)
├── src/
│   ├── config.py                   # Configuration and hyperparameters
│   ├── data_loader.py              # Data loading utilities
│   ├── preprocessing.py            # Data preprocessing and feature engineering
│   ├── metrics_utils.py            # Evaluation metrics
│   ├── train_xgboost.py            # Classical XGBoost training
│   ├── train_qsvm.py               # Quantum SVM training
│   ├── train_vqc.py                # Variational Quantum Classifier training
│   ├── compare_models.py           # Model comparison script
│   └── utils.py                    # General utility functions
└── notebooks/
    └── 01_exploratory_analysis.ipynb   # Exploratory data analysis
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone this repository:
```bash
git clone https://github.com/jerome79/quantumMLFraudDetection.git
cd quantumMLFraudDetection
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
   - Download the Credit Card Fraud Detection dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
   - Place the `creditcard.csv` file in `data/raw/` directory

## 💻 Usage

### Training Individual Models

#### XGBoost (Classical Model)
```bash
cd src
python train_xgboost.py
```

#### Quantum SVM
```bash
cd src
python train_qsvm.py
```

#### Variational Quantum Classifier (VQC)
```bash
cd src
python train_vqc.py
```

### Comparing All Models
```bash
cd src
python compare_models.py
```

This will train all three models and generate a comprehensive comparison report with visualizations.

## 🔬 Models Implemented

### 1. XGBoost (Classical)
- Gradient boosting classifier
- Full feature set (30 features)
- Handles class imbalance

### 2. Quantum Support Vector Machine (QSVM)
- Uses quantum kernel for feature mapping
- ZZ feature map with 2 repetitions
- Reduced feature dimension (4 qubits) via PCA

### 3. Variational Quantum Classifier (VQC)
- Parameterized quantum circuit
- ZZ feature map + RealAmplitudes ansatz
- COBYLA optimizer
- Reduced feature dimension (4 qubits) via PCA

## 📊 Dataset

The project uses the **ULB Credit Card Fraud Detection Dataset**, which contains:
- 284,807 transactions
- 492 fraudulent transactions (0.172% of all transactions)
- 30 numerical features (V1-V28 from PCA, Time, Amount)
- Highly imbalanced dataset

## ⚙️ Configuration

Key parameters can be adjusted in `src/config.py`:
- `RANDOM_STATE`: Random seed for reproducibility
- `TEST_SIZE`: Train/test split ratio
- `FEATURE_DIM`: Number of qubits for quantum models
- `QUANTUM_SHOTS`: Number of shots for quantum circuits
- `SAMPLE_SIZE`: Sample size for quantum model training

## 📈 Evaluation Metrics

Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

## 🔧 Requirements

- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- qiskit >= 0.45.0
- qiskit-machine-learning >= 0.7.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

See `requirements.txt` for complete list.

## 📝 Notes

- Quantum models use a reduced feature set (via PCA) due to qubit limitations
- Quantum models are trained on balanced subsets for computational efficiency
- Training quantum models may take significantly longer than classical models
- Results may vary due to quantum circuit noise and sampling

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🙏 Acknowledgments

- ULB (Université Libre de Bruxelles) for the Credit Card Fraud Detection dataset
- Qiskit team for quantum computing framework
- XGBoost developers for the gradient boosting library
