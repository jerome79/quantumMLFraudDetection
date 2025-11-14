import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Quantum configuration
N_FEATURES_Q = 8          # number of features for quantum models
N_QUBITS = N_FEATURES_Q   # one qubit per feature
FEATURE_MAP_REPS = 2
ANSATZ_REPS = 2

USE_REAL_HW = False  # set to True to use IBMQ (needs account setup)
IBM_BACKEND_NAME = "ibmq_qasm_simulator"  # or a real device, e.g. 'ibmq_manila'

RANDOM_STATE = 42
TEST_SIZE = 0.2
