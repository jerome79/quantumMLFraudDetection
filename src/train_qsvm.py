import json
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel

from .data_loader import load_creditcard_data
from .preprocessing import time_aware_train_test_split, prepare_features
from .metrics_utils import compute_metrics, save_metrics
from .config import (
    RANDOM_STATE,
    N_QUBITS,
    FEATURE_MAP_REPS,
    USE_REAL_HW,
    IBM_BACKEND_NAME,
)
from .utils import result_file


def get_quantum_instance():
    if USE_REAL_HW:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        backend = service.backend(IBM_BACKEND_NAME)
        qi = QuantumInstance(backend=backend, shots=2048)
    else:
        backend = Aer.get_backend("qasm_simulator")
        qi = QuantumInstance(backend=backend, shots=1024, seed_simulator=RANDOM_STATE, seed_transpiler=RANDOM_STATE)
    return qi


def main():
    print("=== Training QSVM (Quantum Kernel SVM) ===")

    df = load_creditcard_data()
    train_df, test_df = time_aware_train_test_split(df)
    data = prepare_features(train_df, test_df)

    X_train_q = data["X_train_q"]
    X_test_q = data["X_test_q"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    print("Quantum feature dimension:", X_train_q.shape[1])

    qi = get_quantum_instance()

    feature_map = ZZFeatureMap(
        feature_dimension=X_train_q.shape[1],
        reps=FEATURE_MAP_REPS,
        entanglement="full",
    )

    quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=qi)

    print("Computing training kernel matrix...")
    kernel_train = quantum_kernel.evaluate(x_vec=X_train_q)
    print("Computing test kernel matrix...")
    kernel_test = quantum_kernel.evaluate(x_vec=X_test_q, y_vec=X_train_q)

    base_svc = SVC(kernel="precomputed", class_weight="balanced", probability=True, random_state=RANDOM_STATE)
    param_grid = {"C": [0.1, 1, 10]}

    grid = GridSearchCV(
        base_svc,
        param_grid=param_grid,
        scoring="average_precision",
        cv=3,
        verbose=1,
    )

    grid.fit(kernel_train, y_train)
    best_svc = grid.best_estimator_
    print("Best QSVM params:", grid.best_params_)

    y_proba = best_svc.predict_proba(kernel_test)[:, 1]

    metrics = compute_metrics(y_test, y_proba, threshold=0.5)
    metrics["best_params"] = grid.best_params_
    metrics["model_type"] = "qsvm"
    metrics["n_qubits"] = N_QUBITS
    metrics["feature_map_reps"] = FEATURE_MAP_REPS
    metrics["selected_features"] = data["selected_feature_names"]

    print(json.dumps(metrics, indent=2))
    save_metrics(metrics, result_file("qsvm_results.json"))


if __name__ == "__main__":
    main()
