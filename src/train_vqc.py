import json
import numpy as np

from qiskit import Aer
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.opflow import Z, StateFn, CircuitStateFn, AerPauliExpectation, CircuitSampler

from sklearn.model_selection import train_test_split

from .data_loader import load_creditcard_data
from .preprocessing import time_aware_train_test_split, prepare_features
from .metrics_utils import compute_metrics, save_metrics
from .config import (
    RANDOM_STATE,
    N_QUBITS,
    FEATURE_MAP_REPS,
    ANSATZ_REPS,
)
from .utils import result_file

from scipy.optimize import minimize


def get_qasm_backend():
    backend = Aer.get_backend("qasm_simulator")
    return backend


def build_vqc_circuit(n_features):
    feature_map = ZZFeatureMap(
        feature_dimension=n_features,
        reps=FEATURE_MAP_REPS,
        entanglement="full",
    )
    ansatz = TwoLocal(
        rotation_blocks="ry",
        entanglement_blocks="cz",
        entanglement="full",
        reps=ANSATZ_REPS,
    )
    return feature_map, ansatz


def vqc_forward(params, X, feature_map, ansatz, sampler):
    """
    Compute model outputs (probability of class 1) for all samples X.
    Very simple implementation using expectation of Z on first qubit.
    """
    probs = []
    for x in X:
        fm_circ = feature_map.bind_parameters(x)
        full_circ = fm_circ.compose(ansatz.bind_parameters(params))
        measurable_expr = StateFn(Z ^ N_QUBITS, is_measurement=True) @ CircuitStateFn(full_circ)
        expectation = AerPauliExpectation().convert(measurable_expr)
        value = sampler.convert(expectation).eval().real
        # Map expectation in [-1,1] to probability [0,1]
        p1 = (1 - value) / 2.0
        probs.append(p1)
    return np.array(probs)


def vqc_loss(params, X, y, feature_map, ansatz, sampler, lambda_reg=0.0):
    y_proba = vqc_forward(params, X, feature_map, ansatz, sampler)
    eps = 1e-10
    # Binary cross-entropy
    loss = -np.mean(y * np.log(y_proba + eps) + (1 - y) * np.log(1 - y_proba + eps))
    loss += lambda_reg * np.linalg.norm(params) ** 2
    return loss


def main():
    print("=== Training VQC ===")
    algorithm_globals.random_seed = RANDOM_STATE

    df = load_creditcard_data()
    train_df, test_df = time_aware_train_test_split(df)
    data = prepare_features(train_df, test_df)

    X_train_q = data["X_train_q"]
    X_test_q = data["X_test_q"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    # For speed, we can sub-sample training data (optional)
    # Here we keep all, but you can sub-sample if needed
    n_features = X_train_q.shape[1]
    print("Quantum feature dimension:", n_features)

    backend = get_qasm_backend()
    qi = QuantumInstance(backend=backend, shots=1024, seed_simulator=RANDOM_STATE, seed_transpiler=RANDOM_STATE)
    sampler = CircuitSampler(qi)

    feature_map, ansatz = build_vqc_circuit(n_features)
    num_params = ansatz.num_parameters
    print("VQC number of parameters:", num_params)

    init_params = 0.01 * (2 * np.random.rand(num_params) - 1)

    # Because the dataset is large, we can use a smaller training subset for VQC
    # to keep runtime manageable
    X_train_sub, _, y_train_sub, _ = train_test_split(
        X_train_q, y_train, test_size=0.8, stratify=y_train, random_state=RANDOM_STATE
    )

    print(f"VQC training subset size: {X_train_sub.shape[0]}")

    def objective(params):
        return vqc_loss(params, X_train_sub, y_train_sub, feature_map, ansatz, sampler)

    print("Optimizing VQC parameters...")
    res = minimize(
        objective,
        x0=init_params,
        method="COBYLA",
        options={"maxiter": 100, "disp": True},
    )

    best_params = res.x
    print("Optimization finished. Final loss:", res.fun)

    # Evaluate on test set
    y_proba_test = vqc_forward(best_params, X_test_q, feature_map, ansatz, sampler)

    metrics = compute_metrics(y_test, y_proba_test, threshold=0.5)
    metrics["model_type"] = "vqc"
    metrics["n_qubits"] = N_QUBITS
    metrics["feature_map_reps"] = FEATURE_MAP_REPS
    metrics["ansatz_reps"] = ANSATZ_REPS
    metrics["selected_features"] = data["selected_feature_names"]

    print(json.dumps(metrics, indent=2))
    save_metrics(metrics, result_file("vqc_results.json"))


if __name__ == "__main__":
    main()
