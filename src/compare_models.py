import json
from pathlib import Path

import pandas as pd
from .config import RESULTS_DIR


def load_result(name: str):
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    xgb = load_result("xgboost_results.json")
    qsvm = load_result("qsvm_results.json")
    vqc = load_result("vqc_results.json")

    rows = []
    for res in [xgb, qsvm, vqc]:
        if res is None:
            continue
        rows.append(
            {
                "model": res["model_type"],
                "auroc": res["auroc"],
                "auprc": res["auprc"],
                "f1_fraud": res["f1_fraud"],
                "recall_fraud": res["recall_fraud"],
                "precision_fraud": res["precision_fraud"],
            }
        )

    df = pd.DataFrame(rows)
    print("=== Model Comparison ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
