import json
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

from .data_loader import load_creditcard_data
from .preprocessing import time_aware_train_test_split, prepare_features
from .metrics_utils import compute_metrics, save_metrics
from .config import RANDOM_STATE
from .utils import result_file


def main():
    print("=== Training XGBoost baseline ===")

    df = load_creditcard_data()
    train_df, test_df = time_aware_train_test_split(df)

    data = prepare_features(train_df, test_df)
    X_train = data["X_train_full"]
    X_test = data["X_test_full"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    # scale_pos_weight to handle imbalance
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    param_dist = {
        "n_estimators": [200, 400, 600],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "gamma": [0, 1, 5],
    }

    search = RandomizedSearchCV(
        xgb,
        param_distributions=param_dist,
        n_iter=20,
        scoring="average_precision",  # AUPRC
        cv=3,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print("Best XGBoost params:", search.best_params_)

    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_proba, threshold=0.5)
    metrics["best_params"] = search.best_params_
    metrics["model_type"] = "xgboost"

    print(json.dumps(metrics, indent=2))
    save_metrics(metrics, result_file("xgboost_results.json"))


if __name__ == "__main__":
    main()
