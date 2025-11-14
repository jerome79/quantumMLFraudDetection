import json
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    y_proba: probability for positive class (fraud = 1)
    """
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {}
    # AUROC / AUPRC
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        metrics["auroc"] = None

    try:
        metrics["auprc"] = float(average_precision_score(y_true, y_proba))
    except ValueError:
        metrics["auprc"] = None

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    # Macro / weighted
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    metrics["precision_macro"] = float(precision_macro)
    metrics["recall_macro"] = float(recall_macro)
    metrics["f1_macro"] = float(f1_macro)
    metrics["precision_weighted"] = float(precision_weighted)
    metrics["recall_weighted"] = float(recall_weighted)
    metrics["f1_weighted"] = float(f1_weighted)

    # Positive class (fraud)
    precision_pos, recall_pos, f1_pos, support_pos = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[1], zero_division=0
    )
    metrics["precision_fraud"] = float(precision_pos[0])
    metrics["recall_fraud"] = float(recall_pos[0])
    metrics["f1_fraud"] = float(f1_pos[0])
    metrics["support_fraud"] = int(support_pos[0])

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # Full classification report as string
    metrics["classification_report"] = classification_report(
        y_true, y_pred, digits=4, zero_division=0
    )

    return metrics


def save_metrics(metrics: Dict, path: Path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
