from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(labels: np.ndarray, risk_scores: np.ndarray, *, threshold: float = 0.5) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    risk_scores = np.asarray(risk_scores, dtype=float)
    if labels.shape != risk_scores.shape:
        raise ValueError("labels and risk_scores must have identical shapes")
    if labels.size == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    predictions = (risk_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(labels, risk_scores)),
        "roc_auc": float(roc_auc_score(labels, risk_scores)) if len(np.unique(labels)) == 2 else None,
        "brier_score": float(brier_score_loss(labels, risk_scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }
