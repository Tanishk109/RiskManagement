from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def calibration_diagnostics(labels: np.ndarray, risk_scores: np.ndarray, *, bins: int = 10) -> dict[str, object]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(risk_scores, dtype=float)
    if labels.shape != scores.shape:
        raise ValueError("labels and risk_scores must have identical shapes")
    observed, predicted = calibration_curve(labels, scores, n_bins=bins, strategy="quantile")
    return {
        "brier_score": float(brier_score_loss(labels, scores)),
        "mean_predicted_probability": predicted.tolist(),
        "observed_fraud_rate": observed.tolist(),
        "bins": bins,
    }
