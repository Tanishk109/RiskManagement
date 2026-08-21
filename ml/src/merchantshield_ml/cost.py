from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostAssumptions:
    currency: str = "INR"
    fraud_loss_fraction: float = 1.0
    chargeback_fixed_cost: float = 0.0
    legitimate_margin_rate: float = 0.20
    false_positive_fixed_cost: float = 0.0
    manual_review_cost: float = 150.0
    review_fraud_catch_rate: float = 0.90
    review_legitimate_approval_rate: float = 0.98

    def __post_init__(self) -> None:
        if self.fraud_loss_fraction < 0 or self.chargeback_fixed_cost < 0 or self.false_positive_fixed_cost < 0 or self.manual_review_cost < 0:
            raise ValueError("Cost values and fractions cannot be negative")
        for name in ("legitimate_margin_rate", "review_fraud_catch_rate", "review_legitimate_approval_rate"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")


def decisions_from_scores(risk_scores: np.ndarray, review_threshold: float, block_threshold: float) -> np.ndarray:
    if not 0 <= review_threshold < block_threshold <= 1:
        raise ValueError("thresholds must satisfy 0 <= review < block <= 1")
    scores = np.asarray(risk_scores, dtype=float)
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("risk scores must be between 0 and 1")
    return np.where(scores >= block_threshold, "BLOCK", np.where(scores >= review_threshold, "REVIEW", "APPROVE"))


def simulate_cost(
    *,
    labels: np.ndarray,
    amounts: np.ndarray,
    risk_scores: np.ndarray,
    review_threshold: float,
    block_threshold: float,
    assumptions: CostAssumptions,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=int)
    amounts = np.asarray(amounts, dtype=float)
    scores = np.asarray(risk_scores, dtype=float)
    if not (labels.shape == amounts.shape == scores.shape):
        raise ValueError("labels, amounts, and risk_scores must have identical shapes")
    if labels.size == 0:
        raise ValueError("Cannot simulate cost over an empty dataset")
    if np.any(amounts < 0):
        raise ValueError("Transaction amounts cannot be negative")
    if np.any(~np.isin(labels, [0, 1])):
        raise ValueError("Labels must contain only 0 or 1")

    decisions = decisions_from_scores(scores, review_threshold, block_threshold)
    fraud = labels == 1
    legitimate = ~fraud
    approved = decisions == "APPROVE"
    reviewed = decisions == "REVIEW"
    blocked = decisions == "BLOCK"

    full_fraud_loss = amounts * assumptions.fraud_loss_fraction + assumptions.chargeback_fixed_cost
    full_legitimate_cost = amounts * assumptions.legitimate_margin_rate + assumptions.false_positive_fixed_cost
    fraud_loss = float(full_fraud_loss[approved & fraud].sum())
    fraud_loss += float((full_fraud_loss[reviewed & fraud] * (1 - assumptions.review_fraud_catch_rate)).sum())
    false_positive_cost = float(full_legitimate_cost[blocked & legitimate].sum())
    false_positive_cost += float((full_legitimate_cost[reviewed & legitimate] * (1 - assumptions.review_legitimate_approval_rate)).sum())
    review_cost = float(reviewed.sum() * assumptions.manual_review_cost)

    true_positives = int((blocked & fraud).sum())
    false_positives = int((blocked & legitimate).sum())
    false_negatives = int((approved & fraud).sum())
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / int(fraud.sum()) if fraud.any() else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "review_volume": int(reviewed.sum()),
        "block_volume": int(blocked.sum()),
        "approve_volume": int(approved.sum()),
        "fraud_loss": fraud_loss,
        "false_positive_cost": false_positive_cost,
        "review_cost": review_cost,
        "total_estimated_cost": fraud_loss + false_positive_cost + review_cost,
    }
