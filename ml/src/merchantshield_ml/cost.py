from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

DECISIONS = ("APPROVE", "REVIEW", "BLOCK")
ESTIMATED_COST_LABEL = "ESTIMATED BUSINESS COST UNDER USER-SUPPLIED ASSUMPTIONS"


@dataclass(frozen=True)
class CostAssumptions:
    """Merchant-supplied scenario inputs; defaults are placeholders, not industry facts."""

    currency: str = "INR"
    fraud_loss_fraction: float = 1.0
    chargeback_fixed_cost: float = 0.0
    legitimate_margin_rate: float = 0.20
    false_positive_fixed_cost: float = 0.0
    manual_review_cost: float = 150.0
    review_fraud_catch_rate: float = 0.90
    review_legitimate_approval_rate: float = 0.98

    def __post_init__(self) -> None:
        if len(self.currency) != 3:
            raise ValueError("currency must be a three-letter code")
        nonnegative = (
            "chargeback_fixed_cost",
            "false_positive_fixed_cost",
            "manual_review_cost",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        rates = (
            "fraud_loss_fraction",
            "legitimate_margin_rate",
            "review_fraud_catch_rate",
            "review_legitimate_approval_rate",
        )
        for name in rates:
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


def validate_thresholds(review_threshold: float, block_threshold: float) -> None:
    if not 0 <= review_threshold < block_threshold <= 1:
        raise ValueError("thresholds must satisfy 0 <= review < block <= 1")


def decisions_from_scores(
    risk_scores: np.ndarray,
    review_threshold: float,
    block_threshold: float,
) -> np.ndarray:
    validate_thresholds(review_threshold, block_threshold)
    scores = np.asarray(risk_scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("risk scores must be one-dimensional")
    if np.any(~np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("risk scores must be finite and between 0 and 1")
    return np.where(
        scores >= block_threshold,
        "BLOCK",
        np.where(scores >= review_threshold, "REVIEW", "APPROVE"),
    )


def _validated_inputs(
    labels: np.ndarray,
    amounts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    normalized_labels = np.asarray(labels, dtype=int)
    normalized_amounts = np.asarray(amounts, dtype=float)
    if normalized_labels.ndim != 1 or normalized_amounts.ndim != 1:
        raise ValueError("labels and amounts must be one-dimensional")
    if normalized_labels.shape != normalized_amounts.shape:
        raise ValueError("labels and amounts must have identical shapes")
    if normalized_labels.size == 0:
        raise ValueError("Cannot simulate cost over an empty dataset")
    if np.any(~np.isfinite(normalized_amounts)) or np.any(normalized_amounts < 0):
        raise ValueError("Transaction amounts must be finite and non-negative")
    if np.any(~np.isin(normalized_labels, [0, 1])):
        raise ValueError("Labels must contain only 0 or 1")
    return normalized_labels, normalized_amounts


def simulate_decisions(
    *,
    labels: np.ndarray,
    amounts: np.ndarray,
    decisions: np.ndarray,
    assumptions: CostAssumptions,
) -> dict[str, Any]:
    labels, amounts = _validated_inputs(labels, amounts)
    normalized_decisions = np.asarray(decisions, dtype=str)
    if normalized_decisions.shape != labels.shape:
        raise ValueError("decisions must have the same shape as labels")
    if np.any(~np.isin(normalized_decisions, DECISIONS)):
        raise ValueError("Decisions must be APPROVE, REVIEW, or BLOCK")

    fraud = labels == 1
    legitimate = ~fraud
    approved = normalized_decisions == "APPROVE"
    reviewed = normalized_decisions == "REVIEW"
    blocked = normalized_decisions == "BLOCK"
    total = len(labels)
    fraud_count = int(fraud.sum())
    legitimate_count = int(legitimate.sum())

    approve_count = int(approved.sum())
    review_count = int(reviewed.sum())
    block_count = int(blocked.sum())
    fraud_approved = int((approved & fraud).sum())
    fraud_reviewed = int((reviewed & fraud).sum())
    fraud_blocked = int((blocked & fraud).sum())
    legitimate_approved = int((approved & legitimate).sum())
    legitimate_reviewed = int((reviewed & legitimate).sum())
    legitimate_blocked = int((blocked & legitimate).sum())

    full_fraud_loss = (
        amounts * assumptions.fraud_loss_fraction + assumptions.chargeback_fixed_cost
    )
    full_legitimate_cost = (
        amounts * assumptions.legitimate_margin_rate
        + assumptions.false_positive_fixed_cost
    )
    reviewed_fraud_residual = full_fraud_loss[reviewed & fraud] * (
        1 - assumptions.review_fraud_catch_rate
    )
    reviewed_legitimate_residual = full_legitimate_cost[reviewed & legitimate] * (
        1 - assumptions.review_legitimate_approval_rate
    )
    approved_fraud_loss = float(full_fraud_loss[approved & fraud].sum())
    reviewed_fraud_loss = float(reviewed_fraud_residual.sum())
    blocked_legitimate_cost = float(full_legitimate_cost[blocked & legitimate].sum())
    reviewed_legitimate_cost = float(reviewed_legitimate_residual.sum())
    fraud_loss = approved_fraud_loss + reviewed_fraud_loss
    false_positive_cost = blocked_legitimate_cost + reviewed_legitimate_cost
    manual_review_cost_total = float(review_count * assumptions.manual_review_cost)
    review_expected_residual_cost = reviewed_fraud_loss + reviewed_legitimate_cost
    review_total_cost = manual_review_cost_total + review_expected_residual_cost
    total_estimated_cost = fraud_loss + false_positive_cost + manual_review_cost_total

    total_fraud_amount = float(amounts[fraud].sum())
    fraud_amount_approved = float(amounts[approved & fraud].sum())
    fraud_amount_reviewed = float(amounts[reviewed & fraud].sum())
    fraud_amount_blocked = float(amounts[blocked & fraud].sum())
    expected_review_caught_fraud_amount = (
        fraud_amount_reviewed * assumptions.review_fraud_catch_rate
    )
    captured_fraud_amount = fraud_amount_blocked + expected_review_caught_fraud_amount
    fraud_amount_capture_rate = (
        captured_fraud_amount / total_fraud_amount if total_fraud_amount else 0.0
    )

    block_precision = fraud_blocked / block_count if block_count else 0.0
    block_recall = fraud_blocked / fraud_count if fraud_count else 0.0
    detected_fraud_recall = (
        (fraud_reviewed + fraud_blocked) / fraud_count if fraud_count else 0.0
    )
    detected_precision = (
        (fraud_reviewed + fraud_blocked) / (review_count + block_count)
        if review_count + block_count
        else 0.0
    )
    return {
        "cost_output_label": ESTIMATED_COST_LABEL,
        "currency": assumptions.currency,
        "transaction_count": total,
        "fraud_count": fraud_count,
        "legitimate_count": legitimate_count,
        "approve_count": approve_count,
        "review_count": review_count,
        "block_count": block_count,
        "approve_volume": approve_count,
        "review_volume": review_count,
        "block_volume": block_count,
        "approve_rate": approve_count / total,
        "review_rate": review_count / total,
        "block_rate": block_count / total,
        "fraud_approved": fraud_approved,
        "fraud_reviewed": fraud_reviewed,
        "fraud_blocked": fraud_blocked,
        "fraud_approve_rate": fraud_approved / fraud_count if fraud_count else 0.0,
        "fraud_review_rate": fraud_reviewed / fraud_count if fraud_count else 0.0,
        "fraud_block_rate": fraud_blocked / fraud_count if fraud_count else 0.0,
        "legitimate_approved": legitimate_approved,
        "legitimate_reviewed": legitimate_reviewed,
        "legitimate_blocked": legitimate_blocked,
        "legitimate_approve_rate": legitimate_approved / legitimate_count
        if legitimate_count
        else 0.0,
        "legitimate_review_rate": legitimate_reviewed / legitimate_count
        if legitimate_count
        else 0.0,
        "legitimate_block_rate": legitimate_blocked / legitimate_count
        if legitimate_count
        else 0.0,
        "precision": block_precision,
        "recall": block_recall,
        "block_precision": block_precision,
        "block_recall": block_recall,
        "detected_precision": detected_precision,
        "detected_fraud_recall": detected_fraud_recall,
        "false_positives": legitimate_blocked,
        "false_negatives": fraud_approved,
        "total_fraud_amount": total_fraud_amount,
        "fraud_amount_approved": fraud_amount_approved,
        "fraud_amount_reviewed": fraud_amount_reviewed,
        "fraud_amount_blocked": fraud_amount_blocked,
        "expected_review_caught_fraud_amount": expected_review_caught_fraud_amount,
        "captured_fraud_amount": captured_fraud_amount,
        "fraud_amount_capture_rate": fraud_amount_capture_rate,
        "approved_fraud_loss": approved_fraud_loss,
        "reviewed_fraud_loss": reviewed_fraud_loss,
        "blocked_legitimate_cost": blocked_legitimate_cost,
        "reviewed_legitimate_cost": reviewed_legitimate_cost,
        "fraud_loss": fraud_loss,
        "false_positive_cost": false_positive_cost,
        "review_cost": manual_review_cost_total,
        "manual_review_cost_total": manual_review_cost_total,
        "review_expected_residual_cost": review_expected_residual_cost,
        "review_total_cost": review_total_cost,
        "total_estimated_cost": total_estimated_cost,
    }


def simulate_cost(
    *,
    labels: np.ndarray,
    amounts: np.ndarray,
    risk_scores: np.ndarray,
    review_threshold: float,
    block_threshold: float,
    assumptions: CostAssumptions,
) -> dict[str, Any]:
    labels, amounts = _validated_inputs(labels, amounts)
    scores = np.asarray(risk_scores, dtype=float)
    if scores.shape != labels.shape:
        raise ValueError("labels, amounts, and risk_scores must have identical shapes")
    decisions = decisions_from_scores(scores, review_threshold, block_threshold)
    return simulate_decisions(
        labels=labels,
        amounts=amounts,
        decisions=decisions,
        assumptions=assumptions,
    )
