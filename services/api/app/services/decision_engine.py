from __future__ import annotations

from typing import Literal

Decision = Literal["APPROVE", "REVIEW", "BLOCK"]
_PRECEDENCE: dict[Decision, int] = {"APPROVE": 0, "REVIEW": 1, "BLOCK": 2}


def decision_from_score(risk_score: float, review_threshold: float, block_threshold: float) -> Decision:
    if not 0 <= risk_score <= 1:
        raise ValueError("risk_score must be between 0 and 1")
    if not 0 <= review_threshold < block_threshold <= 1:
        raise ValueError("thresholds must satisfy 0 <= review < block <= 1")
    if risk_score >= block_threshold:
        return "BLOCK"
    if risk_score >= review_threshold:
        return "REVIEW"
    return "APPROVE"


def combine_decisions(model_decision: Decision, rule_actions: list[Decision]) -> Decision:
    """Rules may escalate a model decision but never silently downgrade it."""
    return max([model_decision, *rule_actions], key=_PRECEDENCE.__getitem__)
