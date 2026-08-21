from __future__ import annotations

import numpy as np
import pytest
from merchantshield_ml.cost import CostAssumptions, decisions_from_scores, simulate_cost
from merchantshield_ml.thresholds import search_thresholds


def test_two_threshold_decisions():
    decisions = decisions_from_scores(np.array([0.1, 0.4, 0.79, 0.8]), 0.4, 0.8)
    assert decisions.tolist() == ["APPROVE", "REVIEW", "REVIEW", "BLOCK"]


def test_invalid_threshold_order_is_rejected():
    with pytest.raises(ValueError, match="review < block"):
        decisions_from_scores(np.array([0.2]), 0.8, 0.8)


def test_cost_calculation_separates_three_decisions():
    assumptions = CostAssumptions(
        fraud_loss_fraction=1,
        chargeback_fixed_cost=0,
        legitimate_margin_rate=0.2,
        false_positive_fixed_cost=0,
        manual_review_cost=10,
        review_fraud_catch_rate=0.5,
        review_legitimate_approval_rate=0.5,
    )
    result = simulate_cost(
        labels=np.array([1, 1, 0, 0]),
        amounts=np.array([100, 200, 300, 400]),
        risk_scores=np.array([0.1, 0.5, 0.9, 0.5]),
        review_threshold=0.4,
        block_threshold=0.8,
        assumptions=assumptions,
    )
    assert result["fraud_loss"] == 200.0
    assert result["false_positive_cost"] == 100.0
    assert result["review_cost"] == 20.0
    assert result["total_estimated_cost"] == 320.0
    assert result["review_volume"] == 2
    assert result["block_volume"] == 1


def test_threshold_search_uses_explicit_validation_provenance():
    result = search_thresholds(
        labels=np.array([0, 1, 0, 1]),
        amounts=np.array([10, 50, 20, 80]),
        risk_scores=np.array([0.1, 0.9, 0.3, 0.7]),
        assumptions=CostAssumptions(manual_review_cost=1),
        review_candidates=[0.2, 0.4],
        block_candidates=[0.6, 0.8],
    )
    assert result["selection_split"] == "validation"
    assert "current" in result["selection_wording"]
    assert len(result["configurations"]) == 4
