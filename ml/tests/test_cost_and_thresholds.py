from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from merchantshield_ml.cost import (
    ESTIMATED_COST_LABEL,
    CostAssumptions,
    decisions_from_scores,
    simulate_cost,
    simulate_decisions,
)
from merchantshield_ml.thresholds import (
    capacity_selections,
    evaluate_threshold_grid,
    search_thresholds,
    select_lowest_cost,
    threshold_candidates,
    validate_threshold_grid_artifact,
)


def assumptions(**overrides: float) -> CostAssumptions:
    values = {
        "fraud_loss_fraction": 0.8,
        "chargeback_fixed_cost": 10,
        "legitimate_margin_rate": 0.2,
        "false_positive_fixed_cost": 5,
        "manual_review_cost": 7,
        "review_fraud_catch_rate": 0.75,
        "review_legitimate_approval_rate": 0.8,
    }
    values.update(overrides)
    return CostAssumptions(**values)


def test_two_threshold_decisions_include_exact_boundaries():
    decisions = decisions_from_scores(np.array([0.1, 0.4, 0.79, 0.8]), 0.4, 0.8)
    assert decisions.tolist() == ["APPROVE", "REVIEW", "REVIEW", "BLOCK"]


@pytest.mark.parametrize(
    ("review_threshold", "block_threshold"),
    [(0.8, 0.8), (0.9, 0.8), (-0.1, 0.8), (0.2, 1.1)],
)
def test_invalid_thresholds_are_rejected(
    review_threshold: float, block_threshold: float
):
    with pytest.raises(ValueError, match="review < block"):
        decisions_from_scores(np.array([0.2]), review_threshold, block_threshold)


def test_invalid_scores_amounts_and_assumptions_are_rejected():
    with pytest.raises(ValueError, match="risk scores"):
        decisions_from_scores(np.array([np.nan]), 0.2, 0.8)
    with pytest.raises(ValueError, match="non-negative"):
        simulate_decisions(
            labels=np.array([0]),
            amounts=np.array([-1]),
            decisions=np.array(["APPROVE"]),
            assumptions=assumptions(),
        )
    with pytest.raises(ValueError, match="between 0 and 1"):
        assumptions(fraud_loss_fraction=1.1)


def test_approve_fraud_cost_formula():
    result = simulate_decisions(
        labels=np.array([1]),
        amounts=np.array([100]),
        decisions=np.array(["APPROVE"]),
        assumptions=assumptions(),
    )
    assert result["approved_fraud_loss"] == 90
    assert result["fraud_loss"] == 90
    assert result["total_estimated_cost"] == 90


def test_block_legitimate_cost_formula():
    result = simulate_decisions(
        labels=np.array([0]),
        amounts=np.array([100]),
        decisions=np.array(["BLOCK"]),
        assumptions=assumptions(),
    )
    assert result["blocked_legitimate_cost"] == 25
    assert result["false_positive_cost"] == 25
    assert result["total_estimated_cost"] == 25


def test_review_cost_formulas_include_each_row_and_expected_residuals():
    result = simulate_decisions(
        labels=np.array([1, 0]),
        amounts=np.array([100, 100]),
        decisions=np.array(["REVIEW", "REVIEW"]),
        assumptions=assumptions(),
    )
    assert result["reviewed_fraud_loss"] == 22.5
    assert result["reviewed_legitimate_cost"] == pytest.approx(5.0)
    assert result["manual_review_cost_total"] == 14
    assert result["review_total_cost"] == pytest.approx(41.5)
    assert result["total_estimated_cost"] == pytest.approx(41.5)


def test_zero_manual_review_cost_and_effectiveness_boundaries():
    perfect = simulate_decisions(
        labels=np.array([1, 0]),
        amounts=np.array([100, 100]),
        decisions=np.array(["REVIEW", "REVIEW"]),
        assumptions=assumptions(
            manual_review_cost=0,
            review_fraud_catch_rate=1,
            review_legitimate_approval_rate=1,
        ),
    )
    ineffective = simulate_decisions(
        labels=np.array([1, 0]),
        amounts=np.array([100, 100]),
        decisions=np.array(["REVIEW", "REVIEW"]),
        assumptions=assumptions(
            manual_review_cost=0,
            review_fraud_catch_rate=0,
            review_legitimate_approval_rate=0,
        ),
    )
    assert perfect["total_estimated_cost"] == 0
    assert ineffective["total_estimated_cost"] == 115


def test_cost_calculation_separates_three_decisions():
    result = simulate_cost(
        labels=np.array([1, 1, 0, 0]),
        amounts=np.array([100, 200, 300, 400]),
        risk_scores=np.array([0.1, 0.5, 0.9, 0.5]),
        review_threshold=0.4,
        block_threshold=0.8,
        assumptions=CostAssumptions(
            fraud_loss_fraction=1,
            legitimate_margin_rate=0.2,
            manual_review_cost=10,
            review_fraud_catch_rate=0.5,
            review_legitimate_approval_rate=0.5,
        ),
    )
    assert result["fraud_loss"] == 200
    assert result["false_positive_cost"] == 100
    assert result["review_cost"] == 20
    assert result["total_estimated_cost"] == 320
    assert result["cost_output_label"] == ESTIMATED_COST_LABEL


def test_amount_capture_uses_full_blocks_and_expected_review_catches():
    result = simulate_decisions(
        labels=np.array([1, 1, 1]),
        amounts=np.array([100, 200, 300]),
        decisions=np.array(["APPROVE", "REVIEW", "BLOCK"]),
        assumptions=assumptions(review_fraud_catch_rate=0.5),
    )
    assert result["fraud_amount_approved"] == 100
    assert result["fraud_amount_reviewed"] == 200
    assert result["fraud_amount_blocked"] == 300
    assert result["captured_fraud_amount"] == 400
    assert result["fraud_amount_capture_rate"] == pytest.approx(2 / 3)


def example_grid() -> list[dict[str, object]]:
    return evaluate_threshold_grid(
        labels=np.array([0, 1, 0, 1, 0]),
        amounts=np.array([10, 50, 20, 80, 30]),
        risk_scores=np.array([0.1, 0.9, 0.3, 0.7, 0.45]),
        assumptions=assumptions(manual_review_cost=1),
        review_candidates=[0.2, 0.4],
        block_candidates=[0.6, 0.8],
    )


def test_threshold_grid_is_deterministic_and_capacity_is_enforced():
    first = example_grid()
    second = example_grid()
    assert first == second
    selection = select_lowest_cost(first, max_review_rate=0.2)
    assert selection["review_rate"] <= 0.2
    capacities = capacity_selections(first, [None, 0.2])
    assert capacities[0]["review_capacity"] is None
    assert capacities[1]["selected_configuration"]["review_rate"] <= 0.2


def test_threshold_candidates_are_inclusive_and_stable():
    assert threshold_candidates(0.05, 0.15, 0.025) == [0.05, 0.075, 0.1, 0.125, 0.15]


def test_threshold_artifact_schema_validation():
    frame = pd.DataFrame(example_grid())
    validate_threshold_grid_artifact(frame)
    with pytest.raises(ValueError, match="missing columns"):
        validate_threshold_grid_artifact(frame.drop(columns=["fraud_loss"]))


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


def test_threshold_analysis_script_has_no_held_out_loader_path():
    script = (Path(__file__).parents[1] / "scripts/analyze_thresholds.py").read_text(
        encoding="utf-8"
    )
    forbidden = ("test.parquet", "load_splits", "load_processed_splits")
    assert not any(token in script for token in forbidden)
