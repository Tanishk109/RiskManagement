from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from merchantshield_ml.final_evaluation import (
    evaluate_frozen_scores,
    validate_frozen_evaluation_spec,
)


def frozen_artifacts() -> tuple[dict, dict]:
    metadata = {
        "status": "validation_candidate",
        "model_name": "CatBoostClassifier",
        "model_version": "catboost-validation-v1",
        "feature_set": "frozen_features",
        "feature_names": ["TransactionAmt", "ProductCD"],
        "categorical_feature_names": ["ProductCD"],
        "actual_tree_count": 17,
        "held_out_test_accessed": False,
    }
    config = {
        "status": "provisional_validation_config",
        "not_final": True,
        "selection_split": "validation",
        "held_out_test_accessed": False,
        "model_version": "catboost-validation-v1",
        "feature_set": "frozen_features",
        "scenario": "moderate",
        "scenario_name": "Moderate merchant",
        "selection_reason": "lowest validation cost within capacity",
        "review_threshold": 0.2,
        "block_threshold": 0.6,
        "assumption_status": "ILLUSTRATIVE MERCHANT ASSUMPTIONS",
        "cost_assumptions": {
            "currency": "INR",
            "fraud_loss_fraction": 0.85,
            "chargeback_fixed_cost": 50,
            "legitimate_margin_rate": 0.18,
            "false_positive_fixed_cost": 20,
            "manual_review_cost": 25,
            "review_fraud_catch_rate": 0.9,
            "review_legitimate_approval_rate": 0.98,
        },
    }
    return metadata, config


def test_frozen_catboost_preflight_uses_validation_provenance() -> None:
    metadata, config = frozen_artifacts()

    spec = validate_frozen_evaluation_spec(
        metadata,
        config,
        saved_model_feature_names=["TransactionAmt", "ProductCD"],
        saved_model_tree_count=17,
    )

    assert spec.model_name == "CatBoostClassifier"
    assert spec.review_threshold == 0.2
    assert spec.block_threshold == 0.6
    assert spec.assumptions.currency == "INR"


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    [
        ("metadata", "held_out_test_accessed", True, "held-out test remained sealed"),
        ("config", "selection_split", "test", "validation-only provenance"),
        ("config", "model_version", "other", "versions do not match"),
        ("config", "block_threshold", 0.1, "0 <= review < block <= 1"),
    ],
)
def test_frozen_catboost_preflight_rejects_integrity_failures(
    target: str,
    field: str,
    value: object,
    message: str,
) -> None:
    metadata, config = frozen_artifacts()
    selected = metadata if target == "metadata" else config
    selected[field] = value

    with pytest.raises(ValueError, match=message):
        validate_frozen_evaluation_spec(
            metadata,
            config,
            saved_model_feature_names=["TransactionAmt", "ProductCD"],
            saved_model_tree_count=17,
        )


def test_frozen_catboost_preflight_requires_exact_saved_schema() -> None:
    metadata, config = frozen_artifacts()

    with pytest.raises(ValueError, match="feature order"):
        validate_frozen_evaluation_spec(
            metadata,
            config,
            saved_model_feature_names=["ProductCD", "TransactionAmt"],
            saved_model_tree_count=17,
        )


def test_final_score_evaluation_applies_frozen_decision_boundaries() -> None:
    metadata, config = frozen_artifacts()
    spec = validate_frozen_evaluation_spec(
        metadata,
        config,
        saved_model_feature_names=["TransactionAmt", "ProductCD"],
        saved_model_tree_count=17,
    )

    binary, costs, decisions = evaluate_frozen_scores(
        labels=np.array([0, 1, 1, 0]),
        amounts=np.array([100.0, 200.0, 300.0, 400.0]),
        risk_scores=np.array([0.1, 0.2, 0.6, 0.9]),
        spec=spec,
    )

    assert decisions.tolist() == ["APPROVE", "REVIEW", "BLOCK", "BLOCK"]
    assert binary["true_positives"] == 1
    assert binary["false_positives"] == 1
    assert costs["review_count"] == 1
    assert costs["block_count"] == 2


def test_input_artifacts_are_not_mutated_by_preflight() -> None:
    metadata, config = frozen_artifacts()
    original_metadata = deepcopy(metadata)
    original_config = deepcopy(config)

    validate_frozen_evaluation_spec(
        metadata,
        config,
        saved_model_feature_names=["TransactionAmt", "ProductCD"],
        saved_model_tree_count=17,
    )

    assert metadata == original_metadata
    assert config == original_config
