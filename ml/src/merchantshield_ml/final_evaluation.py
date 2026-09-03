from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .cost import (
    CostAssumptions,
    decisions_from_scores,
    simulate_cost,
    validate_thresholds,
)
from .evaluate import binary_metrics


@dataclass(frozen=True)
class FrozenEvaluationSpec:
    model_name: str
    model_version: str
    feature_set: str
    feature_names: tuple[str, ...]
    categorical_feature_names: tuple[str, ...]
    review_threshold: float
    block_threshold: float
    scenario: str
    scenario_name: str
    selection_reason: str
    assumptions: CostAssumptions


def validate_frozen_evaluation_spec(
    model_metadata: dict[str, Any],
    operating_config: dict[str, Any],
    *,
    saved_model_feature_names: list[str],
    saved_model_tree_count: int,
) -> FrozenEvaluationSpec:
    """Validate TRAIN-fitted model and VALIDATION-selected policy metadata only."""

    if model_metadata.get("status") != "validation_candidate":
        raise ValueError("Model metadata is not the frozen validation candidate")
    if model_metadata.get("held_out_test_accessed") is not False:
        raise ValueError("Model metadata does not prove that the held-out test remained sealed")
    if model_metadata.get("model_name") != "CatBoostClassifier":
        raise ValueError("Final evaluation requires the selected CatBoost candidate")

    if (
        operating_config.get("status") != "provisional_validation_config"
        or operating_config.get("selection_split") != "validation"
        or operating_config.get("held_out_test_accessed") is not False
        or operating_config.get("not_final") is not True
    ):
        raise ValueError("Operating configuration does not satisfy validation-only provenance")

    model_version = str(model_metadata.get("model_version", ""))
    feature_set = str(model_metadata.get("feature_set", ""))
    if operating_config.get("model_version") != model_version:
        raise ValueError("Model and operating-configuration versions do not match")
    if operating_config.get("feature_set") != feature_set:
        raise ValueError("Model and operating-configuration feature sets do not match")

    feature_names = tuple(model_metadata.get("feature_names") or ())
    categorical_names = tuple(model_metadata.get("categorical_feature_names") or ())
    if not feature_names or len(feature_names) != len(set(feature_names)):
        raise ValueError("Frozen feature schema must be non-empty and contain no duplicates")
    if not set(categorical_names).issubset(feature_names):
        raise ValueError("Categorical feature schema is not a subset of the frozen feature schema")
    if list(feature_names) != list(saved_model_feature_names):
        raise ValueError("Saved CatBoost feature order differs from frozen metadata")
    if int(model_metadata.get("actual_tree_count", -1)) != int(saved_model_tree_count):
        raise ValueError("Saved CatBoost tree count differs from frozen metadata")

    review_threshold = float(operating_config["review_threshold"])
    block_threshold = float(operating_config["block_threshold"])
    validate_thresholds(review_threshold, block_threshold)

    raw_assumptions = operating_config.get("cost_assumptions")
    if not isinstance(raw_assumptions, dict):
        raise TypeError("Operating configuration is missing merchant cost assumptions")
    assumptions = CostAssumptions(**raw_assumptions)
    if operating_config.get("assumption_status") != "ILLUSTRATIVE MERCHANT ASSUMPTIONS":
        raise ValueError("Merchant cost assumptions are not labelled as illustrative")

    return FrozenEvaluationSpec(
        model_name="CatBoostClassifier",
        model_version=model_version,
        feature_set=feature_set,
        feature_names=feature_names,
        categorical_feature_names=categorical_names,
        review_threshold=review_threshold,
        block_threshold=block_threshold,
        scenario=str(operating_config["scenario"]),
        scenario_name=str(operating_config["scenario_name"]),
        selection_reason=str(operating_config["selection_reason"]),
        assumptions=assumptions,
    )


def evaluate_frozen_scores(
    *,
    labels: np.ndarray,
    amounts: np.ndarray,
    risk_scores: np.ndarray,
    spec: FrozenEvaluationSpec,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    """Evaluate already-produced scores without fitting or tuning anything."""

    scores = np.asarray(risk_scores, dtype=float)
    binary = binary_metrics(labels, scores, threshold=spec.block_threshold)
    costs = simulate_cost(
        labels=labels,
        amounts=amounts,
        risk_scores=scores,
        review_threshold=spec.review_threshold,
        block_threshold=spec.block_threshold,
        assumptions=spec.assumptions,
    )
    decisions = decisions_from_scores(
        scores,
        spec.review_threshold,
        spec.block_threshold,
    )
    return binary, costs, decisions
