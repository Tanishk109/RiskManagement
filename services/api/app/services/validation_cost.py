from __future__ import annotations

import json
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from merchantshield_ml.cost import CostAssumptions, simulate_cost, simulate_decisions

from .artifacts import ArtifactUnavailable
from .project_artifacts import ProjectArtifactService, get_project_artifact_service

METRIC_FIELDS = (
    "cost_output_label",
    "currency",
    "transaction_count",
    "fraud_count",
    "legitimate_count",
    "approve_count",
    "review_count",
    "block_count",
    "approve_rate",
    "review_rate",
    "block_rate",
    "fraud_approved",
    "fraud_reviewed",
    "fraud_blocked",
    "fraud_approve_rate",
    "fraud_review_rate",
    "fraud_block_rate",
    "legitimate_approved",
    "legitimate_reviewed",
    "legitimate_blocked",
    "block_precision",
    "block_recall",
    "detected_precision",
    "detected_fraud_recall",
    "false_positives",
    "false_negatives",
    "total_fraud_amount",
    "fraud_amount_approved",
    "fraud_amount_reviewed",
    "fraud_amount_blocked",
    "expected_review_caught_fraud_amount",
    "captured_fraud_amount",
    "fraud_amount_capture_rate",
    "approved_fraud_loss",
    "reviewed_fraud_loss",
    "blocked_legitimate_cost",
    "reviewed_legitimate_cost",
    "fraud_loss",
    "false_positive_cost",
    "manual_review_cost_total",
    "review_expected_residual_cost",
    "review_total_cost",
    "total_estimated_cost",
)


def _required_path(path: Path | None, label: str) -> Path:
    if path is None or not path.is_file():
        raise ArtifactUnavailable(f"Required validation cost artifact is not available: {label}")
    return path


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactUnavailable(f"Validation cost artifact is unreadable: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ArtifactUnavailable(f"Validation cost artifact must contain an object: {path.name}")
    return payload


def _native(value: Any) -> str | int | float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return str(value)


class ValidationCostService:
    """Validation-only threshold simulation over immutable saved predictions."""

    def __init__(self, project: ProjectArtifactService):
        self.project = project

    @cached_property
    def operating_config(self) -> dict[str, Any]:
        path = _required_path(self.project.paths.operating_config, "validation_operating_config.json")
        payload = _json(path)
        self._validate_provenance(payload, path.name)
        if payload.get("not_final") is not True:
            raise ArtifactUnavailable("Validation operating configuration is not marked provisional")
        return payload

    @cached_property
    def analysis(self) -> dict[str, Any]:
        path = _required_path(self.project.paths.threshold_analysis, "threshold_analysis.json")
        payload = _json(path)
        self._validate_provenance(payload, path.name)
        if payload.get("status") != "provisional_validation_analysis":
            raise ArtifactUnavailable("Threshold analysis is not a provisional validation artifact")
        return payload

    @cached_property
    def scenario_config(self) -> dict[str, Any]:
        path = _required_path(self.project.paths.merchant_scenarios, "merchant_scenarios.yaml")
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ArtifactUnavailable("Merchant scenario configuration is unreadable") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("scenarios"), dict):
            raise ArtifactUnavailable("Merchant scenario configuration has an invalid schema")
        if payload.get("assumption_status") != "ILLUSTRATIVE MERCHANT ASSUMPTIONS":
            raise ArtifactUnavailable("Merchant scenarios are not labelled as illustrative assumptions")
        return payload

    @cached_property
    def threshold_grid(self) -> pd.DataFrame:
        path = _required_path(self.project.paths.threshold_grid, "threshold_grid.parquet")
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            raise ArtifactUnavailable("Threshold grid is unreadable") from exc
        required = {
            "analysis_type",
            "scenario_id",
            "review_threshold",
            "block_threshold",
            "review_rate",
            "total_estimated_cost",
        }
        if not required.issubset(frame.columns):
            raise ArtifactUnavailable("Threshold grid has an invalid schema")
        scenarios = frame.loc[frame["analysis_type"] == "merchant_scenario"]
        expected_rows = sum(
            int(scenario["evaluated_configuration_count"])
            for scenario in self.analysis["scenarios"].values()
        )
        if len(scenarios) != expected_rows:
            raise ArtifactUnavailable("Threshold grid row count does not match the analysis artifact")
        return frame

    @staticmethod
    def _validate_provenance(payload: dict[str, Any], name: str) -> None:
        if payload.get("selection_split") != "validation":
            raise ArtifactUnavailable(f"{name} is not a validation-only artifact")
        if payload.get("held_out_test_accessed") is not False:
            raise ArtifactUnavailable(f"{name} does not prove that the held-out test stayed sealed")

    @cached_property
    def scenarios_by_id(self) -> dict[str, dict[str, Any]]:
        configured = self.scenario_config["scenarios"]
        analyzed = self.analysis.get("scenarios")
        if not isinstance(analyzed, dict) or set(configured) != set(analyzed):
            raise ArtifactUnavailable("Scenario configuration does not match threshold analysis")
        for scenario_id, scenario in configured.items():
            if scenario.get("assumptions") != analyzed[scenario_id].get("assumptions"):
                raise ArtifactUnavailable(f"Assumption mismatch for scenario {scenario_id}")
        return configured

    def list_scenarios(self) -> dict[str, Any]:
        scenarios = []
        for scenario_id, config in self.scenarios_by_id.items():
            analyzed = self.analysis["scenarios"][scenario_id]
            lowest = analyzed["lowest_estimated_cost"]
            scenarios.append(
                {
                    "id": scenario_id,
                    "name": config["name"],
                    "description": config["description"],
                    "assumptions": config["assumptions"],
                    "validation_configuration": {
                        "review_threshold": float(lowest["review_threshold"]),
                        "block_threshold": float(lowest["block_threshold"]),
                    },
                }
            )
        return {
            "status": "provisional_validation_analysis",
            "split": "validation",
            "held_out_test_status": "sealed_not_evaluated",
            "assumption_status": self.scenario_config["assumption_status"],
            "cost_output_label": self.scenario_config["cost_output_label"],
            "default_scenario_id": self.operating_config["scenario"],
            "default_review_threshold": float(self.operating_config["review_threshold"]),
            "default_block_threshold": float(self.operating_config["block_threshold"]),
            "default_review_capacity": float(self.operating_config["review_capacity_limit"]),
            "review_capacities": self.scenario_config["review_capacity_rates"],
            "scenarios": scenarios,
        }

    def _assumptions(self, scenario_id: str) -> CostAssumptions:
        try:
            values = self.scenarios_by_id[scenario_id]["assumptions"]
        except KeyError as exc:
            raise ValueError(f"Unknown merchant scenario: {scenario_id}") from exc
        return CostAssumptions(**values)

    @cached_property
    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frame = self.project.validation_frame
        return (
            frame["actual_label"].to_numpy(dtype=int),
            frame["TransactionAmt"].to_numpy(dtype=float),
            frame["fraud_probability"].to_numpy(dtype=float),
        )

    def lowest_cost_feasible(
        self, scenario_id: str, review_capacity: float | None
    ) -> dict[str, Any]:
        if scenario_id not in self.scenarios_by_id:
            raise ValueError(f"Unknown merchant scenario: {scenario_id}")
        frame = self.threshold_grid.loc[
            (self.threshold_grid["analysis_type"] == "merchant_scenario")
            & (self.threshold_grid["scenario_id"] == scenario_id)
        ]
        if review_capacity is not None:
            frame = frame.loc[frame["review_rate"] <= review_capacity + 1e-12]
        if frame.empty:
            raise ValueError("No validation threshold configuration meets this review capacity")
        row = frame.sort_values(
            ["total_estimated_cost", "review_rate", "block_rate", "review_threshold", "block_threshold"],
            kind="stable",
        ).iloc[0]
        return {key: _native(row[key]) for key in METRIC_FIELDS if key in row.index} | {
            "review_threshold": float(row["review_threshold"]),
            "block_threshold": float(row["block_threshold"]),
        }

    def simulate(
        self,
        *,
        scenario_id: str,
        review_threshold: float,
        block_threshold: float,
        review_capacity: float | None,
    ) -> dict[str, Any]:
        assumptions = self._assumptions(scenario_id)
        labels, amounts, scores = self.arrays
        metrics = simulate_cost(
            labels=labels,
            amounts=amounts,
            risk_scores=scores,
            review_threshold=review_threshold,
            block_threshold=block_threshold,
            assumptions=assumptions,
        )
        approve_all = simulate_decisions(
            labels=labels,
            amounts=amounts,
            decisions=np.full(len(labels), "APPROVE"),
            assumptions=assumptions,
        )
        binary = simulate_decisions(
            labels=labels,
            amounts=amounts,
            decisions=np.where(scores >= self.project.classification_threshold, "BLOCK", "APPROVE"),
            assumptions=assumptions,
        )
        lowest = self.lowest_cost_feasible(scenario_id, review_capacity)
        capacity_met = review_capacity is None or metrics["review_rate"] <= review_capacity + 1e-12
        baseline_cost = float(approve_all["total_estimated_cost"])
        binary_cost = float(binary["total_estimated_cost"])
        current_cost = float(metrics["total_estimated_cost"])
        return {
            "status": "provisional_validation_simulation",
            "split": "validation",
            "held_out_test_status": "sealed_not_evaluated",
            "provisional": True,
            "assumption_status": self.scenario_config["assumption_status"],
            "cost_output_label": self.scenario_config["cost_output_label"],
            "scenario": {
                "id": scenario_id,
                "name": self.scenarios_by_id[scenario_id]["name"],
                "description": self.scenarios_by_id[scenario_id]["description"],
                "assumptions": self.scenarios_by_id[scenario_id]["assumptions"],
            },
            "review_threshold": review_threshold,
            "block_threshold": block_threshold,
            "review_capacity": review_capacity,
            "capacity_met": capacity_met,
            "metrics": metrics,
            "policy_comparison": [
                {"policy": "Approve all", "total_estimated_cost": baseline_cost},
                {
                    "policy": f"Binary model @ {self.project.classification_threshold:.2f}",
                    "total_estimated_cost": binary_cost,
                },
                {"policy": "Three-way policy", "total_estimated_cost": current_cost},
            ],
            "estimated_reduction_vs_approve_all": (
                (baseline_cost - current_cost) / baseline_cost if baseline_cost else 0.0
            ),
            "estimated_reduction_vs_binary": (
                (binary_cost - current_cost) / binary_cost if binary_cost else 0.0
            ),
            "lowest_cost_feasible": lowest,
            "sensitivity_analysis": self.analysis["sensitivity_analysis"],
            "failure_slices": self.analysis["failure_slices"],
            "high_value_fraud": self.analysis["high_value_fraud"],
            "provenance": (
                "Recomputed from saved chronological validation predictions only. "
                "Costs use illustrative merchant assumptions; the final test remains sealed."
            ),
        }

    def summary(self) -> dict[str, Any]:
        config = self.operating_config
        result = self.simulate(
            scenario_id=str(config["scenario"]),
            review_threshold=float(config["review_threshold"]),
            block_threshold=float(config["block_threshold"]),
            review_capacity=float(config["review_capacity_limit"]),
        )
        expected = config["validation_metrics"]
        for field in ("approve_count", "review_count", "block_count", "total_estimated_cost"):
            if not np.isclose(float(result["metrics"][field]), float(expected[field]), atol=1e-8):
                raise ArtifactUnavailable(f"Validation operating metric mismatch: {field}")
        result.update(
            {
                "selection_reason": config["selection_reason"],
                "limitations": config["limitations"],
                "sensitivity_analysis": self.analysis["sensitivity_analysis"],
                "failure_slices": self.analysis["failure_slices"],
                "high_value_fraud": self.analysis["high_value_fraud"],
            }
        )
        return result

    def residual_risk(self) -> dict[str, Any]:
        summary = self.summary()
        metrics = summary["metrics"]
        return {
            "status": "provisional_validation_analysis",
            "split": "validation",
            "held_out_test_status": "sealed_not_evaluated",
            "scenario": summary["scenario"],
            "thresholds": {
                "review": summary["review_threshold"],
                "block": summary["block_threshold"],
            },
            "approved_fraud": {
                "count": metrics["fraud_approved"],
                "transaction_amount": metrics["fraud_amount_approved"],
                "estimated_loss": metrics["approved_fraud_loss"],
            },
            "fraud_detection": {
                "fraud_count_detected": metrics["fraud_reviewed"] + metrics["fraud_blocked"],
                "fraud_count_total": metrics["fraud_count"],
                "count_detection_rate": metrics["detected_fraud_recall"],
                "captured_fraud_amount": metrics["captured_fraud_amount"],
                "total_fraud_amount": metrics["total_fraud_amount"],
                "amount_capture_rate": metrics["fraud_amount_capture_rate"],
            },
            "failure_slices": self.analysis["failure_slices"],
            "high_value_fraud": self.analysis["high_value_fraud"],
            "sensitivity_analysis": self.analysis["sensitivity_analysis"],
            "provenance": summary["provenance"],
        }


@lru_cache(maxsize=1)
def get_validation_cost_service() -> ValidationCostService:
    return ValidationCostService(get_project_artifact_service())
