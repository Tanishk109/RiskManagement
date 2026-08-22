from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .cost import CostAssumptions, simulate_cost

THRESHOLD_GRID_REQUIRED_COLUMNS = frozenset(
    {
        "review_threshold",
        "block_threshold",
        "review_rate",
        "block_rate",
        "block_precision",
        "detected_fraud_recall",
        "false_positives",
        "false_negatives",
        "fraud_loss",
        "false_positive_cost",
        "manual_review_cost_total",
        "total_estimated_cost",
    }
)


def threshold_candidates(start: float, stop: float, step: float) -> list[float]:
    """Return an inclusive decimal grid without floating-point accumulation drift."""
    if not 0 <= start <= stop <= 1:
        raise ValueError("threshold bounds must satisfy 0 <= start <= stop <= 1")
    if not 0 < step <= 1:
        raise ValueError("threshold step must be greater than 0 and at most 1")
    count = int(np.floor((stop - start) / step + 1e-12))
    values = [round(float(start + index * step), 10) for index in range(count + 1)]
    if values[-1] < stop and np.isclose(values[-1] + step, stop):
        values.append(round(float(stop), 10))
    return values


def evaluate_threshold_grid(
    *,
    labels: np.ndarray,
    amounts: np.ndarray,
    risk_scores: np.ndarray,
    assumptions: CostAssumptions,
    review_candidates: Iterable[float],
    block_candidates: Iterable[float],
    minimum_gap: float = 0.0,
) -> list[dict[str, Any]]:
    """Evaluate every valid threshold pair in deterministic ascending order."""
    if not 0 <= minimum_gap < 1:
        raise ValueError("minimum_gap must be between 0 (inclusive) and 1 (exclusive)")
    review_values = sorted({round(float(value), 10) for value in review_candidates})
    block_values = sorted({round(float(value), 10) for value in block_candidates})
    rows: list[dict[str, Any]] = []
    for review_threshold in review_values:
        for block_threshold in block_values:
            if block_threshold - review_threshold + 1e-12 < minimum_gap:
                continue
            if review_threshold >= block_threshold:
                continue
            outcome = simulate_cost(
                labels=labels,
                amounts=amounts,
                risk_scores=risk_scores,
                review_threshold=review_threshold,
                block_threshold=block_threshold,
                assumptions=assumptions,
            )
            rows.append(
                {
                    "review_threshold": review_threshold,
                    "block_threshold": block_threshold,
                    **outcome,
                }
            )
    if not rows:
        raise ValueError("Threshold search produced no valid review/block pairs")
    return rows


def select_lowest_cost(
    rows: Sequence[dict[str, Any]],
    *,
    max_review_rate: float | None = None,
) -> dict[str, Any]:
    """Select the lowest-cost row, resolving ties toward less operational friction."""
    if max_review_rate is not None and not 0 <= max_review_rate <= 1:
        raise ValueError("max_review_rate must be between 0 and 1")
    feasible = [
        row
        for row in rows
        if max_review_rate is None
        or float(row["review_rate"]) <= max_review_rate + 1e-12
    ]
    if not feasible:
        raise ValueError("No threshold configuration satisfies the review capacity")
    return min(
        feasible,
        key=lambda row: (
            float(row["total_estimated_cost"]),
            float(row["review_rate"]),
            float(row["block_rate"]),
            float(row["review_threshold"]),
            float(row["block_threshold"]),
        ),
    )


def select_highest_metric(
    rows: Sequence[dict[str, Any]],
    metric: str,
    *,
    max_review_rate: float | None = None,
    minimum_detected_recall: float | None = None,
) -> dict[str, Any] | None:
    """Select a frontier row with deterministic cost/friction tie-breakers."""
    feasible = list(rows)
    if max_review_rate is not None:
        feasible = [
            row
            for row in feasible
            if float(row["review_rate"]) <= max_review_rate + 1e-12
        ]
    if minimum_detected_recall is not None:
        feasible = [
            row
            for row in feasible
            if float(row["detected_fraud_recall"]) + 1e-12 >= minimum_detected_recall
        ]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda row: (
            -float(row[metric]),
            float(row["total_estimated_cost"]),
            float(row["review_rate"]),
            float(row["block_rate"]),
            float(row["review_threshold"]),
            float(row["block_threshold"]),
        ),
    )


def select_lowest_false_positive_cost(
    rows: Sequence[dict[str, Any]],
    *,
    minimum_detected_recall: float,
) -> dict[str, Any] | None:
    feasible = [
        row
        for row in rows
        if float(row["detected_fraud_recall"]) + 1e-12 >= minimum_detected_recall
    ]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda row: (
            float(row["false_positive_cost"]),
            float(row["total_estimated_cost"]),
            float(row["review_rate"]),
            float(row["block_rate"]),
            float(row["review_threshold"]),
            float(row["block_threshold"]),
        ),
    )


def capacity_selections(
    rows: Sequence[dict[str, Any]],
    capacities: Iterable[float | None],
) -> list[dict[str, Any]]:
    selections: list[dict[str, Any]] = []
    for capacity in capacities:
        selected = select_lowest_cost(rows, max_review_rate=capacity)
        feasible_count = sum(
            capacity is None or float(row["review_rate"]) <= capacity + 1e-12
            for row in rows
        )
        selections.append(
            {
                "review_capacity": capacity,
                "feasible_configuration_count": feasible_count,
                "selected_configuration": selected,
            }
        )
    return selections


def validate_threshold_grid_artifact(frame: pd.DataFrame) -> None:
    missing = sorted(THRESHOLD_GRID_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            f"Threshold grid artifact is missing columns: {', '.join(missing)}"
        )
    if frame.empty:
        raise ValueError("Threshold grid artifact cannot be empty")
    key_columns = [
        column
        for column in (
            "analysis_type",
            "scenario_id",
            "sensitivity_parameter",
            "sensitivity_value",
        )
        if column in frame.columns
    ] + ["review_threshold", "block_threshold"]
    if frame.duplicated(subset=key_columns).any():
        raise ValueError(
            "Threshold grid artifact contains duplicate threshold configurations"
        )
    if not (frame["review_threshold"] < frame["block_threshold"]).all():
        raise ValueError("Threshold grid artifact contains invalid threshold ordering")
    if not frame["review_rate"].between(0, 1).all():
        raise ValueError("Threshold grid artifact contains an invalid review rate")


def search_thresholds(
    *,
    labels: np.ndarray,
    amounts: np.ndarray,
    risk_scores: np.ndarray,
    assumptions: CostAssumptions,
    review_candidates: Iterable[float] | None = None,
    block_candidates: Iterable[float] | None = None,
) -> dict[str, object]:
    """Compatibility wrapper for validation-only threshold selection."""
    review_values = review_candidates or threshold_candidates(0.10, 0.80, 0.05)
    block_values = block_candidates or threshold_candidates(0.20, 0.95, 0.05)
    rows = evaluate_threshold_grid(
        labels=labels,
        amounts=amounts,
        risk_scores=risk_scores,
        assumptions=assumptions,
        review_candidates=review_values,
        block_candidates=block_values,
    )
    lowest = select_lowest_cost(rows)
    return {
        "selection_split": "validation",
        "selection_wording": (
            "Lowest estimated cost under the currently selected merchant assumptions."
        ),
        "lowest_estimated_cost_configuration": lowest,
        "configurations": rows,
    }
