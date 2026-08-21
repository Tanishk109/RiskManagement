from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .cost import CostAssumptions, simulate_cost


def search_thresholds(
    *,
    labels: np.ndarray,
    amounts: np.ndarray,
    risk_scores: np.ndarray,
    assumptions: CostAssumptions,
    review_candidates: Iterable[float] | None = None,
    block_candidates: Iterable[float] | None = None,
) -> dict[str, object]:
    """Search validation predictions only; callers enforce the split policy."""
    review_values = list(review_candidates if review_candidates is not None else np.arange(0.10, 0.81, 0.05))
    block_values = list(block_candidates if block_candidates is not None else np.arange(0.20, 1.00, 0.05))
    rows: list[dict[str, float | int]] = []
    for review_threshold in review_values:
        for block_threshold in block_values:
            if review_threshold >= block_threshold:
                continue
            outcome = simulate_cost(
                labels=labels,
                amounts=amounts,
                risk_scores=risk_scores,
                review_threshold=float(review_threshold),
                block_threshold=float(block_threshold),
                assumptions=assumptions,
            )
            rows.append({
                "review_threshold": round(float(review_threshold), 6),
                "block_threshold": round(float(block_threshold), 6),
                **outcome,
            })
    if not rows:
        raise ValueError("Threshold search produced no valid review/block pairs")
    lowest = min(rows, key=lambda row: float(row["total_estimated_cost"]))
    return {
        "selection_split": "validation",
        "selection_wording": "Lowest estimated cost under the currently selected merchant assumptions.",
        "lowest_estimated_cost_configuration": lowest,
        "configurations": rows,
    }
