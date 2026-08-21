from __future__ import annotations

import pandas as pd
from merchantshield_ml.cost import CostAssumptions as MLCostAssumptions
from merchantshield_ml.cost import simulate_cost

from ..config import get_settings
from ..schemas.risk import CostOutcome, CostSimulationRequest, CostSimulationResponse
from .artifacts import ArtifactUnavailable, load_model_metadata

REQUIRED_PREDICTION_COLUMNS = {"TransactionAmt", "isFraud", "risk_score"}


def _outcome(result: dict[str, float | int]) -> CostOutcome:
    return CostOutcome(**{field: result[field] for field in CostOutcome.model_fields})


def simulate_from_held_out(payload: CostSimulationRequest) -> CostSimulationResponse:
    settings = get_settings()
    if not settings.predictions_path.is_file():
        raise ArtifactUnavailable("Held-out prediction artifact is not available")
    frame = pd.read_csv(settings.predictions_path)
    missing = sorted(REQUIRED_PREDICTION_COLUMNS.difference(frame.columns))
    if missing:
        raise ArtifactUnavailable(f"Held-out prediction artifact is missing columns: {', '.join(missing)}")

    assumptions = MLCostAssumptions(**payload.assumptions.model_dump())
    metadata = load_model_metadata()
    frozen = metadata["thresholds"]
    current = simulate_cost(
        labels=frame["isFraud"].to_numpy(),
        amounts=frame["TransactionAmt"].to_numpy(),
        risk_scores=frame["risk_score"].to_numpy(),
        review_threshold=float(frozen["review"]),
        block_threshold=float(frozen["block"]),
        assumptions=assumptions,
    )
    proposed = simulate_cost(
        labels=frame["isFraud"].to_numpy(),
        amounts=frame["TransactionAmt"].to_numpy(),
        risk_scores=frame["risk_score"].to_numpy(),
        review_threshold=payload.review_threshold,
        block_threshold=payload.block_threshold,
        assumptions=assumptions,
    )
    return CostSimulationResponse(
        evaluated=True,
        provenance="Calculated from the held-out temporal test predictions using the submitted merchant assumptions.",
        current=_outcome(current),
        proposed=_outcome(proposed),
    )
