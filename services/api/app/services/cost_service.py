from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pandas as pd
from merchantshield_ml.cost import CostAssumptions as MLCostAssumptions
from merchantshield_ml.cost import simulate_cost
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import CostSimulation, ThresholdConfig
from ..schemas.risk import CostOutcome, CostSimulationRequest, CostSimulationResponse
from .artifacts import ArtifactUnavailable
from .evidence_store import active_threshold, latest_model_run, upsert_cost_config

REQUIRED_PREDICTION_COLUMNS = {"TransactionAmt", "isFraud", "risk_score"}


def _outcome(result: dict[str, float | int]) -> CostOutcome:
    return CostOutcome(**{field: result[field] for field in CostOutcome.model_fields})


def _history_row(
    *,
    group_id: str,
    scenario: str,
    cost_config_id: int,
    threshold_config: ThresholdConfig | None,
    transaction_count: int,
    review_threshold: float,
    block_threshold: float,
    result: dict[str, float | int],
) -> CostSimulation:
    return CostSimulation(
        simulation_group_id=group_id,
        scenario=scenario,
        cost_config_id=cost_config_id,
        threshold_config_id=threshold_config.id if threshold_config else None,
        evaluation_split="test",
        transaction_count=transaction_count,
        review_threshold=review_threshold,
        block_threshold=block_threshold,
        precision=float(result["precision"]),
        recall=float(result["recall"]),
        false_positives=int(result["false_positives"]),
        false_negatives=int(result["false_negatives"]),
        review_volume=int(result["review_volume"]),
        block_volume=int(result["block_volume"]),
        approve_volume=int(result["approve_volume"]),
        fraud_loss=Decimal(str(result["fraud_loss"])),
        false_positive_cost=Decimal(str(result["false_positive_cost"])),
        review_cost=Decimal(str(result["review_cost"])),
        total_estimated_cost=Decimal(str(result["total_estimated_cost"])),
    )


def simulate_from_held_out(payload: CostSimulationRequest, db: Session) -> CostSimulationResponse:
    settings = get_settings()
    if not settings.predictions_path.is_file():
        raise ArtifactUnavailable("Held-out prediction artifact is not available")
    frame = pd.read_csv(settings.predictions_path)
    missing = sorted(REQUIRED_PREDICTION_COLUMNS.difference(frame.columns))
    if missing:
        raise ArtifactUnavailable(f"Held-out prediction artifact is missing columns: {', '.join(missing)}")

    model_run = latest_model_run(db)
    threshold = active_threshold(model_run) if model_run else None
    if model_run is None or model_run.evaluation_status != "COMPLETE" or threshold is None:
        raise ArtifactUnavailable("Held-out model evidence has not been synced to PostgreSQL")

    assumption_values = payload.assumptions.model_dump()
    assumptions = MLCostAssumptions(**assumption_values)
    current = simulate_cost(
        labels=frame["isFraud"].to_numpy(),
        amounts=frame["TransactionAmt"].to_numpy(),
        risk_scores=frame["risk_score"].to_numpy(),
        review_threshold=threshold.review_threshold,
        block_threshold=threshold.block_threshold,
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

    cost_config = upsert_cost_config(db, assumption_values, name_prefix="simulation")
    group_id = str(uuid4())
    db.add_all(
        [
            _history_row(
                group_id=group_id,
                scenario="CURRENT",
                cost_config_id=cost_config.id,
                threshold_config=threshold,
                transaction_count=len(frame),
                review_threshold=threshold.review_threshold,
                block_threshold=threshold.block_threshold,
                result=current,
            ),
            _history_row(
                group_id=group_id,
                scenario="PROPOSED",
                cost_config_id=cost_config.id,
                threshold_config=None,
                transaction_count=len(frame),
                review_threshold=payload.review_threshold,
                block_threshold=payload.block_threshold,
                result=proposed,
            ),
        ]
    )
    db.commit()
    return CostSimulationResponse(
        evaluated=True,
        provenance=(
            "Calculated from the held-out temporal test predictions using the submitted merchant assumptions; "
            "the scenario was recorded in PostgreSQL."
        ),
        current=_outcome(current),
        proposed=_outcome(proposed),
        simulation_group_id=group_id,
    )
