from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib
import pandas as pd
from merchantshield_ml.explain import top_contributions
from sqlalchemy.orm import Session

from ..config import get_settings
from ..schemas.risk import Factor, ScoreRequest, ScoreResponse
from .artifacts import ArtifactUnavailable, load_model_metadata
from .decision_engine import combine_decisions, decision_from_score
from .evidence_store import sync_evidence_artifacts
from .repository import persist_scored_transaction
from .rules_engine import evaluate_rules, load_rules


@lru_cache(maxsize=1)
def _load_model_bundle() -> dict[str, Any]:
    path = get_settings().model_path
    if not path.is_file():
        raise ArtifactUnavailable("Frozen model artifact is not available")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ArtifactUnavailable("Model artifact does not match the expected bundle schema")
    return bundle


def score_transaction(payload: ScoreRequest, db: Session) -> ScoreResponse:
    metadata = load_model_metadata()
    bundle = _load_model_bundle()
    expected_features = list(metadata.get("feature_names") or [])
    if not expected_features:
        raise ArtifactUnavailable("Model metadata does not contain a feature schema")
    missing = sorted(set(expected_features).difference(payload.features))
    unexpected = sorted(set(payload.features).difference(expected_features))
    if missing or unexpected:
        detail = []
        if missing:
            detail.append(f"missing: {', '.join(missing)}")
        if unexpected:
            detail.append(f"unexpected: {', '.join(unexpected)}")
        raise ValueError("Feature schema mismatch (" + "; ".join(detail) + ")")

    frame = pd.DataFrame([{name: payload.features[name] for name in expected_features}])
    pipeline = bundle["pipeline"]
    probability = float(pipeline.predict_proba(frame)[0, 1])
    thresholds = metadata["thresholds"]
    model_decision = decision_from_score(probability, float(thresholds["review"]), float(thresholds["block"]))
    hits = evaluate_rules(payload.features, load_rules(get_settings().rules_path))
    actions = [hit.action for hit in hits if hit.action != "NONE"]
    final_decision = combine_decisions(model_decision, actions)
    factors = [Factor(**item) for item in top_contributions(pipeline, frame, limit=5)]
    version = str(metadata["model_version"])
    config_id = str(metadata.get("threshold_config_id", "frozen-thresholds"))

    if payload.persist:
        if not payload.transaction_id:
            raise ValueError("transaction_id is required when persist=true")
        model_run, threshold_config = sync_evidence_artifacts(db)
        persist_scored_transaction(
            db,
            transaction_id=payload.transaction_id,
            transaction_dt=int(payload.features.get("TransactionDT") or 0),
            amount=float(payload.features.get("TransactionAmt") or 0),
            risk_score=probability,
            decision=final_decision,
            model_run=model_run,
            threshold_config=threshold_config,
            rule_hits=hits,
            factors=factors,
        )

    return ScoreResponse(
        risk_score=probability,
        decision=final_decision,
        rules_triggered=[hit.rule_id for hit in hits],
        top_factors=factors,
        model_version=version,
        threshold_config_id=config_id,
    )
