from __future__ import annotations

import math
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Literal

import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..models import ModelRun, ReviewCase, ThresholdConfig, Transaction
from ..schemas.risk import ReviewDecisionRequest
from .evidence_store import parse_datetime, upsert_cost_config
from .project_artifacts import ProjectArtifactService
from .validation_cost import ValidationCostService

ReviewOrder = Literal["highest_amount", "highest_risk", "fraud", "legitimate"]


class ValidationReviewService:
    """Materialize only reviewed validation examples into the operational database."""

    def __init__(self, project: ProjectArtifactService, cost: ValidationCostService):
        self.project = project
        self.cost = cost

    @property
    def review_frame(self) -> pd.DataFrame:
        config = self.cost.operating_config
        frame = self.project.validation_frame
        review = float(config["review_threshold"])
        block = float(config["block_threshold"])
        return frame.loc[
            (frame["fraud_probability"] >= review) & (frame["fraud_probability"] < block)
        ].copy()

    def _row(self, transaction_id: str) -> pd.Series:
        matched = self.review_frame.loc[
            self.review_frame["TransactionID"].astype(str) == transaction_id
        ]
        if len(matched) != 1:
            raise KeyError(transaction_id)
        return matched.iloc[0]

    def list_reviews(
        self, db: Session, *, order: ReviewOrder, page: int, page_size: int
    ) -> dict[str, Any]:
        frame = self.review_frame
        if order == "fraud":
            frame = frame.loc[frame["actual_label"] == 1].sort_values(
                ["fraud_probability", "TransactionAmt"], ascending=[False, False]
            )
        elif order == "legitimate":
            frame = frame.loc[frame["actual_label"] == 0].sort_values(
                ["fraud_probability", "TransactionAmt"], ascending=[False, False]
            )
        elif order == "highest_risk":
            frame = frame.sort_values(
                ["fraud_probability", "TransactionAmt"], ascending=[False, False]
            )
        else:
            frame = frame.sort_values(
                ["TransactionAmt", "fraud_probability"], ascending=[False, False]
            )
        total = len(frame)
        page_frame = frame.iloc[(page - 1) * page_size : page * page_size]
        ids = [str(int(value)) for value in page_frame["TransactionID"]]
        decisions: dict[str, ReviewCase] = {}
        persistence_status = "available"
        if ids:
            try:
                rows = db.scalars(
                    select(ReviewCase)
                    .join(ReviewCase.transaction)
                    .where(Transaction.transaction_id.in_(ids))
                )
                decisions = {row.transaction.transaction_id: row for row in rows}
            except SQLAlchemyError:
                db.rollback()
                persistence_status = "postgresql_unavailable"
        config = self.cost.operating_config
        return {
            "status": "validation_review_queue",
            "split": "validation",
            "held_out_test_status": "sealed_not_evaluated",
            "scenario_id": config["scenario"],
            "review_threshold": float(config["review_threshold"]),
            "block_threshold": float(config["block_threshold"]),
            "order": order,
            "page": page,
            "page_size": page_size,
            "total": total,
            "page_count": math.ceil(total / page_size) if total else 0,
            "ground_truth_hidden": True,
            "persistence_status": persistence_status,
            "items": [
                self._public_item(row, decisions.get(str(int(row["TransactionID"]))))
                for _, row in page_frame.iterrows()
            ],
            "provenance": (
                "Rows are selected from the provisional validation review band. "
                "Reviewer actions are operational records and do not change model artifacts or metrics."
            ),
        }

    @staticmethod
    def _public_item(row: pd.Series, review: ReviewCase | None) -> dict[str, Any]:
        features = {
            name: None if pd.isna(row[name]) else str(row[name])
            for name in ("ProductCD", "card4", "card6", "C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3")
        }
        return {
            "transaction_id": str(int(row["TransactionID"])),
            "transaction_dt": int(row["TransactionDT"]),
            "transaction_amount": float(row["TransactionAmt"]),
            "fraud_probability": float(row["fraud_probability"]),
            "business_decision": "REVIEW",
            "status": review.status if review else "OPEN",
            "reviewer_decision": review.reviewer_decision if review else None,
            "reviewer_note": review.reviewer_reason if review else None,
            "reviewed_at": review.reviewed_at if review else None,
            "ground_truth": None,
            "features": features,
        }

    def reveal_ground_truth(self, db: Session, transaction_id: str) -> dict[str, Any]:
        row = self._row(transaction_id)
        try:
            persisted = db.scalar(
                select(ReviewCase).join(ReviewCase.transaction).where(
                    Transaction.transaction_id == transaction_id
                )
            )
        except SQLAlchemyError:
            db.rollback()
            persisted = None
        actual = int(row["actual_label"])
        return {
            "transaction_id": transaction_id,
            "split": "validation",
            "actual_label": actual,
            "ground_truth": "FRAUD" if actual == 1 else "LEGITIMATE",
            "reviewer_decision": persisted.reviewer_decision if persisted else None,
            "reviewer_correct": (
                None
                if persisted is None or persisted.reviewer_decision is None
                else (persisted.reviewer_decision == "BLOCK") == bool(actual)
            ),
            "note": "Ground truth is revealed on explicit request and was never used to make the review decision.",
        }

    def decide(
        self,
        db: Session,
        *,
        transaction_id: str,
        payload: ReviewDecisionRequest,
    ) -> dict[str, Any]:
        row = self._row(transaction_id)
        review = db.scalar(
            select(ReviewCase).join(ReviewCase.transaction).where(
                Transaction.transaction_id == transaction_id
            )
        )
        if review is not None and review.status != "OPEN":
            raise ValueError("Review case has already been decided")
        if review is None:
            transaction = db.scalar(
                select(Transaction).where(Transaction.transaction_id == transaction_id)
            )
            if transaction is not None and transaction.source != "VALIDATION_REVIEW_DEMO":
                raise ValueError("Transaction ID already belongs to a different operational source")
            if transaction is None:
                transaction = self._persist_transaction(db, row)
            review = ReviewCase(status="OPEN", model_decision="REVIEW")
            transaction.review_case = review
            db.flush()
        review.status = "DECIDED"
        review.reviewer_decision = payload.decision
        review.reviewer_reason = payload.reason or None
        review.reviewer_id = payload.reviewer_id
        review.reviewed_at = datetime.now(timezone.utc)
        try:
            db.commit()
        except SQLAlchemyError:
            db.rollback()
            raise
        return self._public_item(row, review)

    def _persist_transaction(self, db: Session, row: pd.Series) -> Transaction:
        config = self.cost.operating_config
        analysis_model = self.cost.analysis["model"]
        version = str(config["model_version"])
        model_run = db.scalar(select(ModelRun).where(ModelRun.model_version == version))
        if model_run is None:
            model_run = ModelRun(
                model_name=str(config["model_name"]),
                model_version=version,
                trained_at=parse_datetime(config["generated_at"]),
                feature_set=str(config["feature_set"]),
                evaluation_status="NOT_EVALUATED",
                evaluation_split=None,
                active_rule_count=0,
                metadata_json={
                    "experiment_id": config["experiment_id"],
                    "selection_split": "validation",
                    "feature_names": analysis_model.get("feature_names", []),
                    "provisional": True,
                    "held_out_test_accessed": False,
                },
            )
            db.add(model_run)
            db.flush()
        cost_config = upsert_cost_config(
            db, dict(config["cost_assumptions"]), name_prefix="illustrative-validation"
        )
        threshold_key = f"{version}-{config['scenario']}-provisional-validation"
        threshold = db.scalar(
            select(ThresholdConfig).where(ThresholdConfig.config_key == threshold_key)
        )
        if threshold is None:
            threshold = ThresholdConfig(
                config_key=threshold_key,
                model_run_id=model_run.id,
                cost_config_id=cost_config.id,
                review_threshold=float(config["review_threshold"]),
                block_threshold=float(config["block_threshold"]),
                selection_split="validation",
                objective=str(config["selection_reason"]),
                is_active=False,
            )
            db.add(threshold)
            db.flush()
        transaction = Transaction(
            transaction_id=str(int(row["TransactionID"])),
            transaction_dt=int(row["TransactionDT"]),
            amount=Decimal(str(float(row["TransactionAmt"]))),
            actual_label=int(row["actual_label"]),
            risk_score=float(row["fraud_probability"]),
            decision="REVIEW",
            model_run_id=model_run.id,
            threshold_config_id=threshold.id,
            source="VALIDATION_REVIEW_DEMO",
        )
        db.add(transaction)
        db.flush()
        return transaction


def validation_review_service(
    project: ProjectArtifactService, cost: ValidationCostService
) -> ValidationReviewService:
    return ValidationReviewService(project, cost)
