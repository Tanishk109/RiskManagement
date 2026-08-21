from __future__ import annotations

import argparse
from decimal import Decimal

import pandas as pd
from app.database import SessionLocal
from app.models import ReviewCase, Transaction
from app.services.evidence_store import sync_evidence_artifacts
from common import ARTIFACTS
from sqlalchemy import select


def select_cases(frame: pd.DataFrame, per_group: int) -> pd.DataFrame:
    fraud = frame[frame["isFraud"] == 1]
    legitimate = frame[frame["isFraud"] == 0]
    groups = [
        fraud.nlargest(per_group, "risk_score"),
        legitimate.nlargest(per_group, "risk_score"),
        legitimate[legitimate["decision"] == "BLOCK"].nlargest(per_group, "risk_score"),
        fraud[fraud["decision"] == "APPROVE"].nsmallest(per_group, "risk_score"),
        frame[frame["decision"] == "REVIEW"].sort_values("risk_score").head(per_group),
    ]
    return pd.concat(groups, ignore_index=True).drop_duplicates("TransactionID").sort_values("TransactionDT")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed local PostgreSQL from genuine held-out predictions")
    parser.add_argument("--local-only", action="store_true", help="Required acknowledgement that competition rows remain local")
    parser.add_argument("--per-group", type=int, default=8)
    args = parser.parse_args()
    if not args.local_only:
        raise ValueError("Refusing to export competition rows without --local-only")
    if args.per_group < 1 or args.per_group > 25:
        raise ValueError("per-group must be between 1 and 25")

    prediction_path = ARTIFACTS / "metrics/final_test_predictions.csv"
    if not prediction_path.is_file():
        raise FileNotFoundError("Run final held-out evaluation before seeding demo cases")
    cases = select_cases(pd.read_csv(prediction_path), args.per_group)
    inserted = 0
    with SessionLocal() as db:
        model_run, threshold = sync_evidence_artifacts(db, require_metrics=True)
        for row in cases.to_dict(orient="records"):
            transaction_id = str(int(row["TransactionID"]))
            if db.scalar(select(Transaction).where(Transaction.transaction_id == transaction_id)):
                continue
            transaction = Transaction(
                transaction_id=transaction_id,
                transaction_dt=int(row["TransactionDT"]),
                amount=Decimal(str(row["TransactionAmt"])),
                actual_label=int(row["isFraud"]),
                risk_score=float(row["risk_score"]),
                decision=str(row["decision"]),
                model_run_id=model_run.id,
                threshold_config_id=threshold.id,
                source="HELD_OUT_DEMO",
            )
            if transaction.decision == "REVIEW":
                transaction.review_case = ReviewCase(status="OPEN", model_decision="REVIEW")
            db.add(transaction)
            inserted += 1
        db.commit()
    print(f"Inserted {inserted} local held-out evaluation cases. No row file was published.")


if __name__ == "__main__":
    main()
