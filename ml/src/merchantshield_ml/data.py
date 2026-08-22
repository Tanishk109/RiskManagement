from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

TRANSACTION_REQUIRED_COLUMNS = {"TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "isFraud"}
IDENTITY_REQUIRED_COLUMNS = {"TransactionID"}
IDENTITY_AVAILABLE_COLUMN = "identity_available"


@dataclass(frozen=True)
class DatasetValidation:
    transaction_rows: int
    identity_rows: int
    merged_rows: int
    fraud_rows: int
    legitimate_rows: int
    fraud_percentage: float
    transaction_dt_min: float
    transaction_dt_max: float
    transaction_amount_summary: dict[str, int | float]
    matched_identity_rows: int
    identity_coverage: float
    identity_coverage_fraud: float
    identity_coverage_legitimate: float
    status: str = "VALID"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _columns(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required IEEE-CIS file not found: {path}")
    return list(pd.read_csv(path, nrows=0).columns)


def _selected_columns(available: list[str], requested: Iterable[str] | None, required: set[str]) -> list[str]:
    available_set = set(available)
    missing_required = sorted(required.difference(available_set))
    if missing_required:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_required)}")
    if requested is None:
        return available
    selected = required.union(set(requested).intersection(available_set))
    return [column for column in available if column in selected]


def load_ieee_cis(
    transaction_path: str | Path,
    identity_path: str | Path,
    *,
    feature_names: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, DatasetValidation]:
    """Load labeled IEEE-CIS train files and left-join identity data.

    The original Kaggle test files are intentionally unsupported here because
    they do not contain labels and cannot power honest held-out reporting.
    """

    transaction_path = Path(transaction_path)
    identity_path = Path(identity_path)
    transaction_columns = _columns(transaction_path)
    identity_columns = _columns(identity_path)
    if IDENTITY_AVAILABLE_COLUMN in transaction_columns or IDENTITY_AVAILABLE_COLUMN in identity_columns:
        raise ValueError(f"Raw IEEE-CIS files must not contain derived column {IDENTITY_AVAILABLE_COLUMN}")
    transaction_usecols = _selected_columns(transaction_columns, feature_names, TRANSACTION_REQUIRED_COLUMNS)
    identity_usecols = _selected_columns(identity_columns, feature_names, IDENTITY_REQUIRED_COLUMNS)

    transactions = pd.read_csv(transaction_path, usecols=transaction_usecols)
    identity = pd.read_csv(identity_path, usecols=identity_usecols)
    if transactions["TransactionID"].duplicated().any():
        raise ValueError("train_transaction.csv contains duplicate TransactionID values")
    if identity["TransactionID"].duplicated().any():
        raise ValueError("train_identity.csv contains duplicate TransactionID values")
    if transactions["isFraud"].isna().any():
        raise ValueError("train_transaction.csv contains missing isFraud labels")
    invalid_labels = sorted(set(transactions["isFraud"].unique()).difference({0, 1}))
    if invalid_labels:
        raise ValueError(f"train_transaction.csv contains invalid isFraud labels: {invalid_labels}")

    identity_ids = set(identity["TransactionID"])
    merged = transactions.merge(identity, on="TransactionID", how="left", validate="one_to_one")
    if len(merged) != len(transactions):
        raise AssertionError("Left join changed the number of transaction rows")
    has_identity = merged["TransactionID"].isin(identity_ids)
    merged.insert(1, IDENTITY_AVAILABLE_COLUMN, has_identity.astype(bool))

    fraud_rows = int(merged["isFraud"].sum())
    legitimate_rows = len(merged) - fraud_rows
    amount = merged["TransactionAmt"]
    amount_summary = {
        "count": int(amount.count()),
        "mean": float(amount.mean()),
        "std": float(amount.std()),
        "min": float(amount.min()),
        "p25": float(amount.quantile(0.25)),
        "median": float(amount.median()),
        "p75": float(amount.quantile(0.75)),
        "p90": float(amount.quantile(0.90)),
        "p95": float(amount.quantile(0.95)),
        "p99": float(amount.quantile(0.99)),
        "max": float(amount.max()),
    }

    def coverage(mask: pd.Series) -> float:
        rows = int(mask.sum())
        return float(100 * has_identity[mask].mean()) if rows else 0.0

    report = DatasetValidation(
        transaction_rows=len(transactions),
        identity_rows=len(identity),
        merged_rows=len(merged),
        fraud_rows=fraud_rows,
        legitimate_rows=legitimate_rows,
        fraud_percentage=float(100 * merged["isFraud"].mean()),
        transaction_dt_min=float(merged["TransactionDT"].min()),
        transaction_dt_max=float(merged["TransactionDT"].max()),
        transaction_amount_summary=amount_summary,
        matched_identity_rows=int(has_identity.sum()),
        identity_coverage=float(100 * has_identity.mean()),
        identity_coverage_fraud=coverage(merged["isFraud"] == 1),
        identity_coverage_legitimate=coverage(merged["isFraud"] == 0),
    )
    return merged, report
