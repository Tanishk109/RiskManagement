from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .split import TemporalSplits

PARTITION_NAMES = ("train", "validation", "test")
CATEGORICAL_DISTRIBUTION_COLUMNS = ("ProductCD", "card4", "card6", "DeviceType", "identity_available")
INITIAL_STABILITY_COLUMNS = (
    "TransactionAmt",
    "TransactionDT",
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "identity_available",
    "DeviceType",
    "DeviceInfo",
    "R_emaildomain",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "D1",
    "D2",
    "D3",
)
MISSINGNESS_SHIFT_THRESHOLD_PERCENTAGE_POINTS = 5.0


def _frames(splits: TemporalSplits) -> dict[str, pd.DataFrame]:
    return {"train": splits.train, "validation": splits.validation, "test": splits.test}


def _amount_summary(frame: pd.DataFrame) -> dict[str, int | float]:
    amount = frame["TransactionAmt"].dropna()
    return {
        "count": int(amount.count()),
        "mean": float(amount.mean()),
        "std": float(amount.std()),
        "min": float(amount.min()),
        "p25": float(amount.quantile(0.25)),
        "median": float(amount.median()),
        "p75": float(amount.quantile(0.75)),
        "p95": float(amount.quantile(0.95)),
        "p99": float(amount.quantile(0.99)),
        "max": float(amount.max()),
    }


def _coverage_percent(mask: pd.Series) -> float:
    return float(100 * mask.mean()) if len(mask) else 0.0


def _identity_summary(frame: pd.DataFrame) -> dict[str, int | float]:
    available = frame["identity_available"].astype(bool)
    fraud = frame["isFraud"] == 1
    legitimate = frame["isFraud"] == 0
    return {
        "available_count": int(available.sum()),
        "overall_coverage_percent": _coverage_percent(available),
        "fraud_available_count": int((available & fraud).sum()),
        "fraud_coverage_percent": _coverage_percent(available[fraud]),
        "legitimate_available_count": int((available & legitimate).sum()),
        "legitimate_coverage_percent": _coverage_percent(available[legitimate]),
    }


def _category_shares(frame: pd.DataFrame, column: str) -> list[dict[str, int | float | str]]:
    values = frame[column].astype("object").where(frame[column].notna(), "<MISSING>")
    counts = values.value_counts(dropna=False)
    return [
        {
            "category": str(category),
            "count": int(count),
            "share_percent": float(100 * count / len(frame)),
        }
        for category, count in counts.items()
    ]


def _category_share_shifts(
    partition_analysis: dict[str, dict[str, Any]],
    column: str,
) -> list[dict[str, float | str]]:
    shares = {
        partition: {
            str(row["category"]): float(row["share_percent"])
            for row in partition_analysis[partition]["category_shares"][column]
        }
        for partition in PARTITION_NAMES
    }
    categories = sorted(set().union(*(set(partition_shares) for partition_shares in shares.values())))
    rows = []
    for category in categories:
        values = [shares[partition].get(category, 0.0) for partition in PARTITION_NAMES]
        rows.append(
            {
                "category": category,
                "train_share_percent": values[0],
                "validation_share_percent": values[1],
                "test_share_percent": values[2],
                "max_minus_min_percentage_points": max(values) - min(values),
            }
        )
    return sorted(rows, key=lambda row: (-float(row["max_minus_min_percentage_points"]), str(row["category"])))


def build_temporal_descriptive_analysis(splits: TemporalSplits) -> dict[str, Any]:
    frames = _frames(splits)
    required = set(INITIAL_STABILITY_COLUMNS).union(CATEGORICAL_DISTRIBUTION_COLUMNS, {"isFraud"})
    for partition, frame in frames.items():
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{partition} partition is missing descriptive-analysis columns: {', '.join(missing)}")

    partition_analysis: dict[str, dict[str, Any]] = {}
    for partition, frame in frames.items():
        partition_analysis[partition] = {
            "transaction_amount": _amount_summary(frame),
            "identity_availability": _identity_summary(frame),
            "category_shares": {
                column: _category_shares(frame, column) for column in CATEGORICAL_DISTRIBUTION_COLUMNS
            },
            "missingness_percent": {
                column: float(100 * frame[column].isna().mean()) for column in INITIAL_STABILITY_COLUMNS
            },
        }

    missingness_rows: list[dict[str, float | str | bool]] = []
    for column in INITIAL_STABILITY_COLUMNS:
        values = [float(partition_analysis[name]["missingness_percent"][column]) for name in PARTITION_NAMES]
        spread = max(values) - min(values)
        missingness_rows.append(
            {
                "column": column,
                "train_percent": values[0],
                "validation_percent": values[1],
                "test_percent": values[2],
                "max_minus_min_percentage_points": spread,
                "substantial_shift": spread >= MISSINGNESS_SHIFT_THRESHOLD_PERCENTAGE_POINTS,
            }
        )
    missingness_rows.sort(
        key=lambda row: (-float(row["max_minus_min_percentage_points"]), str(row["column"]))
    )

    return {
        "partition_summaries": partition_analysis,
        "categorical_share_shifts": {
            column: _category_share_shifts(partition_analysis, column)
            for column in CATEGORICAL_DISTRIBUTION_COLUMNS
        },
        "missingness_stability": {
            "substantial_shift_threshold_percentage_points": MISSINGNESS_SHIFT_THRESHOLD_PERCENTAGE_POINTS,
            "ranked_columns": missingness_rows,
            "flagged_columns": [row for row in missingness_rows if bool(row["substantial_shift"])],
        },
    }


def write_temporal_split_report(metadata: dict[str, Any], path: str | Path) -> None:
    analysis = metadata["descriptive_analysis"]
    partitions = analysis["partition_summaries"]
    missingness = analysis["missingness_stability"]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# IEEE-CIS Chronological Split Report",
        "",
        "## Why chronological splitting was chosen",
        "",
        "Fraud detection predicts future transactions. A random split would mix earlier and later regimes and could make evaluation unrealistically easy. Rows were therefore ordered by `TransactionDT`, using `TransactionID` only as a deterministic secondary key.",
        "",
        "## Exact split boundaries",
        "",
        f"Boundary policy: {metadata['boundary_policy']}.",
        "",
        "| Partition | TransactionDT minimum | TransactionDT maximum |",
        "| --- | ---: | ---: |",
        f"| Train | {float(metadata['train_transaction_dt_min']):,.0f} | {float(metadata['train_transaction_dt_max']):,.0f} |",
        f"| Validation | {float(metadata['validation_transaction_dt_min']):,.0f} | {float(metadata['validation_transaction_dt_max']):,.0f} |",
        f"| Held-out test | {float(metadata['test_transaction_dt_min']):,.0f} | {float(metadata['test_transaction_dt_max']):,.0f} |",
        "",
        "The selected boundaries are clean: identical `TransactionDT` values are not split across partitions.",
        "",
        "## Partition sizes and fraud distribution",
        "",
        "| Partition | Rows | Actual share | Fraud | Legitimate | Fraud prevalence |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {"train": "Train", "validation": "Validation", "test": "Held-out test"}
    for partition in PARTITION_NAMES:
        lines.append(
            f"| {labels[partition]} | {int(metadata[f'{partition}_rows']):,} | "
            f"{100 * float(metadata[f'{partition}_fraction_actual']):.6f}% | "
            f"{int(metadata[f'{partition}_fraud_count']):,} | "
            f"{int(metadata[f'{partition}_legitimate_count']):,} | "
            f"{100 * float(metadata[f'{partition}_fraud_rate']):.6f}% |"
        )

    lines.extend(
        [
            "",
            "No partition was stratified or rebalanced; the observed prevalence changes are preserved.",
            "",
            "## Transaction amount distribution",
            "",
            "| Partition | Mean | P25 | Median | P75 | P95 | P99 | Max |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for partition in PARTITION_NAMES:
        amount = partitions[partition]["transaction_amount"]
        lines.append(
            f"| {labels[partition]} | {float(amount['mean']):,.4f} | {float(amount['p25']):,.4f} | "
            f"{float(amount['median']):,.4f} | {float(amount['p75']):,.4f} | "
            f"{float(amount['p95']):,.4f} | {float(amount['p99']):,.4f} | {float(amount['max']):,.4f} |"
        )

    lines.extend(
        [
            "",
            "## Identity availability stability",
            "",
            "`identity_available` is true when `TransactionID` matched a row in `train_identity.csv`; it is not inferred from any nullable identity feature.",
            "",
            "| Partition | Overall coverage | Fraud coverage | Legitimate coverage |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for partition in PARTITION_NAMES:
        identity = partitions[partition]["identity_availability"]
        lines.append(
            f"| {labels[partition]} | {float(identity['overall_coverage_percent']):.6f}% | "
            f"{float(identity['fraud_coverage_percent']):.6f}% | "
            f"{float(identity['legitimate_coverage_percent']):.6f}% |"
        )

    lines.extend(
        [
            "",
            "## Observed temporal distribution changes",
            "",
            "The largest category-share change for each requested field is shown below. These are descriptive shifts, not model metrics.",
            "",
            "| Field | Category | Train | Validation | Test | Maximum spread |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for column, rows in analysis["categorical_share_shifts"].items():
        row = rows[0]
        lines.append(
            f"| `{column}` | {row['category']} | {float(row['train_share_percent']):.6f}% | "
            f"{float(row['validation_share_percent']):.6f}% | {float(row['test_share_percent']):.6f}% | "
            f"{float(row['max_minus_min_percentage_points']):.6f} pp |"
        )

    lines.extend(
        [
            "",
            "## Missingness stability",
            "",
            f"A substantial shift is defined before inspection as an absolute max-minus-min difference of at least {float(missingness['substantial_shift_threshold_percentage_points']):.1f} percentage points across partitions.",
            "",
        ]
    )
    flagged = missingness["flagged_columns"]
    if flagged:
        lines.extend(
            [
                "| Column | Train missing | Validation missing | Test missing | Maximum spread |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in flagged:
            lines.append(
                f"| `{row['column']}` | {float(row['train_percent']):.6f}% | "
                f"{float(row['validation_percent']):.6f}% | {float(row['test_percent']):.6f}% | "
                f"{float(row['max_minus_min_percentage_points']):.6f} pp |"
            )
    else:
        lines.append("No initial proposed feature crossed the documented 5 percentage-point threshold.")

    lines.extend(
        [
            "",
            "## Evaluation policy",
            "",
            "```text",
            "TRAIN",
            "→ fit preprocessing and model",
            "",
            "VALIDATION",
            "→ model comparison, feature decisions, calibration,",
            "  thresholds, cost optimization and rule design",
            "",
            "HELD-OUT TEST",
            "→ final reporting only",
            "```",
            "",
            "No model or preprocessing object was fitted while creating these partitions. Precision, recall, F1, PR-AUC, FP/FN, thresholds, and merchant-cost metrics remain **Not evaluated yet**.",
        ]
    )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
