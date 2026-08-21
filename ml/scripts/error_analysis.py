from __future__ import annotations

import json

import joblib
import pandas as pd
from common import ARTIFACTS, load_splits


def _group_errors(frame: pd.DataFrame, column: str) -> list[dict[str, object]]:
    grouped = frame.groupby(column, dropna=False).agg(
        group_size=("isFraud", "size"),
        fraud_count=("isFraud", "sum"),
        false_negative_count=("false_negative", "sum"),
        fraud_caught=("true_positive", "sum"),
    )
    grouped = grouped[grouped["group_size"] >= max(20, int(len(frame) * 0.001))]
    grouped["recall"] = grouped["fraud_caught"] / grouped["fraud_count"].replace(0, pd.NA)
    grouped = grouped.sort_values(["false_negative_count", "group_size"], ascending=False).head(12)
    return [
        {
            "group": "<MISSING>" if pd.isna(index) else str(index),
            "group_size": int(row["group_size"]),
            "fraud_count": int(row["fraud_count"]),
            "false_negative_count": int(row["false_negative_count"]),
            "recall": None if pd.isna(row["recall"]) else float(row["recall"]),
        }
        for index, row in grouped.iterrows()
    ]


def main() -> None:
    models_dir = ARTIFACTS / "models"
    metrics_dir = ARTIFACTS / "metrics"
    bundle = joblib.load(models_dir / "selected_model.joblib")
    features = list(bundle["feature_names"])
    splits = load_splits(features)
    validation = splits.validation.copy()
    validation["risk_score"] = bundle["pipeline"].predict_proba(validation[features])[:, 1]
    validation["prediction"] = (validation["risk_score"] >= 0.5).astype(int)
    validation["false_negative"] = ((validation["isFraud"] == 1) & (validation["prediction"] == 0)).astype(int)
    validation["false_positive"] = ((validation["isFraud"] == 0) & (validation["prediction"] == 1)).astype(int)
    validation["true_positive"] = ((validation["isFraud"] == 1) & (validation["prediction"] == 1)).astype(int)
    validation["amount_band"] = pd.qcut(validation["TransactionAmt"], q=5, duplicates="drop").astype(str)
    validation["temporal_band"] = pd.qcut(validation["TransactionDT"], q=5, duplicates="drop").astype(str)

    group_columns = [column for column in ["ProductCD", "DeviceType", "amount_band", "temporal_band"] if column in validation.columns]
    analysis = {
        "split": "validation",
        "threshold": 0.5,
        "false_negative_count": int(validation["false_negative"].sum()),
        "false_positive_count": int(validation["false_positive"].sum()),
        "groups": {column: _group_errors(validation, column) for column in group_columns},
    }
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "validation_error_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Failure Analysis",
        "",
        "Generated from validation predictions only. The held-out test set is not used for rule design.",
        "",
        f"- False negatives at 0.50: {analysis['false_negative_count']:,}",
        f"- False positives at 0.50: {analysis['false_positive_count']:,}",
        "",
    ]
    for column, groups in analysis["groups"].items():
        lines.extend([f"## {column}", "", "| Group | Rows | Fraud | False negatives | Recall |", "|---|---:|---:|---:|---:|"])
        for row in groups:
            recall = "N/A" if row["recall"] is None else f"{float(row['recall']):.4f}"
            lines.append(f"| {row['group']} | {row['group_size']} | {row['fraud_count']} | {row['false_negative_count']} | {recall} |")
        lines.append("")
    lines.extend([
        "## Rule decisions",
        "",
        "No rule is enabled automatically. Any proposed rule must be backtested against these validation errors and document rows affected, fraud caught, legitimate rows affected, and cost impact.",
    ])
    (ARTIFACTS.parent / "docs/failure-analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
