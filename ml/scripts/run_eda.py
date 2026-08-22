from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "merchantshield-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import ARTIFACTS, IEEE_CIS_RAW_DATA
from merchantshield_ml.data import load_ieee_cis

CATEGORICAL_FIELDS = ["ProductCD", "card4", "card6", "DeviceType", "P_emaildomain", "R_emaildomain"]
EDA_FEATURES = ["TransactionAmt", *CATEGORICAL_FIELDS]
TIME_BUCKET_COUNT = 20
CSV_CHUNK_SIZE = 25_000


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def _series_summary(series: pd.Series) -> dict[str, int | float]:
    values = series.dropna()
    return {
        "count": int(values.count()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "p75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }


def _full_join_missingness(
    transaction_path: Path,
    identity_path: Path,
    transaction_ids: set[int],
) -> tuple[pd.Series, dict[str, int]]:
    """Compute left-joined missingness in chunks without materializing all columns."""

    transaction_columns = list(pd.read_csv(transaction_path, nrows=0).columns)
    identity_columns = list(pd.read_csv(identity_path, nrows=0).columns)
    overlapping = sorted((set(transaction_columns) & set(identity_columns)).difference({"TransactionID"}))
    if overlapping:
        raise ValueError(f"Unexpected overlapping transaction/identity columns: {', '.join(overlapping)}")

    transaction_missing = pd.Series(0, index=transaction_columns, dtype="int64")
    transaction_rows = 0
    for chunk in pd.read_csv(transaction_path, chunksize=CSV_CHUNK_SIZE, low_memory=False):
        transaction_rows += len(chunk)
        transaction_missing = transaction_missing.add(chunk.isna().sum(), fill_value=0).astype("int64")

    identity_feature_columns = [column for column in identity_columns if column != "TransactionID"]
    identity_missing_when_matched = pd.Series(0, index=identity_feature_columns, dtype="int64")
    identity_rows = 0
    matched_identity_rows = 0
    for chunk in pd.read_csv(identity_path, chunksize=CSV_CHUNK_SIZE, low_memory=False):
        identity_rows += len(chunk)
        matched = chunk[chunk["TransactionID"].isin(transaction_ids)]
        matched_identity_rows += len(matched)
        identity_missing_when_matched = identity_missing_when_matched.add(
            matched[identity_feature_columns].isna().sum(), fill_value=0
        ).astype("int64")

    unmatched_transaction_rows = transaction_rows - matched_identity_rows
    identity_missing_after_left_join = identity_missing_when_matched + unmatched_transaction_rows
    missing_counts = pd.concat([transaction_missing, identity_missing_after_left_join])
    missingness = (100 * missing_counts / transaction_rows).sort_values(ascending=False)
    metadata = {
        "joined_column_count": len(missingness),
        "transaction_rows_scanned": int(transaction_rows),
        "identity_rows_scanned": int(identity_rows),
        "matched_identity_rows": int(matched_identity_rows),
        "orphan_identity_rows": int(identity_rows - matched_identity_rows),
    }
    return missingness, metadata


def _time_buckets(frame: pd.DataFrame) -> list[dict[str, int | float]]:
    dt = frame["TransactionDT"].astype(float)
    dt_min = float(dt.min())
    dt_max = float(dt.max())
    width = (dt_max - dt_min) / TIME_BUCKET_COUNT
    if width <= 0:
        raise ValueError("TransactionDT must span more than one value for temporal EDA")
    bucket = np.floor((dt - dt_min) / width).clip(0, TIME_BUCKET_COUNT - 1).astype(int)
    grouped = frame.assign(_time_bucket=bucket).groupby("_time_bucket")["isFraud"].agg(["size", "sum", "mean"])

    records: list[dict[str, int | float]] = []
    for index in range(TIME_BUCKET_COUNT):
        row = grouped.loc[index] if index in grouped.index else pd.Series({"size": 0, "sum": 0, "mean": 0.0})
        start = dt_min + index * width
        end = dt_max if index == TIME_BUCKET_COUNT - 1 else dt_min + (index + 1) * width
        records.append(
            {
                "bucket": index + 1,
                "transaction_dt_start": start,
                "transaction_dt_end": end,
                "transaction_count": int(row["size"]),
                "fraud_count": int(row["sum"]),
                "fraud_rate": float(row["mean"]),
            }
        )
    return records


def _categorical_rates(frame: pd.DataFrame, minimum_support: int) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {}
    for column in CATEGORICAL_FIELDS:
        values = frame[column].astype("object").where(frame[column].notna(), "<MISSING>")
        grouped = frame.assign(_category=values).groupby("_category")["isFraud"].agg(["size", "sum", "mean"])
        grouped = grouped[grouped["size"] >= minimum_support].sort_values(
            ["mean", "size"], ascending=[False, False]
        )
        results[column] = [
            {
                "category": str(category),
                "transaction_count": int(row["size"]),
                "fraud_count": int(row["sum"]),
                "fraud_rate": float(row["mean"]),
            }
            for category, row in grouped.iterrows()
        ]
    return results


def _write_figures(
    figures: Path,
    frame: pd.DataFrame,
    missingness: pd.Series,
    time_buckets: list[dict[str, int | float]],
    identity_by_label: dict[str, dict[str, int | float]],
    categorical_rates: dict[str, list[dict[str, Any]]],
) -> None:
    class_counts = frame["isFraud"].value_counts().reindex([0, 1], fill_value=0)
    plt.figure(figsize=(7.2, 4.8))
    bars = plt.bar(["Legitimate", "Fraud"], class_counts.values, color=["#35c6a1", "#ff6b6b"])
    plt.ylabel("Transactions")
    plt.title("IEEE-CIS labeled training class balance")
    for bar, count in zip(bars, class_counts.values, strict=True):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(count):,}", ha="center", va="bottom")
    _save_figure(figures / "class_balance.png")

    if (frame["TransactionAmt"].dropna() < 0).any():
        raise ValueError("TransactionAmt contains negative values; log1p amount plot is invalid")
    plt.figure(figsize=(9, 4.8))
    for label, color, name in [(0, "#35c6a1", "Legitimate"), (1, "#ff6b6b", "Fraud")]:
        amount = np.log1p(frame.loc[frame["isFraud"] == label, "TransactionAmt"].dropna())
        plt.hist(amount, bins=80, alpha=0.58, density=True, label=name, color=color)
    plt.xlabel("log1p(TransactionAmt), all non-missing rows")
    plt.ylabel("Density")
    plt.title("Transaction amount distribution by actual label")
    plt.legend()
    _save_figure(figures / "transaction_amount_distribution.png")

    plt.figure(figsize=(9, 4.8))
    midpoints = [
        (float(row["transaction_dt_start"]) + float(row["transaction_dt_end"])) / 2 for row in time_buckets
    ]
    rates = [100 * float(row["fraud_rate"]) for row in time_buckets]
    plt.plot(midpoints, rates, marker="o", linewidth=1.8, color="#ff6b6b")
    plt.xlabel("TransactionDT (relative seconds; not a calendar timestamp)")
    plt.ylabel("Fraud rate (%)")
    plt.title("Fraud prevalence across 20 chronological duration buckets")
    _save_figure(figures / "fraud_rate_over_time.png")

    plt.figure(figsize=(9, 6.8))
    missingness.head(25).sort_values().plot.barh(color="#7b8cff")
    plt.xlabel("Missing values after left join (%)")
    plt.title("Top 25 of all joined columns by missingness")
    _save_figure(figures / "top_missingness.png")

    labels = ["Legitimate", "Fraud"]
    coverage = [
        float(identity_by_label["legitimate"]["coverage_percent"]),
        float(identity_by_label["fraud"]["coverage_percent"]),
    ]
    plt.figure(figsize=(7.2, 4.8))
    bars = plt.bar(labels, coverage, color=["#35c6a1", "#ff6b6b"])
    plt.ylabel("Transactions with an identity row (%)")
    plt.ylim(0, max(100, max(coverage) * 1.12))
    plt.title("Identity-table availability by actual label")
    for bar, value in zip(bars, coverage, strict=True):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}%", ha="center", va="bottom")
    _save_figure(figures / "identity_coverage_by_label.png")

    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, column in zip(axes.flat, CATEGORICAL_FIELDS, strict=True):
        top = categorical_rates[column][:10]
        categories = [str(row["category"]) for row in reversed(top)]
        rates_for_plot = [100 * float(row["fraud_rate"]) for row in reversed(top)]
        axis.barh(categories, rates_for_plot, color="#7b8cff")
        axis.set_title(column)
        axis.set_xlabel("Fraud rate (%)")
    figure.suptitle("Highest supported categorical fraud rates", y=1.01, fontsize=14)
    figure.tight_layout()
    figure.savefig(figures / "categorical_fraud_rates.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    validation = summary["dataset_validation"]
    amount = summary["transaction_amount"]
    identity = summary["identity_availability"]
    temporal = summary["temporal_analysis"]
    missingness = summary["missingness"]
    source_files = summary["source_files"]

    lines = [
        "# IEEE-CIS EDA Summary",
        "",
        "Generated from the official local labeled IEEE-CIS training files. The Kaggle test files were not used.",
        "No model was trained and no temporal train/validation/test split was created in this phase.",
        "",
        "## Evidence status",
        "",
        "- Precision, recall, F1, PR-AUC, FP/FN, thresholds, and cost savings: **Not evaluated yet**",
        "- Merchant-facing ML metrics fabricated: **No**",
        "",
        "## Source files",
        "",
        f"- `train_transaction.csv`: {int(source_files['train_transaction.csv']['bytes']):,} bytes",
        f"- `train_identity.csv`: {int(source_files['train_identity.csv']['bytes']):,} bytes",
        "",
        "## Dataset validation",
        "",
        f"- Transactions: {int(validation['transaction_rows']):,}",
        f"- Identity rows: {int(validation['identity_rows']):,}",
        f"- Fraud: {int(validation['fraud_rows']):,}",
        f"- Legitimate: {int(validation['legitimate_rows']):,}",
        f"- Fraud prevalence: {float(validation['fraud_percentage']):.6f}%",
        f"- `TransactionDT` range: {float(validation['transaction_dt_min']):,.0f} to {float(validation['transaction_dt_max']):,.0f}",
        f"- Identity join coverage: {float(validation['identity_coverage']):.6f}% ({int(validation['matched_identity_rows']):,} transactions)",
        f"- Joined feature columns inspected for missingness: {int(missingness['joined_column_count']):,}",
        "",
        "## TransactionAmt summary",
        "",
        "| Group | Count | Mean | Median | P95 | P99 | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ["overall", "legitimate", "fraud"]:
        row = amount[name]
        lines.append(
            f"| {name.title()} | {int(row['count']):,} | {float(row['mean']):,.4f} | "
            f"{float(row['median']):,.4f} | {float(row['p95']):,.4f} | "
            f"{float(row['p99']):,.4f} | {float(row['max']):,.4f} |"
        )

    lines.extend(
        [
            "",
            "The amount-distribution figure uses `log1p(TransactionAmt)` only for display; it includes every non-missing row and does not cap or remove outliers.",
            "",
            "## Identity availability by actual label",
            "",
            "| Label | Transactions | With identity | Coverage |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name in ["legitimate", "fraud"]:
        row = identity["by_label"][name]
        lines.append(
            f"| {name.title()} | {int(row['transaction_count']):,} | {int(row['with_identity_count']):,} | "
            f"{float(row['coverage_percent']):.6f}% |"
        )

    lines.extend(
        [
            "",
            "## Temporal prevalence",
            "",
            (
                f"Across {int(temporal['bucket_count'])} equal-duration chronological buckets, fraud prevalence ranges "
                f"from {100 * float(temporal['minimum_fraud_rate']):.6f}% to "
                f"{100 * float(temporal['maximum_fraud_rate']):.6f}%."
            ),
            "`TransactionDT` is treated only as a relative ordering variable, not converted to a calendar date.",
            "",
            "## Highest missingness after the left join",
            "",
            f"{int(missingness['columns_over_90_percent'])} of {int(missingness['joined_column_count'])} columns are more than 90% missing.",
            "",
            "| Column | Missing |",
            "| --- | ---: |",
        ]
    )
    for column, value in list(missingness["percent_by_column"].items())[:20]:
        lines.append(f"| `{column}` | {float(value):.6f}% |")

    lines.extend(
        [
            "",
            "## Supported categorical fraud rates",
            "",
            (
                f"Categories below meet the minimum support of {int(summary['minimum_categorical_support']):,} "
                "transactions. The five highest fraud-rate categories per field are shown; the JSON report contains all "
                "supported categories."
            ),
            "",
        ]
    )
    for column, rows in summary["categorical_fraud_rates"].items():
        lines.extend(
            [
                f"### {column}",
                "",
                "| Category | Transactions | Fraud | Fraud rate |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in rows[:5]:
            lines.append(
                f"| {row['category']} | {int(row['transaction_count']):,} | {int(row['fraud_count']):,} | "
                f"{100 * float(row['fraud_rate']):.6f}% |"
            )
        lines.append("")

    lines.extend(
        [
            "## Figures",
            "",
            "- `artifacts/figures/class_balance.png`",
            "- `artifacts/figures/transaction_amount_distribution.png`",
            "- `artifacts/figures/fraud_rate_over_time.png`",
            "- `artifacts/figures/top_missingness.png`",
            "- `artifacts/figures/identity_coverage_by_label.png`",
            "- `artifacts/figures/categorical_fraud_rates.png`",
            "",
            "## Leakage guardrails before modeling",
            "",
            "- `isFraud` is the target and must never enter a feature matrix or rule condition.",
            "- `TransactionID` is an identifier aligned with dataset order and is excluded as a model feature.",
            "- `TransactionDT` may encode time/regime drift; it is retained only as an available-at-transaction-time candidate and requires chronological validation.",
            "- Any future aggregate or velocity feature must use only transactions strictly earlier than the scored row.",
            "- Identity availability and masked `C*`, `D*`, and `V*` fields require scoring-time availability checks before retention.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    figures = ARTIFACTS / "figures"
    reports = ARTIFACTS / "reports"
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    transaction_path = IEEE_CIS_RAW_DATA / "train_transaction.csv"
    identity_path = IEEE_CIS_RAW_DATA / "train_identity.csv"

    frame, validation = load_ieee_cis(transaction_path, identity_path, feature_names=EDA_FEATURES)
    transaction_ids = set(frame["TransactionID"].astype(int))
    identity_ids = set(pd.read_csv(identity_path, usecols=["TransactionID"])["TransactionID"].astype(int))
    has_identity = frame["TransactionID"].isin(identity_ids)

    missingness, missingness_metadata = _full_join_missingness(transaction_path, identity_path, transaction_ids)
    if missingness_metadata["transaction_rows_scanned"] != validation.transaction_rows:
        raise AssertionError("Chunked missingness scan did not cover every transaction")
    if missingness_metadata["identity_rows_scanned"] != validation.identity_rows:
        raise AssertionError("Chunked missingness scan did not cover every identity row")
    if missingness_metadata["matched_identity_rows"] != validation.matched_identity_rows:
        raise AssertionError("Chunked identity match count differs from loader validation")

    time_buckets = _time_buckets(frame)
    minimum_support = max(100, math.ceil(len(frame) * 0.001))
    categorical_rates = _categorical_rates(frame, minimum_support)
    amount_summary = {
        "overall": _series_summary(frame["TransactionAmt"]),
        "legitimate": _series_summary(frame.loc[frame["isFraud"] == 0, "TransactionAmt"]),
        "fraud": _series_summary(frame.loc[frame["isFraud"] == 1, "TransactionAmt"]),
    }
    identity_by_label: dict[str, dict[str, int | float]] = {}
    for label, name in [(0, "legitimate"), (1, "fraud")]:
        label_mask = frame["isFraud"] == label
        with_identity = int((label_mask & has_identity).sum())
        count = int(label_mask.sum())
        identity_by_label[name] = {
            "transaction_count": count,
            "with_identity_count": with_identity,
            "coverage_percent": float(100 * with_identity / count),
        }

    _write_figures(figures, frame, missingness, time_buckets, identity_by_label, categorical_rates)

    fraud_rates = [float(row["fraud_rate"]) for row in time_buckets]
    summary: dict[str, Any] = {
        "source": "Official local IEEE-CIS Fraud Detection labeled training files",
        "source_files": {
            transaction_path.name: {
                "path": str(transaction_path.relative_to(IEEE_CIS_RAW_DATA.parents[2])),
                "bytes": transaction_path.stat().st_size,
            },
            identity_path.name: {
                "path": str(identity_path.relative_to(IEEE_CIS_RAW_DATA.parents[2])),
                "bytes": identity_path.stat().st_size,
            },
        },
        "dataset_validation": validation.to_dict(),
        "selected_eda_frame_memory_bytes": int(frame.memory_usage(index=True, deep=True).sum()),
        "class_balance": {
            "fraud_count": validation.fraud_rows,
            "legitimate_count": validation.legitimate_rows,
            "fraud_prevalence": validation.fraud_percentage / 100,
        },
        "transaction_amount": amount_summary,
        "identity_availability": {
            "overall_coverage_percent": validation.identity_coverage,
            "by_label": identity_by_label,
        },
        "missingness": {
            **missingness_metadata,
            "columns_over_90_percent": int((missingness > 90).sum()),
            "completely_missing_columns": int((missingness == 100).sum()),
            "percent_by_column": {str(column): float(value) for column, value in missingness.items()},
        },
        "temporal_analysis": {
            "bucket_definition": "20 equal-duration chronological buckets over TransactionDT",
            "bucket_count": TIME_BUCKET_COUNT,
            "minimum_fraud_rate": min(fraud_rates),
            "maximum_fraud_rate": max(fraud_rates),
            "buckets": time_buckets,
        },
        "minimum_categorical_support": minimum_support,
        "categorical_fraud_rates": categorical_rates,
        "leakage_review": {
            "target_column": "isFraud — excluded from features",
            "identifier_column": "TransactionID — excluded from features",
            "time_column": "TransactionDT — relative ordering signal requiring chronological validation",
            "future_aggregates": "Not present; any later aggregates must be strictly past-only",
            "masked_features": "C*, D*, and V* require scoring-time availability checks",
        },
        "ml_performance": {
            "model_trained": False,
            "temporal_splits_created": False,
            "precision": "Not evaluated yet",
            "recall": "Not evaluated yet",
            "f1": "Not evaluated yet",
            "pr_auc": "Not evaluated yet",
            "false_positives": "Not evaluated yet",
            "false_negatives": "Not evaluated yet",
            "threshold_recommendation": "Not evaluated yet",
            "cost_savings": "Not evaluated yet",
        },
    }
    json_path = reports / "eda_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_markdown(reports / "eda_summary.md", summary)
    print(json.dumps({"report": str(json_path), "validation": validation.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
