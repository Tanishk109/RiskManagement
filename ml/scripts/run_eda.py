from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import ARTIFACTS, RAW_DATA, ROOT, replace_marked_section
from merchantshield_ml.data import load_ieee_cis

CATEGORICAL_FIELDS = ["ProductCD", "card4", "card6", "DeviceType", "P_emaildomain"]
EDA_FEATURES = ["TransactionAmt", *CATEGORICAL_FIELDS]


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    figures = ARTIFACTS / "figures"
    reports = ARTIFACTS / "reports"
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    frame, validation = load_ieee_cis(
        RAW_DATA / "train_transaction.csv",
        RAW_DATA / "train_identity.csv",
        feature_names=EDA_FEATURES,
    )

    time_bins = pd.qcut(frame["TransactionDT"], q=20, duplicates="drop")
    time_rates = frame.groupby(time_bins, observed=True)["isFraud"].agg(["size", "sum", "mean"]).reset_index(drop=True)
    plt.figure(figsize=(9, 4.8))
    plt.plot(np.arange(len(time_rates)), time_rates["mean"] * 100, marker="o", linewidth=1.6)
    plt.xlabel("Increasing TransactionDT bin")
    plt.ylabel("Fraud rate (%)")
    plt.title("IEEE-CIS fraud rate over time")
    _save_figure(figures / "fraud_rate_over_time.png")

    plt.figure(figsize=(9, 4.8))
    capped = frame["TransactionAmt"].clip(upper=frame["TransactionAmt"].quantile(0.99))
    plt.hist(capped[frame["isFraud"] == 0], bins=60, alpha=0.62, density=True, label="Legitimate")
    plt.hist(capped[frame["isFraud"] == 1], bins=60, alpha=0.62, density=True, label="Fraud")
    plt.xlabel("TransactionAmt (capped at dataset 99th percentile for display)")
    plt.ylabel("Density")
    plt.title("Transaction amount distributions")
    plt.legend()
    _save_figure(figures / "transaction_amount_distribution.png")

    missingness = (frame.isna().mean() * 100).sort_values(ascending=False)
    top_missing = missingness.head(25)
    plt.figure(figsize=(9, 6))
    top_missing.sort_values().plot.barh()
    plt.xlabel("Missing values (%)")
    plt.title("Top columns by missingness")
    _save_figure(figures / "top_missingness.png")

    categorical_results: dict[str, list[dict[str, object]]] = {}
    minimum_support = max(100, int(len(frame) * 0.001))
    for column in CATEGORICAL_FIELDS:
        if column not in frame.columns:
            continue
        grouped = frame.groupby(column, dropna=False)["isFraud"].agg(["size", "sum", "mean"])
        grouped = grouped[grouped["size"] >= minimum_support].sort_values("mean", ascending=False).head(20)
        categorical_results[column] = [
            {
                "value": "<MISSING>" if pd.isna(index) else str(index),
                "support": int(row["size"]),
                "fraud_count": int(row["sum"]),
                "fraud_rate": float(row["mean"]),
            }
            for index, row in grouped.iterrows()
        ]

    summary = {
        "source": "Local IEEE-CIS labeled train files",
        "dataset_validation": validation.to_dict(),
        "minimum_categorical_support": minimum_support,
        "top_missingness_percent": {str(key): float(value) for key, value in top_missing.items()},
        "categorical_fraud_rates": categorical_results,
        "time_bins": [
            {"bin": int(index), "transactions": int(row["size"]), "fraud_count": int(row["sum"]), "fraud_rate": float(row["mean"])}
            for index, row in time_rates.iterrows()
        ],
    }
    (reports / "eda_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# EDA Summary",
        "",
        "Generated from the local labeled IEEE-CIS train files. No fixture metrics are included.",
        "",
        "## Dataset",
        "",
        f"- Transactions: {validation.transaction_rows:,}",
        f"- Fraud transactions: {validation.fraud_rows:,}",
        f"- Fraud percentage: {validation.fraud_percentage:.6f}%",
        f"- Identity rows: {validation.identity_rows:,}",
        f"- Identity coverage: {validation.identity_coverage:.6f}%",
        f"- TransactionDT range: {validation.transaction_dt_min:.0f} to {validation.transaction_dt_max:.0f}",
        "",
        "## Generated figures",
        "",
        "- `artifacts/figures/fraud_rate_over_time.png`",
        "- `artifacts/figures/transaction_amount_distribution.png`",
        "- `artifacts/figures/top_missingness.png`",
        "",
        f"Categorical rates use a minimum support of {minimum_support:,} rows to reduce small-group overinterpretation.",
    ]
    (reports / "eda_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    replace_marked_section(
        ROOT / "docs/modeling-decisions.md",
        "EDA",
        "\n".join([
            f"Generated from actual local data: {validation.transaction_rows:,} transactions, {validation.fraud_rows:,} fraud labels ({validation.fraud_percentage:.6f}%), and {validation.identity_coverage:.6f}% identity coverage.",
            "",
            "Detailed time, amount, missingness, and minimum-support categorical analyses are in `artifacts/reports/eda_summary.md` and `artifacts/reports/eda_summary.json`.",
        ]),
    )


if __name__ == "__main__":
    main()
