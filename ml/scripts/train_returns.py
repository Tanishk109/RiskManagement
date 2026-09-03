from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import catboost
import joblib
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from merchantshield_ml.returns import (
    RETURN_CATEGORICAL_FEATURES,
    RETURN_DATA_SOURCE,
    RETURN_FEATURES,
    RETURN_HIGH_THRESHOLD,
    RETURN_MEDIUM_THRESHOLD,
    RETURN_MODEL_VERSION,
    RETURN_PROXY_DISCLOSURE,
    build_order_dataset,
    chronological_return_split,
    normalize_return_features,
    return_binary_metrics,
)
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data/raw/online-retail-ii/online_retail_II.xlsx"
PROCESSED = ROOT / "data/processed/online-retail-ii"
METRICS_PATH = ROOT / "artifacts/metrics/returns_evaluation.json"
METADATA_PATH = ROOT / "artifacts/models/return_risk_catboost_metadata.json"
MODEL_PATH = ROOT / "artifacts/models/return_risk_catboost.cbm"
BASELINE_PATH = ROOT / "artifacts/models/return_risk_logistic.joblib"
REPORT_PATH = ROOT / "artifacts/reports/returns_model.md"


def _read_uci_workbook(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Official UCI workbook not found: {path}")
    sheets = pd.read_excel(path, sheet_name=None)
    frames: list[pd.DataFrame] = []
    for sheet_name, frame in sheets.items():
        renamed = frame.rename(
            columns={
                "Invoice": "invoice_id",
                "StockCode": "stock_code",
                "Quantity": "quantity",
                "InvoiceDate": "order_datetime",
                "Price": "unit_price",
                "Customer ID": "customer_id",
                "Country": "country",
            }
        )
        renamed["source_sheet"] = sheet_name
        frames.append(renamed)
    return pd.concat(frames, ignore_index=True)


def _baseline() -> Pipeline:
    numeric = [feature for feature in RETURN_FEATURES if feature not in RETURN_CATEGORICAL_FEATURES]
    preprocessing = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20),
                list(RETURN_CATEGORICAL_FEATURES),
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _partition_summary(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": len(frame),
        "cancellations": int(frame["is_cancellation_proxy"].sum()),
        "non_cancellations": int(len(frame) - frame["is_cancellation_proxy"].sum()),
        "prevalence": float(frame["is_cancellation_proxy"].mean()),
        "start": frame["order_datetime"].min().isoformat(),
        "end": frame["order_datetime"].max().isoformat(),
    }


def _write_report(artifact: dict[str, object]) -> None:
    splits = artifact["splits"]
    baseline = artifact["models"]["logistic_regression"]
    candidate = artifact["models"]["catboost"]
    lines = [
        "# Return-risk cancellation proxy evaluation",
        "",
        f"Data source: {RETURN_DATA_SOURCE}.",
        "",
        f"> {RETURN_PROXY_DISCLOSURE}",
        "",
        (
            "The unit of modeling is an invoice/order, not a line item. Invoice identifiers are used "
            "only to construct the proxy label and are excluded from model features. Signed "
            "quantities are converted to magnitudes because negative quantity directly exposes "
            "cancellation rows."
        ),
        "",
        "## Chronological partitions",
        "",
        "| Partition | Orders | Cancellations | Prevalence | Time range |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name in ("train", "validation", "test"):
        item = splits[name]
        lines.append(
            f"| {name.title()} | {item['rows']:,} | {item['cancellations']:,} | "
            f"{item['prevalence']:.2%} | {item['start']} to {item['end']} |"
        )
    lines.extend(
        [
            "",
            "## Measured performance",
            "",
            (
                "Classification metrics use the documented HIGH boundary of "
                f"{RETURN_HIGH_THRESHOLD:.2f}. Average Precision is threshold-free."
            ),
            "",
            "| Model / partition | Precision | Recall | F1 | Average Precision |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for model_name, model in (("Logistic baseline", baseline), ("CatBoost candidate", candidate)):
        for partition in ("validation", "test"):
            item = model[partition]
            lines.append(
                f"| {model_name} / {partition} | {item['precision']:.4f} | "
                f"{item['recall']:.4f} | {item['f1']:.4f} | {item['average_precision']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Leakage controls",
            "",
            "- Partitions are chronological and the UCI test partition is never used for fitting or early stopping.",
            "- Customer history uses cumulative values shifted to exclude the current and all future orders.",
            "- Missing customer IDs never share one synthetic history bucket.",
            "- Invoice number, description, cancellation prefix, signed quantity, and outcome are excluded from features.",
            "- The payment-fraud IEEE-CIS model and held-out test were not opened or modified.",
            "",
            "## Product interpretation",
            "",
            (
                f"LOW is below {RETURN_MEDIUM_THRESHOLD:.2f}, MEDIUM is "
                f"{RETURN_MEDIUM_THRESHOLD:.2f} to below {RETURN_HIGH_THRESHOLD:.2f}, and HIGH is "
                f"at least {RETURN_HIGH_THRESHOLD:.2f}. These labels prioritize merchant review "
                "and never automatically reject an order."
            ),
            "",
            (
                "The proxy may include reversals, corrections, and administrative cancellations "
                "that are not physical returns. Performance must not be presented as physical-return "
                "performance."
            ),
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = _read_uci_workbook(RAW_PATH)
    orders = build_order_dataset(lines)
    split = chronological_return_split(orders)
    orders.to_parquet(PROCESSED / "orders.parquet", index=False)
    split.train.to_parquet(PROCESSED / "train.parquet", index=False)
    split.validation.to_parquet(PROCESSED / "validation.parquet", index=False)
    split.test.to_parquet(PROCESSED / "test.parquet", index=False)

    x_train = normalize_return_features(split.train)
    x_validation = normalize_return_features(split.validation)
    x_test = normalize_return_features(split.test)
    y_train = split.train["is_cancellation_proxy"].astype(int)
    y_validation = split.validation["is_cancellation_proxy"].astype(int)
    y_test = split.test["is_cancellation_proxy"].astype(int)

    baseline = _baseline()
    baseline.fit(x_train, y_train)
    baseline_validation = baseline.predict_proba(x_validation)[:, 1]
    baseline_test = baseline.predict_proba(x_test)[:, 1]
    joblib.dump(baseline, BASELINE_PATH)

    candidate = CatBoostClassifier(
        iterations=500,
        depth=7,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="PRAUC:type=Classic",
        auto_class_weights="Balanced",
        random_seed=42,
        allow_writing_files=False,
        verbose=False,
    )
    candidate.fit(
        x_train,
        y_train,
        cat_features=list(RETURN_CATEGORICAL_FEATURES),
        eval_set=(x_validation, y_validation),
        early_stopping_rounds=70,
        use_best_model=True,
    )
    candidate_validation = candidate.predict_proba(x_validation)[:, 1]
    candidate_test = candidate.predict_proba(x_test)[:, 1]
    candidate.save_model(MODEL_PATH)

    artifact: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": RETURN_DATA_SOURCE,
        "uci_dataset_id": 502,
        "proxy_disclosure": RETURN_PROXY_DISCLOSURE,
        "raw_line_rows": len(lines),
        "order_rows": len(orders),
        "overall_cancellations": int(orders["is_cancellation_proxy"].sum()),
        "overall_prevalence": float(orders["is_cancellation_proxy"].mean()),
        "splits": {
            "train": _partition_summary(split.train),
            "validation": _partition_summary(split.validation),
            "test": _partition_summary(split.test),
        },
        "models": {
            "logistic_regression": {
                "name": "LogisticRegression",
                "validation": return_binary_metrics(y_validation, baseline_validation),
                "test": return_binary_metrics(y_test, baseline_test),
            },
            "catboost": {
                "name": "CatBoostClassifier",
                "model_version": RETURN_MODEL_VERSION,
                "tree_count": int(candidate.tree_count_),
                "best_iteration": int(candidate.get_best_iteration()),
                "validation": return_binary_metrics(y_validation, candidate_validation),
                "test": return_binary_metrics(y_test, candidate_test),
            },
        },
        "risk_bands": {
            "low": f"p < {RETURN_MEDIUM_THRESHOLD}",
            "medium": f"{RETURN_MEDIUM_THRESHOLD} <= p < {RETURN_HIGH_THRESHOLD}",
            "high": f"p >= {RETURN_HIGH_THRESHOLD}",
            "automatic_rejection": False,
        },
        "feature_schema": list(RETURN_FEATURES),
        "categorical_features": list(RETURN_CATEGORICAL_FEATURES),
        "leakage_controls": [
            "Chronological train/validation/test partitions",
            "Prior customer aggregates use shifted cumulative history only",
            "Missing customer IDs receive isolated history keys",
            "Invoice identifier and label prefix excluded from model features",
            "Signed quantities converted to magnitudes",
        ],
        "ieee_cis_model_modified": False,
        "ieee_cis_held_out_test_accessed": False,
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "catboost": catboost.__version__,
        },
    }
    METRICS_PATH.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "model_version": RETURN_MODEL_VERSION,
        "model_type": "CatBoostClassifier",
        "data_source": RETURN_DATA_SOURCE,
        "proxy_disclosure": RETURN_PROXY_DISCLOSURE,
        "feature_schema": list(RETURN_FEATURES),
        "categorical_features": list(RETURN_CATEGORICAL_FEATURES),
        "medium_threshold": RETURN_MEDIUM_THRESHOLD,
        "high_threshold": RETURN_HIGH_THRESHOLD,
        "automatic_rejection": False,
        "tree_count": int(candidate.tree_count_),
        "training_partition": "chronological train",
        "early_stopping_partition": "chronological validation",
        "final_evaluation_partition": "chronological test",
        "test_metrics": artifact["models"]["catboost"]["test"],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _write_report(artifact)
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
