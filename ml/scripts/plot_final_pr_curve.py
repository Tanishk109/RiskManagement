from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "merchantshield-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import ARTIFACTS
from sklearn.metrics import precision_recall_curve

PREDICTIONS_PATH = ARTIFACTS / "metrics/final_test_predictions.csv"
METRICS_PATH = ARTIFACTS / "metrics/final_test_metrics.json"
OUTPUT_PATH = ARTIFACTS / "figures/final_test_precision_recall_curve.png"
REQUIRED_PREDICTION_COLUMNS = {"isFraud", "risk_score"}


def _load_final_artifacts() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(
            "Local-only final_test_predictions.csv is unavailable. Run this script only "
            "where the completed one-time evaluation artifact already exists; it will not "
            "regenerate predictions or access held-out source rows."
        )
    if not METRICS_PATH.is_file():
        raise FileNotFoundError(
            "final_test_metrics.json is unavailable. This script will not rerun final evaluation."
        )

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    if metrics.get("evaluation_status") != "complete" or metrics.get("split") != "test":
        raise ValueError("Refusing to plot from incomplete or non-test metrics")
    if metrics.get("held_out_test_accessed") is not True:
        raise ValueError("Final metrics do not record the completed held-out evaluation")
    if metrics.get("threshold_selection_split") != "validation":
        raise ValueError("The frozen block threshold was not selected on validation")

    predictions = pd.read_csv(PREDICTIONS_PATH)
    missing_columns = REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns)
    if missing_columns:
        raise ValueError(
            "Final prediction artifact is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if len(predictions) != int(metrics["test_transaction_count"]):
        raise ValueError("Final prediction row count differs from final_test_metrics.json")

    labels = pd.to_numeric(predictions["isFraud"], errors="raise").to_numpy(dtype=int)
    scores = pd.to_numeric(predictions["risk_score"], errors="raise").to_numpy(dtype=float)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("isFraud must contain only binary labels")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("risk_score must contain finite probabilities in [0, 1]")
    if int(labels.sum()) != int(metrics["fraud_count"]):
        raise ValueError("Fraud count differs from final_test_metrics.json")

    return predictions, metrics


def _frozen_operating_point(
    labels: np.ndarray,
    scores: np.ndarray,
    block_threshold: float,
) -> tuple[float, float]:
    blocked = scores >= block_threshold
    true_positives = int(np.count_nonzero(blocked & (labels == 1)))
    false_positives = int(np.count_nonzero(blocked & (labels == 0)))
    false_negatives = int(np.count_nonzero(~blocked & (labels == 1)))
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    return precision, recall


def _assert_reported_point(
    metrics: dict[str, Any],
    precision: float,
    recall: float,
) -> None:
    if not np.isclose(precision, float(metrics["precision"]), rtol=0, atol=1e-12):
        raise ValueError("Frozen-point precision differs from final_test_metrics.json")
    if not np.isclose(recall, float(metrics["recall"]), rtol=0, atol=1e-12):
        raise ValueError("Frozen-point recall differs from final_test_metrics.json")


def _assert_point_is_on_curve(
    curve_precision: np.ndarray,
    curve_recall: np.ndarray,
    thresholds: np.ndarray,
    block_threshold: float,
    frozen_precision: float,
    frozen_recall: float,
) -> None:
    index = int(np.searchsorted(thresholds, block_threshold, side="left"))
    if index >= len(thresholds):
        index = len(curve_precision) - 1
    if not (
        np.isclose(curve_precision[index], frozen_precision, rtol=0, atol=1e-12)
        and np.isclose(curve_recall[index], frozen_recall, rtol=0, atol=1e-12)
    ):
        raise ValueError("Frozen operating point does not align with the computed PR curve")


def _save_figure(
    curve_precision: np.ndarray,
    curve_recall: np.ndarray,
    frozen_precision: float,
    frozen_recall: float,
    block_threshold: float,
    base_rate: float,
    average_precision: float,
) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.8, 5.4))
    plt.plot(
        curve_recall,
        curve_precision,
        color="#6f7bf7",
        linewidth=2,
        label=f"Held-out CatBoost (PR-AUC={average_precision:.4f})",
    )
    plt.axhline(
        base_rate,
        color="#8b96a0",
        linestyle="--",
        linewidth=1.6,
    )
    plt.scatter(
        [frozen_recall],
        [frozen_precision],
        marker="*",
        s=220,
        color="#d95f59",
        edgecolor="white",
        linewidth=0.9,
        zorder=5,
        label=(
            f"Frozen block threshold {block_threshold:.3f} "
            "— selected on VALIDATION"
        ),
    )
    plt.annotate(
        f"Precision {frozen_precision:.2%}\nRecall {frozen_recall:.2%}",
        xy=(frozen_recall, frozen_precision),
        xytext=(12, 14),
        textcoords="offset points",
        fontsize=9,
    )
    plt.annotate(
        f"PR-AUC: {average_precision:.4f}",
        xy=(0.98, 0.73),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=10,
    )
    plt.annotate(
        f"Base rate: {base_rate:.2%}",
        xy=(0.98, base_rate),
        xycoords="data",
        xytext=(0, 6),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#68737d",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Held-out precision–recall with frozen operating point")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.2)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    predictions, metrics = _load_final_artifacts()
    labels = predictions["isFraud"].to_numpy(dtype=int)
    scores = predictions["risk_score"].to_numpy(dtype=float)
    block_threshold = float(metrics["block_threshold"])
    curve_precision, curve_recall, thresholds = precision_recall_curve(labels, scores)
    frozen_precision, frozen_recall = _frozen_operating_point(
        labels,
        scores,
        block_threshold,
    )
    _assert_reported_point(metrics, frozen_precision, frozen_recall)
    _assert_point_is_on_curve(
        curve_precision,
        curve_recall,
        thresholds,
        block_threshold,
        frozen_precision,
        frozen_recall,
    )

    base_rate = float(labels.mean())
    lift = frozen_precision / base_rate
    average_precision = float(metrics["average_precision"])
    _save_figure(
        curve_precision,
        curve_recall,
        frozen_precision,
        frozen_recall,
        block_threshold,
        base_rate,
        average_precision,
    )

    print(f"Rows: {len(labels):,}")
    print(f"Fraud cases: {int(labels.sum()):,}")
    print(f"Base rate: {base_rate:.4%}")
    print(f"Frozen block threshold: {block_threshold:.3f}")
    print(f"Frozen precision: {frozen_precision:.4%}")
    print(f"Frozen recall: {frozen_recall:.4%}")
    print(f"Precision lift over base rate: {lift:.2f}x")
    print(f"Average precision / PR-AUC: {average_precision:.4f}")
    print(f"Wrote {OUTPUT_PATH.relative_to(ARTIFACTS.parent)}")


if __name__ == "__main__":
    main()
