from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from common import ARTIFACTS, PROCESSED_DATA, ROOT
from merchantshield_ml.calibration import calibration_diagnostics
from merchantshield_ml.catboost_candidate import normalize_catboost_features
from merchantshield_ml.final_evaluation import (
    FrozenEvaluationSpec,
    evaluate_frozen_scores,
    validate_frozen_evaluation_spec,
)

MODEL_BUNDLE = ARTIFACTS / "models/catboost_candidate.cbm"
MODEL_METADATA = ARTIFACTS / "models/catboost_candidate_metadata.json"
OPERATING_CONFIG = ARTIFACTS / "models/validation_operating_config.json"
SPLIT_METADATA = PROCESSED_DATA / "split_metadata.json"
TEST_DATA = PROCESSED_DATA / "test.parquet"
FINAL_METRICS = ARTIFACTS / "metrics/final_test_metrics.json"
TEST_ACCESS_RECORD = ARTIFACTS / "metrics/held_out_test_access.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _package_versions() -> dict[str, str]:
    packages = ["pandas", "numpy", "scikit-learn", "catboost"]
    return {name: importlib.metadata.version(name) for name in packages}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _enabled_rule_count() -> int:
    payload = yaml.safe_load((ROOT / "rules/merchant_rules.yaml").read_text(encoding="utf-8")) or {}
    return sum(1 for rule in payload.get("rules", []) if rule.get("enabled"))


def _assert_not_previously_started() -> None:
    if TEST_ACCESS_RECORD.exists():
        raise RuntimeError(
            "Held-out evaluation already has an access record. Refusing to rerun; audit "
            "artifacts/metrics/held_out_test_access.json."
        )
    if FINAL_METRICS.exists():
        raise RuntimeError("Final held-out metrics already exist. Refusing to rerun the test.")


def _load_frozen_inputs() -> tuple[
    CatBoostClassifier,
    FrozenEvaluationSpec,
    dict[str, Any],
    dict[str, Any],
]:
    _assert_not_previously_started()
    if not MODEL_BUNDLE.is_file():
        raise FileNotFoundError(
            "Frozen CatBoost candidate is missing. Restore catboost_candidate.cbm; do not retrain."
        )
    if not TEST_DATA.is_file():
        raise FileNotFoundError("Frozen held-out test parquet is missing. Run prepare-data, not training.")

    model_metadata = _read_json(MODEL_METADATA)
    operating_config = _read_json(OPERATING_CONFIG)
    split_metadata = _read_json(SPLIT_METADATA)
    if int(split_metadata.get("test_rows", 0)) <= 0:
        raise ValueError("Split metadata does not contain a valid held-out test row count")
    test_file = (split_metadata.get("parquet_files") or {}).get("test") or {}
    if test_file.get("path") != TEST_DATA.name or int(test_file.get("rows", -1)) != int(
        split_metadata["test_rows"]
    ):
        raise ValueError("Held-out parquet and split metadata do not match")

    model = CatBoostClassifier()
    model.load_model(MODEL_BUNDLE)
    spec = validate_frozen_evaluation_spec(
        model_metadata,
        operating_config,
        saved_model_feature_names=list(model.feature_names_),
        saved_model_tree_count=int(model.tree_count_),
    )
    if _enabled_rule_count() != 0:
        raise ValueError(
            "Enabled rules are present, but the final evaluator does not apply rule overrides. "
            "Freeze a rule-free policy or implement validated rule evaluation before test access."
        )
    return model, spec, model_metadata, split_metadata


def _preflight_summary(spec: FrozenEvaluationSpec, split_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ready",
        "held_out_test_accessed": False,
        "model_name": spec.model_name,
        "model_version": spec.model_version,
        "feature_set": spec.feature_set,
        "feature_count": len(spec.feature_names),
        "scenario": spec.scenario,
        "review_threshold": spec.review_threshold,
        "block_threshold": spec.block_threshold,
        "expected_test_rows": int(split_metadata["test_rows"]),
        "enabled_rules": 0,
        "message": "Preflight passed. No held-out parquet rows were read.",
    }


def _mark_test_access(*, status: str, spec: FrozenEvaluationSpec, error: str | None = None) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "model_version": spec.model_version,
        "model_sha256": _sha256(MODEL_BUNDLE),
        "review_threshold": spec.review_threshold,
        "block_threshold": spec.block_threshold,
        "selection_split": "validation",
        "held_out_test_path": str(TEST_DATA.relative_to(ROOT)),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if error:
        payload["error"] = error
    TEST_ACCESS_RECORD.parent.mkdir(parents=True, exist_ok=True)
    TEST_ACCESS_RECORD.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_and_validate_test(spec: FrozenEvaluationSpec, split_metadata: dict[str, Any]) -> pd.DataFrame:
    columns = list(
        dict.fromkeys(
            ["TransactionID", "TransactionDT", "TransactionAmt", "isFraud", *spec.feature_names]
        )
    )
    frame = pd.read_parquet(TEST_DATA, columns=columns)
    expected_rows = int(split_metadata["test_rows"])
    if len(frame) != expected_rows or frame["TransactionID"].nunique() != expected_rows:
        raise ValueError("Held-out test row count or TransactionID uniqueness differs from split metadata")
    if not frame["isFraud"].isin([0, 1]).all():
        raise ValueError("Held-out labels must contain only 0 and 1")
    amounts = pd.to_numeric(frame["TransactionAmt"], errors="coerce")
    if amounts.isna().any() or (amounts < 0).any():
        raise ValueError("Held-out TransactionAmt values must be finite and non-negative")
    boundaries = split_metadata.get("split_boundaries") or {}
    if (
        float(frame["TransactionDT"].min()) != float(boundaries.get("test_dt_min"))
        or float(frame["TransactionDT"].max()) != float(boundaries.get("test_dt_max"))
    ):
        raise ValueError("Held-out TransactionDT range differs from frozen split metadata")
    return frame


def _write_outputs(
    *,
    frame: pd.DataFrame,
    scores: np.ndarray,
    decisions: np.ndarray,
    binary: dict[str, Any],
    costs: dict[str, Any],
    spec: FrozenEvaluationSpec,
    candidate_metadata: dict[str, Any],
    split_metadata: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    metrics_dir = ARTIFACTS / "metrics"
    models_dir = ARTIFACTS / "models"
    reports_dir = ARTIFACTS / "reports"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    counts = {name: int((decisions == name).sum()) for name in ("APPROVE", "REVIEW", "BLOCK")}
    total = len(decisions)
    final_metrics = {
        "evaluation_status": "complete",
        "generated_at": generated_at,
        "split": "test",
        "held_out_test_accessed": True,
        "model_retrained_for_final_evaluation": False,
        "metric_definition": (
            "Binary precision/recall/F1 use the frozen block threshold; review-band outcomes "
            "are represented separately in cost calculations."
        ),
        "test_transaction_count": total,
        "fraud_count": int(frame["isFraud"].sum()),
        "precision": binary["precision"],
        "recall": binary["recall"],
        "f1": binary["f1"],
        "average_precision": binary["average_precision"],
        "roc_auc": binary["roc_auc"],
        "brier_score": binary["brier_score"],
        "true_positives": binary["true_positives"],
        "false_positives": binary["false_positives"],
        "true_negatives": binary["true_negatives"],
        "false_negatives": binary["false_negatives"],
        "approve_count": counts["APPROVE"],
        "review_count": counts["REVIEW"],
        "block_count": counts["BLOCK"],
        "decision_distribution": {
            name.lower(): {"count": counts[name], "share": counts[name] / total}
            for name in counts
        },
        "confusion_matrix": {
            "true_positives": binary["true_positives"],
            "false_positives": binary["false_positives"],
            "true_negatives": binary["true_negatives"],
            "false_negatives": binary["false_negatives"],
        },
        "detected_fraud_recall": costs["detected_fraud_recall"],
        "fraud_amount_capture_rate": costs["fraud_amount_capture_rate"],
        "false_positive_estimated_cost": costs["false_positive_cost"],
        "false_negative_estimated_cost": costs["fraud_loss"],
        "review_cost": costs["review_cost"],
        "review_expected_residual_cost": costs["review_expected_residual_cost"],
        "total_estimated_cost": costs["total_estimated_cost"],
        "review_threshold": spec.review_threshold,
        "block_threshold": spec.block_threshold,
        "threshold_selection_split": "validation",
        "threshold_selection_reason": spec.selection_reason,
        "business_assumptions": spec.assumptions.to_dict(),
        "business_assumption_status": "ILLUSTRATIVE MERCHANT ASSUMPTIONS",
        "active_rule_count": 0,
    }

    predictions = frame[["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]].copy()
    predictions["risk_score"] = scores
    predictions["decision"] = decisions
    predictions.to_csv(metrics_dir / "final_test_predictions.csv", index=False)
    (metrics_dir / "final_test_calibration.json").write_text(
        json.dumps(calibration_diagnostics(frame["isFraud"].to_numpy(dtype=int), scores), indent=2)
        + "\n",
        encoding="utf-8",
    )

    metadata = {
        "status": "final_evaluated",
        "model_name": spec.model_name,
        "model_version": spec.model_version,
        "trained_at": candidate_metadata["training_timestamp"],
        "evaluated_at": generated_at,
        "feature_set": spec.feature_set,
        "feature_names": list(spec.feature_names),
        "categorical_feature_names": list(spec.categorical_feature_names),
        "training_split": "first 70% by TransactionDT",
        "selection_split": "next 15% by TransactionDT",
        "evaluation_split": "final 15% by TransactionDT",
        "thresholds": {"review": spec.review_threshold, "block": spec.block_threshold},
        "threshold_config_id": f"{spec.model_version}-{spec.scenario}-final",
        "calibration": "No post-hoc calibrator; frozen raw CatBoost probabilities.",
        "model_artifact": str(MODEL_BUNDLE.relative_to(ROOT)),
        "model_sha256": _sha256(MODEL_BUNDLE),
        "model_retrained_for_final_evaluation": False,
        "held_out_test_accessed": True,
    }
    (models_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "generated_at": generated_at,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "random_seed": candidate_metadata["random_seed"],
        "model_parameters": candidate_metadata["catboost_parameters"],
        "best_iteration": candidate_metadata["best_iteration"],
        "actual_tree_count": candidate_metadata["actual_tree_count"],
        "feature_names": list(spec.feature_names),
        "split_boundaries": split_metadata["split_boundaries"],
        "test_policy": "Held-out test used after TRAIN fit and VALIDATION-only selection freeze.",
    }
    (models_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    report = [
        "# Final Held-Out Evaluation",
        "",
        "Generated from the final chronological 15% of the local labeled IEEE-CIS training data.",
        "The saved CatBoost candidate and validation-selected thresholds were used without retraining.",
        "",
        "## Model-derived results",
        "",
        f"- Test transactions: {total:,}",
        f"- Fraud transactions: {int(frame['isFraud'].sum()):,}",
        f"- Precision at block threshold: {float(binary['precision']):.6f}",
        f"- Recall at block threshold: {float(binary['recall']):.6f}",
        f"- F1: {float(binary['f1']):.6f}",
        f"- Average precision / PR-AUC: {float(binary['average_precision']):.6f}",
        f"- ROC-AUC: {float(binary['roc_auc']):.6f}",
        (
            f"- TP / FP / TN / FN: {binary['true_positives']} / {binary['false_positives']} / "
            f"{binary['true_negatives']} / {binary['false_negatives']}"
        ),
        f"- APPROVE / REVIEW / BLOCK: {counts['APPROVE']} / {counts['REVIEW']} / {counts['BLOCK']}",
        "",
        "## Frozen validation operating point",
        "",
        f"- Scenario: {spec.scenario_name}",
        f"- Review threshold: {spec.review_threshold:.4f}",
        f"- Block threshold: {spec.block_threshold:.4f}",
        f"- Selection reason: {spec.selection_reason}.",
        "",
        "## Estimated business cost",
        "",
        "All monetary values below use illustrative merchant assumptions, not industry facts:",
        "",
        f"- Fraud loss: {float(costs['fraud_loss']):.2f} {spec.assumptions.currency}",
        f"- False-positive cost: {float(costs['false_positive_cost']):.2f} {spec.assumptions.currency}",
        f"- Manual review cost: {float(costs['review_cost']):.2f} {spec.assumptions.currency}",
        f"- Total estimated cost: {float(costs['total_estimated_cost']):.2f} {spec.assumptions.currency}",
    ]
    (reports_dir / "final_evaluation.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    FINAL_METRICS.write_text(json.dumps(final_metrics, indent=2) + "\n", encoding="utf-8")
    return final_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen CatBoost candidate once")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate frozen artifacts without reading held-out parquet rows",
    )
    args = parser.parse_args()

    model, spec, candidate_metadata, split_metadata = _load_frozen_inputs()
    summary = _preflight_summary(spec, split_metadata)
    if args.preflight_only:
        print(json.dumps(summary, indent=2))
        return

    generated_at = datetime.now(timezone.utc).isoformat()
    _mark_test_access(status="started", spec=spec)
    try:
        frame = _load_and_validate_test(spec, split_metadata)
        prepared = normalize_catboost_features(
            frame,
            list(spec.feature_names),
            list(spec.categorical_feature_names),
        )
        scores = np.asarray(model.predict_proba(prepared)[:, 1], dtype=float)
        if len(scores) != len(frame) or not np.isfinite(scores).all():
            raise ValueError("Frozen CatBoost candidate returned invalid held-out probabilities")
        binary, costs, decisions = evaluate_frozen_scores(
            labels=frame["isFraud"].to_numpy(dtype=int),
            amounts=frame["TransactionAmt"].to_numpy(dtype=float),
            risk_scores=scores,
            spec=spec,
        )
        final_metrics = _write_outputs(
            frame=frame,
            scores=scores,
            decisions=decisions,
            binary=binary,
            costs=costs,
            spec=spec,
            candidate_metadata=candidate_metadata,
            split_metadata=split_metadata,
            generated_at=generated_at,
        )
    except Exception as exc:
        _mark_test_access(status="failed", spec=spec, error=f"{type(exc).__name__}: {exc}")
        raise
    _mark_test_access(status="complete", spec=spec)
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
