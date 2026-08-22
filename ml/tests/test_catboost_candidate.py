from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from catboost import CatBoostClassifier
from merchantshield_ml.catboost_candidate import (
    CATBOOST_EXPERIMENT_FIELDS,
    CATEGORICAL_MISSING_VALUE,
    normalize_catboost_features,
    prediction_artifact,
    run_catboost_experiment,
    validate_catboost_experiment_record,
    validate_catboost_feature_schema,
    validate_prediction_artifact,
)
from merchantshield_ml.data import load_ieee_cis
from merchantshield_ml.processed import load_baseline_partition

FIXTURES = Path(__file__).parent / "fixtures"
FEATURES = ["TransactionAmt", "D1", "ProductCD", "identity_available", "DeviceType"]
CATEGORICAL_FEATURES = ["ProductCD", "identity_available", "DeviceType"]
PARAMETERS = {
    "loss_function": "Logloss",
    "eval_metric": "PRAUC:type=Classic",
    "iterations": 20,
    "learning_rate": 0.1,
    "depth": 3,
    "l2_leaf_reg": 3.0,
    "early_stopping_rounds": 5,
    "thread_count": 1,
    "allow_writing_files": False,
}


def _fixture_frame():
    frame, _ = load_ieee_cis(
        FIXTURES / "train_transaction.csv",
        FIXTURES / "train_identity.csv",
        feature_names=FEATURES,
    )
    return frame


def _fixture_run():
    frame = _fixture_frame()
    validation = frame.iloc[9:].copy()
    validation.loc[validation.index[0], "ProductCD"] = "VALIDATION_ONLY"
    return frame, validation, run_catboost_experiment(
        frame.iloc[:9],
        validation,
        experiment_id="cb-fixture",
        feature_set="fixture",
        features=FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        auto_class_weights=None,
        parameters=PARAMETERS,
        random_seed=42,
    )


def test_categorical_missing_values_are_normalized_consistently():
    frame = _fixture_frame()
    prepared = normalize_catboost_features(frame, FEATURES, CATEGORICAL_FEATURES)

    assert prepared["DeviceType"].isna().sum() == 0
    assert CATEGORICAL_MISSING_VALUE in set(prepared["DeviceType"])
    assert set(prepared["identity_available"]) == {"False", "True"}
    assert prepared["D1"].isna().sum() == frame["D1"].isna().sum()


@pytest.mark.parametrize("forbidden", ["isFraud", "TransactionID"])
def test_catboost_schema_rejects_forbidden_predictors(forbidden):
    frame = _fixture_frame()
    with pytest.raises(ValueError, match=forbidden):
        validate_catboost_feature_schema(
            frame,
            [*FEATURES, forbidden],
            CATEGORICAL_FEATURES,
        )


def test_catboost_schema_requires_categorical_fields_in_features():
    frame = _fixture_frame()
    with pytest.raises(ValueError, match="absent from the feature set"):
        validate_catboost_feature_schema(frame, FEATURES, [*CATEGORICAL_FEATURES, "DeviceInfo"])


def test_catboost_probability_schema_serialization_and_deterministic_inference(tmp_path: Path):
    _, validation, run = _fixture_run()

    assert run.fraud_probabilities.shape == (len(validation),)
    assert np.isfinite(run.fraud_probabilities).all()
    assert ((run.fraud_probabilities >= 0) & (run.fraud_probabilities <= 1)).all()
    validate_catboost_experiment_record(run.record)
    assert set(CATBOOST_EXPERIMENT_FIELDS).issubset(run.record)

    path = tmp_path / "candidate.cbm"
    run.model.save_model(path)
    restored = CatBoostClassifier()
    restored.load_model(path)
    prepared = normalize_catboost_features(validation, run.features, run.categorical_features)
    restored_probabilities = restored.predict_proba(prepared)[:, 1]
    np.testing.assert_allclose(restored_probabilities, run.fraud_probabilities, rtol=0, atol=0)


def test_validation_prediction_artifact_structure():
    _, validation, run = _fixture_run()
    artifact = prediction_artifact(validation, run)

    validate_prediction_artifact(artifact, expected_rows=len(validation))
    assert list(artifact.columns) == [
        "TransactionID",
        "actual_label",
        "fraud_probability",
        "predicted_label_at_0_5",
        "experiment_id",
        "model_version",
    ]


def test_catboost_uses_the_sealed_model_selection_loader(tmp_path: Path):
    with pytest.raises(ValueError, match="may load only train, validation"):
        load_baseline_partition(tmp_path, "test", FEATURES)


def test_catboost_training_script_has_no_held_out_loader_path():
    script = (Path(__file__).parents[1] / "scripts/train_catboost.py").read_text(encoding="utf-8")

    assert "test.parquet" not in script
    assert "load_splits" not in script
    assert "load_processed_splits" not in script
