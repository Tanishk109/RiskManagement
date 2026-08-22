from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
from merchantshield_ml.baseline import fit_baseline
from merchantshield_ml.baseline_analysis import (
    EXPERIMENT_FIELDS,
    run_logistic_experiment,
    validate_experiment_record,
)
from merchantshield_ml.data import load_ieee_cis
from merchantshield_ml.features import validate_feature_schema, validate_model_features

FIXTURES = Path(__file__).parent / "fixtures"


def test_missing_values_and_unseen_categories_are_supported(tmp_path: Path):
    features = ["TransactionAmt", "D1", "ProductCD", "DeviceType"]
    frame, _ = load_ieee_cis(
        FIXTURES / "train_transaction.csv",
        FIXTURES / "train_identity.csv",
        feature_names=features,
    )
    train = frame.iloc[:9]
    model = fit_baseline(train, features)
    sample = frame.iloc[[10]].copy()
    sample.loc[sample.index[0], "ProductCD"] = "UNSEEN"
    probability = model.predict_proba(sample[features])[0, 1]
    assert 0 <= probability <= 1
    probabilities = model.predict_proba(frame[features])
    assert probabilities.shape == (len(frame), 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))

    path = tmp_path / "model.joblib"
    joblib.dump({"pipeline": model, "feature_names": features}, path)
    restored = joblib.load(path)
    assert restored["pipeline"].predict_proba(sample[features]).shape == (1, 2)


def test_feature_schema_rejects_label_leakage():
    frame, _ = load_ieee_cis(
        FIXTURES / "train_transaction.csv",
        FIXTURES / "train_identity.csv",
        feature_names=["TransactionAmt"],
    )
    with pytest.raises(ValueError, match="isFraud"):
        validate_feature_schema(frame, ["TransactionAmt", "isFraud"])


@pytest.mark.parametrize("forbidden", ["isFraud", "TransactionID"])
def test_model_features_reject_forbidden_audit_fields(forbidden):
    with pytest.raises(ValueError, match=forbidden):
        validate_model_features(["TransactionAmt", forbidden])


def test_preprocessor_statistics_and_categories_are_fit_on_train_only():
    features = ["TransactionAmt", "D1", "ProductCD"]
    frame, _ = load_ieee_cis(
        FIXTURES / "train_transaction.csv",
        FIXTURES / "train_identity.csv",
        feature_names=features,
    )
    train = frame.iloc[:9].copy()
    validation = frame.iloc[9:].copy()
    validation.loc[validation.index[0], "ProductCD"] = "VALIDATION_ONLY"
    validation.loc[validation.index[0], "TransactionAmt"] = 999_999.0

    model = fit_baseline(train, features, random_state=42, max_iter=5_000)
    preprocessor = model.named_steps["preprocessor"]
    numeric_imputer = preprocessor.named_transformers_["numeric"].named_steps["impute"]
    categorical_encoder = preprocessor.named_transformers_["categorical"].named_steps["encode"]

    assert numeric_imputer.statistics_[0] == pytest.approx(train["TransactionAmt"].median())
    assert "VALIDATION_ONLY" not in categorical_encoder.categories_[0]
    assert model.predict_proba(validation[features]).shape == (len(validation), 2)


def test_baseline_training_is_deterministic_for_fixed_seed():
    features = ["TransactionAmt", "D1", "ProductCD"]
    frame, _ = load_ieee_cis(
        FIXTURES / "train_transaction.csv",
        FIXTURES / "train_identity.csv",
        feature_names=features,
    )
    train = frame.iloc[:9]
    first = fit_baseline(train, features, random_state=17, max_iter=5_000)
    second = fit_baseline(train, features, random_state=17, max_iter=5_000)

    np.testing.assert_allclose(
        first.predict_proba(frame[features]),
        second.predict_proba(frame[features]),
        rtol=0,
        atol=1e-12,
    )


def test_logistic_experiment_artifact_schema():
    features = ["TransactionAmt", "D1", "ProductCD"]
    frame, _ = load_ieee_cis(
        FIXTURES / "train_transaction.csv",
        FIXTURES / "train_identity.csv",
        feature_names=features,
    )
    run = run_logistic_experiment(
        frame.iloc[:9],
        frame.iloc[9:],
        experiment_id="fixture-logistic",
        feature_set="fixture",
        features=features,
        class_weight="balanced",
        random_state=42,
        solver="saga",
        max_iter=5_000,
    )

    validate_experiment_record(run.record)
    assert set(EXPERIMENT_FIELDS).issubset(run.record)
    assert run.record["training_rows"] == 9
    assert run.record["validation_rows"] == 3
    assert len(run.fraud_probabilities) == 3
