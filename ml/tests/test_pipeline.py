from __future__ import annotations

from pathlib import Path

import joblib
from merchantshield_ml.baseline import fit_baseline
from merchantshield_ml.data import load_ieee_cis
from merchantshield_ml.features import validate_feature_schema

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
    try:
        validate_feature_schema(frame, ["TransactionAmt", "isFraud"])
    except ValueError as exc:
        assert "label" in str(exc)
    else:
        raise AssertionError("Feature schema accepted isFraud")
