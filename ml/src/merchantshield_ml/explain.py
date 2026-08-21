from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def _feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    return [str(name).replace("numeric__", "").replace("categorical__", "") for name in preprocessor.get_feature_names_out()]


def top_contributions(pipeline: Pipeline, frame: pd.DataFrame, *, limit: int = 5) -> list[dict[str, Any]]:
    """Return per-row contributions only when the fitted estimator exposes them.

    Masked feature names remain untouched; this function never assigns invented
    business semantics to IEEE-CIS columns.
    """
    if len(frame) != 1:
        raise ValueError("top_contributions expects exactly one transaction")
    preprocessor = pipeline.named_steps.get("preprocessor")
    classifier = pipeline.named_steps.get("classifier")
    if preprocessor is None or classifier is None:
        return []
    transformed = preprocessor.transform(frame)
    names = _feature_names(pipeline)

    values: np.ndarray | None = None
    if hasattr(classifier, "coef_"):
        row = transformed.toarray()[0] if hasattr(transformed, "toarray") else np.asarray(transformed)[0]
        values = row * np.asarray(classifier.coef_[0])
    elif classifier.__class__.__name__.startswith("XGB"):
        try:
            import xgboost as xgb

            contributions = classifier.get_booster().predict(xgb.DMatrix(transformed), pred_contribs=True)
            values = np.asarray(contributions[0][:-1])
        except (AttributeError, ValueError, TypeError):
            return []
    if values is None or len(values) != len(names):
        return []

    ranked = np.argsort(np.abs(values))[::-1][:limit]
    return [
        {
            "feature_name": names[index],
            "feature_value": None,
            "contribution": float(values[index]),
        }
        for index in ranked
        if float(abs(values[index])) > 0
    ]
