from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import get_settings


class ArtifactUnavailable(RuntimeError):
    """Raised when an operation requires evidence that has not been generated."""


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactUnavailable(f"Required artifact is not available: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactUnavailable(f"Artifact is unreadable: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ArtifactUnavailable(f"Artifact must contain a JSON object: {path.name}")
    return payload


def load_metrics() -> dict[str, Any]:
    payload = read_json(get_settings().metrics_path)
    if payload.get("split") != "test" or payload.get("evaluation_status") != "complete":
        raise ArtifactUnavailable("Final held-out test evaluation is not complete")
    return payload


def load_model_metadata() -> dict[str, Any]:
    payload = read_json(get_settings().model_metadata_path)
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict) or "review" not in thresholds or "block" not in thresholds:
        raise ArtifactUnavailable("Frozen model metadata does not contain both thresholds")
    return payload
