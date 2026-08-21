from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from merchantshield_ml.cost import CostAssumptions
from merchantshield_ml.data import load_ieee_cis
from merchantshield_ml.features import load_feature_sets
from merchantshield_ml.split import TemporalSplits, temporal_split

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = ROOT / "data/raw"
ARTIFACTS = ROOT / "artifacts"
FEATURE_CONFIG = ROOT / "ml/configs/feature_sets.yaml"
TRAINING_CONFIG = ROOT / "ml/configs/training.yaml"
COST_CONFIG = ROOT / "ml/configs/cost_assumptions.yaml"


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML object in {path}")
    return payload


def feature_sets() -> dict[str, list[str]]:
    return load_feature_sets(FEATURE_CONFIG)


def training_config() -> dict[str, Any]:
    return read_yaml(TRAINING_CONFIG)


def cost_assumptions() -> CostAssumptions:
    return CostAssumptions(**read_yaml(COST_CONFIG))


def load_splits(features: list[str]) -> TemporalSplits:
    config = training_config()
    frame, _ = load_ieee_cis(
        RAW_DATA / "train_transaction.csv",
        RAW_DATA / "train_identity.csv",
        feature_names=features,
    )
    return temporal_split(
        frame,
        train_fraction=float(config["train_fraction"]),
        validation_fraction=float(config["validation_fraction"]),
    )


def replace_marked_section(path: Path, marker: str, body: str) -> None:
    start = f"<!-- {marker}:START -->"
    end = f"<!-- {marker}:END -->"
    content = path.read_text(encoding="utf-8")
    replacement = f"{start}\n{body.rstrip()}\n{end}"
    updated, count = re.subn(re.escape(start) + r".*?" + re.escape(end), replacement, content, flags=re.DOTALL)
    if count != 1:
        raise ValueError(f"Expected one {marker} marker pair in {path}")
    path.write_text(updated, encoding="utf-8")
