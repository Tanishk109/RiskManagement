from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    app_name: str = "MerchantShield API"
    environment: str = "development"
    database_url: str = "postgresql+psycopg://merchantshield:merchantshield@localhost:5432/merchantshield"
    model_path: Path = REPOSITORY_ROOT / "artifacts/models/model.joblib"
    model_metadata_path: Path = REPOSITORY_ROOT / "artifacts/models/model_metadata.json"
    metrics_path: Path = REPOSITORY_ROOT / "artifacts/metrics/final_test_metrics.json"
    predictions_path: Path = REPOSITORY_ROOT / "artifacts/metrics/final_test_predictions.csv"
    rules_path: Path = REPOSITORY_ROOT / "rules/merchant_rules.yaml"
    eda_summary_path: Path = REPOSITORY_ROOT / "artifacts/reports/eda_summary.json"
    split_metadata_path: Path = REPOSITORY_ROOT / "data/processed/ieee-cis/split_metadata.json"
    baseline_metrics_path: Path = REPOSITORY_ROOT / "artifacts/metrics/baseline_validation.json"
    catboost_metrics_path: Path = REPOSITORY_ROOT / "artifacts/metrics/catboost_validation.json"
    catboost_metadata_path: Path = REPOSITORY_ROOT / "artifacts/models/catboost_candidate_metadata.json"
    catboost_model_path: Path = REPOSITORY_ROOT / "artifacts/models/catboost_candidate.cbm"
    experiments_path: Path = REPOSITORY_ROOT / "artifacts/metrics/experiments.csv"
    feature_importance_path: Path = REPOSITORY_ROOT / "artifacts/reports/catboost_feature_importance.csv"
    baseline_validation_predictions_path: Path = (
        REPOSITORY_ROOT / "artifacts/predictions/baseline_validation.parquet"
    )
    catboost_validation_predictions_path: Path = (
        REPOSITORY_ROOT / "artifacts/predictions/catboost_identity_ablation_validation.parquet"
    )
    validation_data_path: Path = REPOSITORY_ROOT / "data/processed/ieee-cis/validation.parquet"
    threshold_analysis_path: Path = REPOSITORY_ROOT / "artifacts/metrics/threshold_analysis.json"
    threshold_grid_path: Path = REPOSITORY_ROOT / "artifacts/metrics/threshold_grid.parquet"
    validation_operating_config_path: Path = (
        REPOSITORY_ROOT / "artifacts/models/validation_operating_config.json"
    )
    merchant_scenarios_path: Path = REPOSITORY_ROOT / "ml/configs/merchant_scenarios.yaml"
    evidence_storage_root: Path = REPOSITORY_ROOT / "data/uploads/chargebacks"
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:3001",
            "http://localhost:3002",
        ]
    )


def _path_from_env(name: str, fallback: Path) -> Path:
    return Path(os.getenv(name, str(fallback))).expanduser().resolve()


def operational_database_url(value: str) -> str:
    """Normalize provider URLs onto the installed psycopg v3 SQLAlchemy driver."""

    if value.startswith("postgres://"):
        return "postgresql+psycopg://" + value.removeprefix("postgres://")
    if value.startswith("postgresql://"):
        return "postgresql+psycopg://" + value.removeprefix("postgresql://")
    if value.startswith("postgresql+psycopg://"):
        return value
    raise ValueError("DATABASE_URL must point to PostgreSQL (local Docker, Neon, or Supabase)")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    defaults = Settings()
    origins = [
        item.strip()
        for item in os.getenv(
            "CORS_ORIGINS", "http://localhost:3000,http://localhost:3001,http://localhost:3002"
        ).split(",")
        if item.strip()
    ]
    return Settings(
        environment=os.getenv("ENVIRONMENT", defaults.environment),
        database_url=os.getenv("DATABASE_URL", defaults.database_url),
        model_path=_path_from_env("MODEL_PATH", defaults.model_path),
        model_metadata_path=_path_from_env("MODEL_METADATA_PATH", defaults.model_metadata_path),
        metrics_path=_path_from_env("METRICS_PATH", defaults.metrics_path),
        predictions_path=_path_from_env("PREDICTIONS_PATH", defaults.predictions_path),
        rules_path=_path_from_env("RULES_PATH", defaults.rules_path),
        eda_summary_path=_path_from_env("EDA_SUMMARY_PATH", defaults.eda_summary_path),
        split_metadata_path=_path_from_env("SPLIT_METADATA_PATH", defaults.split_metadata_path),
        baseline_metrics_path=_path_from_env("BASELINE_METRICS_PATH", defaults.baseline_metrics_path),
        catboost_metrics_path=_path_from_env("CATBOOST_METRICS_PATH", defaults.catboost_metrics_path),
        catboost_metadata_path=_path_from_env("CATBOOST_METADATA_PATH", defaults.catboost_metadata_path),
        catboost_model_path=_path_from_env("CATBOOST_MODEL_PATH", defaults.catboost_model_path),
        experiments_path=_path_from_env("EXPERIMENTS_PATH", defaults.experiments_path),
        feature_importance_path=_path_from_env(
            "FEATURE_IMPORTANCE_PATH", defaults.feature_importance_path
        ),
        baseline_validation_predictions_path=_path_from_env(
            "BASELINE_VALIDATION_PREDICTIONS_PATH", defaults.baseline_validation_predictions_path
        ),
        catboost_validation_predictions_path=_path_from_env(
            "CATBOOST_VALIDATION_PREDICTIONS_PATH", defaults.catboost_validation_predictions_path
        ),
        validation_data_path=_path_from_env("VALIDATION_DATA_PATH", defaults.validation_data_path),
        threshold_analysis_path=_path_from_env(
            "THRESHOLD_ANALYSIS_PATH", defaults.threshold_analysis_path
        ),
        threshold_grid_path=_path_from_env("THRESHOLD_GRID_PATH", defaults.threshold_grid_path),
        validation_operating_config_path=_path_from_env(
            "VALIDATION_OPERATING_CONFIG_PATH", defaults.validation_operating_config_path
        ),
        merchant_scenarios_path=_path_from_env(
            "MERCHANT_SCENARIOS_PATH", defaults.merchant_scenarios_path
        ),
        evidence_storage_root=_path_from_env(
            "EVIDENCE_STORAGE_ROOT", defaults.evidence_storage_root
        ),
        cors_origins=origins,
    )
