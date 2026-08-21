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
    database_url: str = "sqlite+pysqlite:///./merchantshield.db"
    model_path: Path = REPOSITORY_ROOT / "artifacts/models/model.joblib"
    model_metadata_path: Path = REPOSITORY_ROOT / "artifacts/models/model_metadata.json"
    metrics_path: Path = REPOSITORY_ROOT / "artifacts/metrics/final_test_metrics.json"
    predictions_path: Path = REPOSITORY_ROOT / "artifacts/metrics/final_test_predictions.csv"
    rules_path: Path = REPOSITORY_ROOT / "rules/merchant_rules.yaml"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])


def _path_from_env(name: str, fallback: Path) -> Path:
    return Path(os.getenv(name, str(fallback))).expanduser().resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    defaults = Settings()
    origins = [item.strip() for item in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if item.strip()]
    return Settings(
        environment=os.getenv("ENVIRONMENT", defaults.environment),
        database_url=os.getenv("DATABASE_URL", defaults.database_url),
        model_path=_path_from_env("MODEL_PATH", defaults.model_path),
        model_metadata_path=_path_from_env("MODEL_METADATA_PATH", defaults.model_metadata_path),
        metrics_path=_path_from_env("METRICS_PATH", defaults.metrics_path),
        predictions_path=_path_from_env("PREDICTIONS_PATH", defaults.predictions_path),
        rules_path=_path_from_env("RULES_PATH", defaults.rules_path),
        cors_origins=origins,
    )
