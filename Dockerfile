FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/ml/src:/workspace/services/api

WORKDIR /workspace

# Install runtime dependencies first for Docker layer caching
COPY services/api/requirements.txt /tmp/requirements.txt

RUN pip install \
    --no-cache-dir \
    --default-timeout=300 \
    --retries 10 \
    -r /tmp/requirements.txt

# Copy runtime application code and configuration.
COPY ml/src ./ml/src
COPY ml/configs ./ml/configs
COPY ml/scripts/sync_runtime_evidence.py ./ml/scripts/sync_runtime_evidence.py
COPY services/api ./services/api
COPY rules ./rules

# Copy approved runtime model and aggregate evidence artifacts. Docker context
# policy excludes every dataset, Parquet, CSV prediction export, and joblib file.
COPY artifacts ./artifacts
COPY data/processed/ieee-cis/split_metadata.json ./data/processed/ieee-cis/split_metadata.json
COPY data/processed/ieee-cis/validation.parquet ./data/processed/ieee-cis/validation.parquet
RUN test -f /workspace/artifacts/models/catboost_candidate.cbm \
    && test -f /workspace/artifacts/models/catboost_candidate_metadata.json \
    && test -f /workspace/artifacts/models/validation_operating_config.json \
    && test -f /workspace/artifacts/models/return_risk_catboost.cbm \
    && test -f /workspace/artifacts/metrics/final_test_metrics.json \
    && test -f /workspace/artifacts/metrics/experiments.csv \
    && test -f /workspace/artifacts/metrics/threshold_grid.parquet \
    && test -f /workspace/artifacts/predictions/baseline_validation.parquet \
    && test -f /workspace/artifacts/predictions/catboost_identity_ablation_validation.parquet \
    && test -f /workspace/artifacts/reports/catboost_feature_importance.csv \
    && test -f /workspace/data/processed/ieee-cis/split_metadata.json \
    && test -f /workspace/data/processed/ieee-cis/validation.parquet \
    && test ! -e /workspace/data/raw \
    && test ! -e /workspace/data/processed/ieee-cis/test.parquet \
    && test ! -e /workspace/artifacts/metrics/final_test_predictions.csv

# Non-root runtime user
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /workspace

USER appuser

EXPOSE 10000

CMD ["sh", "-c", "python -m alembic -c /workspace/services/api/alembic.ini upgrade head && python /workspace/ml/scripts/sync_runtime_evidence.py && exec uvicorn app.main:app --app-dir /workspace/services/api --host 0.0.0.0 --port ${PORT:-10000}"]
