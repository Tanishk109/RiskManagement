FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace
COPY ml ./ml
COPY services/api ./services/api
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 600 ./ml ./services/api

EXPOSE 8000
CMD ["sh", "-c", "alembic -c services/api/alembic.ini upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
