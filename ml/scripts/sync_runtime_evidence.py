from __future__ import annotations

from app.database import SessionLocal
from app.services.evidence_store import sync_evidence_artifacts


def main() -> None:
    with SessionLocal() as db:
        model_run, threshold = sync_evidence_artifacts(db)
    print(
        "Synced model "
        f"{model_run.model_version} and threshold config {threshold.config_key} to PostgreSQL."
    )


if __name__ == "__main__":
    main()
