# Architecture

## Decision path

```text
transaction features
  │
  ├─ frozen preprocessing + model ── risk probability
  │                                   │
  └─ validation-derived YAML rules ───┤
                                      ▼
                       dual-threshold decision engine
                           APPROVE / REVIEW / BLOCK
                                      │
                     ┌────────────────┴──────────────┐
                     ▼                               ▼
               PostgreSQL record              Cost simulation
                     │                    held-out labels + amounts
                     ▼
              analyst review + reason
```

## Boundaries

- `ml/src/merchantshield_ml` owns dataset loading, chronological splitting, preprocessing, model training, evaluation, explanation, thresholds, and the cost formula.
- `services/api` owns artifact provenance, score orchestration, rules, persistence, review decisions, and HTTP validation.
- `apps/web` is the frontend package boundary. OpenAI Sites currently requires the vinext application at the repository root, so the root `app/` is the deployable adapter and `apps/web/app/page.tsx` re-exports it.
- PostgreSQL is the only operational persistence layer. In-memory SQLite is used only by isolated unit tests and is never a runtime option.
- The hosted same-origin API adapter returns an unevaluated state because protected models and IEEE-CIS rows are not embedded in the public build.

## Source-of-truth rules

- Model metadata and final metrics: normalized `model_runs` columns in PostgreSQL after artifact sync.
- Frozen thresholds: `threshold_configs` in PostgreSQL, linked to their model and validation-time cost configuration.
- Row-level evaluation predictions: local-only `artifacts/metrics/final_test_predictions.csv`.
- Raw and processed ML rows: ignored CSV under `data/raw/` and Parquet under `data/processed/`; never PostgreSQL.
- Analyst decisions: `review_cases` in PostgreSQL.
- Merchant assumptions: `cost_configs`, always labeled separately from model-derived values.
- Model and metric JSON artifacts are an ML-to-runtime handoff, not the application read path.

## Database tables

- `transactions`: selected identifiers, time, amount, optional evaluation label, score, decision, and foreign keys to model/threshold configuration.
- `prediction_reasons`: ranked source feature name/value and contribution.
- `rule_hits`: one row per triggered defensive rule and transaction.
- `review_cases`: open/decided status, model decision, reviewer decision/reason/identity/timestamp.
- `model_runs`: versioned model metadata and typed held-out metric columns.
- `threshold_configs`: immutable threshold identities, values, provenance, and active status.
- `cost_configs`: deduplicated merchant assumptions in typed columns.
- `cost_simulations`: paired current/proposed scenario history with typed outcomes.

Only `model_runs.metadata_json` uses PostgreSQL JSONB, for genuinely variable metadata such as the selected feature list and textual split descriptions. It is not used for predictions, rules, thresholds, costs, or review state.

## Security and fraud-safety

Requests are validated with Pydantic; SQL access uses SQLAlchemy; secrets and protected data remain ignored. Scoring rejects fraud labels and named future-outcome fields. Rules reject those fields too. The system does not provide attack generation, evasion, or offensive simulation.
