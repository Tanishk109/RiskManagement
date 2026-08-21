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
- PostgreSQL is the full local persistence layer. SQLite is an explicit development/test fallback only.
- The hosted same-origin API adapter returns an unevaluated state because protected models and IEEE-CIS rows are not embedded in the public build.

## Source-of-truth rules

- Model metrics: `artifacts/metrics/final_test_metrics.json`, created from the final chronological 15% only.
- Thresholds and model feature schema: `artifacts/models/model_metadata.json`.
- Row-level evaluation predictions: local-only `artifacts/metrics/final_test_predictions.csv`.
- Analyst decisions: `review_cases` in PostgreSQL.
- Merchant assumptions: request payload/config, always labeled separately from model-derived values.

## Database tables

- `transactions`: identifiers, time, amount, label when evaluation mode permits, score, model version, decision, rule hits.
- `prediction_reasons`: source feature name/value and contribution.
- `review_cases`: open/decided status, model decision, reviewer decision/reason/timestamp.
- `model_runs`: versioned feature set and metrics document.
- `cost_configs`: named merchant assumptions.

## Security and fraud-safety

Requests are validated with Pydantic; SQL access uses SQLAlchemy; secrets and protected data remain ignored. Scoring rejects fraud labels and named future-outcome fields. Rules reject those fields too. The system does not provide attack generation, evasion, or offensive simulation.
