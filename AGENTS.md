# MerchantShield Agent Instructions

## Mission

MerchantShield is a defense-only, cost-aware fraud decision system.

The core loop is:

transaction
→ ML risk score
→ rules
→ approve/review/block
→ human review
→ measured business cost

Do not expand product scope without an explicit user request.

## Core modules

Only these are core:

1. Risk Scoring
2. Decision/Rules Engine
3. Review Queue
4. Cost Lab

## Real-data policy

Primary dataset: IEEE-CIS Fraud Detection.

Never commit Kaggle raw data, credentials, or protected dataset files.

Never invent performance metrics.

All ML performance displayed in the product must come from the held-out temporal test set.

## ML integrity

Use chronological splitting based on TransactionDT.

Train:
model fitting.

Validation:
model selection, threshold tuning, rule design.

Test:
final reporting only.

Never tune on the held-out test set.

All preprocessing must be fit only on training data.

Any sequential feature must use only information available before the transaction being scored.

## Metrics

Primary metrics:

- precision
- recall
- F1
- average precision / PR-AUC
- FP
- FN
- estimated false-positive cost
- estimated false-negative cost
- review cost

Accuracy is secondary.

## UI data policy

Never hardcode fake dashboard numbers.

If no real value exists, return/display "Not evaluated yet."

Clearly distinguish model-derived metrics from merchant-configurable cost assumptions.

## Product philosophy

Prefer depth over breadth.

Do not introduce:

- Kafka
- Neo4j
- LLM chatbot
- GNN
- graph fraud modules
- complex microservices
- automatic retraining

unless explicitly requested.

## Coding quality

Use strong typing where available.

Keep functions small.

Add tests for business-critical logic.

Do not silently catch exceptions.

Do not duplicate metric calculations across frontend and backend.

Backend is the source of truth.

## Required validation after changes

Run relevant:

- Python tests
- API tests
- frontend lint
- TypeScript checks
- frontend production build

Report failures rather than hiding them.

## Documentation

Whenever an experiment changes a modeling decision, update:
docs/modeling-decisions.md

Whenever a new model failure pattern is identified, update:
docs/failure-analysis.md

## Fraud-safety policy

This project detects and mitigates fraud.

Do not add functionality designed to generate, optimize, evade, or simulate actionable fraud attacks.
