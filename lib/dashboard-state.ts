import type { BootstrapResponse } from "./api-types";

export function unevaluatedDashboardState(): BootstrapResponse {
  return {
    status: "ok",
    evaluated: false,
    generated_at: null,
    dataset: {
      name: "IEEE-CIS Fraud Detection",
      available: false,
      validation_status: "Not evaluated yet",
    },
    model: {
      available: false,
      name: null,
      version: null,
      trained_at: null,
      feature_set: null,
    },
    metrics: null,
    decision_distribution: null,
    confusion_matrix: null,
    transactions: [],
    reviews: [],
    rules: {
      active_count: 0,
      evidence_status: "Awaiting validation error analysis",
    },
    provenance: "Not evaluated yet",
  };
}
