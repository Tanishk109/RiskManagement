export type Decision = "APPROVE" | "REVIEW" | "BLOCK";

export type CostAssumptions = {
  currency: string;
  fraud_loss_fraction: number;
  chargeback_fixed_cost: number;
  legitimate_margin_rate: number;
  false_positive_fixed_cost: number;
  manual_review_cost: number;
  review_fraud_catch_rate: number;
  review_legitimate_approval_rate: number;
};

export type TransactionSummary = {
  transaction_id: string;
  transaction_dt: number;
  amount: number;
  currency: string;
  risk_score: number;
  decision: Decision;
  actual_label: 0 | 1 | null;
  model_error: boolean | null;
  rules_triggered: string[];
  top_factors: Array<{ feature_name: string; feature_value: string | number | null; contribution: number }>;
};

export type MetricSummary = {
  transactions_evaluated: number;
  fraud_cases: number;
  precision: number;
  recall: number;
  f1: number;
  average_precision: number;
  false_positives: number;
  false_negatives: number;
  estimated_total_cost: number;
};

export type CostOutcome = {
  precision: number;
  recall: number;
  false_positives: number;
  false_negatives: number;
  review_volume: number;
  block_volume: number;
  fraud_loss: number;
  false_positive_cost: number;
  review_cost: number;
  total_estimated_cost: number;
};

export type CostSimulationResponse = {
  evaluated: boolean;
  provenance: string;
  current: CostOutcome | null;
  proposed: CostOutcome | null;
  simulation_group_id?: string | null;
};

export type BootstrapResponse = {
  status: "loading" | "ok" | "unavailable";
  evaluated: boolean;
  generated_at: string | null;
  dataset: { name: string; available: boolean; validation_status: string };
  model: { available: boolean; name: string | null; version: string | null; trained_at: string | null; feature_set: string | null };
  metrics: MetricSummary | null;
  decision_distribution: Record<"approve" | "review" | "block", { count: number; share: number }> | null;
  confusion_matrix: { true_positives: number; false_positives: number; true_negatives: number; false_negatives: number } | null;
  transactions: TransactionSummary[];
  reviews: Array<{ id: number | string; transaction_id: string; risk_score: number; primary_factors: string[] }>;
  rules: { active_count: number; evidence_status: string };
  provenance: string;
};
