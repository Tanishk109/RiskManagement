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

export type ProjectStatusResponse = {
  dataset: {
    status: string;
    name: string;
    transactions: number;
    fraud_transactions: number;
    legitimate_transactions: number;
    fraud_prevalence: number;
    identity_rows: number;
    identity_coverage: number;
  };
  split: {
    status: string;
    strategy: string;
    train_rows: number;
    validation_rows: number;
    test_rows: number;
    train_fraction: number;
    validation_fraction: number;
    test_fraction: number;
    train_transaction_dt_min: number;
    train_transaction_dt_max: number;
    validation_transaction_dt_min: number;
    validation_transaction_dt_max: number;
    test_transaction_dt_min: number;
    test_transaction_dt_max: number;
    test_status: string;
  };
  baseline: { status: string; experiment_id: string };
  candidate_model: { status: string; name: string; experiment_id: string };
  threshold_analysis: { status: string };
  rules: { status: string };
  operational_thresholds: { status: string };
  final_test: { status: string; test_status: string };
};

export type ValidationMetrics = {
  average_precision: number;
  roc_auc: number;
  precision_at_0_5: number;
  recall_at_0_5: number;
  f1_at_0_5: number;
  false_positives: number;
  false_negatives: number;
  true_positives: number;
  true_negatives: number;
  threshold: number;
};

export type ModelComparisonResponse = {
  status: string;
  split: "validation";
  held_out_test_status: "sealed";
  threshold: number;
  logistic_regression: { name: string; experiment_id: string; metrics: ValidationMetrics };
  catboost: { name: string; experiment_id: string; metrics: ValidationMetrics };
  average_precision_relative_improvement: number;
  candidate_details: {
    status: string;
    feature_count: number;
    identity_fields_included: boolean;
    class_weight: string;
    identity_ap_loss: number;
    selection_reason: string;
  };
  failure_analysis: {
    label: string;
    slices: Array<{
      slice: string;
      fraud_support: number;
      logistic_recall: number;
      catboost_recall: number;
      absolute_improvement: number;
    }>;
    false_negatives: {
      count: number;
      transaction_amount_total: number;
      transaction_amount_max: number;
    };
  };
  precision_recall_curves: Array<{
    model: string;
    points: Array<{ recall: number; precision: number }>;
  }>;
  provenance: string;
};

export type FeatureImportanceResponse = {
  status: string;
  model: string;
  items: Array<{ feature: string; importance: number }>;
  note: string;
};

export type ValidationFilter =
  | "all"
  | "true_fraud"
  | "true_legitimate"
  | "true_positive"
  | "false_positive"
  | "false_negative"
  | "true_negative"
  | "high_risk"
  | "high_value";

export type ValidationTransaction = {
  transaction_id: string;
  transaction_dt: number;
  transaction_amount: number;
  actual_label: 0 | 1;
  fraud_probability: number;
  predicted_label_at_0_5: 0 | 1;
  outcome: "TRUE_POSITIVE" | "FALSE_POSITIVE" | "FALSE_NEGATIVE" | "TRUE_NEGATIVE";
  model_error: boolean;
  features: Record<string, string | number | null>;
};

export type ValidationTransactionPage = {
  status: string;
  split: "validation";
  threshold: number;
  filter: ValidationFilter;
  page: number;
  page_size: number;
  total: number;
  page_count: number;
  items: ValidationTransaction[];
};

export type InterestingCasesResponse = {
  status: string;
  split: "validation";
  cases: Array<ValidationTransaction & { case_type: string }>;
};
