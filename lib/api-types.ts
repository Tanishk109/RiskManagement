export type Decision = "APPROVE" | "REVIEW" | "BLOCK";

export type RiskFeatureName =
  | "TransactionAmt"
  | "ProductCD"
  | "card4"
  | "card6"
  | "P_emaildomain"
  | "C1"
  | "C2"
  | "C3"
  | "C4"
  | "C5"
  | "D1"
  | "D2"
  | "D3";

export type RiskFeaturePayload = Record<RiskFeatureName, string | number | null>;

export type RiskScoreResponse = {
  fraud_probability: number;
  risk_score: number;
  decision: Decision;
  rules_triggered: string[];
  top_factors: [];
  model_version: string;
  threshold_config_id: string;
  threshold_configuration: {
    id: string;
    status: string;
    selection_split: "validation";
    scenario: string;
    review_threshold: number;
    block_threshold: number;
    provisional: true;
  };
  feature_schema: RiskFeatureName[];
  held_out_test_accessed: false;
};

export type ValidationScoringTransaction = {
  status: string;
  split: "validation";
  held_out_test_status: "sealed";
  transaction_id: string;
  transaction_dt: number;
  features: RiskFeaturePayload;
  ground_truth_revealed: false;
  note: string;
};

export type ValidationRiskCheckCases = {
  status: string;
  split: "validation";
  held_out_test_status: "sealed";
  ground_truth_hidden: true;
  cases: Array<{
    case_type: string;
    label: string;
    description: string;
    transaction_id: string;
    transaction_amount: number;
  }>;
};

export type ValidationRiskGroundTruth = {
  transaction_id: string;
  split: "validation";
  actual_label: 0 | 1;
  ground_truth: "FRAUD" | "LEGITIMATE";
  note: string;
};

export type BatchScoreResponse = {
  summary: {
    rows_received: number;
    rows_processed: number;
    approved: number;
    reviewed: number;
    blocked: number;
    invalid_rows: number;
  };
  results: Array<{
    row: number;
    transaction_id: string;
    fraud_probability: number;
    decision: Decision;
  }>;
  invalid_rows: Array<{
    row: number;
    transaction_id: string | null;
    errors: string[];
  }>;
  model_version: string;
  threshold_configuration: RiskScoreResponse["threshold_configuration"];
  feature_schema: RiskFeatureName[];
  upload_persisted: false;
  held_out_test_accessed: false;
};

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

export type LegacyCostSimulationResponse = {
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
  business_decision: Decision | null;
  review_threshold: number | null;
  block_threshold: number | null;
  scenario_id: string | null;
  scenario_name: string | null;
  estimated_decision_cost: number | null;
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

export type CostScenario = {
  id: string;
  name: string;
  description: string;
  assumptions: CostAssumptions;
  validation_configuration: { review_threshold: number; block_threshold: number };
};

export type CostScenariosResponse = {
  status: string;
  split: "validation";
  held_out_test_status: string;
  assumption_status: string;
  cost_output_label: string;
  default_scenario_id: string;
  default_review_threshold: number;
  default_block_threshold: number;
  default_review_capacity: number;
  review_capacities: Array<number | null>;
  scenarios: CostScenario[];
};

export type ValidationCostMetrics = {
  currency: string;
  transaction_count: number;
  fraud_count: number;
  legitimate_count: number;
  approve_count: number;
  review_count: number;
  block_count: number;
  approve_rate: number;
  review_rate: number;
  block_rate: number;
  fraud_approved: number;
  fraud_reviewed: number;
  fraud_blocked: number;
  legitimate_blocked: number;
  block_precision: number;
  block_recall: number;
  detected_precision: number;
  detected_fraud_recall: number;
  false_positives: number;
  false_negatives: number;
  total_fraud_amount: number;
  fraud_amount_approved: number;
  fraud_amount_reviewed: number;
  fraud_amount_blocked: number;
  captured_fraud_amount: number;
  fraud_amount_capture_rate: number;
  approved_fraud_loss: number;
  fraud_loss: number;
  false_positive_cost: number;
  manual_review_cost_total: number;
  review_total_cost: number;
  total_estimated_cost: number;
};

export type CostSimulationResponse = {
  status: string;
  split: "validation";
  held_out_test_status: string;
  provisional: true;
  assumption_status: string;
  cost_output_label: string;
  scenario: CostScenario;
  review_threshold: number;
  block_threshold: number;
  review_capacity: number | null;
  capacity_met: boolean;
  metrics: ValidationCostMetrics;
  policy_comparison: Array<{ policy: string; total_estimated_cost: number }>;
  estimated_reduction_vs_approve_all: number;
  estimated_reduction_vs_binary: number;
  lowest_cost_feasible: ValidationCostMetrics & { review_threshold: number; block_threshold: number };
  provenance: string;
  selection_reason?: string;
  limitations?: string[];
  sensitivity_analysis?: Array<{
    parameter: string;
    value: number;
    lowest_estimated_cost: ValidationCostMetrics & { review_threshold: number; block_threshold: number };
  }>;
  failure_slices?: Record<string, {
    fraud_rows: number;
    approve: { count: number; rate: number; transaction_amount: number };
    review: { count: number; rate: number; transaction_amount: number };
    block: { count: number; rate: number; transaction_amount: number };
  }>;
  high_value_fraud?: {
    fraud_rows: number;
    approve: number;
    review: number;
    block: number;
    highest_value_approved_fraud_examples: Array<{
      TransactionID: string | number;
      TransactionAmt: number;
      fraud_probability: number;
      ProductCD: string;
      card4: string;
    }>;
  };
};

export type ValidationReviewItem = {
  transaction_id: string;
  transaction_dt: number;
  transaction_amount: number;
  fraud_probability: number;
  business_decision: "REVIEW";
  status: "OPEN" | "DECIDED";
  reviewer_decision: "APPROVE" | "BLOCK" | null;
  reviewer_note: string | null;
  reviewed_at: string | null;
  ground_truth: null;
  features: Record<string, string | null>;
};

export type ValidationReviewPage = {
  status: string;
  split: "validation";
  held_out_test_status: string;
  scenario_id: string;
  review_threshold: number;
  block_threshold: number;
  order: ReviewOrder;
  page: number;
  page_size: number;
  total: number;
  page_count: number;
  ground_truth_hidden: true;
  persistence_status: "available" | "postgresql_unavailable";
  items: ValidationReviewItem[];
  provenance: string;
};

export type ReviewOrder = "highest_amount" | "highest_risk" | "fraud" | "legitimate";

export type GroundTruthResponse = {
  transaction_id: string;
  split: "validation";
  actual_label: 0 | 1;
  ground_truth: "FRAUD" | "LEGITIMATE";
  reviewer_decision: "APPROVE" | "BLOCK" | null;
  reviewer_correct: boolean | null;
  note: string;
};
