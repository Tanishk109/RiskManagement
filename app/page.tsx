"use client";

import {
  AlertTriangle,
  Activity,
  ArrowRight,
  BarChart3,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleDollarSign,
  Clock3,
  FileText,
  Database,
  FileWarning,
  Filter,
  FlaskConical,
  Eye,
  Gauge,
  Info,
  Layers3,
  ListChecks,
  LockKeyhole,
  Menu,
  Network,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  TableProperties,
  TrendingUp,
  RotateCcw,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import RiskCheck from "../components/risk-check";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type {
  FeatureImportanceResponse,
  CostScenariosResponse,
  CostSimulationResponse,
  GroundTruthResponse,
  InterestingCasesResponse,
  ModelComparisonResponse,
  ProjectStatusResponse,
  ValidationFilter,
  ValidationReviewItem,
  ValidationReviewPage,
  ReviewOrder,
  ValidationTransaction,
  ValidationTransactionPage,
} from "../lib/api-types";

type Section = "overview" | "risk" | "transactions" | "reviews" | "cost";

const sections: Array<{ id: Section; label: string; icon: typeof BarChart3 }> = [
  { id: "overview", label: "Overview", icon: BarChart3 },
  { id: "risk", label: "Risk Check", icon: ShieldCheck },
  { id: "transactions", label: "Transactions", icon: TableProperties },
  { id: "reviews", label: "Review Queue", icon: ListChecks },
  { id: "cost", label: "Cost Lab", icon: SlidersHorizontal },
];

const validationFilters: Array<{ id: ValidationFilter; label: string }> = [
  { id: "all", label: "All" },
  { id: "true_fraud", label: "True Fraud" },
  { id: "true_legitimate", label: "True Legitimate" },
  { id: "true_positive", label: "True Positive" },
  { id: "false_positive", label: "False Positive" },
  { id: "false_negative", label: "False Negative" },
  { id: "true_negative", label: "True Negative" },
  { id: "high_risk", label: "High Risk" },
  { id: "high_value", label: "High Value" },
];

const interestingLabels: Record<string, string> = {
  highest_value_false_negative: "Highest-value false negative",
  highest_confidence_false_positive: "Highest-confidence false positive",
  highest_confidence_true_fraud: "Highest-confidence true fraud",
  highest_confidence_legitimate: "Highest-confidence legitimate",
};

const apiBase = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "");

function apiPath(path: string) {
  return `${apiBase}${path}`;
}

async function fetchJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(apiPath(path), { signal });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail ?? `Evidence API returned ${response.status}`);
  }
  return response.json() as Promise<T>;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(apiPath(path), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail ?? `Evidence API returned ${response.status}`);
  }
  return response.json() as Promise<T>;
}

function formatNumber(value: number, maximumFractionDigits = 0) {
  return new Intl.NumberFormat("en-IN", { maximumFractionDigits }).format(value);
}

function formatPercent(value: number, digits = 2) {
  return `${(value * 100).toFixed(digits)}%`;
}

function formatAmount(value: number) {
  return new Intl.NumberFormat("en-IN", {
    maximumFractionDigits: 2,
  }).format(value);
}

function formatCurrency(value: number, currency = "INR") {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency,
    maximumFractionDigits: 0,
  }).format(value);
}

function labelOutcome(outcome: ValidationTransaction["outcome"]) {
  return outcome.replaceAll("_", " ");
}

function MetricCard({
  label,
  value,
  detail,
  icon: Icon,
}: {
  label: string;
  value: string;
  detail: string;
  icon: typeof Database;
}) {
  return (
    <article className="metric-card">
      <div className="metric-icon"><Icon size={18} /></div>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  );
}

function LockedValue({ label }: { label: string }) {
  return <div className="locked-value"><span>{label}</span><strong>Not evaluated yet</strong></div>;
}

function LoadingOverview() {
  return (
    <div className="loading-grid" aria-label="Loading project evidence">
      {[0, 1, 2, 3].map((item) => <span key={item} />)}
    </div>
  );
}

function EvidenceError({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="error-state" role="alert">
      <FileWarning size={20} />
      <div><strong>Project evidence is unavailable</strong><span>{message}</span></div>
      <button onClick={onRetry}>Retry</button>
    </div>
  );
}

function TransactionTable({
  rows,
  threshold,
  onSelect,
}: {
  rows: ValidationTransaction[];
  threshold: number;
  onSelect: (row: ValidationTransaction) => void;
}) {
  return (
    <div className="table-shell">
      <div className="transaction-row transaction-head">
        <span>Transaction</span><span>TransactionAmt</span><span>Fraud probability</span><span>Business decision</span><span>Model @{threshold.toFixed(2)}</span><span>Actual label</span><span>Outcome</span><span />
      </div>
      {rows.map((row) => (
        <button className="transaction-row" key={row.transaction_id} onClick={() => onSelect(row)}>
          <strong>{row.transaction_id}</strong>
          <span>{formatAmount(row.transaction_amount)}</span>
          <span className="risk-value">{formatPercent(row.fraud_probability)}</span>
          <span className={`decision decision-${row.business_decision?.toLowerCase()}`}>{row.business_decision ?? "Pending"}</span>
          <span>{row.predicted_label_at_0_5 ? "Fraud" : "Legitimate"}</span>
          <span>{row.actual_label ? "Fraud" : "Legitimate"}</span>
          <span><i className={`outcome-dot ${row.outcome.toLowerCase()}`} />{labelOutcome(row.outcome)}</span>
          <ArrowRight size={15} />
        </button>
      ))}
    </div>
  );
}

export default function Home() {
  const [active, setActive] = useState<Section>("overview");
  const [mobileOpen, setMobileOpen] = useState(false);
  const [status, setStatus] = useState<ProjectStatusResponse | null>(null);
  const [comparison, setComparison] = useState<ModelComparisonResponse | null>(null);
  const [importance, setImportance] = useState<FeatureImportanceResponse | null>(null);
  const [interesting, setInteresting] = useState<InterestingCasesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const [transactions, setTransactions] = useState<ValidationTransactionPage | null>(null);
  const [transactionsLoading, setTransactionsLoading] = useState(false);
  const [transactionError, setTransactionError] = useState<string | null>(null);
  const [transactionReloadKey, setTransactionReloadKey] = useState(0);
  const [filter, setFilter] = useState<ValidationFilter>("all");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState<ValidationTransaction | null>(null);
  const [costScenarios, setCostScenarios] = useState<CostScenariosResponse | null>(null);
  const [costSimulation, setCostSimulation] = useState<CostSimulationResponse | null>(null);
  const [costLoading, setCostLoading] = useState(false);
  const [costError, setCostError] = useState<string | null>(null);
  const [costReloadKey, setCostReloadKey] = useState(0);
  const [scenarioId, setScenarioId] = useState("");
  const [reviewThreshold, setReviewThreshold] = useState(0);
  const [blockThreshold, setBlockThreshold] = useState(0);
  const [reviewCapacity, setReviewCapacity] = useState<number | null>(null);
  const [reviewQueue, setReviewQueue] = useState<ValidationReviewPage | null>(null);
  const [reviewOrder, setReviewOrder] = useState<ReviewOrder>("highest_amount");
  const [reviewPage, setReviewPage] = useState(1);
  const [reviewLoading, setReviewLoading] = useState(false);
  const [reviewError, setReviewError] = useState<string | null>(null);
  const [reviewReloadKey, setReviewReloadKey] = useState(0);
  const [selectedReview, setSelectedReview] = useState<ValidationReviewItem | null>(null);
  const [reviewNote, setReviewNote] = useState("");
  const [groundTruth, setGroundTruth] = useState<GroundTruthResponse | null>(null);
  const [reviewSubmitting, setReviewSubmitting] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      fetchJson<ProjectStatusResponse>("/api/v1/project/status", controller.signal),
      fetchJson<ModelComparisonResponse>("/api/v1/model-comparison", controller.signal),
      fetchJson<FeatureImportanceResponse>("/api/v1/model/feature-importance?limit=13", controller.signal),
      fetchJson<InterestingCasesResponse>("/api/v1/validation/interesting-cases", controller.signal),
      fetchJson<CostScenariosResponse>("/api/v1/cost/scenarios", controller.signal),
      fetchJson<CostSimulationResponse>("/api/v1/cost/validation-summary", controller.signal),
    ])
      .then(([nextStatus, nextComparison, nextImportance, nextInteresting, nextScenarios, nextCost]) => {
        setStatus(nextStatus);
        setComparison(nextComparison);
        setImportance(nextImportance);
        setInteresting(nextInteresting);
        setCostScenarios(nextScenarios);
        setCostSimulation(nextCost);
        setScenarioId(nextScenarios.default_scenario_id);
        setReviewThreshold(nextScenarios.default_review_threshold);
        setBlockThreshold(nextScenarios.default_block_threshold);
        setReviewCapacity(nextScenarios.default_review_capacity);
      })
      .catch((caught: unknown) => {
        if (caught instanceof DOMException && caught.name === "AbortError") return;
        setError(caught instanceof Error ? caught.message : "Project evidence could not be loaded");
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [reloadKey]);

  useEffect(() => {
    if (active !== "cost" || !scenarioId || reviewThreshold >= blockThreshold) return;
    const controller = new AbortController();
    const handle = window.setTimeout(() => {
      setCostLoading(true);
      setCostError(null);
      fetch(apiPath("/api/v1/cost/simulate"), {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          scenario_id: scenarioId,
          review_threshold: reviewThreshold,
          block_threshold: blockThreshold,
          review_capacity: reviewCapacity,
        }),
        signal: controller.signal,
      })
        .then(async (response) => {
          if (!response.ok) {
            const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
            throw new Error(payload?.detail ?? `Evidence API returned ${response.status}`);
          }
          return response.json() as Promise<CostSimulationResponse>;
        })
        .then(setCostSimulation)
        .catch((caught: unknown) => {
          if (caught instanceof DOMException && caught.name === "AbortError") return;
          setCostError(caught instanceof Error ? caught.message : "Cost simulation could not be loaded");
        })
        .finally(() => setCostLoading(false));
    }, 180);
    return () => {
      controller.abort();
      window.clearTimeout(handle);
    };
  }, [active, scenarioId, reviewThreshold, blockThreshold, reviewCapacity, costReloadKey]);

  useEffect(() => {
    if (active !== "reviews") return;
    const controller = new AbortController();
    const handle = window.setTimeout(() => {
      setReviewLoading(true);
      setReviewError(null);
      fetchJson<ValidationReviewPage>(
        `/api/v1/reviews/validation?order=${reviewOrder}&page=${reviewPage}&page_size=25`,
        controller.signal,
      )
        .then(setReviewQueue)
        .catch((caught: unknown) => {
          if (caught instanceof DOMException && caught.name === "AbortError") return;
          setReviewError(caught instanceof Error ? caught.message : "Review queue could not be loaded");
        })
        .finally(() => setReviewLoading(false));
    }, 0);
    return () => {
      controller.abort();
      window.clearTimeout(handle);
    };
  }, [active, reviewOrder, reviewPage, reviewReloadKey]);

  useEffect(() => {
    if (active !== "transactions") return;
    const controller = new AbortController();
    const handle = window.setTimeout(() => {
      setTransactionsLoading(true);
      setTransactionError(null);
      const parameters = new URLSearchParams({
        page: String(page),
        page_size: "25",
        filter,
      });
      if (search.trim()) parameters.set("search", search.trim());
      fetchJson<ValidationTransactionPage>(
        `/api/v1/validation/transactions?${parameters}`,
        controller.signal,
      )
        .then(setTransactions)
        .catch((caught: unknown) => {
          if (caught instanceof DOMException && caught.name === "AbortError") return;
          setTransactionError(caught instanceof Error ? caught.message : "Transactions could not be loaded");
        })
        .finally(() => setTransactionsLoading(false));
    }, 180);
    return () => {
      controller.abort();
      window.clearTimeout(handle);
    };
  }, [active, filter, page, search, transactionReloadKey]);

  const performanceData = useMemo(() => {
    if (!comparison) return [];
    const logistic = comparison.logistic_regression.metrics;
    const catboost = comparison.catboost.metrics;
    return [
      { metric: "Avg precision", logistic: logistic.average_precision, catboost: catboost.average_precision },
      { metric: "Recall", logistic: logistic.recall_at_0_5, catboost: catboost.recall_at_0_5 },
      { metric: "F1", logistic: logistic.f1_at_0_5, catboost: catboost.f1_at_0_5 },
    ];
  }, [comparison]);

  const curveData = useMemo(() => {
    if (!comparison) return [];
    const logistic = comparison.precision_recall_curves.find((series) => series.model.includes("Logistic"));
    const catboost = comparison.precision_recall_curves.find((series) => series.model.includes("CatBoost"));
    if (!logistic || !catboost) return [];
    return catboost.points.map((point, index) => ({
      recall: point.recall,
      catboost: point.precision,
      logistic: logistic.points[index]?.precision ?? null,
    }));
  }, [comparison]);

  const costBreakdown = useMemo(() => {
    if (!costSimulation) return [];
    return [
      { component: "Fraud loss", value: costSimulation.metrics.fraud_loss },
      { component: "False-positive cost", value: costSimulation.metrics.false_positive_cost },
      { component: "Manual review", value: costSimulation.metrics.manual_review_cost_total },
    ];
  }, [costSimulation]);

  function changeFilter(next: ValidationFilter) {
    setFilter(next);
    setPage(1);
  }

  function reloadEvidence() {
    setLoading(true);
    setError(null);
    setReloadKey((value) => value + 1);
  }

  function changeScenario(nextId: string) {
    const scenario = costScenarios?.scenarios.find((item) => item.id === nextId);
    if (!scenario) return;
    setScenarioId(nextId);
    setReviewThreshold(scenario.validation_configuration.review_threshold);
    setBlockThreshold(scenario.validation_configuration.block_threshold);
  }

  function restoreLowestCost() {
    if (!costSimulation) return;
    setReviewThreshold(costSimulation.lowest_cost_feasible.review_threshold);
    setBlockThreshold(costSimulation.lowest_cost_feasible.block_threshold);
  }

  function useProvisionalValidationConfig() {
    const scenario = costScenarios?.scenarios.find((item) => item.id === scenarioId);
    if (!scenario) return;
    setReviewThreshold(scenario.validation_configuration.review_threshold);
    setBlockThreshold(scenario.validation_configuration.block_threshold);
  }

  function openReview(item: ValidationReviewItem) {
    setSelectedReview(item);
    setReviewNote(item.reviewer_note ?? "");
    setGroundTruth(null);
    setReviewError(null);
  }

  async function revealReviewGroundTruth() {
    if (!selectedReview) return;
    try {
      const result = await fetchJson<GroundTruthResponse>(
        `/api/v1/reviews/validation/${selectedReview.transaction_id}/ground-truth`,
      );
      setGroundTruth(result);
    } catch (caught) {
      setReviewError(caught instanceof Error ? caught.message : "Ground truth could not be loaded");
    }
  }

  async function submitReviewDecision(decision: "APPROVE" | "BLOCK") {
    if (!selectedReview) return;
    setReviewSubmitting(true);
    setReviewError(null);
    try {
      const result = await postJson<ValidationReviewItem>(
        `/api/v1/reviews/validation/${selectedReview.transaction_id}/decision`,
        { decision, reason: reviewNote.trim() || null },
      );
      setSelectedReview(result);
      setReviewQueue((current) => current ? {
        ...current,
        items: current.items.map((item) => item.transaction_id === result.transaction_id ? result : item),
      } : current);
      setGroundTruth(null);
    } catch (caught) {
      setReviewError(caught instanceof Error ? caught.message : "Decision could not be saved");
    } finally {
      setReviewSubmitting(false);
    }
  }

  const activeLabel = sections.find((section) => section.id === active)?.label;
  const evidenceLabel = status
    ? "Validation candidate ready"
    : loading
      ? "Loading project evidence"
      : "Project evidence unavailable";
  const candidateDescription = comparison
    ? `${comparison.catboost.name} · ${comparison.candidate_details.feature_count} features`
    : "Artifact-backed project state";
  const thresholdAnalysisReady = status?.threshold_analysis.status === "validation_analysis_ready";

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileOpen ? "open" : ""}`}>
        <div className="brand-row">
          <div className="brand-mark"><ShieldCheck size={21} /></div>
          <div><strong>MerchantShield</strong><span>Cost-aware fraud decisions</span></div>
          <button className="icon-button close-nav" onClick={() => setMobileOpen(false)} aria-label="Close navigation"><X size={18} /></button>
        </div>

        <div className="candidate-badge">
          <span className="candidate-dot" />
          <div><strong>{evidenceLabel}</strong><small>Held-out test sealed</small></div>
        </div>

        <nav aria-label="MerchantShield modules">
          <span className="nav-label">Fraud risk</span>
          {sections.map((section) => {
            const Icon = section.icon;
            return (
              <button
                key={section.id}
                className={active === section.id ? "active" : ""}
                onClick={() => { setActive(section.id); setMobileOpen(false); }}
              >
                <Icon size={17} /><span>{section.label}</span>
              </button>
            );
          })}
          <span className="nav-label nav-label-spaced">Loss prevention</span>
          <a className="suite-nav-link" href="/chargebacks"><FileText size={17} /><span>Chargebacks</span></a>
          <a className="suite-nav-link" href="/fraud-pulse"><Activity size={17} /><span>Fraud Pulse</span></a>
          <a className="suite-nav-link" href="/abuse-rings"><Network size={17} /><span>Abuse Rings</span></a>
        </nav>

        <div className="sidebar-note"><LockKeyhole size={16} /><div><strong>Defense-only evidence</strong><span>Validation is visible. Final performance stays locked until the single held-out evaluation.</span></div></div>
      </aside>
      {mobileOpen && <button className="nav-scrim" onClick={() => setMobileOpen(false)} aria-label="Close navigation" />}

      <main className="main-area">
        <header className="topbar">
          <button className="icon-button menu-button" onClick={() => setMobileOpen(true)} aria-label="Open navigation"><Menu size={19} /></button>
          <div className="breadcrumb"><span>MerchantShield</span><ArrowRight size={12} /><strong>{activeLabel}</strong></div>
          <div className="top-status"><span />{evidenceLabel.toUpperCase()}</div>
        </header>

        <div className="page-content">
          {active === "overview" && (
            <section className="module-page overview-page">
              <div className="hero-heading">
                <div>
                  <span className="eyebrow">MerchantShield</span>
                  <h1>Cost-Aware Fraud<br />Decision Engine</h1>
                  <p>Real IEEE-CIS evidence, chronological validation, and transparent model limitations—without presenting development results as final performance.</p>
                </div>
                <div className="hero-status"><ShieldCheck size={20} /><div><strong>{evidenceLabel}</strong><span>{candidateDescription}</span></div></div>
              </div>

              {loading && <LoadingOverview />}
              {error && <EvidenceError message={error} onRetry={reloadEvidence} />}

              {status && comparison && importance && (
                <>
                  <section className="section-block">
                    <div className="section-heading"><div><span>REAL IEEE-CIS DATASET</span><h2>Evidence starts with the data</h2></div><Database size={20} /></div>
                    <div className="dataset-grid">
                      <MetricCard icon={Database} label="Transactions" value={formatNumber(status.dataset.transactions)} detail="Official labeled training rows" />
                      <MetricCard icon={ShieldCheck} label="Fraud transactions" value={formatNumber(status.dataset.fraud_transactions)} detail="Observed target labels" />
                      <MetricCard icon={TrendingUp} label="Fraud prevalence" value={formatPercent(status.dataset.fraud_prevalence, 3)} detail="Strongly imbalanced target" />
                      <MetricCard icon={Layers3} label="Identity coverage" value={formatPercent(status.dataset.identity_coverage, 2)} detail={`${formatNumber(status.dataset.identity_rows)} identity rows`} />
                    </div>
                  </section>

                  <section className="overview-pair">
                    <article className="panel progress-panel">
                      <div className="panel-title"><div><span>PROJECT STATUS</span><h2>Evidence readiness</h2></div><Clock3 size={19} /></div>
                      <div className="progress-list">
                        {([
                          ["Data ingestion", "Complete", true, false],
                          ["EDA", "Complete", true, false],
                          ["Temporal split", "Complete", true, false],
                          ["Logistic baseline", "Complete", true, false],
                          ["CatBoost candidate", "Complete", true, false],
                          ["Validation cost + threshold analysis", thresholdAnalysisReady ? "Complete" : "Pending", thresholdAnalysisReady, false],
                          ["Rule evaluation", "Pending", false, false],
                          ["Held-out final evaluation", "Sealed", false, true],
                        ] as Array<[string, string, boolean, boolean]>).map(([label, state, done, locked]) => (
                          <div key={String(label)} className={done ? "complete" : locked ? "sealed" : "pending"}>
                            <i>{done ? <Check size={13} /> : locked ? <LockKeyhole size={12} /> : null}</i>
                            <span>{label}</span><b>{state}</b>
                          </div>
                        ))}
                      </div>
                    </article>

                    <article className="panel final-panel">
                      <div className="panel-title"><div><span>FINAL HELD-OUT EVALUATION</span><h2>Still deliberately locked</h2></div><LockKeyhole size={19} /></div>
                      <div className="locked-grid">
                        {[
                          "Average Precision", "Precision", "Recall", "F1", "False Positives", "False Negatives", "Estimated cost",
                        ].map((label) => <LockedValue key={label} label={label} />)}
                      </div>
                      <p><LockKeyhole size={13} /> Held-out test remains sealed. Validation evidence is never relabelled as final performance.</p>
                    </article>
                  </section>

                  {costSimulation && (
                    <section className="section-block provisional-card">
                      <div className="section-heading"><div><span>PROVISIONAL VALIDATION OPERATING CARD</span><h2>{costSimulation.scenario.name}</h2></div><Gauge size={20} /></div>
                      <div className="validation-assumption-badge">VALIDATION / ILLUSTRATIVE ASSUMPTIONS</div>
                      <div className="provisional-grid">
                        <div><span>Review threshold</span><strong>{costSimulation.review_threshold.toFixed(3)}</strong></div>
                        <div><span>Block threshold</span><strong>{costSimulation.block_threshold.toFixed(3)}</strong></div>
                        <div><span>Validation review rate</span><strong>{formatPercent(costSimulation.metrics.review_rate)}</strong></div>
                        <div><span>Estimated validation cost</span><strong>{formatCurrency(costSimulation.metrics.total_estimated_cost, costSimulation.metrics.currency)}</strong></div>
                        <div><span>Estimated reduction vs approve-all</span><strong>{formatPercent(costSimulation.estimated_reduction_vs_approve_all)}</strong></div>
                      </div>
                      <p><Info size={14} /> Provisional validation configuration under illustrative merchant assumptions. This is not a final threshold recommendation or realized savings claim.</p>
                    </section>
                  )}

                  <section className="section-block split-section">
                    <div className="section-heading"><div><span>CHRONOLOGICAL EVALUATION</span><h2>Train early. Validate later. Seal the future.</h2></div><Clock3 size={20} /></div>
                    <div className="split-timeline">
                      <div className="train" title={`TransactionDT ${formatNumber(status.split.train_transaction_dt_min)}–${formatNumber(status.split.train_transaction_dt_max)}`}><span>TRAIN</span><strong>{formatPercent(status.split.train_fraction, 0)}</strong><small>{formatNumber(status.split.train_rows)} transactions</small><em>TransactionDT {formatNumber(status.split.train_transaction_dt_min)}–{formatNumber(status.split.train_transaction_dt_max)}</em></div>
                      <div className="validation" title={`TransactionDT ${formatNumber(status.split.validation_transaction_dt_min)}–${formatNumber(status.split.validation_transaction_dt_max)}`}><span>VALIDATION</span><strong>{formatPercent(status.split.validation_fraction, 0)}</strong><small>{formatNumber(status.split.validation_rows)} transactions</small><em>TransactionDT {formatNumber(status.split.validation_transaction_dt_min)}–{formatNumber(status.split.validation_transaction_dt_max)}</em></div>
                      <div className="test" title={`TransactionDT ${formatNumber(status.split.test_transaction_dt_min)}–${formatNumber(status.split.test_transaction_dt_max)}`}><span><LockKeyhole size={12} /> HELD-OUT TEST</span><strong>{formatPercent(status.split.test_fraction, 0)}</strong><small>{formatNumber(status.split.test_rows)} transactions</small><em>SEALED · TransactionDT {formatNumber(status.split.test_transaction_dt_min)}–{formatNumber(status.split.test_transaction_dt_max)}</em></div>
                    </div>
                    <div className="why-card"><Info size={17} /><p>Fraud patterns evolve over time. MerchantShield trains on earlier transactions and validates on later transactions rather than mixing future and past observations through a random split.</p></div>
                  </section>

                  <section className="section-block validation-section">
                    <div className="validation-label"><span>VALIDATION RESULTS</span><small title="These metrics were measured on the chronological validation partition. The held-out test set remains sealed."><Info size={13} /> Not final held-out performance</small></div>
                    <div className="section-heading"><div><span>MODEL VALIDATION COMPARISON</span><h2>Nonlinear ranking captures more signal</h2></div><div className="improvement-chip"><TrendingUp size={15} /><span>Average Precision improvement<strong>+{formatPercent(comparison.average_precision_relative_improvement, 2)}</strong></span></div></div>
                    <div className="comparison-layout">
                      <div className="comparison-table">
                        <div className="comparison-row comparison-head"><span>Metric</span><strong>Logistic Regression</strong><strong>CatBoost</strong></div>
                        {[
                          ["Average Precision", "average_precision", "decimal"],
                          ["ROC-AUC", "roc_auc", "decimal"],
                          [`Precision @${comparison.threshold.toFixed(2)}`, "precision_at_0_5", "percent"],
                          [`Recall @${comparison.threshold.toFixed(2)}`, "recall_at_0_5", "percent"],
                          [`F1 @${comparison.threshold.toFixed(2)}`, "f1_at_0_5", "percent"],
                          ["False Positives", "false_positives", "integer"],
                          ["False Negatives", "false_negatives", "integer"],
                        ].map(([label, key, kind]) => {
                          const logistic = comparison.logistic_regression.metrics[key as keyof typeof comparison.logistic_regression.metrics] as number;
                          const catboost = comparison.catboost.metrics[key as keyof typeof comparison.catboost.metrics] as number;
                          const value = (input: number) => kind === "percent" ? formatPercent(input) : kind === "integer" ? formatNumber(input) : input.toFixed(4);
                          return <div className="comparison-row" key={label}><span>{label}</span><strong>{value(logistic)}</strong><strong className="candidate-value">{value(catboost)}</strong></div>;
                        })}
                      </div>
                      <div className="chart-card">
                        <h3>Core validation metrics</h3>
                        <ResponsiveContainer width="100%" height={285}>
                          <BarChart data={performanceData} margin={{ top: 16, right: 8, left: -20, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e7ebe8" />
                            <XAxis dataKey="metric" tick={{ fontSize: 10 }} axisLine={false} tickLine={false} />
                            <YAxis domain={[0, 0.5]} tickFormatter={(value) => `${Math.round(Number(value) * 100)}%`} tick={{ fontSize: 9 }} axisLine={false} tickLine={false} />
                            <Tooltip formatter={(value) => formatPercent(Number(value))} />
                            <Legend wrapperStyle={{ fontSize: 10 }} />
                            <Bar dataKey="logistic" name="Logistic Regression" fill="#aeb8b2" radius={[4, 4, 0, 0]} />
                            <Bar dataKey="catboost" name="CatBoost" fill="#237454" radius={[4, 4, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </section>

                  <section className="overview-pair model-row">
                    <article className="panel curve-panel">
                      <div className="panel-title"><div><span>VALIDATION PRECISION-RECALL CURVE</span><h2>Ranking quality across thresholds</h2></div><BarChart3 size={19} /></div>
                      <ResponsiveContainer width="100%" height={310}>
                        <LineChart data={curveData} margin={{ top: 18, right: 15, left: -12, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#e7ebe8" />
                          <XAxis type="number" dataKey="recall" domain={[0, 1]} tickFormatter={(value) => `${Math.round(Number(value) * 100)}%`} tick={{ fontSize: 9 }} />
                          <YAxis domain={[0, 1]} tickFormatter={(value) => `${Math.round(Number(value) * 100)}%`} tick={{ fontSize: 9 }} />
                          <Tooltip formatter={(value) => formatPercent(Number(value))} labelFormatter={(value) => `Recall ${formatPercent(Number(value))}`} />
                          <Legend wrapperStyle={{ fontSize: 10 }} />
                          <Line dataKey="logistic" name="Logistic Regression" stroke="#98a39d" strokeWidth={2} dot={false} />
                          <Line dataKey="catboost" name="CatBoost" stroke="#237454" strokeWidth={2.5} dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </article>

                    <article className="panel candidate-panel">
                      <div className="panel-title"><div><span>SELECTED CANDIDATE</span><h2>CatBoostClassifier</h2></div><ShieldCheck size={19} /></div>
                      <div className="candidate-specs">
                        <div><span>Status</span><strong>Validation candidate</strong></div>
                        <div><span>Feature count</span><strong>{comparison.candidate_details.feature_count}</strong></div>
                        <div><span>Identity fields</span><strong>{comparison.candidate_details.identity_fields_included ? "Included" : "Excluded"}</strong></div>
                        <div><span>Class weighting</span><strong>{comparison.candidate_details.class_weight}</strong></div>
                      </div>
                      <div className="selection-reason"><Info size={16} /><div><strong>Why selected</strong><p>Removing identity information reduced AP by only {comparison.candidate_details.identity_ap_loss.toFixed(4)} while reducing dependence on a temporally shifting enrichment process.</p></div></div>
                    </article>
                  </section>

                  <section className="overview-pair model-row">
                    <article className="panel importance-panel">
                      <div className="panel-title"><div><span>FEATURE IMPORTANCE</span><h2>Top predictive associations</h2></div><Layers3 size={19} /></div>
                      <ResponsiveContainer width="100%" height={390}>
                        <BarChart data={importance.items} layout="vertical" margin={{ top: 12, right: 20, left: 25, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e7ebe8" />
                          <XAxis type="number" tick={{ fontSize: 9 }} axisLine={false} />
                          <YAxis type="category" dataKey="feature" width={100} tick={{ fontSize: 9 }} axisLine={false} tickLine={false} />
                          <Tooltip formatter={(value) => Number(value).toFixed(3)} />
                          <Bar dataKey="importance" name="Importance" fill="#237454" radius={[0, 5, 5, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                      <p className="chart-note">{importance.note} Masked fields retain their source names; no undocumented meanings are invented.</p>
                    </article>

                    <article className="panel failure-panel">
                      <div className="panel-title"><div><span>WHERE THE MODEL STILL STRUGGLES</span><h2>Failure slices remain visible</h2></div><AlertTriangle size={19} /></div>
                      <span className="analysis-label">{comparison.failure_analysis.label}</span>
                      <div className="failure-table">
                        <div className="failure-row failure-head"><span>Slice</span><span>Logistic</span><span>CatBoost</span><span>Improvement</span></div>
                        {comparison.failure_analysis.slices.map((slice) => <div className="failure-row" key={slice.slice}><strong>{slice.slice}</strong><span>{formatPercent(slice.logistic_recall)}</span><span>{formatPercent(slice.catboost_recall)}</span><span className="positive">+{formatPercent(slice.absolute_improvement)}</span></div>)}
                      </div>
                      <div className="fn-summary">
                        <div><span>False negatives</span><strong>{formatNumber(comparison.failure_analysis.false_negatives.count)}</strong></div>
                        <div><span>Total FN amount</span><strong>{formatNumber(comparison.failure_analysis.false_negatives.transaction_amount_total, 3)}</strong></div>
                        <div><span>Maximum FN amount</span><strong>{formatNumber(comparison.failure_analysis.false_negatives.transaction_amount_max, 3)}</strong></div>
                      </div>
                      <div className="limitation-card"><AlertTriangle size={16} /><p><strong>Known limitation.</strong> At threshold {comparison.threshold.toFixed(2)} the model still misses many fraudulent transactions. This threshold is used only for validation comparison and is not the final operating threshold.</p></div>
                    </article>
                  </section>
                </>
              )}
            </section>
          )}

          {active === "risk" && <RiskCheck />}

          {active === "transactions" && (
            <section className="module-page transactions-page">
              <div className="compact-heading"><div><span className="eyebrow">Chronological validation partition</span><h1>Validation Transactions</h1><p>Real saved CatBoost predictions joined to a deliberately bounded set of documented validation fields.</p></div><div className="validation-pill"><ShieldCheck size={15} /> VALIDATION ONLY</div></div>

              {interesting && (
                <div className="interesting-section">
                  <div className="section-heading small"><div><span>INTERESTING VALIDATION CASES</span><h2>Programmatically selected</h2></div></div>
                  <div className="interesting-grid">
                    {interesting.cases.map((item) => <button key={item.case_type} onClick={() => setSelected(item)}><span>{interestingLabels[item.case_type]}</span><strong>{item.transaction_id}</strong><small>TransactionAmt {formatAmount(item.transaction_amount)} · {formatPercent(item.fraud_probability)} risk</small><ArrowRight size={14} /></button>)}
                  </div>
                </div>
              )}

              <div className="filter-bar">
                <div className="filter-tabs" aria-label="Validation transaction filters">
                  {validationFilters.map((item) => <button key={item.id} className={filter === item.id ? "active" : ""} onClick={() => changeFilter(item.id)}>{item.label}</button>)}
                </div>
                <label className="search-field"><Search size={15} /><input value={search} onChange={(event) => { setSearch(event.target.value); setPage(1); }} placeholder="Search TransactionID" /></label>
              </div>

              {transactionError && <EvidenceError message={transactionError} onRetry={() => setTransactionReloadKey((value) => value + 1)} />}
              {transactionsLoading && <div className="table-loading"><span /><span /><span /><span /></div>}
              {!transactionsLoading && transactions && transactions.items.length > 0 && <TransactionTable rows={transactions.items} threshold={transactions.threshold} onSelect={setSelected} />}
              {!transactionsLoading && transactions && transactions.items.length === 0 && <div className="empty-state"><Filter size={22} /><h2>No matching validation transactions</h2><p>Change the filter or TransactionID search. No placeholder rows are substituted.</p></div>}
              {transactions && (
                <div className="pagination"><span>{formatNumber(transactions.total)} matching rows · Page {transactions.page} of {Math.max(transactions.page_count, 1)}</span><div><button disabled={page <= 1} onClick={() => setPage((value) => value - 1)}><ChevronLeft size={15} /> Previous</button><button disabled={page >= transactions.page_count} onClick={() => setPage((value) => value + 1)}>Next <ChevronRight size={15} /></button></div></div>
              )}
              <p className="transaction-note"><Info size={13} /> High Risk uses the saved validation comparison threshold{transactions ? ` (${transactions.threshold.toFixed(2)})` : ""}. High Value means TransactionAmt ≥ 500. Neither is an operational recommendation.</p>
            </section>
          )}

          {active === "reviews" && (
            <section className="module-page reviews-page">
              <div className="compact-heading"><div><span className="eyebrow">VALIDATION REVIEW SIMULATION</span><h1>Review Queue</h1><p>These cases come from the chronological validation set using the provisional operating thresholds. Analyst decisions below are demonstration feedback and do not alter frozen validation metrics.</p></div><div className="validation-pill"><Eye size={15} /> LABELS HIDDEN</div></div>
              <div className="review-toolbar">
                <div className="filter-tabs" aria-label="Review queue ordering">
                  {([
                    ["highest_amount", "Highest amount"],
                    ["highest_risk", "Highest risk"],
                    ["fraud", "Fraud cases"],
                    ["legitimate", "Legitimate cases"],
                  ] as Array<[ReviewOrder, string]>).map(([id, label]) => <button key={id} className={reviewOrder === id ? "active" : ""} onClick={() => { setReviewOrder(id); setReviewPage(1); }}>{label}</button>)}
                </div>
                {reviewQueue && <span>{formatNumber(reviewQueue.total)} validation cases · band {reviewQueue.review_threshold.toFixed(3)}–{reviewQueue.block_threshold.toFixed(3)}{reviewQueue.persistence_status !== "available" ? " · PostgreSQL unavailable: decisions disabled" : ""}</span>}
              </div>
              {reviewError && <EvidenceError message={reviewError} onRetry={() => setReviewReloadKey((value) => value + 1)} />}
              {reviewLoading && <div className="table-loading"><span /><span /><span /><span /></div>}
              {!reviewLoading && reviewQueue && (
                <div className="review-grid">
                  {reviewQueue.items.map((item) => (
                    <button key={item.transaction_id} className="review-card" onClick={() => openReview(item)}>
                      <div><span className="review-id">TX {item.transaction_id}</span><span className={`review-status ${item.status.toLowerCase()}`}>{item.status}</span></div>
                      <strong>{formatAmount(item.transaction_amount)}</strong>
                      <div className="review-features"><span>{item.features.ProductCD ?? "Product missing"}</span><span>{item.features.card4 ?? "card4 missing"}</span><span>{item.features.card6 ?? "card6 missing"}</span></div>
                      <div className="risk-track"><i style={{ width: `${item.fraud_probability * 100}%` }} /></div>
                      <small>{formatPercent(item.fraud_probability)} fraud probability</small>
                      <footer><span>{item.reviewer_decision ? `Reviewer: ${item.reviewer_decision}` : "Awaiting reviewer"}</span><ArrowRight size={14} /></footer>
                    </button>
                  ))}
                </div>
              )}
              {reviewQueue && <div className="pagination"><span>Page {reviewQueue.page} of {Math.max(reviewQueue.page_count, 1)}</span><div><button disabled={reviewPage <= 1} onClick={() => setReviewPage((value) => value - 1)}><ChevronLeft size={15} /> Previous</button><button disabled={reviewPage >= reviewQueue.page_count} onClick={() => setReviewPage((value) => value + 1)}>Next <ChevronRight size={15} /></button></div></div>}
              <p className="transaction-note"><Info size={13} /> Reviewer decisions are persisted as application records. They do not alter validation predictions, threshold artifacts, or any model metric.</p>
            </section>
          )}

          {active === "cost" && (
            <section className="module-page cost-page">
              <div className="compact-heading"><div><span className="eyebrow">VALIDATION SIMULATION READY</span><h1>Cost Lab</h1><p>Thresholds and business-cost estimates below were selected on the chronological validation partition. The held-out test remains sealed.</p></div><div className="validation-pill"><FlaskConical size={14} /> VALIDATION ONLY</div></div>
              {costError && <EvidenceError message={costError} onRetry={() => setCostReloadKey((value) => value + 1)} />}
              {costScenarios && costSimulation && (
                <>
                  <div className="cost-top-grid">
                    <article className="panel cost-controls">
                      <div className="panel-title"><div><span>POLICY CONFIGURATION</span><h2>Move thresholds, inspect trade-offs</h2></div><SlidersHorizontal size={19} /></div>
                      <label className="select-label"><span>Illustrative merchant scenario</span><select value={scenarioId} onChange={(event) => changeScenario(event.target.value)}>{costScenarios.scenarios.map((scenario) => <option key={scenario.id} value={scenario.id}>{scenario.name}</option>)}</select></label>
                      <label className="range-label"><span>Review threshold <b>{reviewThreshold.toFixed(3)}</b></span><input type="range" min="0.05" max={Math.max(0.05, blockThreshold - 0.025)} step="0.025" value={reviewThreshold} onChange={(event) => setReviewThreshold(Number(event.target.value))} /></label>
                      <label className="range-label"><span>Block threshold <b>{blockThreshold.toFixed(3)}</b></span><input type="range" min={Math.min(0.95, reviewThreshold + 0.025)} max="0.95" step="0.025" value={blockThreshold} onChange={(event) => setBlockThreshold(Number(event.target.value))} /></label>
                      <label className="select-label"><span>Maximum validation review rate</span><select value={reviewCapacity === null ? "none" : String(reviewCapacity)} onChange={(event) => setReviewCapacity(event.target.value === "none" ? null : Number(event.target.value))}>{costScenarios.review_capacities.map((capacity) => <option key={capacity ?? "none"} value={capacity ?? "none"}>{capacity === null ? "No limit" : formatPercent(capacity, 0)}</option>)}</select></label>
                      <div className="config-actions"><button className="restore-button" title="Lowest estimated-cost threshold pair found on validation under the selected assumptions." onClick={useProvisionalValidationConfig}><RotateCcw size={14} /> Use Provisional Validation Configuration</button><button className="restore-button secondary" onClick={restoreLowestCost}><Gauge size={14} /> Use Lowest-Cost Feasible Configuration</button></div>
                      <div className={`capacity-callout ${costSimulation.capacity_met ? "met" : "missed"}`}><Gauge size={15} /><span>{costSimulation.capacity_met ? "Within selected review capacity" : "Selected review capacity exceeded"} · current review rate {formatPercent(costSimulation.metrics.review_rate)}</span></div>
                      <div className="assumptions-panel"><span>ILLUSTRATIVE MERCHANT ASSUMPTIONS</span>{([
                        ["Fraud loss fraction", "fraud_loss_fraction", "percent"],
                        ["Fixed fraud cost", "chargeback_fixed_cost", "currency"],
                        ["Legitimate margin rate", "legitimate_margin_rate", "percent"],
                        ["False-positive fixed cost", "false_positive_fixed_cost", "currency"],
                        ["Manual review cost", "manual_review_cost", "currency"],
                        ["Review fraud catch rate", "review_fraud_catch_rate", "percent"],
                        ["Legitimate review approval rate", "review_legitimate_approval_rate", "percent"],
                      ] as Array<[string, keyof typeof costSimulation.scenario.assumptions, "percent" | "currency"]>).map(([label, key, kind]) => <div key={key}><small>{label}</small><strong>{kind === "percent" ? formatPercent(Number(costSimulation.scenario.assumptions[key])) : formatCurrency(Number(costSimulation.scenario.assumptions[key]), costSimulation.scenario.assumptions.currency)}</strong></div>)}</div>
                    </article>
                    <article className="panel cost-hero">
                      <div className="panel-title"><div><span>ESTIMATED TOTAL COST</span><h2>Validation simulation</h2></div><CircleDollarSign size={19} /></div>
                      <strong className="cost-total">{formatCurrency(costSimulation.metrics.total_estimated_cost, costSimulation.metrics.currency)}</strong>
                      <span className="cost-total-note">Scenario: {costSimulation.scenario.name} · {formatNumber(costSimulation.metrics.transaction_count)} validation transactions</span>
                      <ResponsiveContainer width="100%" height={190}>
                        <BarChart data={costBreakdown} layout="vertical" margin={{ top: 15, right: 10, left: 35, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e7ebe8" />
                          <XAxis type="number" hide /><YAxis type="category" dataKey="component" width={105} tick={{ fontSize: 9 }} axisLine={false} tickLine={false} />
                          <Tooltip formatter={(value) => formatCurrency(Number(value), costSimulation.metrics.currency)} />
                          <Bar dataKey="value" fill="#237454" radius={[0, 5, 5, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                      <div className="assumption-badge"><Info size={14} /> {costSimulation.assumption_status}. Estimated—not realized—merchant economics.</div>
                    </article>
                  </div>

                  <section className="section-block policy-section">
                    <div className="section-heading"><div><span>COMPARE DECISION STRATEGIES</span><h2>Same validation rows, three decision policies</h2></div>{costLoading && <small>Recomputing…</small>}</div>
                    <div className="policy-grid">{costSimulation.policy_comparison.map((policy) => <article key={policy.policy}><span>{policy.policy}</span><strong>{formatCurrency(policy.total_estimated_cost, costSimulation.metrics.currency)}</strong><small>Estimated validation cost</small></article>)}</div>
                    <div className="reduction-grid"><div><span>Estimated reduction vs approve all</span><strong>{formatPercent(costSimulation.estimated_reduction_vs_approve_all)}</strong></div><div><span>Estimated reduction vs binary @0.50</span><strong>{formatPercent(costSimulation.estimated_reduction_vs_binary)}</strong></div></div>
                    <p className="transaction-note"><Info size={13} /> Under current illustrative assumptions. These are estimated cost reductions, not factual merchant savings.</p>
                  </section>

                  <section className="tradeoff-grid" aria-label="Live threshold trade-offs">
                    <div><span>Fraud detection</span><strong>{formatPercent(costSimulation.metrics.detected_fraud_recall)}</strong></div>
                    <div><span>Fraud amount capture</span><strong>{formatPercent(costSimulation.metrics.fraud_amount_capture_rate)}</strong></div>
                    <div><span>False positives</span><strong>{formatNumber(costSimulation.metrics.false_positives)}</strong></div>
                    <div><span>False negatives</span><strong>{formatNumber(costSimulation.metrics.false_negatives)}</strong></div>
                    <div><span>Review rate</span><strong>{formatPercent(costSimulation.metrics.review_rate)}</strong></div>
                    <div><span>Block rate</span><strong>{formatPercent(costSimulation.metrics.block_rate)}</strong></div>
                  </section>

                  <section className="decision-metrics-grid">
                    <article className="panel"><div className="panel-title"><div><span>DECISION DISTRIBUTION</span><h2>Customer friction</h2></div><ListChecks size={18} /></div><div className="mini-metrics"><div><span>Approve</span><strong>{formatNumber(costSimulation.metrics.approve_count)}</strong><small>{formatPercent(costSimulation.metrics.approve_rate)}</small></div><div><span>Review</span><strong>{formatNumber(costSimulation.metrics.review_count)}</strong><small>{formatPercent(costSimulation.metrics.review_rate)}</small></div><div><span>Block</span><strong>{formatNumber(costSimulation.metrics.block_count)}</strong><small>{formatPercent(costSimulation.metrics.block_rate)}</small></div></div></article>
                    <article className="panel"><div className="panel-title"><div><span>FRAUD COUNT DETECTION</span><h2>Transactions found</h2></div><ShieldCheck size={18} /></div><strong className="spotlight-metric">{formatPercent(costSimulation.metrics.detected_fraud_recall)}</strong><p>{formatNumber(costSimulation.metrics.fraud_reviewed + costSimulation.metrics.fraud_blocked)} of {formatNumber(costSimulation.metrics.fraud_count)} fraudulent validation transactions sent to review or block.</p></article>
                    <article className="panel"><div className="panel-title"><div><span>FRAUD AMOUNT CAPTURE</span><h2>Value expected captured</h2></div><TrendingUp size={18} /></div><strong className="spotlight-metric">{formatPercent(costSimulation.metrics.fraud_amount_capture_rate)}</strong><p>{formatCurrency(costSimulation.metrics.captured_fraud_amount, costSimulation.metrics.currency)} of {formatCurrency(costSimulation.metrics.total_fraud_amount, costSimulation.metrics.currency)} under the scenario&apos;s review-catch assumption.</p></article>
                  </section>

                  {costSimulation.failure_slices && (
                    <section className="section-block residual-section">
                      <div className="section-heading"><div><span>RESIDUAL FRAUD EVIDENCE</span><h2>Approved fraud remains visible</h2></div><AlertTriangle size={19} /></div>
                      <h3 className="residual-subtitle">Fraudulent transaction value by decision</h3><div className="residual-hero"><div><span>Approved fraud</span><strong>{formatNumber(costSimulation.metrics.fraud_approved)}</strong><small>{formatCurrency(costSimulation.metrics.fraud_amount_approved, costSimulation.metrics.currency)} transaction value</small></div><div><span>Reviewed fraud</span><strong>{formatNumber(costSimulation.metrics.fraud_reviewed)}</strong><small>{formatCurrency(costSimulation.metrics.fraud_amount_reviewed, costSimulation.metrics.currency)} transaction value</small></div><div><span>Blocked fraud</span><strong>{formatNumber(costSimulation.metrics.fraud_blocked)}</strong><small>{formatCurrency(costSimulation.metrics.fraud_amount_blocked, costSimulation.metrics.currency)} transaction value</small></div></div>
                      <div className="slice-table"><div className="slice-row slice-head"><span>Failure slice</span><span>Fraud rows</span><span>Approve</span><span>Review</span><span>Block</span></div>{Object.entries(costSimulation.failure_slices).map(([name, slice]) => <div className="slice-row" key={name}><strong>{name}</strong><span>{formatNumber(slice.fraud_rows)}</span><span>{formatNumber(slice.approve.count)}</span><span>{formatNumber(slice.review.count)}</span><span>{formatNumber(slice.block.count)}</span></div>)}</div>
                      {costSimulation.high_value_fraud && <div className="high-value-list"><h3>Highest-Value Residual Fraud Cases</h3>{costSimulation.high_value_fraud.highest_value_approved_fraud_examples.slice(0, 5).map((item) => <div key={item.TransactionID}><strong>{item.TransactionID}</strong><span>{formatCurrency(item.TransactionAmt, costSimulation.metrics.currency)}</span><small>{formatPercent(item.fraud_probability)} risk · approved because score is below the review threshold</small></div>)}</div>}
                      <p className="transaction-note"><Info size={13} /> Failure-slice dispositions and example cases come from the provisional Scenario B operating configuration (review {costScenarios.default_review_threshold.toFixed(3)}, block {costScenarios.default_block_threshold.toFixed(3)}); the residual totals above follow the current controls.</p>
                    </section>
                  )}

                  {costSimulation.sensitivity_analysis && <section className="section-block sensitivity-section"><div className="section-heading"><div><span>WHY THRESHOLDS CHANGE</span><h2>Threshold selection is a business decision, not only a model decision.</h2></div><FlaskConical size={18} /></div><div className="sensitivity-grid">{costSimulation.sensitivity_analysis.map((item, index) => <article key={`${item.parameter}-${item.value}-${index}`}><span>{item.parameter.replaceAll("_", " ")}</span><strong>{item.value}</strong><small>Review {item.lowest_estimated_cost.review_threshold.toFixed(3)} · Block {item.lowest_estimated_cost.block_threshold.toFixed(3)}</small></article>)}</div><p className="transaction-note"><Info size={13} /> One assumption varies at a time; all other Scenario B assumptions remain fixed. These are validation sensitivity results, not universal recommendations.</p></section>}
                </>
              )}
            </section>
          )}
        </div>
      </main>

      {selected && (
        <div className="drawer-scrim">
          <button className="drawer-backdrop" onClick={() => setSelected(null)} aria-label="Close transaction detail" />
          <aside className="detail-drawer">
            <button className="icon-button drawer-close" onClick={() => setSelected(null)} aria-label="Close transaction"><X size={18} /></button>
            <span className="eyebrow">Validation transaction</span>
            <h2>{selected.transaction_id}</h2>
            {selected.model_error && <div className="model-error"><AlertTriangle size={14} /> MODEL ERROR</div>}
            <div className={`outcome-banner ${selected.outcome.toLowerCase()}`}>{labelOutcome(selected.outcome)}</div>
            <dl>
              <div><dt>TransactionAmt</dt><dd>{formatAmount(selected.transaction_amount)}</dd></div>
              <div><dt>Fraud probability</dt><dd>{formatPercent(selected.fraud_probability)}</dd></div>
              <div><dt>Model classification @0.50</dt><dd>{selected.predicted_label_at_0_5 ? "Fraud" : "Legitimate"}</dd></div>
              <div><dt>Business decision</dt><dd>{selected.business_decision ?? "Not configured"}</dd></div>
              <div><dt>Actual validation label</dt><dd>{selected.actual_label ? "Fraud" : "Legitimate"}</dd></div>
              <div><dt>TransactionDT</dt><dd>{formatNumber(selected.transaction_dt)}</dd></div>
              <div><dt>Partition</dt><dd>Validation</dd></div>
              <div><dt>Review / block thresholds</dt><dd>{selected.review_threshold?.toFixed(3) ?? "—"} / {selected.block_threshold?.toFixed(3) ?? "—"}</dd></div>
              <div><dt>Illustrative scenario</dt><dd>{selected.scenario_name ?? "—"}</dd></div>
              <div><dt>Estimated decision cost</dt><dd>{selected.estimated_decision_cost === null ? "—" : formatCurrency(selected.estimated_decision_cost)}</dd></div>
            </dl>
            <h3>Selected model inputs</h3>
            <div className="feature-list">
              {Object.entries(selected.features).map(([feature, value]) => <div key={feature}><span>{feature}</span><strong>{value === null ? "Missing" : String(value)}</strong></div>)}
            </div>
            <p className="drawer-note"><Info size={13} /> Masked IEEE-CIS fields retain their source names. MerchantShield does not invent meanings for C* or D* variables.</p>
          </aside>
        </div>
      )}

      {selectedReview && (
        <div className="drawer-scrim">
          <button className="drawer-backdrop" onClick={() => setSelectedReview(null)} aria-label="Close review workstation" />
          <aside className="detail-drawer review-drawer">
            <button className="icon-button drawer-close" onClick={() => setSelectedReview(null)} aria-label="Close review"><X size={18} /></button>
            <span className="eyebrow">Validation review workstation</span>
            <h2>{selectedReview.transaction_id}</h2>
            <div className="review-risk-hero"><span>Fraud probability</span><strong>{formatPercent(selectedReview.fraud_probability)}</strong><small>Provisional business decision: REVIEW</small></div>
            {reviewError && <div className="inline-error"><AlertTriangle size={14} /> {reviewError}</div>}
            <dl>
              <div><dt>TransactionAmt</dt><dd>{formatAmount(selectedReview.transaction_amount)}</dd></div>
              <div><dt>TransactionDT</dt><dd>{formatNumber(selectedReview.transaction_dt)}</dd></div>
              <div><dt>Ground truth</dt><dd>{groundTruth?.ground_truth ?? "Hidden"}</dd></div>
              <div><dt>Review status</dt><dd>{selectedReview.status}</dd></div>
            </dl>
            <h3>Selected validation fields</h3>
            <div className="feature-list">{Object.entries(selectedReview.features).map(([feature, value]) => <div key={feature}><span>{feature}</span><strong>{value ?? "Missing"}</strong></div>)}</div>
            {selectedReview.status === "OPEN" ? (
              <>
                <label className="review-note"><span>Reviewer note <small>Optional</small></span><textarea value={reviewNote} maxLength={1000} onChange={(event) => setReviewNote(event.target.value)} placeholder="Record the evidence behind your decision…" /></label>
                <div className="review-actions"><button disabled={reviewSubmitting} className="approve-action" onClick={() => submitReviewDecision("APPROVE")}><Check size={15} /> Approve</button><button disabled={reviewSubmitting} className="block-action" onClick={() => submitReviewDecision("BLOCK")}><X size={15} /> Block</button></div>
              </>
            ) : <div className="saved-decision"><Check size={15} /><span>Persisted reviewer decision</span><strong>{selectedReview.reviewer_decision}</strong>{selectedReview.reviewer_note && <small>{selectedReview.reviewer_note}</small>}</div>}
            <button className="reveal-button" onClick={revealReviewGroundTruth}><Eye size={15} /> {groundTruth ? "Refresh ground truth" : "Reveal Ground Truth"}</button>
            {groundTruth && <div className={`ground-truth ${groundTruth.actual_label ? "fraud" : "legitimate"}`}><span>ACTUAL VALIDATION LABEL</span><strong>{groundTruth.ground_truth}</strong>{groundTruth.reviewer_correct !== null && <small>{groundTruth.reviewer_correct ? "Reviewer decision matches the label" : "Reviewer decision does not match the label"}</small>}</div>}
            <p className="drawer-note"><Info size={13} /> Ground truth is excluded from the queue and appears only after this explicit reveal. The reviewer action does not update the model or its evidence.</p>
          </aside>
        </div>
      )}
    </div>
  );
}
