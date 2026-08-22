"use client";

import {
  AlertTriangle,
  ArrowRight,
  BarChart3,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleDollarSign,
  Clock3,
  Database,
  FileWarning,
  Filter,
  FlaskConical,
  Info,
  Layers3,
  ListChecks,
  LockKeyhole,
  Menu,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  TableProperties,
  TrendingUp,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
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
  InterestingCasesResponse,
  ModelComparisonResponse,
  ProjectStatusResponse,
  ValidationFilter,
  ValidationTransaction,
  ValidationTransactionPage,
} from "../lib/api-types";

type Section = "overview" | "transactions" | "reviews" | "cost";

const sections: Array<{ id: Section; label: string; icon: typeof BarChart3 }> = [
  { id: "overview", label: "Overview", icon: BarChart3 },
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
        <span>Transaction</span><span>TransactionAmt</span><span>Fraud probability</span><span>Prediction @{threshold.toFixed(2)}</span><span>Actual label</span><span>Outcome</span><span />
      </div>
      {rows.map((row) => (
        <button className="transaction-row" key={row.transaction_id} onClick={() => onSelect(row)}>
          <strong>{row.transaction_id}</strong>
          <span>{formatAmount(row.transaction_amount)}</span>
          <span className="risk-value">{formatPercent(row.fraud_probability)}</span>
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

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      fetchJson<ProjectStatusResponse>("/api/v1/project/status", controller.signal),
      fetchJson<ModelComparisonResponse>("/api/v1/model-comparison", controller.signal),
      fetchJson<FeatureImportanceResponse>("/api/v1/model/feature-importance?limit=13", controller.signal),
      fetchJson<InterestingCasesResponse>("/api/v1/validation/interesting-cases", controller.signal),
    ])
      .then(([nextStatus, nextComparison, nextImportance, nextInteresting]) => {
        setStatus(nextStatus);
        setComparison(nextComparison);
        setImportance(nextImportance);
        setInteresting(nextInteresting);
      })
      .catch((caught: unknown) => {
        if (caught instanceof DOMException && caught.name === "AbortError") return;
        setError(caught instanceof Error ? caught.message : "Project evidence could not be loaded");
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [reloadKey]);

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

  function changeFilter(next: ValidationFilter) {
    setFilter(next);
    setPage(1);
  }

  function reloadEvidence() {
    setLoading(true);
    setError(null);
    setReloadKey((value) => value + 1);
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
          <span className="nav-label">Project evidence</span>
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
                          ["Validation threshold analysis", thresholdAnalysisReady ? "Complete" : "Pending", thresholdAnalysisReady, false],
                          ["Rules", "Pending", false, false],
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
                          "Precision", "Recall", "F1", "PR-AUC", "Estimated cost",
                        ].map((label) => <LockedValue key={label} label={label} />)}
                      </div>
                      <p><LockKeyhole size={13} /> Held-out test remains sealed. Validation evidence is never relabelled as final performance.</p>
                    </article>
                  </section>

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
            <section className="module-page locked-page">
              <div className="compact-heading"><div><span className="eyebrow">Human-in-the-loop operations</span><h1>Review Queue</h1><p>The product surface is ready, but no provisional validation decision is presented as an operational case.</p></div></div>
              <div className="locked-empty"><div className="lock-orbit"><LockKeyhole size={26} /></div><span>OPERATIONAL THRESHOLDS LOCKED</span><h2>Review thresholds are not activated yet.</h2><p>{thresholdAnalysisReady ? "Validation threshold analysis exists, but operational activation requires the model, rules, governance, and final held-out evaluation to be frozen." : "Operational activation requires validated thresholds, rules, governance, and final held-out evaluation."} No fake review cases are shown.</p><div className="locked-steps"><div className={status ? "done" : ""}>{status ? <Check size={14} /> : <Clock3 size={14} />}<span>Validation model evidence</span></div><div className={thresholdAnalysisReady ? "done" : ""}>{thresholdAnalysisReady ? <Check size={14} /> : <Clock3 size={14} />}<span>Validation threshold analysis</span></div><div><Clock3 size={14} /><span>Rule design and final evaluation</span></div></div></div>
            </section>
          )}

          {active === "cost" && (
            <section className="module-page cost-page">
              <div className="compact-heading"><div><span className="eyebrow">Merchant decision economics</span><h1>Cost Lab</h1><p>Assumptions and controls remain visible as product structure while operational monetary claims stay disabled.</p></div><div className="locked-pill"><LockKeyhole size={14} /> OPERATIONAL SIMULATION LOCKED</div></div>
              <div className="cost-locked-grid">
                <article className="panel disabled-controls">
                  <div className="panel-title"><div><span>CONFIGURATION</span><h2>Awaiting operational activation</h2></div><SlidersHorizontal size={19} /></div>
                  {["Review threshold", "Block threshold", "Merchant margin", "Review cost", "Fraud loss assumption"].map((label) => <label key={label}><span>{label}<b>—</b></span><input type="range" min="0" max="1" value="0" disabled readOnly /></label>)}
                  <div className="disabled-note"><Info size={15} /><p>Cost simulation will be enabled after validation methodology, rules, and operational thresholds are frozen. No rupee estimate is displayed here.</p></div>
                </article>
                <article className="panel locked-result">
                  <div className="panel-title"><div><span>FINAL COST</span><h2>Not evaluated yet</h2></div><CircleDollarSign size={19} /></div>
                  <div className="result-lock"><LockKeyhole size={28} /><strong>No monetary result shown</strong><p>Validation assumptions must not be presented as realized merchant savings or final held-out cost.</p></div>
                  <div className="formula-strip"><FlaskConical size={16} /><span>Future output: fraud loss + false-positive cost + review cost, under explicit merchant assumptions.</span></div>
                </article>
              </div>
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
              <div><dt>Saved prediction label</dt><dd>{selected.predicted_label_at_0_5 ? "Fraud" : "Legitimate"}</dd></div>
              <div><dt>Actual validation label</dt><dd>{selected.actual_label ? "Fraud" : "Legitimate"}</dd></div>
              <div><dt>TransactionDT</dt><dd>{formatNumber(selected.transaction_dt)}</dd></div>
              <div><dt>Partition</dt><dd>Validation</dd></div>
            </dl>
            <h3>Selected model inputs</h3>
            <div className="feature-list">
              {Object.entries(selected.features).map(([feature, value]) => <div key={feature}><span>{feature}</span><strong>{value === null ? "Missing" : String(value)}</strong></div>)}
            </div>
            <p className="drawer-note"><Info size={13} /> Masked IEEE-CIS fields retain their source names. MerchantShield does not invent meanings for C* or D* variables.</p>
          </aside>
        </div>
      )}
    </div>
  );
}
