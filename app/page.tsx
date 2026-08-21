"use client";

import {
  ArrowRight,
  BadgeCheck,
  BarChart3,
  BookOpen,
  Check,
  ChevronDown,
  CircleDollarSign,
  Database,
  FileCheck2,
  FileWarning,
  Filter,
  FlaskConical,
  Gauge,
  Info,
  ListChecks,
  Menu,
  ReceiptIndianRupee,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  TableProperties,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type {
  BootstrapResponse,
  CostAssumptions,
  CostSimulationResponse,
  Decision,
  TransactionSummary,
} from "../lib/api-types";

type Section = "overview" | "transactions" | "reviews" | "cost";

const sections: Array<{ id: Section; label: string; icon: typeof BarChart3 }> = [
  { id: "overview", label: "Overview", icon: BarChart3 },
  { id: "transactions", label: "Transactions", icon: TableProperties },
  { id: "reviews", label: "Review Queue", icon: ListChecks },
  { id: "cost", label: "Cost Lab", icon: SlidersHorizontal },
];

const defaultAssumptions: CostAssumptions = {
  currency: "INR",
  fraud_loss_fraction: 1,
  chargeback_fixed_cost: 0,
  legitimate_margin_rate: 0.2,
  false_positive_fixed_cost: 0,
  manual_review_cost: 150,
  review_fraud_catch_rate: 0.9,
  review_legitimate_approval_rate: 0.98,
};

const apiBase = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "");

function apiPath(path: string) {
  return `${apiBase}${path}`;
}

const blankBootstrap: BootstrapResponse = {
  status: "loading",
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

function formatMetric(value: number | null | undefined, kind: "number" | "percent" | "currency" = "number") {
  if (value === null || value === undefined) return "Not evaluated yet";
  if (kind === "percent") return `${(value * 100).toFixed(1)}%`;
  if (kind === "currency") {
    return new Intl.NumberFormat("en-IN", {
      style: "currency",
      currency: "INR",
      maximumFractionDigits: 0,
    }).format(value);
  }
  return new Intl.NumberFormat("en-IN").format(value);
}

function decisionClass(value: string) {
  return value.toLowerCase().replaceAll("_", "-");
}

function MetricCard({ label, value, note, icon: Icon }: { label: string; value: string; note: string; icon: typeof Gauge }) {
  const unavailable = value === "Not evaluated yet";
  return (
    <article className={`metric-card ${unavailable ? "unavailable" : ""}`}>
      <div className="metric-card-top"><span>{label}</span><Icon size={17} /></div>
      <strong>{value}</strong>
      <small><Info size={12} /> {note}</small>
    </article>
  );
}

function EmptyEvidence({ title, detail, action }: { title: string; detail: string; action?: React.ReactNode }) {
  return (
    <div className="empty-evidence">
      <div className="empty-icon"><FileWarning size={22} /></div>
      <h3>{title}</h3>
      <p>{detail}</p>
      {action}
    </div>
  );
}

function TransactionTable({ rows, onSelect }: { rows: TransactionSummary[]; onSelect: (row: TransactionSummary) => void }) {
  return (
    <div className="table-shell">
      <div className="tx-row tx-head">
        <span>Transaction ID</span><span>Amount</span><span>Risk</span><span>Decision</span><span>Actual label</span><span />
      </div>
      {rows.map((row) => (
        <button className="tx-row" key={row.transaction_id} onClick={() => onSelect(row)}>
          <strong>{row.transaction_id}</strong>
          <span>{new Intl.NumberFormat("en-IN", { style: "currency", currency: row.currency, maximumFractionDigits: 0 }).format(row.amount)}</span>
          <span>{(row.risk_score * 100).toFixed(1)}%</span>
          <span><i className={`decision-dot ${decisionClass(row.decision)}`} />{row.decision}</span>
          <span>{row.actual_label === 1 ? "Fraud" : "Legitimate"}</span>
          <ArrowRight size={15} />
        </button>
      ))}
    </div>
  );
}

export default function Home() {
  const [active, setActive] = useState<Section>("overview");
  const [mobileOpen, setMobileOpen] = useState(false);
  const [data, setData] = useState<BootstrapResponse>(blankBootstrap);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [decision, setDecision] = useState<"ALL" | Decision>("ALL");
  const [selected, setSelected] = useState<TransactionSummary | null>(null);
  const [reviewThreshold, setReviewThreshold] = useState(0.4);
  const [blockThreshold, setBlockThreshold] = useState(0.8);
  const [assumptions, setAssumptions] = useState(defaultAssumptions);
  const [simulation, setSimulation] = useState<CostSimulationResponse | null>(null);
  const [simulating, setSimulating] = useState(false);
  const [selectedReview, setSelectedReview] = useState<BootstrapResponse["reviews"][number] | null>(null);
  const [reviewNote, setReviewNote] = useState("");
  const [reviewSaving, setReviewSaving] = useState(false);
  const [reviewMessage, setReviewMessage] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    fetch(apiPath("/api/v1/bootstrap"), { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Dashboard API returned ${response.status}`);
        return response.json() as Promise<BootstrapResponse>;
      })
      .then(setData)
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setData({ ...blankBootstrap, status: "unavailable" });
        setLoadError(error instanceof Error ? error.message : "Dashboard API unavailable");
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const handle = window.setTimeout(async () => {
      setSimulating(true);
      try {
        const response = await fetch(apiPath("/api/v1/cost/simulate"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({ review_threshold: reviewThreshold, block_threshold: blockThreshold, assumptions }),
        });
        const payload = await response.json() as CostSimulationResponse & { error?: string };
        if (!response.ok) throw new Error(payload.error ?? `Cost API returned ${response.status}`);
        setSimulation(payload);
      } catch (error: unknown) {
        if (!(error instanceof DOMException && error.name === "AbortError")) {
          setSimulation({ evaluated: false, provenance: error instanceof Error ? error.message : "Cost API unavailable", current: null, proposed: null });
        }
      } finally {
        setSimulating(false);
      }
    }, 180);
    return () => { controller.abort(); window.clearTimeout(handle); };
  }, [reviewThreshold, blockThreshold, assumptions]);

  const filteredTransactions = useMemo(() => data.transactions.filter((row) => {
    const matchesQuery = row.transaction_id.toLowerCase().includes(search.toLowerCase());
    return matchesQuery && (decision === "ALL" || row.decision === decision);
  }), [data.transactions, decision, search]);

  const metrics = data.metrics;
  const provenance = data.evaluated
    ? data.provenance
    : "Awaiting final held-out temporal evaluation on local IEEE-CIS labels.";

  function updateReviewThreshold(next: number) {
    setReviewThreshold(Math.min(next, blockThreshold - 0.01));
  }

  function updateBlockThreshold(next: number) {
    setBlockThreshold(Math.max(next, reviewThreshold + 0.01));
  }

  async function submitReview(reviewerDecision: "APPROVE" | "DECLINE") {
    if (!selectedReview || reviewNote.trim().length < 3) {
      setReviewMessage("Add a reviewer note with at least 3 characters.");
      return;
    }
    setReviewSaving(true);
    setReviewMessage("");
    try {
      const response = await fetch(apiPath(`/api/v1/reviews/${selectedReview.id}/decision`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ decision: reviewerDecision, reason: reviewNote.trim() }),
      });
      const payload = await response.json() as { detail?: string };
      if (!response.ok) throw new Error(payload.detail ?? `Review API returned ${response.status}`);
      setData((current) => ({ ...current, reviews: current.reviews.filter((review) => review.id !== selectedReview.id) }));
      setSelectedReview(null);
      setReviewNote("");
    } catch (error: unknown) {
      setReviewMessage(error instanceof Error ? error.message : "Review submission failed");
    } finally {
      setReviewSaving(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileOpen ? "open" : ""}`}>
        <div className="brand-row">
          <div className="brand-mark"><ShieldCheck size={21} /></div>
          <div><strong>MerchantShield</strong><span>Risk decision engine</span></div>
          <button className="icon-button close-nav" onClick={() => setMobileOpen(false)} aria-label="Close navigation"><X size={18} /></button>
        </div>

        <div className={`evidence-badge ${data.evaluated ? "ready" : "pending"}`}>
          {data.evaluated ? <FileCheck2 size={16} /> : <Database size={16} />}
          <div><strong>{data.evaluated ? "Evidence ready" : "Evaluation pending"}</strong><span>{data.dataset.name}</span></div>
        </div>

        <nav aria-label="Product modules">
          <span className="nav-label">Decision workflow</span>
          {sections.map((item) => {
            const Icon = item.icon;
            return (
              <button key={item.id} className={active === item.id ? "active" : ""} onClick={() => { setActive(item.id); setMobileOpen(false); }}>
                <Icon size={17} /><span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="sidebar-note">
          <BadgeCheck size={16} />
          <div><strong>Defense-only</strong><span>No attack generation, fake live feed, or hidden model claims.</span></div>
        </div>
      </aside>
      {mobileOpen && <button className="nav-scrim" aria-label="Close navigation" onClick={() => setMobileOpen(false)} />}

      <main className="main-area">
        <header className="topbar">
          <button className="icon-button menu-button" onClick={() => setMobileOpen(true)} aria-label="Open navigation"><Menu size={19} /></button>
          <div className="crumb"><span>Merchant risk</span><ArrowRight size={13} /><strong>{sections.find((item) => item.id === active)?.label}</strong></div>
          <div className={`status-pill ${data.evaluated ? "ready" : "pending"}`}><span />{data.evaluated ? "Held-out results loaded" : "Not evaluated yet"}</div>
        </header>

        <div className="page-content">
          {loadError && <div className="api-alert"><FileWarning size={16} /><span><strong>API unavailable.</strong> {loadError}</span></div>}

          {active === "overview" && (
            <section className="module-page">
              <div className="page-heading">
                <div><span className="eyebrow">Cost-aware fraud decisions</span><h1>Evidence before automation.</h1><p>MerchantShield turns a calibrated fraud score and validation-derived rules into <strong>APPROVE</strong>, <strong>REVIEW</strong>, or <strong>BLOCK</strong>.</p></div>
                <button className="secondary-button" onClick={() => setActive("cost")}><FlaskConical size={15} /> Open Cost Lab</button>
              </div>

              <div className={`provenance-banner ${data.evaluated ? "ready" : "pending"}`}>
                {data.evaluated ? <FileCheck2 size={18} /> : <FileWarning size={18} />}
                <div><strong>{data.evaluated ? "Held-out evidence loaded" : "No result has been fabricated"}</strong><span>{provenance}</span></div>
                <button onClick={() => document.getElementById("readiness")?.scrollIntoView({ behavior: "smooth" })}>View readiness <ArrowRight size={14} /></button>
              </div>

              <div className="metrics-grid">
                <MetricCard icon={Database} label="Transactions evaluated" value={formatMetric(metrics?.transactions_evaluated)} note="Held-out temporal test set" />
                <MetricCard icon={ShieldCheck} label="Fraud cases" value={formatMetric(metrics?.fraud_cases)} note="Ground-truth labels" />
                <MetricCard icon={Gauge} label="Precision" value={formatMetric(metrics?.precision, "percent")} note="Final BLOCK prediction quality" />
                <MetricCard icon={BarChart3} label="Recall" value={formatMetric(metrics?.recall, "percent")} note="Held-out fraud detected" />
                <MetricCard icon={ReceiptIndianRupee} label="Estimated total cost" value={formatMetric(metrics?.estimated_total_cost, "currency")} note="Current merchant assumptions" />
              </div>

              <div className="overview-grid">
                <article className="panel flow-panel">
                  <div className="panel-heading"><div><span>Core decision loop</span><h2>One explainable path</h2></div><ShieldCheck size={18} /></div>
                  <div className="decision-flow">
                    <div><i>01</i><span><strong>Risk score</strong><small>Frozen model probability</small></span></div><ArrowRight size={16} />
                    <div><i>02</i><span><strong>Rules</strong><small>Validation evidence only</small></span></div><ArrowRight size={16} />
                    <div><i>03</i><span><strong>Decision</strong><small>Approve · Review · Block</small></span></div><ArrowRight size={16} />
                    <div><i>04</i><span><strong>Feedback</strong><small>Persisted, never auto-trained</small></span></div>
                  </div>
                  <div className="threshold-strip" style={{ background: `linear-gradient(90deg, #eaf4ee 0 ${reviewThreshold * 100}%, #fff5df ${reviewThreshold * 100}% ${blockThreshold * 100}%, #fcedeb ${blockThreshold * 100}%)` }}>
                    <span>APPROVE</span><i style={{ left: `${reviewThreshold * 100}%` }} /><span>REVIEW</span><i style={{ left: `${blockThreshold * 100}%` }} /><span>BLOCK</span>
                  </div>
                  <small className="threshold-copy">Current proposed thresholds: review at {reviewThreshold.toFixed(2)}, block at {blockThreshold.toFixed(2)}. Merchant-configurable assumptions, not model metrics.</small>
                </article>

                <article className="panel evidence-panel">
                  <div className="panel-heading"><div><span>Held-out evaluation</span><h2>Precision & recall</h2></div><BadgeCheck size={18} /></div>
                  {data.evaluated && metrics ? (
                    <div className="precision-recall">
                      <div style={{ "--meter": `${metrics.precision * 100}%` } as React.CSSProperties}><span>Precision</span><strong>{formatMetric(metrics.precision, "percent")}</strong><i /></div>
                      <div style={{ "--meter": `${metrics.recall * 100}%` } as React.CSSProperties}><span>Recall</span><strong>{formatMetric(metrics.recall, "percent")}</strong><i /></div>
                      <p>{data.provenance}</p>
                    </div>
                  ) : <EmptyEvidence title="Not evaluated yet" detail="Run the real IEEE-CIS pipeline, freeze the model and thresholds on validation, then perform the single held-out test evaluation." />}
                </article>
              </div>

              <div className="overview-grid lower" id="readiness">
                <article className="panel readiness-panel">
                  <div className="panel-heading"><div><span>Implementation state</span><h2>Evidence readiness</h2></div><BookOpen size={18} /></div>
                  <div className="readiness-list">
                    <div className="done"><Check size={14} /><span><strong>Software path</strong><small>Loader, temporal split, models, cost engine, API and UI</small></span><b>READY</b></div>
                    <div className={data.dataset.available ? "done" : "waiting"}><span className="step-dot" /><span><strong>Local dataset</strong><small>train_transaction.csv + train_identity.csv</small></span><b>{data.dataset.available ? "READY" : "WAITING"}</b></div>
                    <div className={data.model.available ? "done" : "waiting"}><span className="step-dot" /><span><strong>Frozen model</strong><small>Selected on validation only</small></span><b>{data.model.available ? "READY" : "WAITING"}</b></div>
                    <div className={data.evaluated ? "done" : "waiting"}><span className="step-dot" /><span><strong>Held-out report</strong><small>Source of truth for product metrics</small></span><b>{data.evaluated ? "READY" : "WAITING"}</b></div>
                  </div>
                </article>

                <article className="panel distribution-panel">
                  <div className="panel-heading"><div><span>Final decisions</span><h2>Decision distribution</h2></div><BarChart3 size={18} /></div>
                  {data.decision_distribution ? (
                    <div className="distribution-bars">
                      {(["approve", "review", "block"] as const).map((key) => <div key={key}><span>{key}</span><i><b style={{ width: `${data.decision_distribution![key].share * 100}%` }} /></i><strong>{formatMetric(data.decision_distribution![key].count)}</strong></div>)}
                    </div>
                  ) : <EmptyEvidence title="Not evaluated yet" detail="Decision counts appear only after two thresholds are frozen and applied to the held-out predictions." />}
                </article>
              </div>
            </section>
          )}

          {active === "transactions" && (
            <section className="module-page">
              <div className="page-heading compact"><div><span className="eyebrow">Held-out explorer</span><h1>Transactions</h1><p>Inspect actual labels, scores, rules, and model errors without hiding failure cases.</p></div></div>
              <div className="toolbar">
                <label className="search-field"><Search size={15} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search transaction ID" /></label>
                <label className="select-field"><Filter size={14} /><select value={decision} onChange={(event) => setDecision(event.target.value as typeof decision)}><option>ALL</option><option>APPROVE</option><option>REVIEW</option><option>BLOCK</option></select><ChevronDown size={14} /></label>
                <span className="result-count">{data.evaluated ? `${filteredTransactions.length} loaded rows` : "No held-out rows loaded"}</span>
              </div>
              {filteredTransactions.length ? <TransactionTable rows={filteredTransactions} onSelect={setSelected} /> : (
                <EmptyEvidence title="No held-out transactions available" detail="Competition rows remain local-only. After evaluation, export a bounded set of permitted examples from real predictions; never invent transaction history for this table." action={<a className="inline-link" href="#setup" onClick={(event) => { event.preventDefault(); setActive("overview"); window.setTimeout(() => document.getElementById("readiness")?.scrollIntoView({ behavior: "smooth" }), 0); }}>See data readiness <ArrowRight size={14} /></a>} />
              )}
            </section>
          )}

          {active === "reviews" && (
            <section className="module-page">
              <div className="page-heading compact"><div><span className="eyebrow">Human-in-the-loop controls</span><h1>Review Queue</h1><p>Analyst feedback is persisted for future iteration. It never triggers automatic retraining.</p></div></div>
              <div className="review-policy"><BadgeCheck size={17} /><div><strong>Review policy</strong><span>Only transactions in the configured review band appear here. A decision and reason are required.</span></div><b>{data.reviews.length} OPEN</b></div>
              {data.reviews.length ? <div className="review-list">{data.reviews.map((review) => <article key={review.id}><span>{review.transaction_id}</span><strong>{(review.risk_score * 100).toFixed(1)}%</strong><p>{review.primary_factors.join(" · ") || "No contribution artifact available"}</p><button onClick={() => { setSelectedReview(review); setReviewNote(""); setReviewMessage(""); }}>Open review <ArrowRight size={14} /></button></article>)}</div> : (
                <EmptyEvidence title="Review queue is empty" detail="The queue will be seeded only from real held-out REVIEW decisions or genuine production events. Synthetic review cases are never shown as historical evidence." />
              )}
            </section>
          )}

          {active === "cost" && (
            <section className="module-page">
              <div className="page-heading compact"><div><span className="eyebrow">Merchant decision economics</span><h1>Cost Lab</h1><p>Explore how review and block thresholds trade fraud loss against customer friction and review operations.</p></div><span className="assumption-tag"><CircleDollarSign size={15} /> Merchant assumptions</span></div>
              <div className="cost-layout">
                <article className="panel controls-panel">
                  <div className="panel-heading"><div><span>Decision thresholds</span><h2>Proposed configuration</h2></div><SlidersHorizontal size={18} /></div>
                  <label className="range-control"><span><b>Review threshold</b><strong>{reviewThreshold.toFixed(2)}</strong></span><input type="range" min="0.01" max="0.94" step="0.01" value={reviewThreshold} onChange={(event) => updateReviewThreshold(Number(event.target.value))} /><small>Scores at or above this value enter manual review.</small></label>
                  <label className="range-control"><span><b>Block threshold</b><strong>{blockThreshold.toFixed(2)}</strong></span><input type="range" min="0.06" max="0.99" step="0.01" value={blockThreshold} onChange={(event) => updateBlockThreshold(Number(event.target.value))} /><small>Must remain greater than the review threshold.</small></label>
                  <div className="assumption-grid">
                    <label><span>Fraud loss fraction</span><div><input type="number" min="0" max="2" step="0.05" value={assumptions.fraud_loss_fraction} onChange={(event) => setAssumptions({ ...assumptions, fraud_loss_fraction: Number(event.target.value) })} /><b>× amount</b></div></label>
                    <label><span>Merchant margin</span><div><input type="number" min="0" max="1" step="0.01" value={assumptions.legitimate_margin_rate} onChange={(event) => setAssumptions({ ...assumptions, legitimate_margin_rate: Number(event.target.value) })} /><b>fraction</b></div></label>
                    <label><span>Review cost</span><div><input type="number" min="0" step="1" value={assumptions.manual_review_cost} onChange={(event) => setAssumptions({ ...assumptions, manual_review_cost: Number(event.target.value) })} /><b>INR</b></div></label>
                    <label><span>Fraud caught in review</span><div><input type="number" min="0" max="1" step="0.01" value={assumptions.review_fraud_catch_rate} onChange={(event) => setAssumptions({ ...assumptions, review_fraud_catch_rate: Number(event.target.value) })} /><b>rate</b></div></label>
                    <label><span>Legitimate approved in review</span><div><input type="number" min="0" max="1" step="0.01" value={assumptions.review_legitimate_approval_rate} onChange={(event) => setAssumptions({ ...assumptions, review_legitimate_approval_rate: Number(event.target.value) })} /><b>rate</b></div></label>
                  </div>
                  <div className="validation-rule"><Check size={14} /><span>Valid: review threshold &lt; block threshold</span></div>
                </article>

                <article className="panel outcome-panel">
                  <div className="panel-heading"><div><span>Calculated over held-out predictions</span><h2>{simulating ? "Recalculating…" : "Configuration impact"}</h2></div><FlaskConical size={18} /></div>
                  {simulation?.evaluated && simulation.proposed ? (
                    <div className="cost-results">
                      <div className="cost-hero"><span>Total estimated cost</span><strong>{formatMetric(simulation.proposed.total_estimated_cost, "currency")}</strong><small>Lowest estimated cost under current assumptions is highlighted only when a validation search artifact is available.</small></div>
                      <div className="cost-metric-grid"><span>Precision<strong>{formatMetric(simulation.proposed.precision, "percent")}</strong></span><span>Recall<strong>{formatMetric(simulation.proposed.recall, "percent")}</strong></span><span>False positives<strong>{formatMetric(simulation.proposed.false_positives)}</strong></span><span>False negatives<strong>{formatMetric(simulation.proposed.false_negatives)}</strong></span><span>Review volume<strong>{formatMetric(simulation.proposed.review_volume)}</strong></span><span>Block volume<strong>{formatMetric(simulation.proposed.block_volume)}</strong></span></div>
                    </div>
                  ) : <EmptyEvidence title="Not evaluated yet" detail={simulation?.provenance ?? "Cost simulation requires real held-out labels, prediction probabilities, and transaction amounts. Your merchant assumptions are editable, but no monetary result is invented."} />}
                  <div className="formula-note"><Info size={15} /><div><strong>Transparent cost formula</strong><span>Fraud loss + legitimate block cost + review cost + residual review outcomes. Every assumption is separated from model-derived counts.</span></div></div>
                </article>
              </div>
            </section>
          )}
        </div>
      </main>

      {selected && (
        <div className="drawer-scrim">
          <button className="drawer-backdrop" onClick={() => setSelected(null)} aria-label="Close transaction details" />
          <aside className="detail-drawer">
            <button className="icon-button drawer-close" onClick={() => setSelected(null)} aria-label="Close transaction"><X size={18} /></button>
            <span className="eyebrow">Held-out transaction</span><h2>{selected.transaction_id}</h2>
            {selected.model_error && <div className="model-error">MODEL ERROR</div>}
            <dl><div><dt>Amount</dt><dd>{selected.amount}</dd></div><div><dt>Risk score</dt><dd>{(selected.risk_score * 100).toFixed(1)}%</dd></div><div><dt>Decision</dt><dd>{selected.decision}</dd></div><div><dt>Actual label</dt><dd>{selected.actual_label === 1 ? "Fraud" : "Legitimate"}</dd></div></dl>
            <h3>Top factors</h3><div className="factor-list">{selected.top_factors.map((factor) => <span key={factor.feature_name}><strong>{factor.feature_name}</strong><small>{factor.contribution.toFixed(4)} contribution</small></span>)}</div>
            <p className="drawer-note">Masked IEEE-CIS fields are shown by their source names. MerchantShield does not invent undocumented business meanings.</p>
          </aside>
        </div>
      )}

      {selectedReview && (
        <div className="review-modal-shell">
          <button className="review-modal-backdrop" onClick={() => setSelectedReview(null)} aria-label="Close review" />
          <section className="review-modal" role="dialog" aria-modal="true" aria-labelledby="review-title">
            <button className="icon-button review-close" onClick={() => setSelectedReview(null)} aria-label="Close review"><X size={18} /></button>
            <span className="eyebrow">Human review</span>
            <h2 id="review-title">{selectedReview.transaction_id}</h2>
            <div className="review-score"><span>Model risk score</span><strong>{(selectedReview.risk_score * 100).toFixed(1)}%</strong></div>
            <div className="review-factors"><span>Primary model factors</span><p>{selectedReview.primary_factors.join(" · ") || "No contribution artifact available"}</p></div>
            <label className="review-note"><span>Reviewer note <b>Required</b></span><textarea value={reviewNote} onChange={(event) => setReviewNote(event.target.value)} placeholder="Record the evidence behind this decision…" maxLength={1000} /></label>
            {reviewMessage && <p className="review-message">{reviewMessage}</p>}
            <div className="review-actions"><button className="approve-button" disabled={reviewSaving} onClick={() => submitReview("APPROVE")}><Check size={15} /> Approve</button><button className="decline-button" disabled={reviewSaving} onClick={() => submitReview("DECLINE")}><X size={15} /> Decline</button></div>
            <small>Reviewer feedback is stored for future iteration and never starts automatic retraining.</small>
          </section>
        </div>
      )}
    </div>
  );
}
