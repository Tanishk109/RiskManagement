"use client";
/* eslint-disable @next/next/no-html-link-for-pages */

import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  FileText,
  Gauge,
  Info,
  ListChecks,
  LockKeyhole,
  Menu,
  Play,
  Radar,
  ShieldCheck,
  SlidersHorizontal,
  TableProperties,
  Upload,
  X,
  Zap,
} from "lucide-react";
import { ChangeEvent, useEffect, useMemo, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const apiBase = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "");
const apiPath = (path: string) => `${apiBase}${path}`;

type DetectorMethod = "rolling_zscore" | "ewma" | "percent_deviation";
type PulseMetric = "transaction_count" | "mean_risk_score" | "high_risk_count" | "high_risk_amount";
type DetectorConfig = { method: DetectorMethod; metric: PulseMetric; window_seconds: number; baseline_windows: number; sensitivity: number; ewma_alpha: number; percent_deviation_threshold: number };
type PulseWindow = { window_index: number; window_start: number; window_end: number; transaction_count: number; mean_risk_score: number; high_risk_count: number; review_count: number; block_count: number; high_risk_amount: number; monitored_value: number; baseline_state: "WARMING_UP" | "READY"; baseline_value: number | null; absolute_change: number | null; percent_deviation: number | null; detector_score: number | null; alert_active: boolean };
type PulseAlert = { window_index: number; window_start: number; metric: PulseMetric; current_value: number; baseline_value: number; absolute_change: number; percent_deviation: number | null; detector_score: number; label: "SPIKE ALERT" };
type PulseResponse = { source: string; data_partition: "validation" | "merchant upload"; evaluation_status: "Not evaluated yet"; detector_is_classifier: false; config: DetectorConfig; model_version: string; review_threshold: number; block_threshold: number; rows_received: number; rows_scored: number; invalid_rows: Array<{ row: number; transaction_id: string | null; errors: string[] }>; windows: PulseWindow[]; alerts: PulseAlert[]; held_out_test_accessed: false; limitations: string[] };
type PulseStatus = { data_source: string; evaluation_status: "Not evaluated yet"; detector_is_classifier: false; model_version: string; review_threshold: number; block_threshold: number; upload_required_columns: string[]; limitations: string[] };

const defaultConfig: DetectorConfig = { method: "rolling_zscore", metric: "high_risk_count", window_seconds: 21600, baseline_windows: 8, sensitivity: 3, ewma_alpha: .3, percent_deviation_threshold: .5 };
const metricLabels: Record<PulseMetric, string> = { transaction_count: "Transaction count", mean_risk_score: "Mean risk score", high_risk_count: "High-risk count", high_risk_amount: "High-risk amount" };
const methodLabels: Record<DetectorMethod, string> = { rolling_zscore: "Rolling z-score", ewma: "EWMA deviation", percent_deviation: "Percent deviation" };

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiPath(path), init);
  if (!response.ok) {
    const payload = await response.json().catch(() => null) as { detail?: string } | null;
    throw new Error(payload?.detail ?? `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

function compact(value: number, metric: PulseMetric) {
  if (metric === "mean_risk_score") return `${(value * 100).toFixed(2)}%`;
  if (metric === "high_risk_amount") return new Intl.NumberFormat("en-IN", { notation: "compact", maximumFractionDigits: 1 }).format(value);
  return new Intl.NumberFormat("en-IN", { maximumFractionDigits: 1 }).format(value);
}

export default function FraudPulse() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [status, setStatus] = useState<PulseStatus | null>(null);
  const [config, setConfig] = useState<DetectorConfig>(defaultConfig);
  const [result, setResult] = useState<PulseResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadName, setUploadName] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api<PulseStatus>("/api/v1/fraud-pulse/status", { signal: controller.signal }).then(setStatus).catch((caught: unknown) => {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      setError(caught instanceof Error ? caught.message : "Fraud Pulse status could not be loaded");
    });
    return () => controller.abort();
  }, []);

  async function replayValidation() {
    setBusy(true); setError(null); setUploadName(null);
    try {
      setResult(await api<PulseResponse>("/api/v1/fraud-pulse/replay", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ config }) }));
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Validation replay could not be analyzed"); }
    finally { setBusy(false); }
  }

  async function uploadCsv(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]; event.target.value = "";
    if (!file) return;
    if (file.size > 1_000_000) { setError("CSV exceeds the documented 1 MB limit"); return; }
    setBusy(true); setError(null); setUploadName(file.name);
    try {
      setResult(await api<PulseResponse>("/api/v1/fraud-pulse/upload", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ csv_content: await file.text(), config }) }));
    } catch (caught) { setError(caught instanceof Error ? caught.message : "CSV could not be scored and analyzed"); }
    finally { setBusy(false); }
  }

  const chartData = useMemo(() => result?.windows.map((window) => ({ name: `W${window.window_index + 1}`, current: window.monitored_value, baseline: window.baseline_value, alertValue: window.alert_active ? window.monitored_value : null })) ?? [], [result]);
  const latest = result?.windows.at(-1);

  return <div className="app-shell">
    <aside className={`sidebar ${mobileOpen ? "open" : ""}`}>
      <div className="brand-row"><div className="brand-mark"><ShieldCheck size={21} /></div><div><strong>MerchantShield</strong><span>Merchant loss prevention</span></div><button className="icon-button close-nav" onClick={() => setMobileOpen(false)} aria-label="Close navigation"><X size={18} /></button></div>
      <div className="candidate-badge"><span className="candidate-dot" /><div><strong>Frozen CatBoost monitor</strong><small>Validation thresholds · no retraining</small></div></div>
      <nav aria-label="MerchantShield modules"><span className="nav-label">Workspace</span><a className="suite-nav-link" href="/"><BarChart3 size={17} /><span>Overview</span></a><span className="nav-label nav-label-spaced">Fraud Risk</span><a className="suite-nav-link" href="/"><Gauge size={17} /><span>Risk Check</span></a><a className="suite-nav-link" href="/"><TableProperties size={17} /><span>Transactions</span></a><a className="suite-nav-link" href="/"><ListChecks size={17} /><span>Review Queue</span></a><a className="suite-nav-link" href="/"><SlidersHorizontal size={17} /><span>Cost Lab</span></a><span className="nav-label nav-label-spaced">Loss prevention</span><a className="suite-nav-link" href="/chargebacks"><FileText size={17} /><span>Chargebacks</span></a><a className="suite-nav-link active" href="/fraud-pulse"><Activity size={17} /><span>Fraud Pulse</span></a></nav>
      <div className="sidebar-note"><LockKeyhole size={16} /><div><strong>Held-out test sealed</strong><span>Pulse replay uses chronological validation probabilities only.</span></div></div>
    </aside>
    {mobileOpen && <button className="nav-scrim" onClick={() => setMobileOpen(false)} aria-label="Close navigation" />}
    <main className="main-area"><header className="topbar"><button className="icon-button menu-button" onClick={() => setMobileOpen(true)} aria-label="Open navigation"><Menu size={19} /></button><div className="breadcrumb"><span>MerchantShield</span><ArrowRight size={12} /><strong>Fraud Pulse</strong></div><div className="top-status neutral-status"><span />{status?.evaluation_status.toUpperCase() ?? "LOADING MONITOR"}</div></header>
      <div className="page-content pulse-page"><section className="module-page">
        <div className="compact-heading"><div><span className="eyebrow">FRAUD-SPIKE DETECTOR</span><h1>See risk move before loss compounds.</h1><p>Monitor changes in frozen CatBoost risk probabilities with a visible rolling baseline. Fraud Pulse is a score-volume monitor—not another fraud classifier.</p></div><div className="validation-pill"><Radar size={14} /> TRANSPARENT DETECTOR</div></div>
        <div className="module-disclosure"><Info size={17} /><div><strong>Data source: {status?.data_source ?? "Frozen validation probabilities and chronological transaction fields"}</strong><p>Model: {status?.model_version ?? "catboost-validation-v1"}. Review/block thresholds: {status ? `${status.review_threshold.toFixed(3)} / ${status.block_threshold.toFixed(3)}` : "0.175 / 0.250"}. Evaluation: <b>Not evaluated yet.</b></p></div></div>
        {error && <div className="error-state"><AlertTriangle size={17} /><div><strong>Pulse notice</strong><span>{error}</span></div><button onClick={() => setError(null)}>Dismiss</button></div>}
        <div className="pulse-layout">
          <aside className="panel pulse-controls"><div className="panel-title"><div><span>DETECTOR CONFIGURATION</span><h2>What change should trigger attention?</h2></div><SlidersHorizontal size={19} /></div>
            <div className="pulse-control-grid"><label><span>Method</span><select value={config.method} onChange={(e) => setConfig({ ...config, method: e.target.value as DetectorMethod })}>{Object.entries(methodLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label><label><span>Monitored metric</span><select value={config.metric} onChange={(e) => setConfig({ ...config, metric: e.target.value as PulseMetric })}>{Object.entries(metricLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label><label><span>Window length</span><select value={config.window_seconds} onChange={(e) => setConfig({ ...config, window_seconds: Number(e.target.value) })}><option value={3600}>1 hour</option><option value={21600}>6 hours</option><option value={43200}>12 hours</option><option value={86400}>24 hours</option></select></label><label><span>Baseline windows</span><input type="number" min={3} max={48} value={config.baseline_windows} onChange={(e) => setConfig({ ...config, baseline_windows: Number(e.target.value) })} /></label>
              {config.method === "percent_deviation" ? <label><span>Alert above baseline</span><input type="number" min={.01} max={10} step={.05} value={config.percent_deviation_threshold} onChange={(e) => setConfig({ ...config, percent_deviation_threshold: Number(e.target.value) })} /><small>{(config.percent_deviation_threshold * 100).toFixed(0)}% increase</small></label> : <label><span>{config.method === "ewma" ? "Relative sensitivity" : "Z-score sensitivity"}</span><input type="number" min={.1} max={20} step={.1} value={config.sensitivity} onChange={(e) => setConfig({ ...config, sensitivity: Number(e.target.value) })} /><small>Trigger at {config.sensitivity.toFixed(1)}{config.method === "rolling_zscore" ? "σ" : "× relative change"}</small></label>}
              {config.method === "ewma" && <label><span>EWMA alpha</span><input type="number" min={.01} max={1} step={.05} value={config.ewma_alpha} onChange={(e) => setConfig({ ...config, ewma_alpha: Number(e.target.value) })} /><small>Higher values react faster</small></label>}
            </div>
            <div className="pulse-actions"><button disabled={busy} onClick={replayValidation}><Play size={15} /> {busy ? "Analyzing…" : "Replay Validation"}</button><label className={busy ? "disabled" : ""}><Upload size={15} /> Upload Transaction CSV<input disabled={busy} type="file" accept=".csv,text/csv" onChange={uploadCsv} /></label></div>
            <div className="upload-spec"><strong>Merchant CSV route</strong><p>Requires EventTime plus the exact 13-field Risk Check schema; TransactionID is optional. Every valid row is scored by the frozen model before aggregation. Maximum 1 MB / 1,000 rows. isFraud and actual_label are forbidden. Uploaded files are not persisted.</p></div>
          </aside>
          <div className="pulse-results">
            {!result && <div className="panel pulse-empty"><Activity size={33} /><strong>Ready for a real signal</strong><p>Replay all chronological validation scores or upload merchant transactions. No sample chart is fabricated.</p></div>}
            {result && <>
              <section className="pulse-summary"><article><span>ROWS SCORED</span><strong>{result.rows_scored.toLocaleString("en-IN")}</strong><small>{result.invalid_rows.length} invalid</small></article><article><span>TIME WINDOWS</span><strong>{result.windows.length}</strong><small>{result.config.window_seconds / 3600}h each</small></article><article className={result.alerts.length ? "alert" : ""}><span>SPIKE ALERTS</span><strong>{result.alerts.length}</strong><small>configuration-dependent</small></article><article><span>BASELINE</span><strong>{latest?.baseline_state === "READY" ? "Ready" : "Warming"}</strong><small>{latest?.baseline_value === null || latest?.baseline_value === undefined ? "collecting prior windows" : compact(latest.baseline_value, result.config.metric)}</small></article></section>
              <section className="panel pulse-chart"><div className="panel-title"><div><span>{result.data_partition.toUpperCase()}</span><h2>{metricLabels[result.config.metric]} against prior-window baseline</h2></div><span className="source-chip">{uploadName ?? result.source}</span></div><div className="pulse-chart-canvas"><ResponsiveContainer width="100%" height="100%"><LineChart data={chartData} margin={{ top: 10, right: 12, left: -16, bottom: 0 }}><CartesianGrid stroke="#edf1ee" vertical={false} /><XAxis dataKey="name" tick={{ fontSize: 8, fill: "#829087" }} interval="preserveStartEnd" /><YAxis tick={{ fontSize: 8, fill: "#829087" }} /><Tooltip formatter={(value) => compact(Number(value), result.config.metric)} contentStyle={{ fontSize: 9, borderRadius: 9, borderColor: "#dce5df" }} /><Line type="monotone" dataKey="baseline" name="Prior baseline" stroke="#99a59e" strokeDasharray="5 5" dot={false} connectNulls /><Line type="monotone" dataKey="current" name="Current window" stroke="#1e7350" strokeWidth={2} dot={{ r: 2, fill: "#1e7350" }} /><Line type="monotone" dataKey="alertValue" name="Change active" stroke="transparent" dot={{ r: 4, fill: "#b43e37", stroke: "#b43e37" }} /></LineChart></ResponsiveContainer></div><div className="chart-state-key"><span><i className="current" /> Current window</span><span><i className="baseline" /> Prior baseline</span><span><i className="alert" /> Change active</span></div></section>
              <section className="panel pulse-window-table"><div className="panel-title"><div><span>RECENT WINDOWS</span><h2>Baseline, change, and alert state</h2></div><Zap size={18} /></div><div className="pulse-window-row head"><span>Window</span><span>Transactions</span><span>Mean score</span><span>Review / Block</span><span>Monitored</span><span>Baseline</span><span>Change</span><span>State</span></div>{result.windows.slice(-10).reverse().map((window) => <div className={`pulse-window-row ${window.alert_active ? "active" : ""}`} key={window.window_index}><strong>W{window.window_index + 1}</strong><span>{window.transaction_count.toLocaleString("en-IN")}</span><span>{(window.mean_risk_score * 100).toFixed(2)}%</span><span>{window.review_count} / {window.block_count}</span><span>{compact(window.monitored_value, result.config.metric)}</span><span>{window.baseline_value === null ? "Warming up" : compact(window.baseline_value, result.config.metric)}</span><span>{window.percent_deviation === null ? "—" : `${window.percent_deviation >= 0 ? "+" : ""}${(window.percent_deviation * 100).toFixed(1)}%`}</span><span className={window.alert_active ? "spike" : "normal"}>{window.alert_active ? "SPIKE ALERT" : window.baseline_state}</span></div>)}</section>
              {result.alerts.length > 0 && <section className="panel alert-feed"><div className="panel-title"><div><span>ACTIVE CHANGES</span><h2>Windows above configured baseline</h2></div><AlertTriangle size={19} /></div>{result.alerts.slice(-8).reverse().map((alert) => <article key={alert.window_index}><span>W{alert.window_index + 1}</span><div><strong>{alert.label} · {metricLabels[alert.metric]}</strong><small>{compact(alert.current_value, alert.metric)} vs {compact(alert.baseline_value, alert.metric)} baseline</small></div><b>{alert.percent_deviation === null ? alert.detector_score.toFixed(2) : `+${(alert.percent_deviation * 100).toFixed(1)}%`}</b></article>)}</section>}
            </>}
          </div>
        </div>
        <div className="module-limitations"><AlertTriangle size={16} /><div><strong>Interpretation boundary</strong><p>Fraud Pulse reports changes in model-score behavior, not confirmed attacks and not new model performance. Replay results are validation-only. Alert counts can change when the merchant changes the window, baseline, monitored metric, or sensitivity.</p></div></div>
      </section></div>
    </main>
  </div>;
}
