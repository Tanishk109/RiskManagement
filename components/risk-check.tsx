"use client";

import {
  AlertTriangle,
  CheckCircle2,
  Download,
  Eye,
  FileSpreadsheet,
  Info,
  LoaderCircle,
  Play,
  Search,
  ShieldCheck,
  Upload,
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useState } from "react";
import type {
  BatchScoreResponse,
  RiskFeatureName,
  RiskFeaturePayload,
  RiskScoreResponse,
  ValidationRiskCheckCases,
  ValidationRiskGroundTruth,
  ValidationScoringTransaction,
} from "../lib/api-types";

type RiskMode = "single" | "validation" | "batch";
type FeatureValues = Record<RiskFeatureName, string>;

const apiBase = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "");
const featureSchema: RiskFeatureName[] = [
  "TransactionAmt",
  "ProductCD",
  "card4",
  "card6",
  "P_emaildomain",
  "C1",
  "C2",
  "C3",
  "C4",
  "C5",
  "D1",
  "D2",
  "D3",
];
const numericFeatures = new Set<RiskFeatureName>([
  "TransactionAmt",
  "C1",
  "C2",
  "C3",
  "C4",
  "C5",
  "D1",
  "D2",
  "D3",
]);
const advancedFeatures: RiskFeatureName[] = ["C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3"];

const blankFeatures = (): FeatureValues => Object.fromEntries(
  featureSchema.map((feature) => [feature, ""]),
) as FeatureValues;

function path(value: string) {
  return `${apiBase}${value}`;
}

function errorMessage(payload: unknown, fallback: string) {
  if (payload && typeof payload === "object" && "detail" in payload) {
    const detail = (payload as { detail?: unknown }).detail;
    if (typeof detail === "string") return detail;
  }
  return fallback;
}

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(path(url));
  const payload = await response.json().catch(() => null);
  if (!response.ok) throw new Error(errorMessage(payload, `Request failed (${response.status})`));
  return payload as T;
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(path(url), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) throw new Error(errorMessage(payload, `Request failed (${response.status})`));
  return payload as T;
}

function featureInput(
  feature: RiskFeatureName,
  label: string,
  values: FeatureValues,
  setValues: (next: FeatureValues) => void,
  options?: string[],
) {
  const inputId = `risk-${feature}`;
  return (
    <label className="risk-field" htmlFor={inputId} key={feature}>
      <span>{label}</span>
      {options ? (
        <>
          <input
            id={inputId}
            list={`${inputId}-options`}
            maxLength={255}
            value={values[feature]}
            onChange={(event) => setValues({ ...values, [feature]: event.target.value })}
            placeholder="Type or select a value"
          />
          <datalist id={`${inputId}-options`}>
            {options.map((option) => <option value={option} key={option} />)}
          </datalist>
        </>
      ) : (
        <input
          id={inputId}
          type={numericFeatures.has(feature) ? "number" : "text"}
          inputMode={numericFeatures.has(feature) ? "decimal" : undefined}
          step="any"
          min={feature === "TransactionAmt" ? "0" : undefined}
          maxLength={numericFeatures.has(feature) ? undefined : 255}
          required={feature === "TransactionAmt"}
          value={values[feature]}
          onChange={(event) => setValues({ ...values, [feature]: event.target.value })}
          placeholder={feature === "TransactionAmt" ? "e.g. 1499.00" : "Optional; blank uses training-time missing handling"}
        />
      )}
    </label>
  );
}

function DecisionResult({ result }: { result: RiskScoreResponse }) {
  const thresholds = result.threshold_configuration;
  return (
    <article className={`risk-result decision-result-${result.decision.toLowerCase()}`} aria-live="polite">
      <div className="risk-result-heading">
        <div><span>RISK SCORE</span><strong>{(result.fraud_probability * 100).toFixed(2)}%</strong></div>
        <div className={`risk-decision risk-decision-${result.decision.toLowerCase()}`}>
          <small>DECISION</small><strong>{result.decision}</strong>
        </div>
      </div>
      <div className="threshold-track" aria-label="Decision thresholds">
        <span className="threshold-fill" style={{ width: `${Math.min(100, result.fraud_probability * 100)}%` }} />
        <i style={{ left: `${thresholds.review_threshold * 100}%` }}><b>Review</b></i>
        <i style={{ left: `${thresholds.block_threshold * 100}%` }}><b>Block</b></i>
      </div>
      <dl className="risk-result-details">
        <div><dt>Review threshold</dt><dd>{thresholds.review_threshold.toFixed(3)}</dd></div>
        <div><dt>Block threshold</dt><dd>{thresholds.block_threshold.toFixed(3)}</dd></div>
        <div><dt>Model version</dt><dd>{result.model_version}</dd></div>
        <div><dt>Threshold status</dt><dd>Provisional · validation</dd></div>
      </dl>
      <p><Info size={13} /> No explanation is shown because this saved candidate has no verified per-row explanation artifact.</p>
    </article>
  );
}

export default function RiskCheck() {
  const [mode, setMode] = useState<RiskMode>("single");
  const [values, setValues] = useState<FeatureValues>(blankFeatures);
  const [result, setResult] = useState<RiskScoreResponse | null>(null);
  const [scoreError, setScoreError] = useState<string | null>(null);
  const [scoring, setScoring] = useState(false);
  const [cases, setCases] = useState<ValidationRiskCheckCases | null>(null);
  const [transactionId, setTransactionId] = useState("");
  const [loadedTransaction, setLoadedTransaction] = useState<ValidationScoringTransaction | null>(null);
  const [groundTruth, setGroundTruth] = useState<ValidationRiskGroundTruth | null>(null);
  const [loadingTransaction, setLoadingTransaction] = useState(false);
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [batchResult, setBatchResult] = useState<BatchScoreResponse | null>(null);
  const [batchError, setBatchError] = useState<string | null>(null);
  const [batchScoring, setBatchScoring] = useState(false);

  useEffect(() => {
    getJson<ValidationRiskCheckCases>("/api/v1/validation/risk-check-cases")
      .then(setCases)
      .catch(() => setCases(null));
  }, []);

  const formPayload = useMemo(() => {
    const payload = {} as RiskFeaturePayload;
    for (const feature of featureSchema) {
      const value = values[feature].trim();
      if (!value) {
        payload[feature] = null;
      } else if (numericFeatures.has(feature)) {
        payload[feature] = Number(value);
      } else {
        payload[feature] = value;
      }
    }
    return payload;
  }, [values]);

  async function runRiskCheck(event?: FormEvent) {
    event?.preventDefault();
    setScoreError(null);
    setResult(null);
    if (!values.TransactionAmt.trim()) {
      setScoreError("Transaction amount is required.");
      return;
    }
    for (const feature of numericFeatures) {
      const raw = values[feature].trim();
      if (raw && !Number.isFinite(Number(raw))) {
        setScoreError(`${feature} must be a valid number.`);
        return;
      }
    }
    setScoring(true);
    try {
      setResult(await postJson<RiskScoreResponse>("/api/v1/score", { features: formPayload }));
    } catch (caught) {
      setScoreError(caught instanceof Error ? caught.message : "Risk Check could not be completed.");
    } finally {
      setScoring(false);
    }
  }

  async function loadValidationTransaction() {
    if (!transactionId.trim()) {
      setScoreError("Enter or select a validation TransactionID.");
      return;
    }
    setLoadingTransaction(true);
    setScoreError(null);
    setResult(null);
    setGroundTruth(null);
    try {
      const loaded = await getJson<ValidationScoringTransaction>(
        `/api/v1/validation/transactions/${encodeURIComponent(transactionId.trim())}`,
      );
      setLoadedTransaction(loaded);
      setTransactionId(loaded.transaction_id);
      setValues(Object.fromEntries(
        featureSchema.map((feature) => [feature, loaded.features[feature] === null ? "" : String(loaded.features[feature])]),
      ) as FeatureValues);
    } catch (caught) {
      setLoadedTransaction(null);
      setScoreError(caught instanceof Error ? caught.message : "Validation transaction could not be loaded.");
    } finally {
      setLoadingTransaction(false);
    }
  }

  async function revealGroundTruth() {
    if (!loadedTransaction || !result) return;
    setScoreError(null);
    try {
      setGroundTruth(await getJson<ValidationRiskGroundTruth>(
        `/api/v1/validation/transactions/${loadedTransaction.transaction_id}/ground-truth`,
      ));
    } catch (caught) {
      setScoreError(caught instanceof Error ? caught.message : "Ground truth could not be revealed.");
    }
  }

  async function runBatch() {
    if (!batchFile) {
      setBatchError("Choose a CSV file first.");
      return;
    }
    if (batchFile.size > 1_000_000) {
      setBatchError("CSV files are limited to 1 MB and 1,000 data rows.");
      return;
    }
    setBatchScoring(true);
    setBatchError(null);
    setBatchResult(null);
    try {
      const csvContent = await batchFile.text();
      setBatchResult(await postJson<BatchScoreResponse>("/api/v1/score/batch", { csv_content: csvContent }));
    } catch (caught) {
      setBatchError(caught instanceof Error ? caught.message : "Batch scoring could not be completed.");
    } finally {
      setBatchScoring(false);
    }
  }

  function exportResults() {
    if (!batchResult) return;
    const safe = (value: string) => {
      const escaped = /^[=+\-@]/.test(value) ? `'${value}` : value;
      return `"${escaped.replaceAll('"', '""')}"`;
    };
    const rows = batchResult.results.map((item) => [
      String(item.row),
      item.transaction_id,
      item.fraud_probability.toFixed(8),
      item.decision,
      batchResult.model_version,
    ].map(safe).join(","));
    const blob = new Blob([[
      ["row", "transaction_id", "fraud_probability", "decision", "model_version"].join(","),
      ...rows,
    ].join("\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "merchantshield-scored-results.csv";
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <section className="module-page risk-check-page">
      <div className="compact-heading risk-check-heading">
        <div><span className="eyebrow">FROZEN CATBOOST · VALIDATION THRESHOLDS</span><h1>Risk Check</h1><p>Submit one transaction or a temporary batch and convert model probability into a merchant action.</p></div>
        <div className="risk-seal"><ShieldCheck size={16} /><span>Inference only</span><small>Held-out test sealed</small></div>
      </div>

      <div className="dataset-limitation" role="note">
        <Info size={18} />
        <p>The current research model was trained on IEEE-CIS. Several high-value predictors (C1-C5 and D1-D3) are anonymized by the dataset provider. In a real merchant integration these would be replaced by documented first-party behavioral/velocity features.</p>
      </div>

      <div className="risk-mode-tabs" role="tablist" aria-label="Risk Check input mode">
        <button role="tab" aria-selected={mode === "single"} className={mode === "single" ? "active" : ""} onClick={() => setMode("single")}><Play size={14} /> Single Transaction</button>
        <button role="tab" aria-selected={mode === "validation"} className={mode === "validation" ? "active" : ""} onClick={() => setMode("validation")}><Search size={14} /> Load Validation Transaction</button>
        <button role="tab" aria-selected={mode === "batch"} className={mode === "batch" ? "active" : ""} onClick={() => setMode("batch")}><FileSpreadsheet size={14} /> Batch CSV Upload</button>
      </div>

      {mode !== "batch" && (
        <div className="risk-workspace">
          <div className="panel risk-form-panel">
            {mode === "validation" && (
              <div className="validation-loader">
                <div><span>VALIDATION DEMO</span><h2>Load without revealing the label</h2><p>Ground truth stays out of the response until you explicitly reveal it after scoring.</p></div>
                {cases && <select aria-label="Interesting validation case" value="" onChange={(event) => setTransactionId(event.target.value)}><option value="">Select an interesting case…</option>{cases.cases.map((item) => <option key={item.case_type} value={item.transaction_id}>{item.label} · ID {item.transaction_id}</option>)}</select>}
                <div className="transaction-id-row"><label><span>TransactionID</span><input value={transactionId} onChange={(event) => setTransactionId(event.target.value)} placeholder="Enter validation TransactionID" /></label><button type="button" onClick={loadValidationTransaction} disabled={loadingTransaction}>{loadingTransaction ? <LoaderCircle className="spin" size={14} /> : <Search size={14} />} Load</button></div>
                {loadedTransaction && <div className="loaded-transaction"><CheckCircle2 size={14} /><span>Loaded validation transaction {loadedTransaction.transaction_id}</span><small>Ground truth hidden</small></div>}
              </div>
            )}

            <form onSubmit={runRiskCheck} className="risk-form">
              <div className="form-section-heading"><div><span>DOCUMENTED INPUTS</span><h2>Transaction details</h2></div><small>Exact 13-field model schema</small></div>
              <div className="risk-field-grid">
                {featureInput("TransactionAmt", "Transaction amount", values, setValues)}
                {featureInput("ProductCD", "Product code", values, setValues, ["W", "C", "R", "H", "S"])}
                {featureInput("card4", "Card network", values, setValues, ["visa", "mastercard", "american express", "discover"])}
                {featureInput("card6", "Card type", values, setValues, ["debit", "credit", "charge card", "debit or credit"])}
                {featureInput("P_emaildomain", "Purchaser email domain", values, setValues, ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com"])}
              </div>
              <details className="advanced-fields">
                <summary><span>Advanced IEEE-CIS Fields</span><small>C1-C5 and D1-D3 · anonymized competition features</small></summary>
                <p>These values retain only their source dataset names. MerchantShield does not assign or invent semantic meanings.</p>
                <div className="risk-field-grid advanced-grid">
                  {advancedFeatures.map((feature) => featureInput(feature, `${feature} — anonymized competition feature`, values, setValues))}
                </div>
              </details>
              {scoreError && <div className="risk-inline-error" role="alert"><AlertTriangle size={14} /> {scoreError}</div>}
              <button className="run-risk-button" disabled={scoring} type="submit">{scoring ? <LoaderCircle className="spin" size={15} /> : <ShieldCheck size={15} />} Run Risk Check</button>
            </form>
          </div>
          <div className="risk-result-column">
            {result ? <DecisionResult result={result} /> : <article className="risk-result-empty"><ShieldCheck size={28} /><strong>Ready for a transaction</strong><p>Your result will show the fraud probability, action, provisional thresholds, and saved model version.</p></article>}
            {mode === "validation" && result && loadedTransaction && (
              <article className="ground-truth-card">
                {!groundTruth ? <><Eye size={19} /><div><strong>Ground truth remains hidden</strong><p>Reveal only after you have seen the model decision.</p></div><button onClick={revealGroundTruth}><Eye size={14} /> Reveal Ground Truth</button></> : <><CheckCircle2 size={19} /><div><span>ACTUAL VALIDATION LABEL</span><strong>{groundTruth.ground_truth}</strong><p>Explicitly revealed. This label was not a scoring input.</p></div></>}
              </article>
            )}
            <p className="decision-copy">MerchantShield converts model risk probability into a business action using the currently selected validation operating thresholds.</p>
          </div>
        </div>
      )}

      {mode === "batch" && (
        <div className="batch-workspace">
          <article className="panel batch-upload-card">
            <div className="panel-title"><div><span>TEMPORARY CSV INGESTION</span><h2>Score up to 1,000 transactions</h2></div><Upload size={19} /></div>
            <p>Required columns: {featureSchema.join(", ")}. TransactionID is optional. Labels such as isFraud are forbidden and never used as scoring inputs.</p>
            <label className="file-drop"><FileSpreadsheet size={26} /><strong>{batchFile?.name ?? "Choose an exact-schema CSV"}</strong><span>Maximum 1 MB · 1,000 data rows · file is not stored permanently</span><input type="file" accept=".csv,text/csv" onChange={(event) => { setBatchFile(event.target.files?.[0] ?? null); setBatchResult(null); setBatchError(null); }} /></label>
            {batchError && <div className="risk-inline-error" role="alert"><AlertTriangle size={14} /> {batchError}</div>}
            <button className="run-risk-button" disabled={batchScoring || !batchFile} onClick={runBatch}>{batchScoring ? <LoaderCircle className="spin" size={15} /> : <ShieldCheck size={15} />} Score CSV</button>
          </article>

          {batchResult && (
            <section className="batch-results">
              <div className="batch-summary">
                <div><span>Rows processed</span><strong>{batchResult.summary.rows_processed}</strong></div>
                <div className="approve"><span>Approved</span><strong>{batchResult.summary.approved}</strong></div>
                <div className="review"><span>Reviewed</span><strong>{batchResult.summary.reviewed}</strong></div>
                <div className="block"><span>Blocked</span><strong>{batchResult.summary.blocked}</strong></div>
                <div className="invalid"><span>Invalid rows</span><strong>{batchResult.summary.invalid_rows}</strong></div>
              </div>
              <div className="batch-table-heading"><div><span>SCORED RESULTS</span><h2>{batchResult.model_version}</h2></div><button onClick={exportResults}><Download size={14} /> Export scored CSV</button></div>
              <div className="batch-table">
                <div className="batch-row batch-head"><span>Row / Transaction ID</span><span>Risk probability</span><span>Decision</span></div>
                {batchResult.results.map((item) => <div className="batch-row" key={`${item.row}-${item.transaction_id}`}><strong>{item.row} / {item.transaction_id}</strong><span>{(item.fraud_probability * 100).toFixed(2)}%</span><span className={`risk-decision risk-decision-${item.decision.toLowerCase()}`}>{item.decision}</span></div>)}
              </div>
              {batchResult.invalid_rows.length > 0 && <div className="invalid-row-list"><strong>Invalid rows</strong>{batchResult.invalid_rows.map((item) => <p key={item.row}>Row {item.row}{item.transaction_id ? ` · ${item.transaction_id}` : ""}: {item.errors.join("; ")}</p>)}</div>}
            </section>
          )}
        </div>
      )}
    </section>
  );
}
