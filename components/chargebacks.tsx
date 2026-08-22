"use client";
/* eslint-disable @next/next/no-html-link-for-pages */

import {
  AlertTriangle,
  Activity,
  ArrowRight,
  BarChart3,
  Check,
  Download,
  FileCheck2,
  FileText,
  Gauge,
  Info,
  ListChecks,
  LockKeyhole,
  Menu,
  Network,
  Paperclip,
  Save,
  ShieldCheck,
  SlidersHorizontal,
  TableProperties,
  Upload,
  X,
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useState } from "react";

const apiBase = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "");
const apiPath = (path: string) => `${apiBase}${path}`;

type EvidenceCategory = "invoice" | "proof_of_delivery" | "tracking" | "customer_communication" | "refund_evidence" | "merchant_policy" | "other";
type DisputeReason = "item_not_received" | "duplicate" | "refund_not_received" | "cancelled_recurring" | "not_as_described" | "other";
type Evidence = { id: number; category: EvidenceCategory; original_filename: string; content_type: string; size_bytes: number };
type Draft = { draft_text: string; generation_method: string; evidence_count: number; missing_categories: EvidenceCategory[]; human_approved: boolean };
type Completeness = { present: EvidenceCategory[]; expected: EvidenceCategory[]; missing: EvidenceCategory[]; present_count: number; expected_count: number; ratio: number; checklist_basis: string };
type ChargebackCase = {
  id: number; dispute_id: string; transaction_id: string; amount: number; currency: string; reason: DisputeReason;
  deadline: string; customer_information: Record<string, string>; order_information: Record<string, string>;
  delivery_information: Record<string, string>; merchant_notes: string | null; status: string;
  evidence: Evidence[]; draft: Draft | null; completeness: Completeness;
};
type ModuleStatus = { data_source: string; evaluation_status: "Not evaluated yet"; checklist_basis: string; limitations: string[]; maximum_file_size_bytes: number; automatic_submission: false };

const reasons: Array<{ value: DisputeReason; label: string }> = [
  { value: "item_not_received", label: "Item not received" },
  { value: "duplicate", label: "Duplicate" },
  { value: "refund_not_received", label: "Refund not received" },
  { value: "cancelled_recurring", label: "Cancelled recurring" },
  { value: "not_as_described", label: "Not as described" },
  { value: "other", label: "Other" },
];
const evidenceCategories: Array<{ value: EvidenceCategory; label: string }> = [
  { value: "invoice", label: "Invoice" }, { value: "proof_of_delivery", label: "Proof of delivery" },
  { value: "tracking", label: "Tracking" }, { value: "customer_communication", label: "Customer communication" },
  { value: "refund_evidence", label: "Refund evidence" }, { value: "merchant_policy", label: "Merchant policy" },
  { value: "other", label: "Other" },
];

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiPath(path), init);
  if (!response.ok) {
    const payload = await response.json().catch(() => null) as { detail?: string } | null;
    throw new Error(payload?.detail ?? `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

function fileToBase64(file: File): Promise<string> {
  return file.arrayBuffer().then((buffer) => {
    const bytes = new Uint8Array(buffer);
    let binary = "";
    for (let offset = 0; offset < bytes.length; offset += 32768) {
      binary += String.fromCharCode(...bytes.subarray(offset, offset + 32768));
    }
    return btoa(binary);
  });
}

function pretty(value: string) { return value.replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase()); }

export default function Chargebacks() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [status, setStatus] = useState<ModuleStatus | null>(null);
  const [cases, setCases] = useState<ChargebackCase[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [draftText, setDraftText] = useState("");
  const [evidenceCategory, setEvidenceCategory] = useState<EvidenceCategory>("invoice");
  const [evidenceFile, setEvidenceFile] = useState<File | null>(null);
  const [form, setForm] = useState({
    dispute_id: "", transaction_id: "", amount: "", currency: "INR", reason: "item_not_received" as DisputeReason,
    deadline: "", customer_name: "", customer_email: "", order_id: "", order_description: "",
    carrier: "", tracking_reference: "", merchant_notes: "",
  });
  const selected = useMemo(() => cases.find((item) => item.id === selectedId) ?? null, [cases, selectedId]);

  function updateCase(next: ChargebackCase) {
    setCases((current) => current.some((item) => item.id === next.id) ? current.map((item) => item.id === next.id ? next : item) : [next, ...current]);
    setSelectedId(next.id);
    setDraftText(next.draft?.draft_text ?? "");
  }

  useEffect(() => {
    const controller = new AbortController();
    api<ModuleStatus>("/api/v1/chargebacks/status", { signal: controller.signal })
      .then(setStatus)
      .catch((caught: unknown) => {
        if (caught instanceof DOMException && caught.name === "AbortError") return;
        setError("Chargeback module status could not be loaded from the API.");
      });
    api<ChargebackCase[]>("/api/v1/chargebacks/cases", { signal: controller.signal }).then((caseList) => {
      setCases(caseList);
      if (caseList.length) { setSelectedId(caseList[0].id); setDraftText(caseList[0].draft?.draft_text ?? ""); }
    }).catch((caught: unknown) => {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      setError("Operational case storage is unavailable. Check PostgreSQL and run the latest Alembic migration.");
    });
    return () => controller.abort();
  }, []);

  async function createCase(event: FormEvent) {
    event.preventDefault(); setBusy(true); setError(null);
    try {
      const result = await api<ChargebackCase>("/api/v1/chargebacks/cases", {
        method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({
          dispute_id: form.dispute_id, transaction_id: form.transaction_id, amount: Number(form.amount), currency: form.currency,
          reason: form.reason, deadline: form.deadline,
          customer_information: { name: form.customer_name, email: form.customer_email },
          order_information: { order_id: form.order_id, description: form.order_description },
          delivery_information: { carrier: form.carrier, tracking_reference: form.tracking_reference },
          merchant_notes: form.merchant_notes || null,
        }),
      });
      updateCase(result);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Case could not be created"); }
    finally { setBusy(false); }
  }

  async function uploadEvidence(event: FormEvent) {
    event.preventDefault();
    if (!selected || !evidenceFile) return;
    setBusy(true); setError(null);
    try {
      const result = await api<ChargebackCase>(`/api/v1/chargebacks/cases/${selected.id}/evidence`, {
        method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({
          category: evidenceCategory, filename: evidenceFile.name, content_type: evidenceFile.type,
          base64_content: await fileToBase64(evidenceFile),
        }),
      });
      updateCase(result); setEvidenceFile(null);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Evidence could not be uploaded"); }
    finally { setBusy(false); }
  }

  async function generateDraft() {
    if (!selected) return; setBusy(true); setError(null);
    try {
      const result = await api<Draft>(`/api/v1/chargebacks/cases/${selected.id}/generate-draft`, { method: "POST" });
      setDraftText(result.draft_text); updateCase({ ...selected, draft: result, status: "READY_FOR_HUMAN_REVIEW" });
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Draft could not be generated"); }
    finally { setBusy(false); }
  }

  async function saveDraft(humanApproved: boolean) {
    if (!selected) return; setBusy(true); setError(null);
    try {
      const result = await api<Draft>(`/api/v1/chargebacks/cases/${selected.id}/draft`, {
        method: "PUT", headers: { "content-type": "application/json" },
        body: JSON.stringify({ draft_text: draftText, human_approved: humanApproved }),
      });
      updateCase({ ...selected, draft: result, status: humanApproved ? "APPROVED_FOR_EXPORT" : "READY_FOR_HUMAN_REVIEW" });
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Draft could not be saved"); }
    finally { setBusy(false); }
  }

  async function exportDraft() {
    if (!selected) return;
    const response = await fetch(apiPath(`/api/v1/chargebacks/cases/${selected.id}/export`));
    if (!response.ok) { setError("Human approval is required before draft export"); return; }
    const url = URL.createObjectURL(await response.blob());
    const anchor = document.createElement("a"); anchor.href = url; anchor.download = `${selected.dispute_id}-draft.txt`; anchor.click();
    URL.revokeObjectURL(url);
  }

  return <div className="app-shell">
    <aside className={`sidebar ${mobileOpen ? "open" : ""}`}>
      <div className="brand-row"><div className="brand-mark"><ShieldCheck size={21} /></div><div><strong>MerchantShield</strong><span>Merchant loss prevention</span></div><button className="icon-button close-nav" onClick={() => setMobileOpen(false)} aria-label="Close navigation"><X size={18} /></button></div>
      <div className="candidate-badge"><span className="candidate-dot" /><div><strong>Loss prevention suite</strong><small>Evidence-backed · defense only</small></div></div>
      <nav aria-label="MerchantShield modules">
        <span className="nav-label">Workspace</span><a className="suite-nav-link" href="/"><BarChart3 size={17} /><span>Overview</span></a>
        <span className="nav-label nav-label-spaced">Fraud Risk</span>
        <a className="suite-nav-link" href="/"><Gauge size={17} /><span>Risk Check</span></a><a className="suite-nav-link" href="/"><TableProperties size={17} /><span>Transactions</span></a><a className="suite-nav-link" href="/"><ListChecks size={17} /><span>Review Queue</span></a><a className="suite-nav-link" href="/"><SlidersHorizontal size={17} /><span>Cost Lab</span></a>
        <span className="nav-label nav-label-spaced">Loss prevention</span><a className="suite-nav-link active" href="/chargebacks"><FileText size={17} /><span>Chargebacks</span></a><a className="suite-nav-link" href="/fraud-pulse"><Activity size={17} /><span>Fraud Pulse</span></a><a className="suite-nav-link" href="/abuse-rings"><Network size={17} /><span>Abuse Rings</span></a>
      </nav>
      <div className="sidebar-note"><LockKeyhole size={16} /><div><strong>No automatic submission</strong><span>Every draft requires merchant review and explicit approval before export.</span></div></div>
    </aside>
    {mobileOpen && <button className="nav-scrim" onClick={() => setMobileOpen(false)} aria-label="Close navigation" />}
    <main className="main-area">
      <header className="topbar"><button className="icon-button menu-button" onClick={() => setMobileOpen(true)} aria-label="Open navigation"><Menu size={19} /></button><div className="breadcrumb"><span>MerchantShield</span><ArrowRight size={12} /><strong>Chargebacks</strong></div><div className="top-status neutral-status"><span />{status?.evaluation_status.toUpperCase() ?? "LOADING MODULE"}</div></header>
      <div className="page-content chargeback-page">
        <section className="module-page">
          <div className="compact-heading"><div><span className="eyebrow">CHARGEBACK EVIDENCE RESPONDER</span><h1>Build an evidence-ready response.</h1><p>Organize merchant-supplied dispute context, see exactly what is missing, and prepare a human-reviewed draft—without claiming a win probability.</p></div><div className="validation-pill"><FileCheck2 size={14} /> HUMAN APPROVAL REQUIRED</div></div>
          <div className="module-disclosure"><Info size={17} /><div><strong>Data source: {status?.data_source ?? "Merchant-entered dispute data and uploaded evidence"}</strong><p>{status?.checklist_basis ?? "Internal merchant preparation checklist; not payment-network rules."} Evaluation: <b>Not evaluated yet.</b></p></div></div>
          {error && <div className="error-state"><AlertTriangle size={17} /><div><strong>Workspace notice</strong><span>{error}</span></div><button onClick={() => setError(null)}>Dismiss</button></div>}
          <div className="chargeback-workspace">
            <section className="panel case-create-panel"><div className="panel-title"><div><span>NEW CASE</span><h2>Dispute intake</h2></div><FileText size={19} /></div>
              <form className="case-form" onSubmit={createCase}>
                <div className="field-grid two"><label><span>Dispute ID</span><input required maxLength={100} value={form.dispute_id} onChange={(e) => setForm({ ...form, dispute_id: e.target.value })} placeholder="DSP-2026-001" /></label><label><span>Transaction ID</span><input required maxLength={100} value={form.transaction_id} onChange={(e) => setForm({ ...form, transaction_id: e.target.value })} placeholder="TXN-10482" /></label></div>
                <div className="field-grid three"><label><span>Amount</span><input required min="0.01" step="0.01" type="number" value={form.amount} onChange={(e) => setForm({ ...form, amount: e.target.value })} /></label><label><span>Currency</span><input required pattern="[A-Za-z]{3}" maxLength={3} value={form.currency} onChange={(e) => setForm({ ...form, currency: e.target.value.toUpperCase() })} /></label><label><span>Deadline</span><input required type="date" value={form.deadline} onChange={(e) => setForm({ ...form, deadline: e.target.value })} /></label></div>
                <label><span>Merchant-selected dispute reason</span><select value={form.reason} onChange={(e) => setForm({ ...form, reason: e.target.value as DisputeReason })}>{reasons.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
                <details><summary>Customer, order & delivery information</summary><div className="details-fields"><div className="field-grid two"><label><span>Customer name</span><input value={form.customer_name} onChange={(e) => setForm({ ...form, customer_name: e.target.value })} /></label><label><span>Customer email</span><input type="email" value={form.customer_email} onChange={(e) => setForm({ ...form, customer_email: e.target.value })} /></label></div><div className="field-grid two"><label><span>Order ID</span><input value={form.order_id} onChange={(e) => setForm({ ...form, order_id: e.target.value })} /></label><label><span>Order description</span><input value={form.order_description} onChange={(e) => setForm({ ...form, order_description: e.target.value })} /></label></div><div className="field-grid two"><label><span>Carrier</span><input value={form.carrier} onChange={(e) => setForm({ ...form, carrier: e.target.value })} /></label><label><span>Tracking reference</span><input value={form.tracking_reference} onChange={(e) => setForm({ ...form, tracking_reference: e.target.value })} /></label></div></div></details>
                <label><span>Merchant notes</span><textarea maxLength={5000} value={form.merchant_notes} onChange={(e) => setForm({ ...form, merchant_notes: e.target.value })} placeholder="Record only known facts and relevant chronology…" /></label>
                <button disabled={busy} className="run-risk-button" type="submit"><FileCheck2 size={15} /> Create evidence case</button>
              </form>
            </section>
            <section className="panel cases-panel"><div className="panel-title"><div><span>CASE WORKSPACE</span><h2>{selected ? selected.dispute_id : "Select a case"}</h2></div><Paperclip size={19} /></div>
              {cases.length > 0 && <div className="case-selector">{cases.map((item) => <button className={item.id === selectedId ? "active" : ""} key={item.id} onClick={() => { setSelectedId(item.id); setDraftText(item.draft?.draft_text ?? ""); }}><span>{item.dispute_id}</span><small>{item.currency} {item.amount.toLocaleString("en-IN")} · due {item.deadline}</small></button>)}</div>}
              {!selected && <div className="empty-case"><FileText size={28} /><strong>No case selected</strong><span>Create a case using real merchant context to begin.</span></div>}
              {selected && <>
                <div className="completeness-card"><div><span>EVIDENCE COMPLETENESS</span><strong>{selected.completeness.present_count} / {selected.completeness.expected_count}</strong><small>expected categories present · not a win probability</small></div><div className="completeness-track"><i style={{ width: `${selected.completeness.ratio * 100}%` }} /></div><div className="evidence-chips">{selected.completeness.expected.map((category) => <span className={selected.completeness.missing.includes(category) ? "missing" : "present"} key={category}>{selected.completeness.missing.includes(category) ? <X size={11} /> : <Check size={11} />}{pretty(category)}</span>)}</div></div>
                <form className="upload-row" onSubmit={uploadEvidence}><select value={evidenceCategory} onChange={(e) => setEvidenceCategory(e.target.value as EvidenceCategory)}>{evidenceCategories.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select><input aria-label="Evidence file" required accept=".pdf,.png,.jpg,.jpeg,application/pdf,image/png,image/jpeg" type="file" onChange={(e) => setEvidenceFile(e.target.files?.[0] ?? null)} /><button disabled={busy || !evidenceFile}><Upload size={14} /> Upload</button></form><p className="upload-note">PDF, PNG or JPG · maximum 10 MB each · bytes are not stored in PostgreSQL.</p>
                <div className="evidence-list">{selected.evidence.map((item) => <div key={item.id}><FileCheck2 size={14} /><span><strong>{item.original_filename}</strong><small>{pretty(item.category)} · {(item.size_bytes / 1024).toFixed(1)} KB</small></span></div>)}</div>
                <button className="generate-draft" disabled={busy} onClick={generateDraft}><FileText size={15} /> Generate evidence-grounded draft</button>
                {selected.draft && <div className="draft-editor"><div className="draft-label"><span>ASSISTANT DRAFT · NOT SUBMITTED</span><small>{selected.draft.human_approved ? "Human approved" : "Human review required"}</small></div><textarea value={draftText} onChange={(e) => setDraftText(e.target.value)} /><div className="draft-actions"><button disabled={busy} onClick={() => saveDraft(false)}><Save size={14} /> Save draft</button><button className="approve-draft" disabled={busy} onClick={() => saveDraft(true)}><Check size={14} /> Approve for export</button><button disabled={!selected.draft.human_approved} onClick={exportDraft}><Download size={14} /> Export .txt</button></div></div>}
              </>}
            </section>
          </div>
          <div className="module-limitations"><AlertTriangle size={16} /><div><strong>Known limitations</strong><p>MerchantShield does not interpret document contents, reproduce network-specific rules, predict dispute outcomes, or submit evidence automatically. Filenames and merchant-entered context must be verified by the reviewing human.</p></div></div>
        </section>
      </div>
    </main>
  </div>;
}
