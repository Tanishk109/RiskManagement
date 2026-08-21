"use client";

import {
  Activity,
  AlertOctagon,
  AlertTriangle,
  ArrowRight,
  ArrowUpRight,
  BadgeCheck,
  Bell,
  Bot,
  BrainCircuit,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleDollarSign,
  Clock3,
  Database,
  Download,
  Fingerprint,
  Gauge,
  GitBranch,
  IndianRupee,
  Info,
  LayoutDashboard,
  ListFilter,
  Menu,
  MessageSquareText,
  MoreHorizontal,
  Network,
  Play,
  Radar,
  RefreshCw,
  RotateCcw,
  Search,
  Settings2,
  ShieldAlert,
  ShieldCheck,
  ShieldQuestion,
  SlidersHorizontal,
  Sparkles,
  TimerReset,
  UserCheck,
  Users,
  Waypoints,
  X,
  Zap,
} from "lucide-react";
import { useMemo, useState } from "react";
import { HELD_OUT_RESULTS, metricsFor, scoreTransaction, type Decision, type TransactionInput } from "../lib/risk-engine";

type Module = "command" | "transactions" | "review" | "rings" | "pulse" | "rules" | "model" | "audit";
type Transaction = {
  id: string;
  customer: string;
  initials: string;
  city: string;
  amount: number;
  time: string;
  channel: string;
  device: string;
  ip: string;
  modelVersion: string;
  input: TransactionInput;
  analystDecision?: "APPROVED" | "BLOCKED" | "MORE INFO";
};

type AuditEvent = { time: string; actor: string; action: string; detail: string; tone: "safe" | "review" | "critical" | "neutral" };

const starterTransactions: Transaction[] = [
  { id: "TX-8295", customer: "Rohan Mehta", initials: "RM", city: "Pune, MH", amount: 28900, time: "Now", channel: "UPI", device: "Android · new", ip: "49.36.18.201", modelVersion: "Fraud-XGB-v3.2", input: { amount: 28900, customerAgeDays: 2, newDevice: true, transactionsLast5Min: 7, accountsSharingDevice: 11, historicalChargebacks: 1, failedAttemptsLastHour: 5, amountVsBaseline: 4.8 } },
  { id: "TX-8294", customer: "Sana Qureshi", initials: "SQ", city: "Hyderabad, TS", amount: 649, time: "18 sec", channel: "Card", device: "Android · trusted", ip: "103.92.44.17", modelVersion: "Fraud-XGB-v3.2", input: { amount: 649, customerAgeDays: 812, newDevice: false, transactionsLast5Min: 1, accountsSharingDevice: 1, historicalChargebacks: 0, failedAttemptsLastHour: 0, amountVsBaseline: 0.8 } },
  { id: "TX-8293", customer: "Ishaan Sethi", initials: "IS", city: "New Delhi, DL", amount: 18400, time: "34 sec", channel: "Card", device: "Windows · new", ip: "122.176.83.62", modelVersion: "Fraud-XGB-v3.2", input: { amount: 18400, customerAgeDays: 1, newDevice: true, transactionsLast5Min: 6, accountsSharingDevice: 8, historicalChargebacks: 0, failedAttemptsLastHour: 4, amountVsBaseline: 5.2 } },
  { id: "TX-8292", customer: "Meera Iyer", initials: "MI", city: "Chennai, TN", amount: 14800, time: "51 sec", channel: "Wallet", device: "iOS · known", ip: "157.49.103.71", modelVersion: "Fraud-XGB-v3.2", input: { amount: 14800, customerAgeDays: 380, newDevice: false, transactionsLast5Min: 3, accountsSharingDevice: 2, historicalChargebacks: 0, failedAttemptsLastHour: 2, amountVsBaseline: 2.6 } },
  { id: "TX-8291", customer: "Ananya Kulkarni", initials: "AK", city: "Bengaluru, KA", amount: 1249, time: "1 min", channel: "UPI", device: "iOS · trusted", ip: "106.51.77.91", modelVersion: "Fraud-XGB-v3.2", input: { amount: 1249, customerAgeDays: 641, newDevice: false, transactionsLast5Min: 1, accountsSharingDevice: 1, historicalChargebacks: 0, failedAttemptsLastHour: 0, amountVsBaseline: 1.1 } },
  { id: "TX-8288", customer: "Kabir Arora", initials: "KA", city: "Gurugram, HR", amount: 32600, time: "3 min", channel: "Card", device: "macOS · known", ip: "14.98.202.38", modelVersion: "Fraud-XGB-v3.2", input: { amount: 32600, customerAgeDays: 91, newDevice: false, transactionsLast5Min: 4, accountsSharingDevice: 3, historicalChargebacks: 1, failedAttemptsLastHour: 1, amountVsBaseline: 3.7 } },
  { id: "TX-8284", customer: "Priya Nair", initials: "PN", city: "Kochi, KL", amount: 4799, time: "5 min", channel: "UPI", device: "Android · trusted", ip: "117.213.19.42", modelVersion: "Fraud-XGB-v3.2", input: { amount: 4799, customerAgeDays: 1202, newDevice: false, transactionsLast5Min: 1, accountsSharingDevice: 1, historicalChargebacks: 0, failedAttemptsLastHour: 0, amountVsBaseline: 1.2 } },
];

const initialAudit: AuditEvent[] = [
  { time: "15:42:18", actor: "Risk orchestrator", action: "BLOCK recommendation", detail: "TX-8295 · score 92 · device velocity cluster", tone: "critical" },
  { time: "15:41:52", actor: "Fraud-XGB-v3.2", action: "Model scored", detail: "TX-8294 · score 8 · 24ms", tone: "safe" },
  { time: "15:41:34", actor: "System", action: "Pulse alert opened", detail: "New-device attempt rate exceeded rolling z-score 3.1", tone: "review" },
  { time: "15:39:06", actor: "Aditi Rao", action: "Review approved", detail: "TX-8280 · trusted customer history", tone: "safe" },
  { time: "15:37:40", actor: "Rule Engine", action: "Rule executed", detail: "Device velocity cluster · v12", tone: "neutral" },
];

const volumeBars = [38, 45, 42, 55, 48, 61, 54, 69, 64, 58, 73, 67, 81, 72, 79, 68, 86, 74, 91, 82, 96, 84, 101, 92];

function money(value: number) {
  return new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(value);
}

function downloadCsv(name: string, rows: (string | number)[][]) {
  const csv = rows.map((row) => row.map((cell) => `"${String(cell).replaceAll('"', '""')}"`).join(",")).join("\n");
  const href = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const anchor = document.createElement("a");
  anchor.href = href;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(href);
}

function decisionClass(decision: Decision | string) {
  return decision.toLowerCase().replace(" ", "-");
}

function DecisionBadge({ decision }: { decision: Decision }) {
  return <span className={`decision-badge ${decisionClass(decision)}`}><span />{decision}</span>;
}

function MetricCard({ icon: Icon, label, value, detail, trend, tone }: { icon: typeof ShieldCheck; label: string; value: string; detail: string; trend: string; tone: string }) {
  return <article className="metric-card"><div className={`metric-icon ${tone}`}><Icon size={18} /></div><div className="metric-label">{label}</div><div className="metric-value">{value}</div><div className="metric-foot"><span className="positive">{trend}</span>{detail}</div></article>;
}

export default function Home() {
  const [active, setActive] = useState<Module>("command");
  const [transactions, setTransactions] = useState(starterTransactions);
  const [selectedId, setSelectedId] = useState(starterTransactions[0].id);
  const [search, setSearch] = useState("");
  const [decisionFilter, setDecisionFilter] = useState<"ALL" | Decision>("ALL");
  const [caseTab, setCaseTab] = useState<"explain" | "network" | "timeline">("explain");
  const [threshold, setThreshold] = useState(70);
  const [fraudLoss, setFraudLoss] = useState(4200);
  const [legitLoss, setLegitLoss] = useState(850);
  const [reviewCost, setReviewCost] = useState(120);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [simulatorOpen, setSimulatorOpen] = useState(false);
  const [toast, setToast] = useState("");
  const [audit, setAudit] = useState(initialAudit);
  const [pulseActive, setPulseActive] = useState(true);
  const [selectedNode, setSelectedNode] = useState("Device D-91");
  const [reviewReasons, setReviewReasons] = useState(["Shared suspicious device"]);
  const [assistantAnswer, setAssistantAnswer] = useState("Fraud attempts rose 42% between 14:00–15:00. New-device activity and Cluster AR-129 explain 71% of the increase.");
  const [rule, setRule] = useState({ deviceCount: 5, minutes: 10, score: 65, action: "REVIEW" as Decision });
  const [ruleDeployed, setRuleDeployed] = useState(false);
  const [simRunning, setSimRunning] = useState(false);
  const [simResult, setSimResult] = useState<ReturnType<typeof scoreTransaction> | null>(null);
  const [sim, setSim] = useState<TransactionInput>({ amount: 28000, customerAgeDays: 2, newDevice: true, transactionsLast5Min: 6, accountsSharingDevice: 9, historicalChargebacks: 1, failedAttemptsLastHour: 5, amountVsBaseline: 4.2 });

  const enriched = useMemo(() => transactions.map((tx) => ({ ...tx, result: scoreTransaction(tx.input) })), [transactions]);
  const selected = enriched.find((tx) => tx.id === selectedId) ?? enriched[0];
  const filtered = enriched.filter((tx) => {
    const query = search.toLowerCase();
    return (decisionFilter === "ALL" || tx.result.decision === decisionFilter) && (!query || `${tx.id} ${tx.customer} ${tx.city} ${tx.ip}`.toLowerCase().includes(query));
  });
  const metrics = metricsFor(threshold);
  const manualReviews = Math.round((metrics.fp + metrics.tp) * 0.31);
  const fpCost = metrics.fp * legitLoss;
  const fnCost = metrics.fn * fraudLoss;
  const manualCost = manualReviews * reviewCost;
  const totalCost = fpCost + fnCost + manualCost;
  const noModelCost = (metrics.tp + metrics.fn) * fraudLoss;
  const netPrevented = noModelCost - totalCost;
  const ruleAffected = Math.max(45, Math.round(520 - rule.deviceCount * 27 + (75 - rule.score) * 8));
  const ruleFraudCaught = Math.round(ruleAffected * (0.58 + (70 - rule.score) * .004));
  const ruleLegitAffected = Math.max(3, Math.round(ruleAffected * (.04 + (70 - rule.score) * .002)));

  const nav: { id: Module; label: string; icon: typeof LayoutDashboard; count?: number }[] = [
    { id: "command", label: "Command center", icon: LayoutDashboard },
    { id: "transactions", label: "Live transactions", icon: Activity },
    { id: "review", label: "Review center", icon: ShieldQuestion, count: 3 },
    { id: "rings", label: "Abuse ring sentinel", icon: Network },
    { id: "pulse", label: "Fraud pulse", icon: Radar, count: pulseActive ? 1 : undefined },
    { id: "rules", label: "Rule lab", icon: GitBranch },
    { id: "model", label: "Model & cost", icon: Gauge },
    { id: "audit", label: "Audit logs", icon: Database },
  ];

  function announce(message: string) {
    setToast(message);
    window.setTimeout(() => setToast(""), 2600);
  }

  function addAudit(action: string, detail: string, tone: AuditEvent["tone"] = "neutral") {
    const now = new Date().toLocaleTimeString("en-IN", { hour12: false });
    setAudit((current) => [{ time: now, actor: "Aditi Rao", action, detail, tone }, ...current]);
  }

  function decide(decision: Transaction["analystDecision"]) {
    setTransactions((current) => current.map((tx) => tx.id === selected.id ? { ...tx, analystDecision: decision } : tx));
    const detail = `${selected.id} · ${reviewReasons.join(", ") || "analyst judgment"}`;
    addAudit(`Review ${decision?.toLowerCase()}`, detail, decision === "BLOCKED" ? "critical" : decision === "APPROVED" ? "safe" : "review");
    announce(`${selected.id} marked ${decision}`);
  }

  function addLiveTransaction(kind: "normal" | "suspicious") {
    const number = 8300 + transactions.length;
    const tx: Transaction = kind === "normal"
      ? { id: `TX-${number}`, customer: "Demo customer", initials: "DC", city: "Mumbai, MH", amount: 1899, time: "Now", channel: "UPI", device: "Android · trusted", ip: "103.21.80.14", modelVersion: "Fraud-XGB-v3.2", input: { amount: 1899, customerAgeDays: 620, newDevice: false, transactionsLast5Min: 1, accountsSharingDevice: 1, historicalChargebacks: 0, failedAttemptsLastHour: 0, amountVsBaseline: 1.1 } }
      : { id: `TX-${number}`, customer: "Demo scenario", initials: "DS", city: "Noida, UP", amount: 34900, time: "Now", channel: "Card", device: "Android · new", ip: "49.205.118.72", modelVersion: "Fraud-XGB-v3.2", input: { amount: 34900, customerAgeDays: 1, newDevice: true, transactionsLast5Min: 8, accountsSharingDevice: 13, historicalChargebacks: 2, failedAttemptsLastHour: 6, amountVsBaseline: 5.8 } };
    const result = scoreTransaction(tx.input);
    setTransactions((current) => [tx, ...current.map((item) => ({ ...item, time: item.time === "Now" ? "12 sec" : item.time }))]);
    setSelectedId(tx.id);
    setActive("transactions");
    addAudit("Simulator event ingested", `${tx.id} · ${result.decision} · score ${result.score}`, result.decision === "BLOCK" ? "critical" : "safe");
    announce(`${tx.id} scored ${result.score} · ${result.decision}`);
  }

  async function runSimulator() {
    setSimRunning(true);
    setSimResult(null);
    try {
      const response = await fetch("/api/score", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(sim) });
      const data = await response.json() as ReturnType<typeof scoreTransaction>;
      window.setTimeout(() => { setSimResult(data); setSimRunning(false); addAudit("Risk simulator scored", `Manual scenario · ${data.decision} · ${data.score}/100`, data.decision === "BLOCK" ? "critical" : data.decision === "REVIEW" ? "review" : "safe"); }, 650);
    } catch {
      const data = scoreTransaction(sim);
      setSimResult(data);
      setSimRunning(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileOpen ? "open" : ""}`}>
        <div className="brand"><div className="brand-mark"><ShieldCheck size={22} strokeWidth={2.3} /></div><div><strong>MerchantShield</strong><span>Autonomous risk ops</span></div><button className="icon-button mobile-close" onClick={() => setMobileOpen(false)} aria-label="Close navigation"><X size={18} /></button></div>
        <div className="workspace-card"><div className="workspace-avatar">VN</div><div><strong>Vistara North</strong><span>India · Production</span></div><ChevronDown size={15} /></div>
        <nav className="main-nav" aria-label="MerchantShield modules"><span className="nav-label">Risk operations</span>{nav.map((item) => { const Icon = item.icon; return <button key={item.id} className={active === item.id ? "active" : ""} onClick={() => { setActive(item.id); setMobileOpen(false); }}><Icon size={17} /><span>{item.label}</span>{item.count && <b>{item.count}</b>}</button>; })}</nav>
        <div className="sidebar-foot"><div className="protection-mini"><div><Radar size={16} /><span>Protection live</span></div><strong>82,491</strong><small>payments scored today</small><div className="mini-progress"><span /></div></div><button className="profile-row" onClick={() => announce("Signed in as Aditi Rao")}><div className="profile-avatar">AR</div><div><strong>Aditi Rao</strong><span>Risk operations lead</span></div><MoreHorizontal size={16} /></button></div>
      </aside>
      {mobileOpen && <button className="sidebar-scrim" onClick={() => setMobileOpen(false)} aria-label="Close navigation" />}

      <main className="main-area">
        <header className="topbar"><div className="topbar-title"><button className="icon-button menu-button" onClick={() => setMobileOpen(true)} aria-label="Open navigation"><Menu size={19} /></button><div className="live-pill"><span /> Live</div><span>Fraud-XGB scoring</span><strong>v3.2</strong></div><div className="topbar-actions"><button className="icon-button notification" onClick={() => setActive("pulse")} aria-label="Alerts"><Bell size={18} /><span /></button><button className="secondary-button" onClick={() => setSimulatorOpen(true)}><Play size={15} /> Risk simulator</button><button className="primary-button" onClick={() => setActive("review")}><ShieldQuestion size={15} /> Review queue <span>3</span></button></div></header>

        <div className="page-content">
          {active === "command" && (
            <section className="module-page">
              <div className="page-heading"><div><div className="eyebrow"><Sparkles size={14} /> Defense-only payment fraud operations</div><h1>Risk command center</h1><p>Real-time ML, rules and graph signals — optimized for <strong>merchant loss, not vanity accuracy.</strong></p></div><button className="date-button"><Clock3 size={15} /> Last 24 hours <ChevronDown size={14} /></button></div>
              <div className="honesty-strip"><BadgeCheck size={16} /><span><strong>Data honesty:</strong> live screens use labeled synthetic demo events. Model metrics come from a separate seeded, temporal synthetic hold-out and are never mixed with live metadata.</span><button onClick={() => setActive("model")}>View evaluation</button></div>
              <section className="metrics-grid"><MetricCard icon={ShieldCheck} label="Net loss prevented" value="₹5.80L" trend="+18.2% " detail="vs prior week" tone="lime" /><MetricCard icon={Activity} label="Payments processed" value="82,491" trend="+7.4% " detail="today" tone="blue" /><MetricCard icon={ShieldAlert} label="Fraud caught" value="451" trend="91.5% " detail="of confirmed attempts" tone="orange" /><MetricCard icon={CircleDollarSign} label="False-positive cost" value="₹34.0K" trend="−11.8% " detail="legitimate value" tone="purple" /></section>

              <section className="command-grid">
                <article className="panel threat-panel"><div className="panel-head"><div><span className="section-kicker">Real-time threat level</span><h2>Coordinated activity detected</h2></div><span className="spike-chip"><AlertTriangle size={13} /> Elevated</span></div><div className="threat-body"><div className="threat-gauge"><div className="gauge-arc"><div><strong>76</strong><span>/ 100</span></div></div><b>HIGH</b></div><div className="threat-copy"><p>New-device velocity from four related IP clusters is <strong>3.1σ above baseline</strong>.</p><div className="threat-stats"><span><strong>14.8</strong> suspicious/min</span><span><strong>+543%</strong> over baseline</span><span><strong>₹1.82L</strong> exposed</span></div><button onClick={() => setActive("pulse")}>Open Fraud Pulse <ArrowRight size={14} /></button></div></div></article>
                <article className="panel ai-analyst"><div className="analyst-head"><div className="analyst-orb"><Bot size={21} /></div><div><span className="section-kicker">Grounded AI analyst</span><h2>What changed?</h2></div><span className="grounded-tag"><Check size={12} /> Aggregates only</span></div><p className="analyst-answer">{assistantAnswer}</p><div className="analyst-points"><span><i /> 19 high-risk payments from 4 related IPs</span><span><i /> Electronics = 48% of blocked value</span><span><i /> AR-129 contributed ₹74,200 attempted fraud</span></div><div className="prompt-row"><button onClick={() => setAssistantAnswer("Cluster AR-129 is the highest-priority investigation: 12 linked accounts, 3 shared devices and ₹1.82L exposure. No customer PII was sent to the assistant.")}>What should I inspect?</button><button onClick={() => setAssistantAnswer("The current spike is concentrated in newly observed devices. Card and UPI rails are both affected; wallet traffic remains within baseline.")}>Which channel changed?</button></div></article>
              </section>

              <section className="command-lower">
                <article className="panel volume-panel"><div className="panel-head"><div><span className="section-kicker">Fraud attempted vs prevented</span><h2>Hourly payment risk</h2></div><div className="chart-legend"><span className="legend-protected" /> Prevented <span className="legend-exposure" /> Missed</div></div><div className="css-bar-chart">{volumeBars.map((height, i) => <div key={i} className={i > 18 ? "hot" : ""}><span style={{ height: `${height}%` }} /><i style={{ height: `${Math.max(5, height * .16)}%` }} /></div>)}</div><div className="chart-times"><span>00:00</span><span>06:00</span><span>12:00</span><span>18:00</span><span>Now</span></div></article>
                <article className="panel live-snapshot"><div className="panel-head"><div><span className="section-kicker">Streaming now</span><h2>Live transactions</h2></div><button className="text-button" onClick={() => setActive("transactions")}>View feed <ArrowRight size={14} /></button></div><div className="snapshot-table">{enriched.slice(0,5).map((tx) => <button key={tx.id} onClick={() => { setSelectedId(tx.id); setActive("transactions"); }}><span className="tx-live-dot" /><strong>{tx.id}</strong><span>{money(tx.amount)}</span><span className={`risk-score-text ${decisionClass(tx.result.decision)}`}>{tx.result.score}</span><DecisionBadge decision={tx.result.decision} /><ChevronRight size={14} /></button>)}</div></article>
              </section>
              <div className="demo-actions"><div><Sparkles size={16} /><span><strong>Judge demo controls</strong><small>Inject a safe or suspicious event into the exact same scoring pipeline.</small></span></div><button className="secondary-button" onClick={() => addLiveTransaction("normal")}><CheckCircle2 size={15} /> Normal payment</button><button className="primary-button danger-primary" onClick={() => addLiveTransaction("suspicious")}><AlertOctagon size={15} /> Suspicious payment</button></div>
            </section>
          )}

          {active === "transactions" && (
            <section className="module-page"><div className="module-heading"><div><span className="section-kicker">Event ingestion + investigation</span><h1>Live transactions</h1><p>Every event is enriched, scored, explained and converted into an operational decision.</p></div><div className="heading-actions"><button className="secondary-button" onClick={() => addLiveTransaction("normal")}><Check size={15} /> Inject normal</button><button className="primary-button danger-primary" onClick={() => addLiveTransaction("suspicious")}><Zap size={15} /> Inject suspicious</button></div></div>
              <div className="pipeline-strip">{[[Database,"Event ingested"],[SlidersHorizontal,"Features built"],[BrainCircuit,"ML scored"],[GitBranch,"Rules checked"],[Network,"Graph enriched"],[ShieldCheck,"Decision"]].map(([Icon,label],i) => { const C = Icon as typeof Database; return <div key={String(label)}><span><C size={14} /></span><b>{label as string}</b>{i < 5 && <ChevronRight size={13} />}</div>; })}</div>
              <div className="queue-toolbar"><div className="search-box"><Search size={16} /><input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search transaction, customer, city or IP" aria-label="Search transactions" /></div><div className="filter-pills">{(["ALL","ALLOW","REVIEW","BLOCK"] as const).map((d) => <button key={d} className={decisionFilter === d ? "active" : ""} onClick={() => setDecisionFilter(d)}>{d === "ALL" ? "All decisions" : d}</button>)}</div><button className="filter-button"><ListFilter size={15} /> Filters</button></div>
              <div className="investigation-layout"><article className="panel transaction-table"><div className="tx-table-head"><span>Transaction</span><span>Customer</span><span>Amount</span><span>Risk</span><span>Decision</span></div>{filtered.map((tx) => <button key={tx.id} className={tx.id === selected.id ? "selected" : ""} onClick={() => setSelectedId(tx.id)}><span><i className={decisionClass(tx.result.decision)} />{tx.id}<small>{tx.time}</small></span><span><b>{tx.customer}</b><small>{tx.city}</small></span><span><b>{money(tx.amount)}</b><small>{tx.channel}</small></span><span className={`risk-number ${decisionClass(tx.result.decision)}`}>{tx.result.score}</span><DecisionBadge decision={tx.result.decision} /></button>)}</article>
                <aside className="panel investigation-panel"><div className="case-head"><div><div className="customer-avatar large">{selected.initials}</div><div><span>{selected.id} · {selected.time}</span><h2>{selected.customer}</h2></div></div><DecisionBadge decision={selected.result.decision} /></div><div className="score-package"><div className={`big-score ${decisionClass(selected.result.decision)}`}><strong>{selected.result.score}</strong><span>/100 risk</span></div><div><span>Model probability <b>{(selected.result.probability*100).toFixed(1)}%</b></span><span>Rules triggered <b>{selected.result.rules.length}</b></span><span>Network risk <b>{selected.result.networkRisk}/100</b></span></div></div><div className="case-tabs"><button className={caseTab === "explain" ? "active" : ""} onClick={() => setCaseTab("explain")}>Explanation</button><button className={caseTab === "network" ? "active" : ""} onClick={() => setCaseTab("network")}>Linked entities</button><button className={caseTab === "timeline" ? "active" : ""} onClick={() => setCaseTab("timeline")}>Timeline</button></div>
                  {caseTab === "explain" && <div className="investigation-content"><div className="reason-list">{selected.result.reasons.map((reason,i) => <div key={reason}><span>{i+1}</span><div><strong>{reason}</strong><small>{i === 0 ? "High contribution" : i === 1 ? "Medium contribution" : "Supporting evidence"}</small></div><b>+{[27,19,12,8][i]}</b></div>)}</div><h3>Triggered rules</h3>{selected.result.rules.length ? selected.result.rules.map((rule) => <div className="triggered-rule" key={rule}><GitBranch size={14} /><span>{rule}</span><small>v12</small></div>) : <div className="no-rule"><CheckCircle2 size={15} /> No deterministic rules triggered</div>}<div className="risk-components">{[["Velocity",selected.result.velocityRisk],["Device",selected.result.deviceRisk],["Network",selected.result.networkRisk],["Model",selected.result.modelRisk]].map(([label,value]) => <div key={String(label)}><span>{label}</span><i><b style={{width:`${value}%`}} /></i><strong>{value}</strong></div>)}</div></div>}
                  {caseTab === "network" && <div className="investigation-content"><div className="mini-network"><div className="mini-node center"><Fingerprint size={15} /><span>{selected.device.split(" · ")[0]}</span></div><div className="mini-node n1">IP<br/>{selected.ip}</div><div className="mini-node n2">Accounts<br/>{selected.input.accountsSharingDevice}</div><div className="mini-node n3">Cards<br/>{Math.max(1,selected.input.accountsSharingDevice-2)}</div><i className="ml1"/><i className="ml2"/><i className="ml3"/></div><button className="full-width-button" onClick={() => setActive("rings")}>Open abuse-ring graph <ArrowRight size={14} /></button></div>}
                  {caseTab === "timeline" && <div className="timeline-list"><div className="done"><span><Check size={12}/></span><div><strong>Event accepted</strong><p>Schema validated · payment API · 0ms</p></div></div><div className="done"><span><SlidersHorizontal size={12}/></span><div><strong>Features computed</strong><p>31 leakage-safe features · 8ms</p></div></div><div className="done"><span><BrainCircuit size={12}/></span><div><strong>Risk package generated</strong><p>{selected.modelVersion} · {selected.result.score}/100 · 24ms</p></div></div><div className={selected.analystDecision ? "done" : "current"}><span><UserCheck size={12}/></span><div><strong>Human decision</strong><p>{selected.analystDecision ?? "Awaiting review when required"}</p></div></div></div>}
                  <div className="investigation-meta"><span><Fingerprint size={13}/> {selected.device}</span><span><Waypoints size={13}/> {selected.ip}</span><span><Database size={13}/> {selected.modelVersion}</span></div></aside></div>
            </section>
          )}

          {active === "review" && (
            <section className="module-page"><div className="module-heading"><div><span className="section-kicker">Human-in-the-loop control</span><h1>Review center</h1><p>Ambiguous payments get evidence, not accusations. Decisions become model feedback.</p></div><button className="secondary-button" onClick={() => downloadCsv("merchantshield-review-queue.csv", [["transaction","customer","amount","risk","decision"],...enriched.filter(t=>t.result.decision!=="ALLOW").map(t=>[t.id,t.customer,t.amount,t.result.score,t.analystDecision??"PENDING"])])}><Download size={15}/> Export queue</button></div>
              <div className="review-summary"><div><span>Queue</span><strong>{enriched.filter(t=>t.result.decision!=="ALLOW"&&!t.analystDecision).length}</strong></div><div><span>Critical</span><strong className="critical-text">2</strong></div><div><span>Median wait</span><strong>01:42</strong></div><div><span>Analyst agreement</span><strong>94.8%</strong></div></div>
              <div className="review-center-layout"><article className="panel review-list"><div className="review-list-head"><span>Prioritized by expected loss</span><small>Highest risk first</small></div>{enriched.filter(t=>t.result.decision!=="ALLOW").map(tx => <button key={tx.id} className={selected.id===tx.id?"selected":""} onClick={()=>setSelectedId(tx.id)}><div className={`review-score ${decisionClass(tx.result.decision)}`}>{tx.result.score}</div><div><strong>{tx.customer}</strong><span>{tx.id} · {tx.city}</span></div><div><strong>{money(tx.amount)}</strong><span>{tx.result.reasons[0]}</span></div><span className={`review-state ${tx.analystDecision?"complete":"pending"}`}>{tx.analystDecision??"PENDING"}</span><ChevronRight size={15}/></button>)}</article>
                <aside className="panel decision-panel"><div className="decision-top"><div><span className="section-kicker">Decision workspace</span><h2>{selected.id} · {money(selected.amount)}</h2></div><DecisionBadge decision={selected.result.decision}/></div><div className="decision-explain"><Sparkles size={15}/><p><strong>Summary:</strong> {selected.result.reasons.slice(0,2).join("; ")}. Recommendation: {selected.result.decision.toLowerCase()} with human override.</p></div><h3>Decision reasons</h3><div className="reason-checks">{["Shared suspicious device","High transaction velocity","Amount anomaly","Known trusted customer"].map(reason=><button key={reason} className={reviewReasons.includes(reason)?"checked":""} onClick={()=>setReviewReasons(current=>current.includes(reason)?current.filter(r=>r!==reason):[...current,reason])}><span>{reviewReasons.includes(reason)&&<Check size={12}/>}</span>{reason}</button>)}</div><div className="decision-facts"><div><span>Velocity risk</span><strong>{selected.result.velocityRisk}/100</strong></div><div><span>Device risk</span><strong>{selected.result.deviceRisk}/100</strong></div><div><span>Accounts linked</span><strong>{selected.input.accountsSharingDevice}</strong></div><div><span>Model version</span><strong>v3.2</strong></div></div><div className="decision-actions"><button className="approve-action" onClick={()=>decide("APPROVED")}><CheckCircle2 size={15}/> Approve</button><button className="block-action" onClick={()=>decide("BLOCKED")}><ShieldAlert size={15}/> Block</button><button className="info-action" onClick={()=>decide("MORE INFO")}><MessageSquareText size={15}/> Need more information</button></div><p className="decision-disclaimer"><Info size={13}/> This action is reversible and logged with analyst identity, reasons, model and rule versions.</p></aside></div>
            </section>
          )}

          {active === "rings" && (
            <section className="module-page"><div className="module-heading"><div><span className="section-kicker">Coordinated-abuse detection</span><h1>Abuse ring sentinel</h1><p>Shared devices, IPs and payment tokens reveal risk invisible at transaction level.</p></div><button className="secondary-button" onClick={()=>announce("Graph refreshed with latest 30 minutes")}><RefreshCw size={15}/> Refresh graph</button></div>
              <div className="ring-banner"><div><AlertOctagon size={20}/><span><strong>Cluster AR-129 · critical</strong><small>12 linked accounts · 7 blocked payments · ₹1.82L exposure</small></span></div><div><span>Cluster risk</span><strong>94</strong><small>/100</small></div></div>
              <div className="ring-layout"><article className="panel graph-panel"><div className="graph-toolbar"><div><span className="dot customer"/>Customer <span className="dot device"/>Device <span className="dot ip"/>IP <span className="dot payment"/>Payment token</div><button><Settings2 size={14}/> Layout</button></div><div className="entity-graph"><i className="edge e1"/><i className="edge e2"/><i className="edge e3"/><i className="edge e4"/><i className="edge e5"/><i className="edge e6"/><i className="edge e7"/><button className="graph-node device center" onClick={()=>setSelectedNode("Device D-91")}><Fingerprint size={18}/><strong>D-91</strong><span>Device</span></button><button className="graph-node ip top" onClick={()=>setSelectedNode("IP 49.36.18.201")}><Waypoints size={16}/><strong>49.36…</strong><span>IP</span></button><button className="graph-node customer c1" onClick={()=>setSelectedNode("Customer 731")}><Users size={15}/><strong>C-731</strong><span>Customer</span></button><button className="graph-node customer c2" onClick={()=>setSelectedNode("Customer 821")}><Users size={15}/><strong>C-821</strong><span>Customer</span></button><button className="graph-node customer c3" onClick={()=>setSelectedNode("Customer 193")}><Users size={15}/><strong>C-193</strong><span>Customer</span></button><button className="graph-node payment p1" onClick={()=>setSelectedNode("Payment token P-44")}><IndianRupee size={15}/><strong>P-44</strong><span>Token</span></button><button className="graph-node payment p2" onClick={()=>setSelectedNode("Payment token P-71")}><IndianRupee size={15}/><strong>P-71</strong><span>Token</span></button></div></article>
                <aside className="panel entity-panel"><span className="section-kicker">Selected entity</span><div className="entity-title"><div><Fingerprint size={20}/></div><span><h2>{selectedNode}</h2><small>First seen 42 minutes ago</small></span></div><div className="entity-risk"><span>Entity risk</span><strong>91/100</strong><i><b/></i></div><dl><div><dt>Linked accounts</dt><dd>11</dd></div><div><dt>Payment tokens</dt><dd>6</dd></div><div><dt>Attempted value</dt><dd>₹1,42,600</dd></div><div><dt>Blocked</dt><dd>7</dd></div><div><dt>First-seen geography</dt><dd>Pune, MH</dd></div></dl><div className="entity-alert"><AlertTriangle size={15}/><span>6 accounts were created within the same 19-minute window.</span></div><button className="primary-button" onClick={()=>{setActive("review");announce("Cluster cases added to analyst queue");}}>Route cluster to review <ArrowRight size={14}/></button></aside></div>
            </section>
          )}

          {active === "pulse" && (
            <section className="module-page"><div className="module-heading"><div><span className="section-kicker">Rolling anomaly detection</span><h1>Fraud pulse</h1><p>Detect abrupt shifts in rate, devices, geographies, categories and network clusters.</p></div><button className={`primary-button ${pulseActive?"":"danger-primary"}`} onClick={()=>{setPulseActive(!pulseActive);announce(pulseActive?"Spike simulation reset":"Fraud spike injected");}}>{pulseActive?<><RotateCcw size={15}/> Reset spike</>:<><Zap size={15}/> Simulate spike</>}</button></div>
              <div className={`spike-alert ${pulseActive?"active":""}`}><div className="spike-icon"><AlertTriangle size={24}/></div><div><span className="section-kicker">{pulseActive?"Fraud spike detected":"No active anomaly"}</span><h2>{pulseActive?"New-device payment velocity is 3.1σ above baseline":"Monitoring 18 aggregate signals"}</h2><p>{pulseActive?"Started 15:32:41 · strongest contributor: Cluster AR-129":"Inject a defensive demo spike to test alerting and response workflows."}</p></div><div className="spike-rate"><span>Current rate</span><strong>{pulseActive?"14.8":"2.3"}</strong><small>suspicious/min</small></div><div className="spike-change"><ArrowUpRight size={15}/>{pulseActive?"+543%":"Baseline"}</div></div>
              <div className="pulse-grid"><article className="panel pulse-chart-panel"><div className="panel-head"><div><span className="section-kicker">Suspicious attempts / minute</span><h2>Rolling 60-minute window</h2></div><span className="baseline-label">Baseline 2.3</span></div><div className="pulse-chart"><div className="pulse-baseline"/><div className="pulse-bars">{Array.from({length:30},(_,i)=>{const h=pulseActive&&i>22?[42,55,69,83,96,88,79][i-23]:(13+(i*7)%15);return <span key={i} className={pulseActive&&i>22?"anomaly":""} style={{height:`${h}%`}}/>})}</div></div><div className="chart-times"><span>14:45</span><span>15:00</span><span>15:15</span><span>15:30</span><span>15:45</span></div></article><article className="panel pulse-drivers"><div className="panel-head"><div><span className="section-kicker">Spike composition</span><h2>Common signals</h2></div></div>{[["Newly observed devices",78,"+61%"],["Electronics category",48,"+34%"],["Linked IP clusters",41,"+28%"],["High amount anomaly",26,"+12%"]].map(([label,value,change])=><div className="driver-row" key={String(label)}><div><span>{label}</span><strong>{change}</strong></div><i><b style={{width:`${value}%`}}/></i><small>{value}%</small></div>)}</article></div>
              <div className="response-strip"><div><ShieldCheck size={18}/><span><strong>Suggested defensive response</strong><small>Temporarily route scores 62–69 to review and inspect Cluster AR-129 first.</small></span></div><button className="secondary-button" onClick={()=>setActive("rings")}>Inspect AR-129</button><button className="primary-button" onClick={()=>{setThreshold(62);setActive("model");announce("Proposed threshold loaded in Cost Lab");}}>Test sensitivity <ArrowRight size={14}/></button></div>
            </section>
          )}

          {active === "rules" && (
            <section className="module-page"><div className="module-heading"><div><span className="section-kicker">Deterministic controls + backtesting</span><h1>Merchant rule lab</h1><p>Build readable safeguards, measure historical impact, then deploy with version control.</p></div><span className="draft-chip">Draft · not live</span></div>
              <div className="rule-layout"><article className="panel rule-builder"><div className="panel-head"><div><span className="section-kicker">Visual rule builder</span><h2>Device velocity protection</h2></div><MoreHorizontal size={18}/></div><div className="rule-sentence"><div><small>WHEN</small><span>transactions from same device</span><select value={rule.deviceCount} onChange={e=>setRule({...rule,deviceCount:Number(e.target.value)})}>{[3,4,5,6,8].map(v=><option key={v} value={v}>&gt; {v}</option>)}</select></div><div><small>WITHIN</small><select value={rule.minutes} onChange={e=>setRule({...rule,minutes:Number(e.target.value)})}>{[5,10,15,30].map(v=><option key={v} value={v}>{v} minutes</option>)}</select></div><div><small>AND</small><span>risk score</span><select value={rule.score} onChange={e=>setRule({...rule,score:Number(e.target.value)})}>{[55,60,65,70,75].map(v=><option key={v} value={v}>&gt; {v}</option>)}</select></div><div className="rule-then"><small>THEN</small><select value={rule.action} onChange={e=>setRule({...rule,action:e.target.value as Decision})}><option>REVIEW</option><option>BLOCK</option><option>ALLOW</option></select></div></div><div className="rule-safety"><ShieldCheck size={15}/><span>Rules can only change merchant-side decisions. Every match is logged and reversible.</span></div></article>
                <article className="panel backtest-panel"><div className="panel-head"><div><span className="section-kicker">Historical backtest</span><h2>Impact on the last 30 days</h2></div><span className="computed-chip"><RefreshCw size={12}/> Recomputed</span></div><div className="backtest-hero"><span>Estimated fraud prevented</span><strong>{money(ruleFraudCaught*2950)}</strong><small>vs. {money(ruleLegitAffected*850)} potential legitimate loss</small></div><div className="backtest-grid"><div><span>Transactions affected</span><strong>{ruleAffected}</strong></div><div><span>Fraud caught</span><strong>{ruleFraudCaught}</strong></div><div><span>Legitimate affected</span><strong>{ruleLegitAffected}</strong></div><div><span>Precision</span><strong>{((ruleFraudCaught/(ruleFraudCaught+ruleLegitAffected))*100).toFixed(1)}%</strong></div></div><button className="primary-button" disabled={ruleDeployed} onClick={()=>{setRuleDeployed(true);addAudit("Rule deployed",`Device velocity protection · v13 · ${rule.action}`,"review");announce("Rule v13 deployed with audit trail");}}>{ruleDeployed?<><Check size={15}/> Rule v13 deployed</>:<><Play size={15}/> Deploy rule</>}</button><p><Info size={13}/> Backtest uses a historical hold-out. No rule is deployed without showing false positives.</p></article></div>
              <div className="rules-list"><div className="rules-list-head"><span>Active rules</span><button>View versions</button></div>{[["Device velocity cluster","REVIEW","v12","1,284 matches"],["New account + high amount","REVIEW","v8","472 matches"],["Repeated payment failures","BLOCK","v5","216 matches"],["Trusted customer fast lane","ALLOW","v4","18,904 matches"]].map(([name,action,version,matches])=><div key={name}><span className={`rule-status ${String(action).toLowerCase()}`}/><strong>{name}</strong><span>{action}</span><span>{version}</span><span>{matches}</span><button><Settings2 size={14}/></button></div>)}</div>
            </section>
          )}

          {active === "model" && (
            <section className="module-page"><div className="module-heading"><div><span className="section-kicker">Frozen temporal hold-out</span><h1>Model performance & cost lab</h1><p>12,500 unseen seeded synthetic transactions · Days 151–180 · positive prevalence 10.8%.</p></div><button className="secondary-button" onClick={()=>downloadCsv("merchantshield-heldout-metrics.csv",[["threshold","tp","fp","tn","fn"],...HELD_OUT_RESULTS.map(r=>[r.threshold,r.tp,r.fp,r.tn,r.fn])])}><Download size={15}/> Download evaluation</button></div>
              <div className="model-banner"><div className="model-banner-icon"><BadgeCheck size={22}/></div><div><strong>Fraud-XGB-v3.2 is within deployment guardrails</strong><span>Temporal test period frozen before threshold selection · P95 latency 31ms · PR-AUC 0.903</span></div><div className="model-status"><span/> Healthy</div></div>
              <div className="model-metric-grid"><article><span>Precision</span><strong>{(metrics.precision*100).toFixed(1)}%</strong><small>Flagged payments that were fraud</small><div className="microbar"><span style={{width:`${metrics.precision*100}%`}}/></div></article><article><span>Recall</span><strong>{(metrics.recall*100).toFixed(1)}%</strong><small>True fraud caught</small><div className="microbar blue"><span style={{width:`${metrics.recall*100}%`}}/></div></article><article><span>F1</span><strong>{(metrics.f1*100).toFixed(1)}%</strong><small>Precision/recall balance</small><div className="microbar purple"><span style={{width:`${metrics.f1*100}%`}}/></div></article><article><span>False-positive rate</span><strong>{(metrics.fpr*100).toFixed(2)}%</strong><small>{metrics.fp} legitimate payments flagged</small><div className="microbar orange"><span style={{width:`${Math.max(4,metrics.fpr*500)}%`}}/></div></article></div>
              <div className="model-layout"><article className="panel threshold-panel"><div className="panel-head"><div><span className="section-kicker">What-if threshold lab</span><h2>Minimize total merchant cost</h2></div><span className="threshold-badge">{threshold}/100</span></div><p className="panel-description">Compare real error counts from the held-out evaluation. Review band remains 40–{threshold-1}; scores ≥ {threshold} recommend block.</p><div className="threshold-control"><input type="range" min="0" max={HELD_OUT_RESULTS.length-1} value={HELD_OUT_RESULTS.findIndex(r=>r.threshold===threshold)} onChange={e=>setThreshold(HELD_OUT_RESULTS[Number(e.target.value)].threshold)} /><div>{HELD_OUT_RESULTS.map(r=><span key={r.threshold}>{r.threshold}</span>)}</div></div><div className="cost-inputs"><label>Fraud loss <span>₹</span><input type="number" value={fraudLoss} onChange={e=>setFraudLoss(Number(e.target.value))}/></label><label>Legitimate loss <span>₹</span><input type="number" value={legitLoss} onChange={e=>setLegitLoss(Number(e.target.value))}/></label><label>Manual review <span>₹</span><input type="number" value={reviewCost} onChange={e=>setReviewCost(Number(e.target.value))}/></label></div><div className="cost-formula"><span>Total risk cost</span><strong>{money(totalCost)}</strong><small>{money(fnCost)} missed fraud + {money(fpCost)} false positives + {money(manualCost)} reviews</small></div><div className="cost-result"><div><span>Net loss prevented</span><strong>{money(netPrevented)}</strong></div><span className="positive"><ArrowUpRight size={13}/> {((netPrevented/noModelCost)*100).toFixed(1)}% vs no controls</span></div></article>
                <article className="panel matrix-panel"><div className="panel-head"><div><span className="section-kicker">Untouched test set</span><h2>Confusion matrix</h2></div><span className="sample-size">n = 12,500</span></div><div className="matrix-wrap"><div className="matrix-y">Actual class</div><div className="matrix-label top"><span>Predicted fraud</span><span>Predicted safe</span></div><div className="matrix-label side"><span>Fraud</span><span>Legitimate</span></div><div className="matrix-cell tp"><span>True positive</span><strong>{metrics.tp.toLocaleString("en-IN")}</strong><small>Fraud caught</small></div><div className="matrix-cell fn"><span>False negative</span><strong>{metrics.fn.toLocaleString("en-IN")}</strong><small>Fraud missed</small></div><div className="matrix-cell fp"><span>False positive</span><strong>{metrics.fp.toLocaleString("en-IN")}</strong><small>Legitimate flagged</small></div><div className="matrix-cell tn"><span>True negative</span><strong>{metrics.tn.toLocaleString("en-IN")}</strong><small>Friction avoided</small></div></div></article></div>
              <div className="drift-grid"><article className="panel drift-panel"><div className="panel-head"><div><span className="section-kicker">Feature drift</span><h2>Population stability</h2></div><span className="healthy-tag"><Check size={12}/> Stable</span></div>{[["transaction_amount",8],["account_age_days",11],["payment_velocity",17],["device_seen_count",6]].map(([label,value])=><div className="drift-row" key={String(label)}><span>{label}</span><div><i style={{width:`${Number(value)*4}%`}}/></div><strong>PSI 0.{String(value).padStart(2,"0")}</strong></div>)}<p><Info size={13}/> Alert threshold PSI ≥ 0.20. Payment velocity is closest and under watch.</p></article><article className="panel audit-panel"><div className="panel-head"><div><span className="section-kicker">Evaluation facts</span><h2>Dataset honesty card</h2></div><Fingerprint size={19}/></div><dl><div><dt>Source</dt><dd>Seeded synthetic benchmark</dd></div><div><dt>Split</dt><dd>Temporal, untouched</dd></div><div><dt>Training</dt><dd>Days 1–120</dd></div><div><dt>Validation</dt><dd>Days 121–150</dd></div><div><dt>Test</dt><dd>Days 151–180</dd></div><div><dt>Graph metadata</dt><dd>Synthetic · demo only</dd></div></dl></article></div>
            </section>
          )}

          {active === "audit" && (
            <section className="module-page"><div className="module-heading"><div><span className="section-kicker">Traceable by design</span><h1>Decision audit log</h1><p>Scores, versions, triggered rules and human actions preserved for every payment.</p></div><button className="secondary-button" onClick={()=>downloadCsv("merchantshield-audit.csv",[["time","actor","action","detail"],...audit.map(e=>[e.time,e.actor,e.action,e.detail])])}><Download size={15}/> Export audit</button></div>
              <div className="audit-summary"><div><Database size={20}/><span><strong>{audit.length.toLocaleString("en-IN")} visible events</strong><small>Append-only demo trail · newest first</small></span></div><span><CheckCircle2 size={14}/> All decision packages complete</span><span><TimerReset size={14}/> Retention: 365 days</span></div>
              <article className="panel audit-table"><div className="audit-table-head"><span>Time</span><span>Actor</span><span>Action</span><span>Evidence</span><span>Integrity</span></div>{audit.map((event,i)=><div key={`${event.time}-${i}`}><span>{event.time}</span><span><i className={`audit-dot ${event.tone}`}/>{event.actor}</span><strong>{event.action}</strong><span>{event.detail}</span><span className="integrity"><Fingerprint size={13}/> Logged</span></div>)}</article><div className="audit-guardrail"><ShieldCheck size={18}/><div><strong>Defense-only operating boundary</strong><p>MerchantShield detects and responds to suspicious payment behavior. It does not reveal evasion tactics, generate attack steps or treat model output as an accusation.</p></div></div>
            </section>
          )}
        </div>
      </main>

      {simulatorOpen && <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="Risk simulator"><button className="modal-dismiss" onClick={()=>setSimulatorOpen(false)} aria-label="Close simulator"/><section className="simulator-modal payment-simulator"><div className="sim-head"><div><span className="section-kicker">End-to-end risk simulator</span><h2>Run a payment risk check</h2><p>The request passes through feature extraction, ML, rules and network analysis.</p></div><button className="icon-button" onClick={()=>setSimulatorOpen(false)}><X size={18}/></button></div><div className="sim-layout"><div className="sim-form"><label>Amount <span>{money(sim.amount)}</span><input type="range" min="500" max="50000" step="500" value={sim.amount} onChange={e=>setSim({...sim,amount:Number(e.target.value)})}/></label><div className="input-pair"><label>Customer age (days)<input type="number" value={sim.customerAgeDays} onChange={e=>setSim({...sim,customerAgeDays:Number(e.target.value)})}/></label><label>Transactions / 5 min<input type="number" value={sim.transactionsLast5Min} onChange={e=>setSim({...sim,transactionsLast5Min:Number(e.target.value)})}/></label></div><div className="input-pair"><label>Accounts sharing device<input type="number" value={sim.accountsSharingDevice} onChange={e=>setSim({...sim,accountsSharingDevice:Number(e.target.value)})}/></label><label>Failed attempts / hour<input type="number" value={sim.failedAttemptsLastHour} onChange={e=>setSim({...sim,failedAttemptsLastHour:Number(e.target.value)})}/></label></div><div className="input-pair"><label>Historical chargebacks<input type="number" value={sim.historicalChargebacks} onChange={e=>setSim({...sim,historicalChargebacks:Number(e.target.value)})}/></label><label>Amount vs baseline<input type="number" step="0.1" value={sim.amountVsBaseline} onChange={e=>setSim({...sim,amountVsBaseline:Number(e.target.value)})}/></label></div><div className="check-options"><button className={sim.newDevice?"checked":""} onClick={()=>setSim({...sim,newDevice:!sim.newDevice})}><span>{sim.newDevice&&<Check size={12}/>}</span> Newly observed device</button></div><button className="primary-button simulator-run" onClick={runSimulator} disabled={simRunning}>{simRunning?<><RefreshCw className="spin" size={15}/> Running pipeline…</>:<><Play size={15}/> Run risk check</>}</button></div><div className={`sim-result ${simResult?decisionClass(simResult.decision):"idle"}`}>{!simResult&&!simRunning&&<div className="sim-empty"><Radar size={35}/><h3>Ready to score</h3><p>Adjust the inputs, then run the same transparent detector used by the live feed.</p></div>}{simRunning&&<div className="pipeline-running"><div className="loader-orb"><BrainCircuit size={28}/></div>{["Feature extraction","ML scoring","Rule evaluation","Network analysis"].map((s,i)=><span key={s} style={{animationDelay:`${i*.12}s`}}><Check size={13}/>{s}</span>)}</div>}{simResult&&!simRunning&&<><div className="sim-score-ring" style={{background:`conic-gradient(${simResult.decision==="BLOCK"?"var(--orange)":simResult.decision==="REVIEW"?"var(--amber)":"var(--teal)"} ${simResult.score}%, #27302b 0)`}}><div><strong>{simResult.score}</strong><span>/ 100</span></div></div><DecisionBadge decision={simResult.decision}/><h3>{simResult.decision==="BLOCK"?"Intervention recommended":simResult.decision==="REVIEW"?"Human review recommended":"Safe to allow"}</h3><p>{simResult.reasons.slice(0,2).join(". ")}.</p><div className="sim-drivers">{simResult.reasons.slice(0,3).map(reason=><span key={reason}><b>+</b>{reason}</span>)}</div></>}</div></div><div className="sim-footer"><span><Info size={13}/> Demo-only scenario. No real customer data.</span><button className="secondary-button" onClick={()=>{setSim({amount:28000,customerAgeDays:2,newDevice:true,transactionsLast5Min:6,accountsSharingDevice:9,historicalChargebacks:1,failedAttemptsLastHour:5,amountVsBaseline:4.2});setSimResult(null);}}><RotateCcw size={14}/> Reset</button><button className="primary-button" disabled={!simResult} onClick={()=>{setSimulatorOpen(false);setActive("transactions");}}>Open investigation <ArrowRight size={14}/></button></div></section></div>}
      {toast && <div className="toast" role="status"><CheckCircle2 size={16}/>{toast}</div>}
    </div>
  );
}
