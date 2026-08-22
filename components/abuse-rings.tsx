"use client";
/* eslint-disable @next/next/no-html-link-for-pages */

import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  CircleDot,
  FileText,
  Gauge,
  Info,
  ListChecks,
  LockKeyhole,
  Menu,
  Network,
  Play,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  TableProperties,
  X,
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useState } from "react";

const apiBase = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "");
const apiPath = (path: string) => `${apiBase}${path}`;
type Decision = "APPROVE" | "REVIEW" | "BLOCK";
type GraphConfig = { min_attribute_degree: number; max_attribute_degree: number; minimum_cluster_transactions: number; minimum_high_risk_transactions: number; minimum_high_risk_share: number; max_clusters: number };
type Cluster = { cluster_id: string; transaction_count: number; shared_attribute_count: number; edge_count: number; connectivity_score: number; high_risk_count: number; high_risk_share: number; average_risk_score: number; maximum_risk_score: number; total_transaction_amount: number; high_risk_amount: number; shared_attribute_types: string[]; example_transaction_ids: string[]; label: "SUSPICIOUS LINKED CLUSTER" };
type GraphNode = { id: string; node_type: "transaction" | "shared_attribute"; label: string; transaction_id: string | null; risk_score: number | null; amount: number | null; decision: Decision | null; source_field: string | null; attribute_type: string | null; attribute_value: string | null };
type GraphEdge = { source: string; target: string; relationship: "SHARES ATTRIBUTE" };
type ClusterGraph = { cluster_id: string; label: "SUSPICIOUS LINKED CLUSTER"; nodes: GraphNode[]; edges: GraphEdge[]; total_nodes: number; total_edges: number; graph_truncated: boolean; limitation: string };
type Analysis = { source: string; data_partition: "validation"; evaluation_status: "Not evaluated yet"; model_version: string; review_threshold: number; block_threshold: number; config: GraphConfig; transaction_rows_considered: number; transaction_nodes: number; shared_attribute_nodes: number; edge_count: number; connected_components: number; suspicious_cluster_count: number; returned_cluster_count: number; suppressed_attribute_values: number; clusters: Cluster[]; held_out_test_accessed: false; confirmed_fraud_ring_claimed: false; limitations: string[] };
type Status = { data_source: string; evaluation_status: "Not evaluated yet"; model_version: string; review_threshold: number; block_threshold: number; considered_attributes: Array<{ source_field: string; documented_label: string }>; default_common_value_suppression: string; terminology: string[]; limitations: string[] };
type Neighborhood = { transaction_id: string; found_in_validation: boolean; connected_through_eligible_attributes: boolean; graph: ClusterGraph | null; message: string; held_out_test_accessed: false };

const defaultConfig: GraphConfig = { min_attribute_degree: 2, max_attribute_degree: 50, minimum_cluster_transactions: 2, minimum_high_risk_transactions: 1, minimum_high_risk_share: .25, max_clusters: 25 };

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiPath(path), init);
  if (!response.ok) { const payload = await response.json().catch(() => null) as { detail?: string } | null; throw new Error(payload?.detail ?? `Request failed (${response.status})`); }
  return response.json() as Promise<T>;
}

function currency(value: number) { return new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 }).format(value); }
function percent(value: number) { return `${(value * 100).toFixed(1)}%`; }

function GraphCanvas({ graph, selected, onSelect }: { graph: ClusterGraph; selected: GraphNode | null; onSelect: (node: GraphNode) => void }) {
  const attributes = graph.nodes.filter((node) => node.node_type === "shared_attribute");
  const transactions = graph.nodes.filter((node) => node.node_type === "transaction");
  const positions = new Map<string, { x: number; y: number }>();
  attributes.forEach((node, index) => positions.set(node.id, { x: 170, y: 38 + index * (344 / Math.max(attributes.length - 1, 1)) }));
  transactions.forEach((node, index) => positions.set(node.id, { x: 630, y: 38 + index * (344 / Math.max(transactions.length - 1, 1)) }));
  function activate(node: GraphNode) { onSelect(node); }
  return <div className="graph-canvas"><svg role="img" aria-label="Suspicious linked cluster graph" viewBox="0 0 800 420" preserveAspectRatio="xMidYMid meet">
    <g className="graph-edges">{graph.edges.map((edge, index) => { const source = positions.get(edge.source); const target = positions.get(edge.target); if (!source || !target) return null; return <line key={`${edge.source}-${edge.target}-${index}`} x1={source.x} y1={source.y} x2={target.x} y2={target.y} />; })}</g>
    {graph.nodes.map((node) => { const point = positions.get(node.id); if (!point) return null; const active = selected?.id === node.id; return <g key={node.id} className={`graph-node ${node.node_type} ${node.decision?.toLowerCase() ?? ""} ${active ? "selected" : ""}`} role="button" tabIndex={0} aria-label={node.label} onClick={() => activate(node)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") activate(node); }} transform={`translate(${point.x} ${point.y})`}>
      {node.node_type === "transaction" ? <circle r={active ? 10 : 7} /> : <rect x={active ? -10 : -8} y={active ? -10 : -8} width={active ? 20 : 16} height={active ? 20 : 16} rx={4} />}
      <text x={node.node_type === "transaction" ? 14 : -14} y={3} textAnchor={node.node_type === "transaction" ? "start" : "end"}>{node.node_type === "transaction" ? node.transaction_id : node.source_field}</text>
    </g>; })}
  </svg></div>;
}

export default function AbuseRings() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [status, setStatus] = useState<Status | null>(null);
  const [config, setConfig] = useState<GraphConfig>(defaultConfig);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [graph, setGraph] = useState<ClusterGraph | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [transactionId, setTransactionId] = useState("");
  const [searchMessage, setSearchMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { const controller = new AbortController(); api<Status>("/api/v1/abuse-rings/status", { signal: controller.signal }).then(setStatus).catch((caught: unknown) => { if (caught instanceof DOMException && caught.name === "AbortError") return; setError(caught instanceof Error ? caught.message : "Linkage status could not be loaded"); }); return () => controller.abort(); }, []);

  async function loadCluster(cluster: Cluster, currentConfig = config) {
    setSelectedNode(null); setSearchMessage(null);
    const detail = await api<ClusterGraph>(`/api/v1/abuse-rings/clusters/${cluster.cluster_id}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ config: currentConfig }) });
    setGraph(detail);
  }

  async function analyzeValidation() {
    setBusy(true); setError(null); setSearchMessage(null); setGraph(null);
    try {
      const result = await api<Analysis>("/api/v1/abuse-rings/analyze-validation", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ config }) });
      setAnalysis(result);
      if (result.clusters.length) await loadCluster(result.clusters[0], result.config);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Validation linkage analysis failed"); }
    finally { setBusy(false); }
  }

  async function searchNeighborhood(event: FormEvent) {
    event.preventDefault(); setBusy(true); setError(null); setSelectedNode(null);
    try {
      const result = await api<Neighborhood>("/api/v1/abuse-rings/neighborhood", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ transaction_id: transactionId, config }) });
      setSearchMessage(result.message); setGraph(result.graph);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Transaction neighborhood could not be loaded"); }
    finally { setBusy(false); }
  }

  const selectedCluster = useMemo(() => analysis?.clusters.find((cluster) => cluster.cluster_id === graph?.cluster_id) ?? null, [analysis, graph]);
  return <div className="app-shell"><aside className={`sidebar ${mobileOpen ? "open" : ""}`}><div className="brand-row"><div className="brand-mark"><ShieldCheck size={21} /></div><div><strong>MerchantShield</strong><span>Merchant loss prevention</span></div><button className="icon-button close-nav" onClick={() => setMobileOpen(false)} aria-label="Close navigation"><X size={18} /></button></div><div className="candidate-badge"><span className="candidate-dot" /><div><strong>NetworkX linkage analysis</strong><small>Shared attributes · validation only</small></div></div><nav aria-label="MerchantShield modules"><span className="nav-label">Workspace</span><a className="suite-nav-link" href="/"><BarChart3 size={17} /><span>Overview</span></a><span className="nav-label nav-label-spaced">Fraud Risk</span><a className="suite-nav-link" href="/"><Gauge size={17} /><span>Risk Check</span></a><a className="suite-nav-link" href="/"><TableProperties size={17} /><span>Transactions</span></a><a className="suite-nav-link" href="/"><ListChecks size={17} /><span>Review Queue</span></a><a className="suite-nav-link" href="/"><SlidersHorizontal size={17} /><span>Cost Lab</span></a><span className="nav-label nav-label-spaced">Loss prevention</span><a className="suite-nav-link" href="/chargebacks"><FileText size={17} /><span>Chargebacks</span></a><a className="suite-nav-link" href="/fraud-pulse"><Activity size={17} /><span>Fraud Pulse</span></a><a className="suite-nav-link active" href="/abuse-rings"><Network size={17} /><span>Abuse Rings</span></a></nav><div className="sidebar-note"><LockKeyhole size={16} /><div><strong>No identity inference</strong><span>Dataset attributes do not establish a person, card number, or account.</span></div></div></aside>
    {mobileOpen && <button className="nav-scrim" onClick={() => setMobileOpen(false)} aria-label="Close navigation" />}
    <main className="main-area"><header className="topbar"><button className="icon-button menu-button" onClick={() => setMobileOpen(true)} aria-label="Open navigation"><Menu size={19} /></button><div className="breadcrumb"><span>MerchantShield</span><ArrowRight size={12} /><strong>Abuse Rings</strong></div><div className="top-status neutral-status"><span />{status?.evaluation_status.toUpperCase() ?? "LOADING LINKAGE"}</div></header><div className="page-content rings-page"><section className="module-page">
      <div className="compact-heading"><div><span className="eyebrow">ABUSE-RING SENTINEL</span><h1>Trace suspicious shared-attribute clusters.</h1><p>Explore validation transactions linked through eligible IEEE-CIS attributes. MerchantShield never treats a shared value as proof of identity or confirmed coordinated abuse.</p></div><div className="validation-pill"><Network size={14} /> NETWORKX · NO NEO4J</div></div>
      <div className="module-disclosure"><Info size={17} /><div><strong>Data source: {status?.data_source ?? "IEEE-CIS validation attributes and frozen validation probabilities"}</strong><p>Common values above the configured degree are suppressed to avoid weak giant components. Evaluation: <b>Not evaluated yet.</b></p></div></div>
      {error && <div className="error-state"><AlertTriangle size={17} /><div><strong>Linkage notice</strong><span>{error}</span></div><button onClick={() => setError(null)}>Dismiss</button></div>}
      <div className="rings-toolbar panel"><div className="ring-controls"><label><span>Maximum shared-attribute degree</span><input type="number" min={2} max={500} value={config.max_attribute_degree} onChange={(e) => setConfig({ ...config, max_attribute_degree: Number(e.target.value) })} /></label><label><span>Minimum high-risk share</span><input type="number" min={0} max={1} step={.05} value={config.minimum_high_risk_share} onChange={(e) => setConfig({ ...config, minimum_high_risk_share: Number(e.target.value) })} /><small>{percent(config.minimum_high_risk_share)}</small></label><button disabled={busy} onClick={analyzeValidation}><Play size={15} /> {busy ? "Building graph…" : "Analyze Validation Links"}</button></div><form className="ring-search" onSubmit={searchNeighborhood}><label><span>Transaction neighborhood</span><div><input required inputMode="numeric" pattern="[0-9]+" value={transactionId} onChange={(e) => setTransactionId(e.target.value)} placeholder="Enter validation TransactionID" /><button disabled={busy}><Search size={14} /> Search</button></div></label><p>Searches the actual chronological validation partition. It never opens held-out test rows.</p></form></div>
      {searchMessage && <div className="search-result-note"><Search size={14} /> {searchMessage}</div>}
      {!analysis && !graph && <section className="panel ring-empty"><Network size={34} /><strong>No graph fabricated</strong><p>Run the real validation linkage analysis or search a validation TransactionID to render a shared-attribute neighborhood.</p></section>}
      {analysis && <section className="ring-summary"><article><span>ROWS CONSIDERED</span><strong>{analysis.transaction_rows_considered.toLocaleString("en-IN")}</strong><small>chronological validation</small></article><article><span>LINKED TX NODES</span><strong>{analysis.transaction_nodes.toLocaleString("en-IN")}</strong><small>{analysis.edge_count.toLocaleString("en-IN")} attribute links</small></article><article><span>ATTRIBUTE NODES</span><strong>{analysis.shared_attribute_nodes.toLocaleString("en-IN")}</strong><small>{analysis.suppressed_attribute_values} common values suppressed</small></article><article><span>SUSPICIOUS CLUSTERS</span><strong>{analysis.suspicious_cluster_count}</strong><small>configuration-dependent · not confirmed</small></article></section>}
      {(graph || analysis) && <div className="rings-workspace"><section className="panel cluster-list"><div className="panel-title"><div><span>RANKED COMPONENTS</span><h2>Suspicious linked clusters</h2></div><Network size={18} /></div>{analysis?.clusters.map((cluster, index) => <button key={cluster.cluster_id} className={graph?.cluster_id === cluster.cluster_id ? "active" : ""} onClick={() => loadCluster(cluster)}><span><i>{index + 1}</i><strong>{cluster.label}</strong><small>{cluster.transaction_count} transactions · {cluster.shared_attribute_count} shared attributes</small></span><span><b>{percent(cluster.high_risk_share)}</b><small>{currency(cluster.high_risk_amount)} high-risk amount</small></span></button>) ?? <div className="search-only"><Search size={18} /> Search result neighborhood</div>}</section><section className="panel graph-panel"><div className="panel-title"><div><span>LIGHTWEIGHT BIPARTITE VIEW</span><h2>{selectedCluster ? `${selectedCluster.transaction_count} transactions connected by shared attributes` : "Transaction neighborhood"}</h2></div>{graph && <span className="graph-count">{graph.total_nodes} nodes · {graph.total_edges} edges</span>}</div>{busy && <div className="graph-loading">Building the eligible NetworkX component…</div>}{graph && !busy && <><GraphCanvas graph={graph} selected={selectedNode} onSelect={setSelectedNode} /><div className="graph-legend"><span><i className="approve" /> Transaction · approve</span><span><i className="review" /> Transaction · review</span><span><i className="block" /> Transaction · block</span><span><i className="attribute" /> Shared attribute</span></div><div className="graph-node-picker" aria-label="Graph node picker">{graph.nodes.slice(0, 12).map((node) => <button className={selectedNode?.id === node.id ? "active" : ""} key={node.id} onClick={() => setSelectedNode(node)}>{node.label}</button>)}</div>{graph.graph_truncated && <p className="graph-note"><Info size={12} /> Display is capped to keep the neighborhood legible; cluster statistics use the complete component.</p>}</>}</section><aside className="panel node-inspector"><div className="panel-title"><div><span>NODE INSPECTOR</span><h2>{selectedNode?.label ?? "Select a node"}</h2></div><CircleDot size={18} /></div>{!selectedNode && <p>Choose a graph node or its label button. Attribute labels retain their dataset source names.</p>}{selectedNode?.node_type === "transaction" && <dl><div><dt>TransactionID</dt><dd>{selectedNode.transaction_id}</dd></div><div><dt>Fraud probability</dt><dd>{percent(selectedNode.risk_score ?? 0)}</dd></div><div><dt>Decision</dt><dd>{selectedNode.decision}</dd></div><div><dt>Amount</dt><dd>{currency(selectedNode.amount ?? 0)}</dd></div></dl>}{selectedNode?.node_type === "shared_attribute" && <dl><div><dt>Source field</dt><dd>{selectedNode.source_field}</dd></div><div><dt>Documented label</dt><dd>{selectedNode.attribute_type}</dd></div><div><dt>Shared value</dt><dd>{selectedNode.attribute_value}</dd></div><div><dt>Relationship</dt><dd>SHARES ATTRIBUTE</dd></div></dl>}<p className="inspector-limit"><AlertTriangle size={12} /> A link does not establish common ownership, identity, or confirmed abuse.</p></aside></div>}
      <section className="attribute-policy panel"><div className="panel-title"><div><span>ATTRIBUTE POLICY</span><h2>Exactly what can create a link</h2></div><LockKeyhole size={18} /></div><div>{(status?.considered_attributes ?? [{ source_field: "card4", documented_label: "card4 shared network attribute" }, { source_field: "card6", documented_label: "card6 shared card-type attribute" }, { source_field: "P_emaildomain", documented_label: "purchaser email-domain attribute" }, { source_field: "R_emaildomain", documented_label: "recipient email-domain attribute" }, { source_field: "DeviceType", documented_label: "device-type attribute" }, { source_field: "DeviceInfo", documented_label: "dataset-provided device-info attribute" }]).map((item) => <span key={item.source_field}><b>{item.source_field}</b><small>{item.documented_label}</small></span>)}</div><p>MerchantShield never renames these fields as card numbers, accounts, customers, or real-world identities.</p></section>
      <div className="module-limitations"><AlertTriangle size={16} /><div><strong>Interpretation boundary</strong><p>Outputs are suspicious linked clusters, not confirmed fraud rings. Common values are suppressed, but remaining shared attributes can still be coincidental. Validation results are not held-out graph-detector metrics.</p></div></div>
    </section></div></main></div>;
}
