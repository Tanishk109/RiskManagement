import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function worker() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}-${Math.random()}`);
  return (await import(workerUrl.href)).default;
}

const env = {
  ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) },
};
const ctx = { waitUntil() {}, passThroughOnException() {} };

test("server-renders the honest five-module MerchantShield surface", async () => {
  const app = await worker();
  const response = await app.fetch(new Request("http://localhost/", { headers: { accept: "text/html" } }), env, ctx);
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /MerchantShield/);
  assert.match(html, /Cost-Aware Fraud/);
  assert.match(html, /Loading project evidence/);
  assert.match(html, /Overview/);
  assert.match(html, /Risk Check/);
  assert.match(html, /Transactions/);
  assert.match(html, /Review Queue/);
  assert.match(html, /Cost Lab/);
  assert.match(html, /Held-out test sealed/);
  assert.doesNotMatch(html, /Abuse ring sentinel|Fraud pulse|AI analyst|Live transactions|₹5\.80L|Fraud-XGB-v3\.2/);
});

test("Risk Check exposes all requested input modes and the exact frozen schema", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const riskCheck = await readFile(new URL("../components/risk-check.tsx", import.meta.url), "utf8");
  assert.match(page, /Risk Check/);
  assert.match(page, /active === "risk"/);
  assert.match(riskCheck, /Single Transaction/);
  assert.match(riskCheck, /Load Validation Transaction/);
  assert.match(riskCheck, /Batch CSV Upload/);
  for (const feature of ["TransactionAmt", "ProductCD", "card4", "card6", "P_emaildomain", "C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3"]) {
    assert.match(riskCheck, new RegExp(`"${feature}"`));
  }
  assert.match(riskCheck, /Run Risk Check/);
  assert.match(riskCheck, /\/api\/v1\/score/);
});

test("validation loading is label-safe and ground truth requires an explicit reveal", async () => {
  const riskCheck = await readFile(new URL("../components/risk-check.tsx", import.meta.url), "utf8");
  assert.match(riskCheck, /Ground truth stays out of the response/);
  assert.match(riskCheck, /Ground truth hidden/);
  assert.match(riskCheck, /Reveal Ground Truth/);
  assert.match(riskCheck, /result && loadedTransaction/);
  assert.match(riskCheck, /ground-truth/);
  assert.match(riskCheck, /This label was not a scoring input/);
});

test("batch upload validates limits, supports export, and never claims persistence", async () => {
  const riskCheck = await readFile(new URL("../components/risk-check.tsx", import.meta.url), "utf8");
  assert.match(riskCheck, /Maximum 1 MB/);
  assert.match(riskCheck, /1,000 data rows/);
  assert.match(riskCheck, /isFraud are forbidden/);
  assert.match(riskCheck, /not stored permanently/);
  assert.match(riskCheck, /Export scored CSV/);
  assert.match(riskCheck, /rows_processed/);
});

test("anonymized fields are disclosed without invented explanations", async () => {
  const riskCheck = await readFile(new URL("../components/risk-check.tsx", import.meta.url), "utf8");
  assert.match(riskCheck, /Advanced IEEE-CIS Fields/);
  assert.match(riskCheck, /anonymized competition features/);
  assert.match(riskCheck, /does not assign or invent semantic meanings/);
  assert.match(riskCheck, /No explanation is shown/);
  assert.match(riskCheck, /held-out test sealed/i);
  assert.doesNotMatch(riskCheck, /C1 — .*velocity|C2 — .*chargeback|D1 — .*account age/i);
});

test("bootstrap adapter exposes null evidence instead of fake metrics", async () => {
  const app = await worker();
  const response = await app.fetch(new Request("http://localhost/api/v1/bootstrap"), env, ctx);
  assert.equal(response.status, 200);
  const result = await response.json();
  assert.equal(result.evaluated, false);
  assert.equal(result.metrics, null);
  assert.deepEqual(result.transactions, []);
  assert.match(result.provenance, /Not evaluated yet/);
});

test("cost adapter validates the new request and never fabricates a fallback result", async () => {
  const app = await worker();
  const invalid = await app.fetch(new Request("http://localhost/api/v1/cost/simulate", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ scenario_id: "moderate", review_threshold: 0.8, block_threshold: 0.4 }),
  }), env, ctx);
  assert.equal(invalid.status, 422);

  const valid = await app.fetch(new Request("http://localhost/api/v1/cost/simulate", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ scenario_id: "moderate", review_threshold: 0.4, block_threshold: 0.8 }),
  }), env, ctx);
  assert.equal(valid.status, 503);
  const result = await valid.json();
  assert.match(result.detail, /No fallback result was fabricated/);
});

test("score adapter refuses to substitute a handcrafted model", async () => {
  const app = await worker();
  const response = await app.fetch(new Request("http://localhost/api/v1/score", { method: "POST" }), env, ctx);
  assert.equal(response.status, 503);
  assert.match((await response.json()).detail, /does not substitute/);
});

test("frontend source contains no former fabricated evaluation constants", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(page, /HELD_OUT_RESULTS|metricsFor\(|82,491|Fraud-XGB-v3\.2|₹5\.80L/);
  assert.doesNotMatch(page, /0\.426003|0\.231434|590,?540|20,?663|378,?333\.733/);
  assert.match(page, /Not evaluated yet/);
});

test("frontend clearly separates validation evidence from final results", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.match(page, /VALIDATION RESULTS/);
  assert.match(page, /Not final held-out performance/);
  assert.match(page, /FINAL HELD-OUT EVALUATION/);
  assert.match(page, /Held-out test remains sealed/);
  assert.match(page, /Not evaluated yet/);
});

test("active validation product surfaces keep costs estimated and final results sealed", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.match(page, /Reveal Ground Truth/);
  assert.match(page, /VALIDATION SIMULATION READY/);
  assert.match(page, /VALIDATION REVIEW SIMULATION/);
  assert.match(page, /Use Provisional Validation Configuration/);
  assert.match(page, /Use Lowest-Cost Feasible Configuration/);
  assert.match(page, /ILLUSTRATIVE|assumption_status|Estimated—not realized/);
  assert.match(page, /FRAUD COUNT DETECTION/);
  assert.match(page, /FRAUD AMOUNT CAPTURE/);
  assert.doesNotMatch(page, /moneySaved|guaranteed savings/i);
});

test("Chargebacks route renders an honest evidence responder", async () => {
  const app = await worker();
  const response = await app.fetch(new Request("http://localhost/chargebacks", { headers: { accept: "text/html" } }), env, ctx);
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /Chargeback Evidence Responder/i);
  assert.match(html, /Not evaluated yet/i);
  assert.match(html, /win probability/i);
  assert.match(html, /human approval required/i);
});

test("Chargeback workflow accepts real input, labels files, and has no auto-submit path", async () => {
  const source = await readFile(new URL("../components/chargebacks.tsx", import.meta.url), "utf8");
  for (const field of ["Dispute ID", "Transaction ID", "Amount", "Deadline", "Customer name", "Order ID", "Carrier", "Merchant notes"]) {
    assert.match(source, new RegExp(field));
  }
  for (const category of ["invoice", "proof_of_delivery", "tracking", "customer_communication", "refund_evidence", "merchant_policy", "other"]) {
    assert.match(source, new RegExp(category));
  }
  assert.match(source, /Create evidence case/);
  assert.match(source, /Generate evidence-grounded draft/);
  assert.match(source, /Approve for export/);
  assert.match(source, /No automatic submission/);
  assert.doesNotMatch(source, /win_probability|success probability|auto-submit/i);
});
