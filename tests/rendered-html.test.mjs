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

test("server-renders the honest four-module MerchantShield surface", async () => {
  const app = await worker();
  const response = await app.fetch(new Request("http://localhost/", { headers: { accept: "text/html" } }), env, ctx);
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /MerchantShield/);
  assert.match(html, /Cost-Aware Fraud/);
  assert.match(html, /Loading project evidence/);
  assert.match(html, /Overview/);
  assert.match(html, /Transactions/);
  assert.match(html, /Review Queue/);
  assert.match(html, /Cost Lab/);
  assert.match(html, /Held-out test sealed/);
  assert.doesNotMatch(html, /Abuse ring sentinel|Fraud pulse|AI analyst|Live transactions|₹5\.80L|Fraud-XGB-v3\.2/);
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

test("cost adapter validates threshold order and withholds unevaluated output", async () => {
  const app = await worker();
  const invalid = await app.fetch(new Request("http://localhost/api/v1/cost/simulate", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ review_threshold: 0.8, block_threshold: 0.4, assumptions: {} }),
  }), env, ctx);
  assert.equal(invalid.status, 422);

  const valid = await app.fetch(new Request("http://localhost/api/v1/cost/simulate", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ review_threshold: 0.4, block_threshold: 0.8, assumptions: {} }),
  }), env, ctx);
  assert.equal(valid.status, 200);
  const result = await valid.json();
  assert.equal(result.evaluated, false);
  assert.equal(result.proposed, null);
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

test("locked product surfaces contain no fake cases or monetary claims", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.match(page, /No fake review cases are shown/);
  assert.match(page, /No monetary result shown/);
  assert.match(page, /No rupee estimate is displayed here/);
  assert.doesNotMatch(page, /₹|reviewCases|estimatedSavings|moneySaved/);
});
