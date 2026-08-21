import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
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

test("server-renders the MerchantShield command center", async () => {
  const app = await worker();
  const response = await app.fetch(new Request("http://localhost/", { headers: { accept: "text/html" } }), env, ctx);
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /MerchantShield/);
  assert.match(html, /Risk command center/);
  assert.match(html, /Defense-only payment fraud operations/);
  assert.match(html, /Live transactions/);
  assert.match(html, /Abuse ring sentinel/);
  assert.match(html, /False-positive cost/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/i);
});

test("scores a suspicious payment through the API", async () => {
  const app = await worker();
  const response = await app.fetch(new Request("http://localhost/api/score", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      amount: 28000,
      customerAgeDays: 2,
      newDevice: true,
      transactionsLast5Min: 6,
      accountsSharingDevice: 9,
      historicalChargebacks: 1,
      failedAttemptsLastHour: 5,
      amountVsBaseline: 4.2,
    }),
  }), env, ctx);
  assert.equal(response.status, 200);
  const result = await response.json();
  assert.equal(result.decision, "BLOCK");
  assert.ok(result.score >= 70);
  assert.ok(result.reasons.length >= 2);
  assert.match(result.modelVersion, /Fraud-XGB/);
});

test("removes all temporary starter assets", async () => {
  const [page, layout, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);
  assert.doesNotMatch(page, /SkeletonPreview|codex-preview/);
  assert.match(layout, /MerchantShield AI/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  await assert.rejects(access(new URL("../app/_sites-preview/SkeletonPreview.tsx", import.meta.url)));
});
