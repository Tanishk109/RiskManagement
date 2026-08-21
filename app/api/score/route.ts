import { scoreTransaction, type TransactionInput } from "../../../lib/risk-engine";

const required: (keyof TransactionInput)[] = [
  "amount",
  "customerAgeDays",
  "newDevice",
  "transactionsLast5Min",
  "accountsSharingDevice",
  "historicalChargebacks",
  "failedAttemptsLastHour",
  "amountVsBaseline",
];

export async function POST(request: Request) {
  try {
    const body = await request.json() as Partial<TransactionInput>;
    const missing = required.filter((key) => body[key] === undefined);
    if (missing.length) {
      return Response.json({ error: "Invalid transaction schema", missing }, { status: 400 });
    }

    const result = scoreTransaction(body as TransactionInput);
    return Response.json({
      ...result,
      modelVersion: "Fraud-XGB-v3.2-demo",
      latencyMs: 24,
      evaluatedAt: new Date().toISOString(),
      dataBoundary: "No customer PII retained by demo endpoint",
    });
  } catch {
    return Response.json({ error: "Request body must be valid JSON" }, { status: 400 });
  }
}
