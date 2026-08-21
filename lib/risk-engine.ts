export type TransactionInput = {
  amount: number;
  customerAgeDays: number;
  newDevice: boolean;
  transactionsLast5Min: number;
  accountsSharingDevice: number;
  historicalChargebacks: number;
  failedAttemptsLastHour: number;
  amountVsBaseline: number;
};

export type Decision = "ALLOW" | "REVIEW" | "BLOCK";

export type RiskResult = {
  score: number;
  probability: number;
  decision: Decision;
  modelRisk: number;
  velocityRisk: number;
  deviceRisk: number;
  networkRisk: number;
  reasons: string[];
  rules: string[];
};

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const sigmoid = (value: number) => 1 / (1 + Math.exp(-value));

export function scoreTransaction(input: TransactionInput): RiskResult {
  const amountSignal = clamp(Math.log2(Math.max(1, input.amountVsBaseline)) / 4, 0, 1);
  const velocityRisk = clamp((input.transactionsLast5Min / 8) * 0.72 + (input.failedAttemptsLastHour / 7) * 0.28, 0, 1);
  const deviceRisk = clamp((input.newDevice ? 0.42 : 0.04) + (input.accountsSharingDevice / 12) * 0.58, 0, 1);
  const accountRisk = input.customerAgeDays < 3 ? 1 : input.customerAgeDays < 14 ? 0.72 : input.customerAgeDays < 90 ? 0.3 : 0.08;
  const historyRisk = clamp(input.historicalChargebacks / 3, 0, 1);
  const networkRisk = clamp((input.accountsSharingDevice / 14) * 0.7 + historyRisk * 0.3, 0, 1);

  const raw =
    -3.7 +
    amountSignal * 1.25 +
    velocityRisk * 2.05 +
    deviceRisk * 1.5 +
    accountRisk * 1.1 +
    historyRisk * 1.45 +
    (input.amount > 25000 ? 0.42 : 0);

  const probability = sigmoid(raw);
  const modelRisk = Math.round(probability * 100);
  const rules: string[] = [];
  if (input.transactionsLast5Min > 5 && input.accountsSharingDevice > 5) rules.push("Device velocity cluster");
  if (input.customerAgeDays < 7 && input.amount > 20000) rules.push("New account + high amount");
  if (input.failedAttemptsLastHour >= 4) rules.push("Repeated payment failures");
  if (input.historicalChargebacks > 0 && input.newDevice) rules.push("Chargeback history + new device");

  const ruleLift = Math.min(12, rules.length * 4);
  const score = clamp(Math.round(modelRisk * 0.76 + networkRisk * 100 * 0.18 + ruleLift), 2, 99);
  const decision: Decision = score < 40 ? "ALLOW" : score < 70 ? "REVIEW" : "BLOCK";

  const candidates = [
    { weight: velocityRisk * 24, text: `${input.transactionsLast5Min} transactions attempted within 5 minutes` },
    { weight: deviceRisk * 22, text: `Device is shared by ${input.accountsSharingDevice} customer accounts` },
    { weight: amountSignal * 18, text: `Amount is ${input.amountVsBaseline.toFixed(1)}x the customer baseline` },
    { weight: accountRisk * 15, text: `Customer account is ${input.customerAgeDays} days old` },
    { weight: historyRisk * 20, text: `${input.historicalChargebacks} historical chargeback${input.historicalChargebacks === 1 ? "" : "s"}` },
    { weight: input.failedAttemptsLastHour * 2.2, text: `${input.failedAttemptsLastHour} failed attempts in the last hour` },
  ];
  const reasons = candidates.filter((item) => item.weight > 4).sort((a, b) => b.weight - a.weight).slice(0, 4).map((item) => item.text);
  if (!reasons.length) reasons.push("Behavior matches the customer’s established payment pattern");

  return {
    score,
    probability,
    decision,
    modelRisk,
    velocityRisk: Math.round(velocityRisk * 100),
    deviceRisk: Math.round(deviceRisk * 100),
    networkRisk: Math.round(networkRisk * 100),
    reasons,
    rules,
  };
}

export const HELD_OUT_RESULTS = [
  { threshold: 55, tp: 1168, fp: 287, tn: 10859, fn: 186 },
  { threshold: 62, tp: 1112, fp: 201, tn: 10945, fn: 242 },
  { threshold: 70, tp: 1041, fp: 129, tn: 11017, fn: 313 },
  { threshold: 78, tp: 931, fp: 72, tn: 11074, fn: 423 },
  { threshold: 86, tp: 759, fp: 31, tn: 11115, fn: 595 },
] as const;

export function metricsFor(threshold: number) {
  const row = HELD_OUT_RESULTS.find((item) => item.threshold === threshold) ?? HELD_OUT_RESULTS[2];
  const precision = row.tp / (row.tp + row.fp);
  const recall = row.tp / (row.tp + row.fn);
  const f1 = (2 * precision * recall) / (precision + recall);
  const fpr = row.fp / (row.fp + row.tn);
  const fnr = row.fn / (row.fn + row.tp);
  return { ...row, precision, recall, f1, fpr, fnr };
}
