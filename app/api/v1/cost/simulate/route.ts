type RequestBody = {
  review_threshold?: number;
  block_threshold?: number;
  assumptions?: Record<string, unknown>;
};

function isProbability(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
}

export async function POST(request: Request) {
  let body: RequestBody;
  try {
    body = await request.json() as RequestBody;
  } catch {
    return Response.json({ error: "Request body must be valid JSON." }, { status: 400 });
  }

  if (!isProbability(body.review_threshold) || !isProbability(body.block_threshold)) {
    return Response.json({ error: "Thresholds must be numbers between 0 and 1." }, { status: 422 });
  }
  if (body.review_threshold >= body.block_threshold) {
    return Response.json({ error: "review_threshold must be less than block_threshold." }, { status: 422 });
  }

  return Response.json({
    evaluated: false,
    provenance: "Not evaluated yet — cost simulation requires held-out IEEE-CIS labels, scores, and transaction amounts.",
    current: null,
    proposed: null,
  });
}
