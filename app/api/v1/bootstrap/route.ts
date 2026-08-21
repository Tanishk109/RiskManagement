import { unevaluatedDashboardState } from "../../../../lib/dashboard-state";

export async function GET() {
  return Response.json(unevaluatedDashboardState(), {
    headers: { "Cache-Control": "no-store" },
  });
}
