export async function POST() {
  return Response.json({
    error: "Model not available",
    detail: "Train and freeze the IEEE-CIS model before requesting a score. MerchantShield does not substitute a handcrafted demo score.",
  }, { status: 503 });
}
