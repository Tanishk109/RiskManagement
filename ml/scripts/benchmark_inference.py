from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
from common import ARTIFACTS, load_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if args.batch_size < 1 or args.iterations < 2:
        raise ValueError("batch-size must be positive and iterations must be at least 2")

    bundle = joblib.load(ARTIFACTS / "models/model.joblib")
    features = list(bundle["feature_names"])
    test = load_splits(features).test
    batch = test[features].head(args.batch_size)
    if len(batch) < args.batch_size:
        raise ValueError("Requested batch size exceeds the held-out test rows")
    pipeline = bundle["pipeline"]
    pipeline.predict_proba(batch)
    timings: list[float] = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        pipeline.predict_proba(batch)
        timings.append((time.perf_counter() - started) * 1000)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "median_latency_ms": float(np.median(timings)),
        "p95_latency_ms": float(np.percentile(timings, 95)),
        "batch_size": args.batch_size,
        "iterations": args.iterations,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "claim": "Offline benchmark only; not a production SLA.",
    }
    path = ARTIFACTS / "metrics/inference_benchmark.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
