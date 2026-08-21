from __future__ import annotations


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_are_honest_when_artifact_is_absent(client):
    response = client.get("/api/v1/metrics/summary")
    assert response.status_code == 200
    assert response.json()["evaluated"] is False
    assert response.json()["metrics"] is None


def test_transaction_listing_and_detail(client, seeded_review):
    listing = client.get("/api/v1/transactions", params={"decision": "REVIEW"})
    assert listing.status_code == 200
    assert len(listing.json()["items"]) == 1
    detail = client.get("/api/v1/transactions/fixture-review-1")
    assert detail.status_code == 200
    assert detail.json()["actual_label"] == 1
    assert detail.json()["top_factors"][0]["feature_name"] == "V17"


def test_review_submission_is_persisted(client, seeded_review):
    response = client.post(
        f"/api/v1/reviews/{seeded_review}/decision",
        json={"decision": "DECLINE", "reason": "Confirmed mismatch after manual evidence review."},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "DECIDED"
    assert response.json()["reviewer_decision"] == "DECLINE"
    repeat = client.post(
        f"/api/v1/reviews/{seeded_review}/decision",
        json={"decision": "APPROVE", "reason": "Second decision must not replace the first."},
    )
    assert repeat.status_code == 409


def test_cost_simulation_rejects_invalid_threshold_order(client):
    response = client.post(
        "/api/v1/cost/simulate",
        json={"review_threshold": 0.8, "block_threshold": 0.4, "assumptions": {}},
    )
    assert response.status_code == 422


def test_cost_simulation_does_not_invent_results(client):
    response = client.post(
        "/api/v1/cost/simulate",
        json={"review_threshold": 0.4, "block_threshold": 0.8, "assumptions": {}},
    )
    assert response.status_code == 200
    assert response.json()["evaluated"] is False
    assert response.json()["proposed"] is None


def test_score_requires_frozen_real_model(client):
    response = client.post("/api/v1/score", json={"features": {"TransactionAmt": 100}, "persist": False})
    assert response.status_code == 503


def test_score_rejects_ground_truth_as_feature(client):
    response = client.post("/api/v1/score", json={"features": {"TransactionAmt": 100, "isFraud": 1}})
    assert response.status_code == 422
