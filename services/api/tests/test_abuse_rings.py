from __future__ import annotations

from pathlib import Path

import pytest
from app.database import Base
from app.schemas.abuse_rings import GraphConfig
from app.services.abuse_graph import (
    AbuseGraphPaths,
    AbuseGraphService,
    get_abuse_graph_service,
)
from app.services.artifacts import ArtifactUnavailable
from app.services.validation_scoring import get_validation_scoring_service


def test_status_documents_attributes_and_forbids_identity_claims(client):
    response = client.get("/api/v1/abuse-rings/status")
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["evaluation_status"] == "Not evaluated yet"
    assert {item["source_field"] for item in result["considered_attributes"]} == {
        "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"
    }
    assert "suspicious linked cluster" in result["terminology"]
    joined = " ".join(result["limitations"]).lower()
    assert "card numbers" in joined
    assert "not confirmed fraud rings" in joined


def test_real_validation_graph_returns_suspicious_components_without_confirmation(client):
    response = client.post("/api/v1/abuse-rings/analyze-validation", json={"config": {}})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["transaction_rows_considered"] == 88581
    assert result["data_partition"] == "validation"
    assert result["transaction_nodes"] > 0
    assert result["shared_attribute_nodes"] > 0
    assert result["suppressed_attribute_values"] > 0
    assert result["suspicious_cluster_count"] >= result["returned_cluster_count"] > 0
    assert result["confirmed_fraud_ring_claimed"] is False
    assert result["held_out_test_accessed"] is False
    assert all(item["label"] == "SUSPICIOUS LINKED CLUSTER" for item in result["clusters"])


def test_cluster_graph_is_bipartite_and_uses_shared_attribute_language(client):
    analysis = client.post("/api/v1/abuse-rings/analyze-validation", json={"config": {}}).json()
    cluster_id = analysis["clusters"][0]["cluster_id"]
    response = client.post(f"/api/v1/abuse-rings/clusters/{cluster_id}", json={"config": {}})
    assert response.status_code == 200, response.text
    result = response.json()
    node_types = {node["node_type"] for node in result["nodes"]}
    assert node_types == {"transaction", "shared_attribute"}
    assert all(edge["relationship"] == "SHARES ATTRIBUTE" for edge in result["edges"])
    attribute_nodes = [node for node in result["nodes"] if node["node_type"] == "shared_attribute"]
    assert all(node["attribute_type"] and node["source_field"] for node in attribute_nodes)


def test_transaction_search_returns_actual_validation_neighborhood(client):
    response = client.post(
        "/api/v1/abuse-rings/neighborhood",
        json={"transaction_id": "3434106", "config": {}},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["found_in_validation"] is True
    assert result["connected_through_eligible_attributes"] is True
    transaction_ids = {
        node["transaction_id"] for node in result["graph"]["nodes"]
        if node["node_type"] == "transaction"
    }
    assert "3434106" in transaction_ids


def test_missing_transaction_search_is_honest(client):
    response = client.post(
        "/api/v1/abuse-rings/neighborhood",
        json={"transaction_id": "1", "config": {}},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["found_in_validation"] is False
    assert result["graph"] is None


def test_held_out_graph_paths_are_rejected_before_read():
    service = AbuseGraphService(
        AbuseGraphPaths(
            validation_predictions=Path("artifacts/predictions/catboost_heldout_predictions.parquet"),
            validation_data=Path("data/processed/ieee-cis/validation.parquet"),
        ),
        get_validation_scoring_service(),
    )
    with pytest.raises(ArtifactUnavailable, match="held-out test"):
        _ = service.validation_frame


def test_abuse_graph_operational_schema_is_normalized():
    run_columns = Base.metadata.tables["abuse_graph_runs"].columns
    cluster_columns = Base.metadata.tables["abuse_cluster_records"].columns
    assert "run_id" in cluster_columns
    assert "confirmed_fraud_ring_claimed" in run_columns
    assert "raw_graph" not in run_columns


def test_networkx_is_the_only_graph_runtime():
    import networkx

    assert tuple(int(part) for part in networkx.__version__.split(".")[:1]) >= (3,)
    assert GraphConfig().max_attribute_degree == 50
    assert get_abuse_graph_service().__class__.__name__ == "AbuseGraphService"
