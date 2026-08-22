from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_attribute_degree: int = Field(default=2, ge=2, le=20)
    max_attribute_degree: int = Field(default=50, ge=2, le=500)
    minimum_cluster_transactions: int = Field(default=2, ge=2, le=100)
    minimum_high_risk_transactions: int = Field(default=1, ge=1, le=100)
    minimum_high_risk_share: float = Field(default=0.25, ge=0, le=1)
    max_clusters: int = Field(default=25, ge=1, le=100)


class GraphAnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: GraphConfig = Field(default_factory=GraphConfig)


class ClusterSummary(BaseModel):
    cluster_id: str
    transaction_count: int
    shared_attribute_count: int
    edge_count: int
    connectivity_score: float
    high_risk_count: int
    high_risk_share: float
    average_risk_score: float
    maximum_risk_score: float
    total_transaction_amount: float
    high_risk_amount: float
    shared_attribute_types: list[str]
    example_transaction_ids: list[str]
    label: Literal["SUSPICIOUS LINKED CLUSTER"]


class GraphNode(BaseModel):
    id: str
    node_type: Literal["transaction", "shared_attribute"]
    label: str
    transaction_id: str | None = None
    risk_score: float | None = None
    amount: float | None = None
    decision: Literal["APPROVE", "REVIEW", "BLOCK"] | None = None
    source_field: str | None = None
    attribute_type: str | None = None
    attribute_value: str | None = None


class GraphEdge(BaseModel):
    source: str
    target: str
    relationship: Literal["SHARES ATTRIBUTE"]


class ClusterGraph(BaseModel):
    cluster_id: str
    label: Literal["SUSPICIOUS LINKED CLUSTER"]
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    total_nodes: int
    total_edges: int
    graph_truncated: bool
    limitation: str


class GraphAnalysisResponse(BaseModel):
    source: Literal["IEEE-CIS chronological validation transactions and frozen CatBoost validation probabilities"]
    data_partition: Literal["validation"]
    evaluation_status: Literal["Not evaluated yet"]
    model_version: str
    review_threshold: float
    block_threshold: float
    config: GraphConfig
    transaction_rows_considered: int
    transaction_nodes: int
    shared_attribute_nodes: int
    edge_count: int
    connected_components: int
    suspicious_cluster_count: int
    returned_cluster_count: int
    suppressed_attribute_values: int
    clusters: list[ClusterSummary]
    held_out_test_accessed: Literal[False]
    confirmed_fraud_ring_claimed: Literal[False]
    limitations: list[str]


class NeighborhoodRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(min_length=1, max_length=100)
    config: GraphConfig = Field(default_factory=GraphConfig)


class NeighborhoodResponse(BaseModel):
    transaction_id: str
    found_in_validation: bool
    connected_through_eligible_attributes: bool
    graph: ClusterGraph | None
    message: str
    held_out_test_accessed: Literal[False]


class GraphStatus(BaseModel):
    module: Literal["Abuse-Ring Sentinel"]
    data_source: str
    evaluation_status: Literal["Not evaluated yet"]
    model_version: str
    review_threshold: float
    block_threshold: float
    considered_attributes: list[dict[str, str]]
    default_common_value_suppression: str
    terminology: list[str]
    limitations: list[str]
