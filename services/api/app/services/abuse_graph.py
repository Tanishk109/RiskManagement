from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from ..config import get_settings
from ..schemas.abuse_rings import (
    ClusterGraph,
    ClusterSummary,
    GraphAnalysisResponse,
    GraphConfig,
    GraphEdge,
    GraphNode,
    NeighborhoodResponse,
)
from .artifacts import ArtifactUnavailable
from .decision_engine import decision_from_score
from .fraud_pulse import _assert_validation_path
from .validation_scoring import ValidationScoringService, get_validation_scoring_service

ATTRIBUTE_FIELDS: dict[str, str] = {
    "card4": "card4 shared network attribute",
    "card6": "card6 shared card-type attribute",
    "P_emaildomain": "purchaser email-domain attribute",
    "R_emaildomain": "recipient email-domain attribute",
    "DeviceType": "device-type attribute",
    "DeviceInfo": "dataset-provided device-info attribute",
}


@dataclass(frozen=True)
class AbuseGraphPaths:
    validation_predictions: Path
    validation_data: Path


def configured_abuse_graph_paths() -> AbuseGraphPaths:
    settings = get_settings()
    return AbuseGraphPaths(
        validation_predictions=settings.catboost_validation_predictions_path,
        validation_data=settings.validation_data_path,
    )


def _clean_attribute(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    cleaned = str(value).strip().casefold()
    return cleaned or None


def _cluster_id(transaction_ids: list[str]) -> str:
    canonical = "|".join(sorted(transaction_ids))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


class AbuseGraphService:
    """Validation-only defensive linkage analysis using NetworkX bipartite graphs."""

    def __init__(self, paths: AbuseGraphPaths, scoring: ValidationScoringService):
        self.paths = paths
        self.scoring = scoring
        self._graph_cache: dict[tuple[int, int], tuple[nx.Graph, int]] = {}

    @cached_property
    def validation_frame(self) -> pd.DataFrame:
        _assert_validation_path(self.paths.validation_predictions, "Linkage predictions")
        _assert_validation_path(self.paths.validation_data, "Linkage transactions")
        if not self.paths.validation_predictions.is_file() or not self.paths.validation_data.is_file():
            raise ArtifactUnavailable("Validation linkage artifacts are unavailable")
        transaction_columns = ["TransactionID", "TransactionAmt", *ATTRIBUTE_FIELDS]
        transactions = pd.read_parquet(self.paths.validation_data, columns=transaction_columns)
        predictions = pd.read_parquet(
            self.paths.validation_predictions,
            columns=["TransactionID", "fraud_probability", "model_version"],
        )
        if transactions["TransactionID"].duplicated().any() or predictions["TransactionID"].duplicated().any():
            raise ArtifactUnavailable("Linkage analysis requires unique TransactionID values")
        joined = transactions.merge(predictions, on="TransactionID", how="inner", validate="one_to_one")
        if len(joined) != len(transactions) or len(joined) != len(predictions):
            raise ArtifactUnavailable("Validation linkage score join is incomplete")
        versions = joined["model_version"].dropna().astype(str).unique().tolist()
        if versions != [str(self.scoring.metadata["model_version"])]:
            raise ArtifactUnavailable("Linkage scores do not match the frozen model version")
        return joined

    def build_graph(self, config: GraphConfig) -> tuple[nx.Graph, int]:
        cache_key = (config.min_attribute_degree, config.max_attribute_degree)
        if cache_key in self._graph_cache:
            return self._graph_cache[cache_key]
        frame = self.validation_frame
        eligible: dict[str, set[int]] = {}
        suppressed = 0
        for field in ATTRIBUTE_FIELDS:
            cleaned = frame[field].map(_clean_attribute)
            counts = cleaned.value_counts(dropna=True)
            for value, count in counts.items():
                if config.min_attribute_degree <= int(count) <= config.max_attribute_degree:
                    eligible[f"attribute::{field}::{value}"] = set(cleaned.index[cleaned == value].tolist())
                elif int(count) > config.max_attribute_degree:
                    suppressed += 1

        graph = nx.Graph()
        for attribute_node, row_indices in eligible.items():
            _, field, value = attribute_node.split("::", 2)
            graph.add_node(
                attribute_node,
                node_type="shared_attribute",
                source_field=field,
                attribute_type=ATTRIBUTE_FIELDS[field],
                attribute_value=value,
            )
            for row_index in row_indices:
                row = frame.loc[row_index]
                transaction_id = str(int(row["TransactionID"]))
                transaction_node = f"transaction::{transaction_id}"
                if transaction_node not in graph:
                    graph.add_node(
                        transaction_node,
                        node_type="transaction",
                        transaction_id=transaction_id,
                        risk_score=float(row["fraud_probability"]),
                        amount=float(row["TransactionAmt"]),
                    )
                graph.add_edge(transaction_node, attribute_node, relationship="SHARES ATTRIBUTE")
        self._graph_cache[cache_key] = (graph, suppressed)
        return graph, suppressed

    def _component_summaries(
        self, graph: nx.Graph, config: GraphConfig
    ) -> tuple[list[ClusterSummary], int]:
        review = float(self.scoring.operating_config["review_threshold"])
        summaries: list[ClusterSummary] = []
        component_count = 0
        for component in nx.connected_components(graph):
            subgraph = graph.subgraph(component)
            transaction_nodes = [node for node, data in subgraph.nodes(data=True) if data["node_type"] == "transaction"]
            attribute_nodes = [node for node, data in subgraph.nodes(data=True) if data["node_type"] == "shared_attribute"]
            if len(transaction_nodes) < config.minimum_cluster_transactions:
                continue
            component_count += 1
            risks = [float(graph.nodes[node]["risk_score"]) for node in transaction_nodes]
            amounts = [float(graph.nodes[node]["amount"]) for node in transaction_nodes]
            high_mask = [risk >= review for risk in risks]
            high_count = sum(high_mask)
            high_share = high_count / len(transaction_nodes)
            if high_count < config.minimum_high_risk_transactions or high_share < config.minimum_high_risk_share:
                continue
            transaction_ids = [str(graph.nodes[node]["transaction_id"]) for node in transaction_nodes]
            attribute_types = sorted({str(graph.nodes[node]["attribute_type"]) for node in attribute_nodes})
            summaries.append(
                ClusterSummary(
                    cluster_id=_cluster_id(transaction_ids),
                    transaction_count=len(transaction_nodes),
                    shared_attribute_count=len(attribute_nodes),
                    edge_count=subgraph.number_of_edges(),
                    connectivity_score=subgraph.number_of_edges() / len(transaction_nodes),
                    high_risk_count=high_count,
                    high_risk_share=high_share,
                    average_risk_score=sum(risks) / len(risks),
                    maximum_risk_score=max(risks),
                    total_transaction_amount=sum(amounts),
                    high_risk_amount=sum(amount for amount, high in zip(amounts, high_mask, strict=True) if high),
                    shared_attribute_types=attribute_types,
                    example_transaction_ids=sorted(transaction_ids)[:8],
                    label="SUSPICIOUS LINKED CLUSTER",
                )
            )
        summaries.sort(
            key=lambda item: (item.high_risk_amount, item.high_risk_count, item.connectivity_score),
            reverse=True,
        )
        return summaries, component_count

    def analyze(self, config: GraphConfig) -> GraphAnalysisResponse:
        graph, suppressed = self.build_graph(config)
        summaries, component_count = self._component_summaries(graph, config)
        thresholds = self.scoring.operating_config
        transaction_nodes = sum(1 for _, data in graph.nodes(data=True) if data["node_type"] == "transaction")
        attribute_nodes = graph.number_of_nodes() - transaction_nodes
        return GraphAnalysisResponse(
            source="IEEE-CIS chronological validation transactions and frozen CatBoost validation probabilities",
            data_partition="validation",
            evaluation_status="Not evaluated yet",
            model_version=str(self.scoring.metadata["model_version"]),
            review_threshold=float(thresholds["review_threshold"]),
            block_threshold=float(thresholds["block_threshold"]),
            config=config,
            transaction_rows_considered=len(self.validation_frame),
            transaction_nodes=transaction_nodes,
            shared_attribute_nodes=attribute_nodes,
            edge_count=graph.number_of_edges(),
            connected_components=component_count,
            suspicious_cluster_count=len(summaries),
            returned_cluster_count=min(len(summaries), config.max_clusters),
            suppressed_attribute_values=suppressed,
            clusters=summaries[: config.max_clusters],
            held_out_test_accessed=False,
            confirmed_fraud_ring_claimed=False,
            limitations=[
                "Linked transactions are suspicious components, not confirmed fraud rings.",
                "IEEE-CIS attributes do not establish a real-world person, card number, or account identity.",
                "Common attribute values are suppressed because they create weak, non-specific links.",
                "Validation analysis is not held-out graph-detector performance evaluation.",
            ],
        )

    def _render_subgraph(
        self,
        graph: nx.Graph,
        transaction_nodes: list[str],
        cluster_id: str,
        focus_node: str | None = None,
    ) -> ClusterGraph:
        selected_transactions = sorted(
            transaction_nodes,
            key=lambda node: float(graph.nodes[node].get("risk_score", 0)),
            reverse=True,
        )[:28]
        if focus_node is not None and focus_node not in selected_transactions:
            selected_transactions = [focus_node, *selected_transactions[:27]]
        selected_attributes = sorted(
            {neighbor for node in selected_transactions for neighbor in graph.neighbors(node)}
        )[:12]
        selected_nodes = set(selected_transactions) | set(selected_attributes)
        subgraph = graph.subgraph(selected_nodes)
        thresholds = self.scoring.operating_config
        review = float(thresholds["review_threshold"])
        block = float(thresholds["block_threshold"])
        nodes: list[GraphNode] = []
        for node, data in subgraph.nodes(data=True):
            if data["node_type"] == "transaction":
                risk = float(data["risk_score"])
                nodes.append(
                    GraphNode(
                        id=node,
                        node_type="transaction",
                        label=f"TX {data['transaction_id']}",
                        transaction_id=str(data["transaction_id"]),
                        risk_score=risk,
                        amount=float(data["amount"]),
                        decision=decision_from_score(risk, review, block),
                    )
                )
            else:
                nodes.append(
                    GraphNode(
                        id=node,
                        node_type="shared_attribute",
                        label=f"Shared {data['source_field']}",
                        source_field=str(data["source_field"]),
                        attribute_type=str(data["attribute_type"]),
                        attribute_value=str(data["attribute_value"]),
                    )
                )
        edges = [
            GraphEdge(source=source, target=target, relationship="SHARES ATTRIBUTE")
            for source, target in subgraph.edges()
        ]
        return ClusterGraph(
            cluster_id=cluster_id,
            label="SUSPICIOUS LINKED CLUSTER",
            nodes=nodes,
            edges=edges,
            total_nodes=subgraph.number_of_nodes(),
            total_edges=subgraph.number_of_edges(),
            graph_truncated=(
                len(selected_transactions) < len(transaction_nodes)
                or any(
                    neighbor not in selected_nodes
                    for node in selected_transactions
                    for neighbor in graph.neighbors(node)
                )
            ),
            limitation="Links mean only that validation transactions share an eligible dataset attribute.",
        )

    def cluster_graph(self, cluster_id: str, config: GraphConfig) -> ClusterGraph:
        graph, _ = self.build_graph(config)
        for component in nx.connected_components(graph):
            transaction_nodes = [node for node in component if graph.nodes[node]["node_type"] == "transaction"]
            transaction_ids = [str(graph.nodes[node]["transaction_id"]) for node in transaction_nodes]
            if _cluster_id(transaction_ids) == cluster_id:
                return self._render_subgraph(graph, transaction_nodes, cluster_id)
        raise ValueError("Suspicious linked cluster not found for this configuration")

    def neighborhood(self, transaction_id: str, config: GraphConfig) -> NeighborhoodResponse:
        normalized = transaction_id.strip()
        if not normalized.isdigit():
            raise ValueError("TransactionID must be numeric for the IEEE-CIS validation demo")
        numeric_id = int(normalized)
        found = bool((self.validation_frame["TransactionID"] == numeric_id).any())
        if not found:
            return NeighborhoodResponse(
                transaction_id=normalized,
                found_in_validation=False,
                connected_through_eligible_attributes=False,
                graph=None,
                message="TransactionID is not present in the chronological validation partition.",
                held_out_test_accessed=False,
            )
        graph, _ = self.build_graph(config)
        node = f"transaction::{numeric_id}"
        if node not in graph:
            return NeighborhoodResponse(
                transaction_id=normalized,
                found_in_validation=True,
                connected_through_eligible_attributes=False,
                graph=None,
                message="Transaction exists, but it has no eligible shared attribute after common-value suppression.",
                held_out_test_accessed=False,
            )
        component = nx.node_connected_component(graph, node)
        transaction_nodes = [item for item in component if graph.nodes[item]["node_type"] == "transaction"]
        transaction_ids = [str(graph.nodes[item]["transaction_id"]) for item in transaction_nodes]
        return NeighborhoodResponse(
            transaction_id=normalized,
            found_in_validation=True,
            connected_through_eligible_attributes=True,
            graph=self._render_subgraph(
                graph,
                transaction_nodes,
                _cluster_id(transaction_ids),
                focus_node=node,
            ),
            message="Neighborhood shows eligible shared dataset attributes only.",
            held_out_test_accessed=False,
        )


@lru_cache(maxsize=1)
def get_abuse_graph_service() -> AbuseGraphService:
    return AbuseGraphService(configured_abuse_graph_paths(), get_validation_scoring_service())
