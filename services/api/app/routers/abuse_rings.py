from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from ..schemas.abuse_rings import (
    ClusterGraph,
    GraphAnalysisResponse,
    GraphAnalyzeRequest,
    GraphStatus,
    NeighborhoodRequest,
    NeighborhoodResponse,
)
from ..services.abuse_graph import (
    ATTRIBUTE_FIELDS,
    AbuseGraphService,
    get_abuse_graph_service,
)
from ..services.artifacts import ArtifactUnavailable

router = APIRouter(prefix="/api/v1/abuse-rings", tags=["abuse-rings"])
GraphServiceDependency = Annotated[AbuseGraphService, Depends(get_abuse_graph_service)]


@router.get("/status", response_model=GraphStatus)
def graph_status(service: GraphServiceDependency) -> GraphStatus:
    try:
        thresholds = service.scoring.operating_config
        return GraphStatus(
            module="Abuse-Ring Sentinel",
            data_source="IEEE-CIS chronological validation attributes joined to frozen CatBoost validation probabilities",
            evaluation_status="Not evaluated yet",
            model_version=str(service.scoring.metadata["model_version"]),
            review_threshold=float(thresholds["review_threshold"]),
            block_threshold=float(thresholds["block_threshold"]),
            considered_attributes=[
                {"source_field": field, "documented_label": label}
                for field, label in ATTRIBUTE_FIELDS.items()
            ],
            default_common_value_suppression="Attribute values linked to more than 50 validation transactions are excluded by default.",
            terminology=[
                "shared attribute",
                "shared identifier",
                "suspicious linked cluster",
            ],
            limitations=[
                "A shared attribute does not prove common ownership or coordinated abuse.",
                "Masked and dataset-provided fields are not described as card numbers, accounts, or people.",
                "Components are not confirmed fraud rings.",
                "Held-out test data is not used.",
            ],
        )
    except ArtifactUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/analyze-validation", response_model=GraphAnalysisResponse)
def analyze_validation(request: GraphAnalyzeRequest, service: GraphServiceDependency) -> GraphAnalysisResponse:
    try:
        return service.analyze(request.config)
    except ArtifactUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/clusters/{cluster_id}", response_model=ClusterGraph)
def cluster_graph(
    cluster_id: str, request: GraphAnalyzeRequest, service: GraphServiceDependency
) -> ClusterGraph:
    try:
        return service.cluster_graph(cluster_id, request.config)
    except ArtifactUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/neighborhood", response_model=NeighborhoodResponse)
def transaction_neighborhood(
    request: NeighborhoodRequest, service: GraphServiceDependency
) -> NeighborhoodResponse:
    try:
        return service.neighborhood(request.transaction_id, request.config)
    except ArtifactUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
