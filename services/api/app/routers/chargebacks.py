from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from ..config import get_settings
from ..database import get_db
from ..models import ChargebackCase, ChargebackDraft, ChargebackEvidence
from ..schemas.chargebacks import (
    ChargebackCaseCreate,
    ChargebackCaseOut,
    ChargebackStatus,
    DraftOut,
    DraftSave,
    EvidenceUpload,
)
from ..services.chargeback_evidence import (
    CHECKLIST_BASIS,
    MAX_FILE_SIZE,
    EvidenceFileError,
    LocalEvidenceStorage,
    completeness,
    generate_evidence_draft,
)

router = APIRouter(prefix="/api/v1/chargebacks", tags=["chargebacks"])


def get_evidence_storage() -> LocalEvidenceStorage:
    return LocalEvidenceStorage(get_settings().evidence_storage_root)


def _case_query():
    return select(ChargebackCase).options(
        selectinload(ChargebackCase.evidence), selectinload(ChargebackCase.draft)
    ).execution_options(populate_existing=True)


def _get_case(db: Session, case_id: int) -> ChargebackCase:
    case = db.scalar(_case_query().where(ChargebackCase.id == case_id))
    if case is None:
        raise HTTPException(status_code=404, detail="Chargeback case not found")
    return case


def _case_out(case: ChargebackCase) -> ChargebackCaseOut:
    payload = {column.name: getattr(case, column.name) for column in ChargebackCase.__table__.columns}
    payload.update(evidence=case.evidence, draft=case.draft, completeness=completeness(case))
    return ChargebackCaseOut.model_validate(payload)


@router.get("/status", response_model=ChargebackStatus)
def chargeback_status() -> ChargebackStatus:
    return ChargebackStatus(
        module="Chargeback Evidence Responder",
        data_source="Merchant-entered dispute data and merchant-uploaded evidence files",
        evaluation_status="Not evaluated yet",
        checklist_basis=CHECKLIST_BASIS,
        limitations=[
            "Completeness is required-evidence coverage, not a win probability.",
            "Drafts only describe supplied metadata and file labels; file contents are not interpreted.",
            "MerchantShield does not submit evidence to a bank or payment network.",
        ],
        accepted_file_types=["application/pdf", "image/png", "image/jpeg"],
        maximum_file_size_bytes=MAX_FILE_SIZE,
        automatic_submission=False,
    )


@router.post("/cases", response_model=ChargebackCaseOut, status_code=status.HTTP_201_CREATED)
def create_case(
    request: ChargebackCaseCreate, db: Annotated[Session, Depends(get_db)]
) -> ChargebackCaseOut:
    case = ChargebackCase(
        dispute_id=request.dispute_id,
        transaction_id=request.transaction_id,
        amount=Decimal(str(request.amount)),
        currency=request.currency.upper(),
        reason=request.reason,
        deadline=request.deadline,
        customer_information=request.customer_information,
        order_information=request.order_information,
        delivery_information=request.delivery_information,
        merchant_notes=request.merchant_notes,
    )
    db.add(case)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Dispute ID already exists") from exc
    return _case_out(_get_case(db, case.id))


@router.get("/cases", response_model=list[ChargebackCaseOut])
def list_cases(db: Annotated[Session, Depends(get_db)]) -> list[ChargebackCaseOut]:
    cases = db.scalars(_case_query().order_by(ChargebackCase.deadline, ChargebackCase.id.desc())).unique()
    return [_case_out(case) for case in cases]


@router.get("/cases/{case_id}", response_model=ChargebackCaseOut)
def get_case(case_id: int, db: Annotated[Session, Depends(get_db)]) -> ChargebackCaseOut:
    return _case_out(_get_case(db, case_id))


@router.post("/cases/{case_id}/evidence", response_model=ChargebackCaseOut)
def upload_evidence(
    case_id: int,
    request: EvidenceUpload,
    db: Annotated[Session, Depends(get_db)],
    storage: Annotated[LocalEvidenceStorage, Depends(get_evidence_storage)],
) -> ChargebackCaseOut:
    case = _get_case(db, case_id)
    try:
        stored = storage.store_base64(
            dispute_id=case.dispute_id,
            content_type=request.content_type,
            encoded=request.base64_content,
        )
    except EvidenceFileError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    evidence = ChargebackEvidence(
        case_id=case.id,
        category=request.category,
        original_filename=Path(request.filename).name,
        content_type=request.content_type,
        size_bytes=stored.size_bytes,
        storage_key=stored.storage_key,
    )
    db.add(evidence)
    case.status = "DRAFT"
    if case.draft is not None:
        case.draft.human_approved = False
    try:
        db.commit()
    except Exception:
        db.rollback()
        storage.delete(stored.storage_key)
        raise
    return _case_out(_get_case(db, case.id))


@router.post("/cases/{case_id}/generate-draft", response_model=DraftOut)
def generate_draft(case_id: int, db: Annotated[Session, Depends(get_db)]) -> DraftOut:
    case = _get_case(db, case_id)
    draft_text, missing = generate_evidence_draft(case)
    if case.draft is None:
        case.draft = ChargebackDraft(
            draft_text=draft_text,
            evidence_count=len(case.evidence),
            missing_categories=missing,
        )
    else:
        case.draft.draft_text = draft_text
        case.draft.evidence_count = len(case.evidence)
        case.draft.missing_categories = missing
        case.draft.human_approved = False
    case.status = "READY_FOR_HUMAN_REVIEW"
    db.commit()
    return DraftOut.model_validate(_get_case(db, case.id).draft)


@router.put("/cases/{case_id}/draft", response_model=DraftOut)
def save_draft(
    case_id: int, request: DraftSave, db: Annotated[Session, Depends(get_db)]
) -> DraftOut:
    case = _get_case(db, case_id)
    if case.draft is None:
        raise HTTPException(status_code=409, detail="Generate an evidence-grounded draft before editing it")
    case.draft.draft_text = request.draft_text
    case.draft.human_approved = request.human_approved
    case.status = "APPROVED_FOR_EXPORT" if request.human_approved else "READY_FOR_HUMAN_REVIEW"
    db.commit()
    return DraftOut.model_validate(_get_case(db, case.id).draft)


@router.get("/cases/{case_id}/export")
def export_draft(case_id: int, db: Annotated[Session, Depends(get_db)]) -> Response:
    case = _get_case(db, case_id)
    if case.draft is None or not case.draft.human_approved:
        raise HTTPException(status_code=409, detail="Human approval is required before draft export")
    safe_name = "".join(character for character in case.dispute_id if character.isalnum() or character in "-_")
    return Response(
        content=case.draft.draft_text,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{safe_name or "chargeback"}-draft.txt"'},
    )
