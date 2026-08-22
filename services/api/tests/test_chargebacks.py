from __future__ import annotations

import base64
from pathlib import Path

from app.main import app
from app.routers.chargebacks import get_evidence_storage
from app.services.chargeback_evidence import LocalEvidenceStorage


def case_payload(dispute_id: str = "DSP-1001") -> dict[str, object]:
    return {
        "dispute_id": dispute_id,
        "transaction_id": "TXN-9001",
        "amount": 1499.50,
        "currency": "inr",
        "reason": "item_not_received",
        "deadline": "2026-09-01",
        "customer_information": {"name": "A. Customer", "email": "buyer@example.test"},
        "order_information": {"order_id": "ORDER-88", "item": "Travel pouch"},
        "delivery_information": {"carrier": "Merchant courier"},
        "merchant_notes": "Customer contacted support on 20 August.",
    }


def create_case(client, dispute_id: str = "DSP-1001") -> dict[str, object]:
    response = client.post("/api/v1/chargebacks/cases", json=case_payload(dispute_id))
    assert response.status_code == 201, response.text
    return response.json()


def valid_png() -> str:
    return base64.b64encode(b"\x89PNG\r\n\x1a\nmerchant-evidence").decode()


def test_status_is_honest_and_disables_automatic_submission(client):
    response = client.get("/api/v1/chargebacks/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["evaluation_status"] == "Not evaluated yet"
    assert payload["automatic_submission"] is False
    assert "not payment-network rules" in payload["checklist_basis"]


def test_case_creation_returns_explicit_completeness(client):
    result = create_case(client)
    assert result["currency"] == "INR"
    assert result["completeness"]["expected"] == ["invoice", "tracking", "proof_of_delivery"]
    assert result["completeness"]["present_count"] == 0
    assert result["completeness"]["ratio"] == 0
    assert result["draft"] is None


def test_duplicate_dispute_id_is_rejected(client):
    create_case(client)
    response = client.post("/api/v1/chargebacks/cases", json=case_payload())
    assert response.status_code == 409


def test_upload_stores_bytes_outside_database_and_updates_completeness(client, tmp_path: Path):
    app.dependency_overrides[get_evidence_storage] = lambda: LocalEvidenceStorage(tmp_path)
    try:
        case = create_case(client)
        response = client.post(
            f"/api/v1/chargebacks/cases/{case['id']}/evidence",
            json={
                "category": "invoice",
                "filename": "invoice.png",
                "content_type": "image/png",
                "base64_content": valid_png(),
            },
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["completeness"]["present_count"] == 1
        assert result["completeness"]["ratio"] == 1 / 3
        assert len(list(tmp_path.rglob("*.png"))) == 1
        assert "storage_key" not in result["evidence"][0]
    finally:
        app.dependency_overrides.pop(get_evidence_storage, None)


def test_upload_rejects_mime_signature_mismatch(client, tmp_path: Path):
    app.dependency_overrides[get_evidence_storage] = lambda: LocalEvidenceStorage(tmp_path)
    try:
        case = create_case(client)
        response = client.post(
            f"/api/v1/chargebacks/cases/{case['id']}/evidence",
            json={
                "category": "invoice",
                "filename": "fake.pdf",
                "content_type": "application/pdf",
                "base64_content": valid_png(),
            },
        )
        assert response.status_code == 422
        assert not list(tmp_path.rglob("*.*"))
    finally:
        app.dependency_overrides.pop(get_evidence_storage, None)


def test_draft_uses_supplied_fields_calls_out_missing_and_requires_human_approval(client):
    case = create_case(client)
    generated = client.post(f"/api/v1/chargebacks/cases/{case['id']}/generate-draft")
    assert generated.status_code == 200
    draft = generated.json()
    assert "HUMAN-REVIEW DRAFT — NOT SUBMITTED" in draft["draft_text"]
    assert "Travel pouch" in draft["draft_text"]
    assert "tracking" in draft["draft_text"]
    assert draft["human_approved"] is False

    before_approval = client.get(f"/api/v1/chargebacks/cases/{case['id']}/export")
    assert before_approval.status_code == 409
    approved = client.put(
        f"/api/v1/chargebacks/cases/{case['id']}/draft",
        json={"draft_text": draft["draft_text"] + "\nReviewed by merchant.", "human_approved": True},
    )
    assert approved.status_code == 200
    assert approved.json()["human_approved"] is True
    exported = client.get(f"/api/v1/chargebacks/cases/{case['id']}/export")
    assert exported.status_code == 200
    assert "Reviewed by merchant" in exported.text
    assert "attachment" in exported.headers["content-disposition"]


def test_case_input_rejects_unknown_fields(client):
    payload = case_payload()
    payload["predicted_win_probability"] = 0.99
    response = client.post("/api/v1/chargebacks/cases", json=payload)
    assert response.status_code == 422


def test_chargeback_schema_keeps_file_content_out_of_postgresql():
    from app.database import Base

    evidence_columns = Base.metadata.tables["chargeback_evidence"].columns
    assert "storage_key" in evidence_columns
    assert "content" not in evidence_columns
    assert "base64_content" not in evidence_columns
    assert "win_probability" not in Base.metadata.tables["chargeback_cases"].columns
