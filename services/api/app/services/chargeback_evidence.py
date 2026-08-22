from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from ..models import ChargebackCase

MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_MIME_TYPES = {
    "application/pdf": (".pdf", b"%PDF-"),
    "image/png": (".png", b"\x89PNG\r\n\x1a\n"),
    "image/jpeg": (".jpg", b"\xff\xd8\xff"),
}
EVIDENCE_CATEGORIES = {
    "invoice",
    "proof_of_delivery",
    "tracking",
    "customer_communication",
    "refund_evidence",
    "merchant_policy",
    "other",
}

# MerchantShield's documented internal workflow checklist. These are deliberately
# not represented as card-network or legal requirements.
DISPUTE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "item_not_received": ("invoice", "tracking", "proof_of_delivery"),
    "duplicate": ("invoice", "customer_communication"),
    "refund_not_received": ("refund_evidence", "customer_communication"),
    "cancelled_recurring": ("customer_communication", "merchant_policy"),
    "not_as_described": ("invoice", "customer_communication", "merchant_policy"),
    "other": ("invoice",),
}
CHECKLIST_BASIS = (
    "MerchantShield internal evidence checklist for merchant preparation; "
    "not payment-network rules and not a prediction of dispute outcome."
)


class EvidenceFileError(ValueError):
    pass


@dataclass(frozen=True)
class StoredEvidence:
    storage_key: str
    size_bytes: int


class LocalEvidenceStorage:
    """Local development adapter; production can supply an object-store implementation."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def store_base64(self, *, dispute_id: str, content_type: str, encoded: str) -> StoredEvidence:
        file_spec = ALLOWED_MIME_TYPES.get(content_type)
        if file_spec is None:
            raise EvidenceFileError("Only PDF, PNG, and JPEG evidence files are accepted")
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise EvidenceFileError("Evidence content is not valid base64") from exc
        if not payload:
            raise EvidenceFileError("Evidence file is empty")
        if len(payload) > MAX_FILE_SIZE:
            raise EvidenceFileError("Evidence file exceeds the 10 MB limit")
        extension, signature = file_spec
        if not payload.startswith(signature):
            raise EvidenceFileError("Evidence file signature does not match its declared content type")

        safe_dispute_id = re.sub(r"[^A-Za-z0-9._-]", "_", dispute_id)[:100]
        storage_key = f"{safe_dispute_id}/{uuid4().hex}{extension}"
        destination = (self.root / storage_key).resolve()
        if self.root not in destination.parents:
            raise EvidenceFileError("Invalid evidence storage destination")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return StoredEvidence(storage_key=storage_key, size_bytes=len(payload))

    def delete(self, storage_key: str) -> None:
        target = (self.root / storage_key).resolve()
        if self.root in target.parents and target.is_file():
            target.unlink()


def completeness(case: ChargebackCase) -> dict[str, object]:
    expected = list(DISPUTE_REQUIREMENTS[case.reason])
    present = sorted({item.category for item in case.evidence if item.category in EVIDENCE_CATEGORIES})
    missing = [category for category in expected if category not in present]
    present_required = len(expected) - len(missing)
    return {
        "present": present,
        "expected": expected,
        "missing": missing,
        "present_count": present_required,
        "expected_count": len(expected),
        "ratio": present_required / len(expected) if expected else 1.0,
        "checklist_basis": CHECKLIST_BASIS,
    }


def _provided_information(title: str, values: dict[str, object]) -> list[str]:
    if not values:
        return [f"{title}: Not provided"]
    output = [f"{title}:"]
    output.extend(f"- {key}: {value}" for key, value in sorted(values.items()))
    return output


def generate_evidence_draft(case: ChargebackCase) -> tuple[str, list[str]]:
    status = completeness(case)
    missing = list(status["missing"])
    lines = [
        "HUMAN-REVIEW DRAFT — NOT SUBMITTED",
        "",
        f"Dispute: {case.dispute_id}",
        f"Transaction: {case.transaction_id}",
        f"Disputed amount: {case.currency} {case.amount}",
        f"Merchant-selected reason: {case.reason.replace('_', ' ')}",
        f"Response deadline: {case.deadline.isoformat()}",
        "",
        *_provided_information("Customer information supplied by merchant", case.customer_information),
        "",
        *_provided_information("Order information supplied by merchant", case.order_information),
        "",
        *_provided_information("Delivery information supplied by merchant", case.delivery_information),
        "",
        "Evidence files supplied:",
    ]
    if case.evidence:
        lines.extend(
            f"- {item.category.replace('_', ' ')}: {item.original_filename}"
            for item in case.evidence
        )
    else:
        lines.append("- None")
    lines.extend(["", "Missing checklist evidence:"])
    lines.extend(f"- {item.replace('_', ' ')}" for item in missing) if missing else lines.append("- None")
    lines.extend(
        [
            "",
            f"Merchant notes: {case.merchant_notes or 'Not provided'}",
            "",
            "Reviewer instruction: verify every statement against the uploaded files before approval.",
            f"Checklist basis: {CHECKLIST_BASIS}",
        ]
    )
    return "\n".join(lines), missing
