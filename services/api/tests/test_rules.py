from __future__ import annotations

from pathlib import Path

import pytest
from app.services.rules_engine import evaluate_rules, load_rules


def test_rule_framework_can_escalate_to_review(tmp_path: Path):
    path = tmp_path / "rules.yaml"
    path.write_text(
        """version: 1
rules:
  - id: validation-rule
    name: Validation evidence rule
    enabled: true
    conditions:
      - field: V17
        operator: gt
        value: 2.5
    action: REVIEW
    reason: Validation-documented masked feature boundary.
    evidence:
      rows_affected: 0
""",
        encoding="utf-8",
    )
    hits = evaluate_rules({"V17": 3.0}, load_rules(path))
    assert [hit.action for hit in hits] == ["REVIEW"]


def test_rule_framework_rejects_label_fields(tmp_path: Path):
    path = tmp_path / "rules.yaml"
    path.write_text(
        """rules:
  - id: leakage
    name: Invalid future rule
    enabled: true
    conditions:
      - field: isFraud
        operator: eq
        value: 1
    action: BLOCK
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="forbidden"):
        load_rules(path)
