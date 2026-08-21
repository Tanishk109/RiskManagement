from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

RuleAction = Literal["NONE", "REVIEW", "BLOCK"]
ALLOWED_OPERATORS = {"eq", "ne", "gt", "gte", "lt", "lte", "in", "is_null"}
FORBIDDEN_FIELDS = {"isFraud", "actual_label", "future_chargeback_outcome", "future_fraud_label"}


@dataclass(frozen=True)
class RuleHit:
    rule_id: str
    action: RuleAction
    reason: str


@dataclass(frozen=True)
class Rule:
    rule_id: str
    name: str
    enabled: bool
    conditions: tuple[dict[str, Any], ...]
    action: RuleAction
    reason: str
    evidence: dict[str, Any]


def _compare(actual: Any, operator: str, expected: Any) -> bool:
    if operator == "is_null":
        return (actual is None) is bool(expected)
    if operator == "eq":
        return actual == expected
    if operator == "ne":
        return actual != expected
    if actual is None:
        return False
    if operator == "gt":
        return actual > expected
    if operator == "gte":
        return actual >= expected
    if operator == "lt":
        return actual < expected
    if operator == "lte":
        return actual <= expected
    if operator == "in":
        return actual in expected
    raise ValueError(f"Unsupported rule operator: {operator}")


def load_rules(path: Path) -> list[Rule]:
    if not path.is_file():
        return []
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_rules = payload.get("rules", [])
    if not isinstance(raw_rules, list):
        raise TypeError("rules file must contain a list under 'rules'")

    rules: list[Rule] = []
    for raw in raw_rules:
        conditions = tuple(raw.get("conditions", []))
        for condition in conditions:
            field = condition.get("field")
            operator = condition.get("operator")
            if field in FORBIDDEN_FIELDS:
                raise ValueError(f"Rule uses forbidden future/label field: {field}")
            if operator not in ALLOWED_OPERATORS:
                raise ValueError(f"Rule uses unsupported operator: {operator}")
        action = raw.get("action", "NONE")
        if action not in {"NONE", "REVIEW", "BLOCK"}:
            raise ValueError(f"Rule uses unsupported action: {action}")
        rules.append(Rule(
            rule_id=str(raw["id"]),
            name=str(raw["name"]),
            enabled=bool(raw.get("enabled", False)),
            conditions=conditions,
            action=action,
            reason=str(raw.get("reason", "")),
            evidence=dict(raw.get("evidence") or {}),
        ))
    return rules


def evaluate_rules(features: dict[str, Any], rules: list[Rule]) -> list[RuleHit]:
    hits: list[RuleHit] = []
    for rule in rules:
        if not rule.enabled:
            continue
        if all(_compare(features.get(condition["field"]), condition["operator"], condition.get("value")) for condition in rule.conditions):
            hits.append(RuleHit(rule_id=rule.rule_id, action=rule.action, reason=rule.reason))
    return hits
