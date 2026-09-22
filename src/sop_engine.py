"""
Deterministic SOP Engine.

This module is the ONLY place where SOPs are evaluated against weather data.
The LLM is never involved in this decision.

Evaluation logic:
1. Load all SOPs from YAML
2. For each SOP:
   a. Check activity relevance (if SOP has activities list, at least one must match)
   b. Evaluate weather conditions using Python comparison operators
3. Return ALL matching SOPs with reasons
4. Caller selects primary SOP using deterministic priority rules
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

SOP_FILE = Path(__file__).parent.parent / "sops" / "sops.yaml"

SEVERITY_RANK = {"low": 1, "moderate": 2, "high": 3, "critical": 4}


@dataclass
class SOP:
    id: str
    category: str
    title: str
    severity: str
    trigger_description: str
    activities: list  # empty list = applies to ALL activities
    conditions: object  # list of dicts (all) OR dict with 'any' key
    advice: str
    fuzzy: bool = False

    def severity_rank(self) -> int:
        return SEVERITY_RANK.get(self.severity, 0)


@dataclass
class SOPMatch:
    sop: SOP
    reasons: list  # human-readable reasons why it matched

    @property
    def id(self):
        return self.sop.id

    @property
    def severity(self):
        return self.sop.severity

    @property
    def severity_rank(self):
        return self.sop.severity_rank()


def load_sops() -> list:
    """Load all SOPs from YAML. No LLM involved."""
    with open(SOP_FILE, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    sops = []
    for item in raw["sops"]:
        sops.append(SOP(
            id=item["id"],
            category=item["category"],
            title=item["title"],
            severity=item["severity"],
            trigger_description=item["trigger_description"],
            activities=item.get("activities", []),
            conditions=item.get("conditions", []),
            advice=item["advice"],
            fuzzy=item.get("fuzzy", False),
        ))
    return sops


def _activity_matches(sop: SOP, activity: str) -> bool:
    """
    Returns True if:
    - The SOP has an empty activities list (applies universally), OR
    - Any activity alias in the SOP matches the user's activity string (case-insensitive)
    """
    if not sop.activities:
        return True  # universal SOP — matches all activities
    activity_lower = activity.lower()
    for alias in sop.activities:
        if alias.lower() in activity_lower or activity_lower in alias.lower():
            return True
    return False


def _eval_single_condition(cond: dict, weather: dict) -> tuple:
    """
    Evaluate one condition dict against weather data.
    Returns (matched: bool, reason: str)
    """
    field_name = cond.get("field")
    op = cond.get("op")
    threshold = cond.get("value")

    actual = weather.get(field_name)
    if actual is None:
        return False, f"{field_name} = None (missing data)"

    ops = {
        "gte": (lambda a, t: a >= t, ">="),
        "gt":  (lambda a, t: a > t,  ">"),
        "lte": (lambda a, t: a <= t, "<="),
        "lt":  (lambda a, t: a < t,  "<"),
        "eq":  (lambda a, t: a == t, "=="),
    }

    if op not in ops:
        return False, f"Unknown operator: {op}"

    fn, symbol = ops[op]
    matched = fn(actual, threshold)
    reason = f"{field_name} = {actual} {symbol} {threshold}"
    return matched, reason


def _eval_conditions(conditions: object, weather: dict) -> tuple:
    """
    Evaluate conditions against weather data.

    conditions can be:
    - A list of dicts (ALL must be true — AND logic)
    - A dict with key 'any' containing a list (ANY must be true — OR logic)
    - Empty list (returns False — non-fuzzy SOPs with no conditions don't auto-match)

    Returns (matched: bool, reasons: list[str])
    """
    if not conditions:
        return False, []

    # OR logic: dict with 'any' key
    if isinstance(conditions, dict) and "any" in conditions:
        for cond in conditions["any"]:
            matched, reason = _eval_single_condition(cond, weather)
            if matched:
                return True, [reason]
        return False, []

    # AND logic: list of conditions
    if isinstance(conditions, list):
        reasons = []
        for cond in conditions:
            matched, reason = _eval_single_condition(cond, weather)
            if not matched:
                return False, []
            reasons.append(reason)
        return True, reasons

    return False, []


def evaluate_sops(
    activity: str,
    weather: dict,
    vulnerable_group: Optional[str] = None,
    all_sops: Optional[list] = None,
) -> list:
    """
    Deterministically evaluate all SOPs against weather + activity.

    Returns a list of SOPMatch objects for every SOP that matches.
    Never calls the LLM. Never uses fuzzy logic — all evaluation is Python.

    Priority (used by select_primary_sop):
    1. Activity-specific match beats universal match (empty activities list)
    2. Higher severity beats lower severity
    3. If still tied, preserve all (caller decides)
    """
    if all_sops is None:
        all_sops = load_sops()

    # Augment activity with vulnerable group info for matching
    full_activity = activity
    if vulnerable_group:
        full_activity = f"{activity} {vulnerable_group}"

    matches = []
    for sop in all_sops:
        # Step 1: Activity relevance check
        if not _activity_matches(sop, full_activity):
            continue

        # Step 2: Condition evaluation (pure Python, no LLM)
        condition_matched, reasons = _eval_conditions(sop.conditions, weather)
        if condition_matched:
            matches.append(SOPMatch(sop=sop, reasons=reasons))

    return matches


def select_primary_sop(matches: list) -> Optional[SOPMatch]:
    """
    Deterministically select the primary SOP from a list of matches.

    Strategy (documented):
    1. Activity-specific SOPs (non-empty activities list) beat universal SOPs
    2. Among remaining, highest severity wins
    3. If severity is tied, first one in YAML order wins (stable)

    This function never calls the LLM.
    """
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    # Prefer activity-specific over universal (empty activities = universal)
    specific = [m for m in matches if m.sop.activities]
    universal = [m for m in matches if not m.sop.activities]

    candidates = specific if specific else universal

    # Among candidates, pick highest severity
    return max(candidates, key=lambda m: m.severity_rank)
