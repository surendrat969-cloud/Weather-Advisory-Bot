"""
Loads SOPs from sops/sops.yaml and provides matching utilities.
Policy changes only require editing the YAML, not this code.
"""
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

SOP_FILE = Path(__file__).parent.parent / "sops" / "sops.yaml"


@dataclass
class SOP:
    id: str
    category: str
    title: str
    severity: str
    trigger_description: str
    conditions: list
    advice: str
    fuzzy: bool = False

    def severity_rank(self) -> int:
        return {"low": 1, "moderate": 2, "high": 3, "critical": 4}.get(
            self.severity, 0
        )


def load_sops() -> list:
    """Load and parse all SOPs from the YAML file."""
    with open(SOP_FILE, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    sops = []
    for item in raw["sops"]:
        sops.append(
            SOP(
                id=item["id"],
                category=item["category"],
                title=item["title"],
                severity=item["severity"],
                trigger_description=item["trigger_description"],
                conditions=item.get("conditions", []),
                advice=item["advice"],
                fuzzy=item.get("fuzzy", False),
            )
        )
    return sops


def check_numeric_conditions(sop: SOP, weather) -> bool:
    """
    Check if all numeric conditions in an SOP are met by the weather data.

    Returns True if all conditions pass (or if no conditions defined for fuzzy SOPs).
    For fuzzy SOPs, always returns False — they are evaluated by the LLM.
    For non-fuzzy SOPs with no conditions, returns False (shouldn't auto-match).
    """
    if sop.fuzzy:
        return False  # fuzzy SOPs are evaluated by LLM, not here

    if not sop.conditions:
        return False  # non-fuzzy with no conditions shouldn't auto-match

    for cond in sop.conditions:
        field_name = cond.get("field")
        operator = cond.get("op")  # gte, lte, gt, lt, eq
        threshold = cond.get("value")

        actual = getattr(weather, field_name, None)
        if actual is None:
            return False  # missing data means condition can't be confirmed

        if operator == "gte" and not (actual >= threshold):
            return False
        elif operator == "gt" and not (actual > threshold):
            return False
        elif operator == "lte" and not (actual <= threshold):
            return False
        elif operator == "lt" and not (actual < threshold):
            return False
        elif operator == "eq" and not (actual == threshold):
            return False

    return True
