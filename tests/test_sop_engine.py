"""
Unit tests for src/sop_engine.py

Run from project root:
    py -m pytest tests/ -v
    or
    py tests/test_sop_engine.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

# Dummy key so graph.py can be imported without a real Groq account
os.environ.setdefault("GROQ_API_KEY", "sk-test-dummy")

from src.sop_engine import (
    load_sops,
    evaluate_sops,
    select_primary_sop,
    _eval_conditions,
    _activity_matches,
    SOP,
    SOPMatch,
    SEVERITY_RANK,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
ALL_SOPS = load_sops()

MILD_WEATHER = {
    "temperature_2m": 24.0,
    "wind_speed_10m": 10.0,
    "precipitation": 0.0,
    "precipitation_probability": 10.0,
    "uv_index": 3.0,
    "relative_humidity_2m": 50.0,
    "apparent_temperature": 24.0,
    "wind_gusts_10m": 15.0,
}

HIGH_WIND_WEATHER = {**MILD_WEATHER, "wind_speed_10m": 55.0, "wind_gusts_10m": 70.0}
HIGH_RAIN_WEATHER = {**MILD_WEATHER, "precipitation_probability": 85.0}
HIGH_UV_WEATHER   = {**MILD_WEATHER, "uv_index": 9.0}
HEAT_WEATHER      = {**MILD_WEATHER, "temperature_2m": 39.0}
STORM_WEATHER     = {**MILD_WEATHER, "precipitation": 6.0, "wind_speed_10m": 40.0}


class TestOperators(unittest.TestCase):
    """Test each supported comparison operator."""

    def _make_sop(self, op, value):
        return SOP(
            id="TEST", category="test", title="T", severity="high",
            trigger_description="", activities=[],
            conditions=[{"field": "wind_speed_10m", "op": op, "value": value}],
            advice="",
        )

    def test_gte_passes(self):
        matched, _ = _eval_conditions(
            [{"field": "wind_speed_10m", "op": "gte", "value": 40}],
            {"wind_speed_10m": 40.0}
        )
        self.assertTrue(matched)

    def test_gte_fails(self):
        matched, _ = _eval_conditions(
            [{"field": "wind_speed_10m", "op": "gte", "value": 40}],
            {"wind_speed_10m": 39.9}
        )
        self.assertFalse(matched)

    def test_gt_passes(self):
        matched, _ = _eval_conditions(
            [{"field": "wind_speed_10m", "op": "gt", "value": 40}],
            {"wind_speed_10m": 40.1}
        )
        self.assertTrue(matched)

    def test_gt_fails_equal(self):
        matched, _ = _eval_conditions(
            [{"field": "wind_speed_10m", "op": "gt", "value": 40}],
            {"wind_speed_10m": 40.0}
        )
        self.assertFalse(matched)

    def test_lte_passes(self):
        matched, _ = _eval_conditions(
            [{"field": "uv_index", "op": "lte", "value": 5}],
            {"uv_index": 5.0}
        )
        self.assertTrue(matched)

    def test_lt_passes(self):
        matched, _ = _eval_conditions(
            [{"field": "uv_index", "op": "lt", "value": 5}],
            {"uv_index": 4.9}
        )
        self.assertTrue(matched)

    def test_eq_passes(self):
        matched, _ = _eval_conditions(
            [{"field": "weather_code", "op": "eq", "value": 0}],
            {"weather_code": 0}
        )
        self.assertTrue(matched)

    def test_missing_field_fails(self):
        matched, _ = _eval_conditions(
            [{"field": "wind_speed_10m", "op": "gte", "value": 40}],
            {"temperature_2m": 25.0}  # no wind_speed_10m
        )
        self.assertFalse(matched)

    def test_or_logic_first_branch(self):
        """OR: first condition triggers."""
        matched, reasons = _eval_conditions(
            {"any": [
                {"field": "precipitation_probability", "op": "gte", "value": 50},
                {"field": "wind_speed_10m", "op": "gt", "value": 30},
            ]},
            {"precipitation_probability": 60.0, "wind_speed_10m": 10.0}
        )
        self.assertTrue(matched)
        self.assertTrue(len(reasons) > 0)

    def test_or_logic_second_branch(self):
        """OR: second condition triggers when first fails."""
        matched, _ = _eval_conditions(
            {"any": [
                {"field": "precipitation_probability", "op": "gte", "value": 50},
                {"field": "wind_speed_10m", "op": "gt", "value": 30},
            ]},
            {"precipitation_probability": 20.0, "wind_speed_10m": 35.0}
        )
        self.assertTrue(matched)

    def test_or_logic_none_trigger(self):
        """OR: no condition triggers."""
        matched, _ = _eval_conditions(
            {"any": [
                {"field": "precipitation_probability", "op": "gte", "value": 50},
                {"field": "wind_speed_10m", "op": "gt", "value": 30},
            ]},
            {"precipitation_probability": 20.0, "wind_speed_10m": 10.0}
        )
        self.assertFalse(matched)

    def test_and_logic_all_pass(self):
        """AND: both conditions must pass."""
        matched, reasons = _eval_conditions(
            [
                {"field": "wind_speed_10m", "op": "gte", "value": 40},
                {"field": "precipitation_probability", "op": "gte", "value": 60},
            ],
            {"wind_speed_10m": 45.0, "precipitation_probability": 65.0}
        )
        self.assertTrue(matched)
        self.assertEqual(len(reasons), 2)

    def test_and_logic_one_fails(self):
        """AND: one condition failing means no match."""
        matched, _ = _eval_conditions(
            [
                {"field": "wind_speed_10m", "op": "gte", "value": 40},
                {"field": "precipitation_probability", "op": "gte", "value": 60},
            ],
            {"wind_speed_10m": 45.0, "precipitation_probability": 30.0}
        )
        self.assertFalse(matched)


class TestActivityMatching(unittest.TestCase):
    """Test activity alias matching."""

    def _sop_with_activities(self, activities):
        return SOP(
            id="T", category="c", title="t", severity="low",
            trigger_description="", activities=activities,
            conditions=[], advice="",
        )

    def test_universal_sop_matches_anything(self):
        sop = self._sop_with_activities([])
        self.assertTrue(_activity_matches(sop, "kite flying"))
        self.assertTrue(_activity_matches(sop, "work from home"))

    def test_exact_alias_match(self):
        sop = self._sop_with_activities(["cycling"])
        self.assertTrue(_activity_matches(sop, "cycling"))

    def test_alias_substring_match(self):
        sop = self._sop_with_activities(["bicycle"])
        self.assertTrue(_activity_matches(sop, "take my bicycle outside"))

    def test_no_alias_match(self):
        sop = self._sop_with_activities(["cycling", "bicycle", "biking"])
        self.assertFalse(_activity_matches(sop, "work from home"))

    def test_case_insensitive(self):
        sop = self._sop_with_activities(["Cycling"])
        self.assertTrue(_activity_matches(sop, "CYCLING"))


class TestEvaluateSops(unittest.TestCase):
    """Integration tests for evaluate_sops + select_primary_sop."""

    def test_oe001_cycling_high_wind(self):
        matches = evaluate_sops("cycling", HIGH_WIND_WEATHER, all_sops=ALL_SOPS)
        ids = [m.id for m in matches]
        self.assertIn("OE-001", ids)

    def test_oe001_not_triggered_work_from_home(self):
        """Activity relevance: cycling SOP should NOT match 'work from home'."""
        matches = evaluate_sops("work from home", HIGH_WIND_WEATHER, all_sops=ALL_SOPS)
        ids = [m.id for m in matches]
        self.assertNotIn("OE-001", ids)

    def test_oe002_running_high_rain(self):
        matches = evaluate_sops("running", HIGH_RAIN_WEATHER, all_sops=ALL_SOPS)
        ids = [m.id for m in matches]
        self.assertIn("OE-002", ids)

    def test_no_sop_kite_flying_mild(self):
        matches = evaluate_sops("kite flying", MILD_WEATHER, all_sops=ALL_SOPS)
        self.assertEqual(matches, [])
        self.assertIsNone(select_primary_sop(matches))

    def test_multiple_sops_returned(self):
        """cycling + high wind + high rain → OE-001 and OE-002 both match."""
        weather = {**HIGH_WIND_WEATHER, "precipitation_probability": 85.0}
        matches = evaluate_sops("cycling", weather, all_sops=ALL_SOPS)
        ids = [m.id for m in matches]
        self.assertIn("OE-001", ids)
        self.assertIn("OE-002", ids)

    def test_select_primary_highest_severity(self):
        """When OE-001 (high) and OE-002 (critical) both match, OE-002 wins."""
        weather = {**HIGH_WIND_WEATHER, "precipitation_probability": 85.0}
        matches = evaluate_sops("cycling", weather, all_sops=ALL_SOPS)
        primary = select_primary_sop(matches)
        self.assertIsNotNone(primary)
        self.assertEqual(primary.id, "OE-002")

    def test_select_primary_activity_specific_beats_universal(self):
        """
        GO-001 is universal (activities: []).
        OE-001 is cycling-specific.
        When both match for a cycling question, OE-001 should win despite GO-001
        also being 'high' severity (tied severity → activity-specific wins).
        """
        weather = {**HIGH_WIND_WEATHER, "precipitation_probability": 65.0}
        matches = evaluate_sops("cycling", weather, all_sops=ALL_SOPS)
        ids = [m.id for m in matches]
        # Both should match
        self.assertIn("OE-001", ids)
        self.assertIn("GO-001", ids)
        primary = select_primary_sop(matches)
        # OE-001 is activity-specific; GO-001 is universal — OE-001 wins
        self.assertEqual(primary.id, "OE-001")

    def test_vulnerable_group_augments_activity(self):
        """Vulnerable group 'children' should help match VG-002."""
        matches = evaluate_sops(
            "playing", HIGH_UV_WEATHER, vulnerable_group="children", all_sops=ALL_SOPS
        )
        ids = [m.id for m in matches]
        self.assertIn("VG-002", ids)

    def test_evidence_reasons_populated(self):
        """Each match must include human-readable reason strings."""
        matches = evaluate_sops("cycling", HIGH_WIND_WEATHER, all_sops=ALL_SOPS)
        oe001 = next((m for m in matches if m.id == "OE-001"), None)
        self.assertIsNotNone(oe001)
        self.assertTrue(len(oe001.reasons) > 0)
        # Reason should mention the field and values
        self.assertIn("wind_speed_10m", oe001.reasons[0])

    def test_malformed_weather_missing_field_does_not_crash(self):
        """Missing weather field should cause condition to fail gracefully, not crash."""
        bad_weather = {"temperature_2m": 25.0}  # no wind_speed_10m
        try:
            matches = evaluate_sops("cycling", bad_weather, all_sops=ALL_SOPS)
            # OE-001 requires wind_speed_10m — should not match
            ids = [m.id for m in matches]
            self.assertNotIn("OE-001", ids)
        except Exception as e:
            self.fail(f"evaluate_sops raised an exception on missing field: {e}")

    def test_go002_picnic_or_logic(self):
        """GO-002 uses OR conditions — triggering any one should match."""
        weather_with_high_wind = {**MILD_WEATHER, "wind_speed_10m": 35.0}
        matches = evaluate_sops("picnic", weather_with_high_wind, all_sops=ALL_SOPS)
        ids = [m.id for m in matches]
        self.assertIn("GO-002", ids)

    def test_severity_rank_ordering(self):
        self.assertLess(SEVERITY_RANK["low"], SEVERITY_RANK["moderate"])
        self.assertLess(SEVERITY_RANK["moderate"], SEVERITY_RANK["high"])
        self.assertLess(SEVERITY_RANK["high"], SEVERITY_RANK["critical"])


class TestSOPYamlExtensibility(unittest.TestCase):
    """Verify YAML loads cleanly and supports the expected schema."""

    def test_all_sops_loaded(self):
        self.assertGreaterEqual(len(ALL_SOPS), 12)

    def test_all_sops_have_required_fields(self):
        for sop in ALL_SOPS:
            self.assertTrue(sop.id, f"SOP missing id: {sop}")
            self.assertTrue(sop.title, f"SOP {sop.id} missing title")
            self.assertIn(sop.severity, ("low", "moderate", "high", "critical"),
                          f"SOP {sop.id} has invalid severity: {sop.severity}")
            self.assertTrue(sop.advice, f"SOP {sop.id} missing advice")

    def test_universal_sops_have_empty_activities(self):
        """GO-001 and GO-003 are universal — activities must be empty."""
        go001 = next((s for s in ALL_SOPS if s.id == "GO-001"), None)
        go003 = next((s for s in ALL_SOPS if s.id == "GO-003"), None)
        self.assertIsNotNone(go001)
        self.assertIsNotNone(go003)
        self.assertEqual(go001.activities, [])
        self.assertEqual(go003.activities, [])

    def test_go002_uses_or_conditions(self):
        """GO-002 must use OR/any conditions, not AND."""
        go002 = next((s for s in ALL_SOPS if s.id == "GO-002"), None)
        self.assertIsNotNone(go002)
        self.assertIsInstance(go002.conditions, dict)
        self.assertIn("any", go002.conditions)


if __name__ == "__main__":
    unittest.main(verbosity=2)
