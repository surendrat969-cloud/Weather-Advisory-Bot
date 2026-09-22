"""
Backward-compatibility shim.
All real SOP logic is now in src/sop_engine.py.
This file re-exports the key names so existing imports don't break.
"""
from src.sop_engine import SOP, load_sops, evaluate_sops, select_primary_sop, SOPMatch


# check_numeric_conditions kept for eval suite backward compat
def check_numeric_conditions(sop: SOP, weather_proxy) -> bool:
    """
    Legacy helper — checks if a single non-fuzzy SOP's conditions are met.
    Used only in the eval suite for unit-testing individual SOPs.
    """
    from src.sop_engine import _eval_conditions
    # Convert weather proxy object to dict
    weather_dict = {}
    for attr in ["temperature_2m", "wind_speed_10m", "precipitation",
                 "precipitation_probability", "uv_index", "weather_code",
                 "relative_humidity_2m", "apparent_temperature", "wind_gusts_10m"]:
        weather_dict[attr] = getattr(weather_proxy, attr, None)
    matched, _ = _eval_conditions(sop.conditions, weather_dict)
    return matched
