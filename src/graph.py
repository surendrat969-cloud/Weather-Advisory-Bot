"""
LangGraph graph for the Weather-Advisory Support Bot.

Graph flow:
  understand_question
       │
  fetch_weather  ──(no location / weather error)──► failure_response → END
       │
  evaluate_sops  (deterministic Python — NO LLM)
       │
  sop_found?
  ├── NO  → no_sop_response → END
  └── YES → compose_response → END

LLM is used ONLY in:
  - understand_question : extract structured intent from natural language
  - compose_response    : format natural-language answer around deterministic results
  - no_sop_response     : format "no policy found" message
  - failure_response    : format "weather unavailable" message

LLM is NEVER used to decide which SOP applies, evaluate thresholds, or invent weather values.
"""
import os
import json
from pathlib import Path
from typing import Optional, TypedDict
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

load_dotenv()
# Explicitly load from project root .env regardless of cwd (needed when running eval suite)
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)


# ---------------------------------------------------------------------------
# Resolve API key — works locally (.env) AND on Streamlit Cloud (st.secrets)
# ---------------------------------------------------------------------------
def _resolve_groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY", "").strip()
        if key:
            os.environ["GROQ_API_KEY"] = key
            return key
    except Exception:
        pass
    raise RuntimeError(
        "GROQ_API_KEY not found. "
        "Add it to .env (local) or Streamlit Cloud secrets (deployment)."
    )


GROQ_API_KEY = _resolve_groq_key()

from src.sop_engine import load_sops, evaluate_sops, select_primary_sop, SOP, SOPMatch
from src.weather import get_weather_for_city
from src.prompts import (
    SYSTEM_PROMPT,
    EXTRACT_INTENT_PROMPT,
    COMPOSE_ANSWER_PROMPT,
    NO_SOP_COMPOSE_PROMPT,
    WEATHER_FAILURE_RESPONSE,
)

# ---------------------------------------------------------------------------
# LLM — used ONLY for intent extraction and response composition
# ---------------------------------------------------------------------------
llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0, api_key=GROQ_API_KEY)

# Load SOPs once at startup — avoids re-reading YAML on every request
ALL_SOPS = load_sops()


# ---------------------------------------------------------------------------
# Structured intent — validated before weather fetch
# ---------------------------------------------------------------------------
@dataclass
class UserIntent:
    """Validated structured intent extracted from user message."""
    location: Optional[str]       # city name, None if unknown
    activity: Optional[str]       # what user wants to do
    vulnerable_group: Optional[str]  # elderly/children/pets, or None
    is_followup: bool             # True if this refers to a prior message
    missing_location: bool        # True when location cannot be determined


# ---------------------------------------------------------------------------
# AgentState — explicitly typed for all fields carried across nodes
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    # Conversation history (LangChain message objects)
    messages: list

    # Current turn input
    current_question: str

    # Structured context — persisted across follow-up turns
    location: Optional[str]           # resolved city name
    latitude: Optional[float]         # from geocoding (stored for auditability)
    longitude: Optional[float]        # from geocoding
    activity: Optional[str]           # user's intended activity
    category: Optional[str]           # SOP category hint
    vulnerable_group: Optional[str]   # elderly / children / pets
    time_reference: Optional[str]     # "today", "tomorrow", "this evening", etc.
    is_followup: bool

    # Weather data from Open-Meteo (never invented)
    weather: Optional[dict]
    weather_error: Optional[str]

    # Deterministic SOP engine results
    sop_matches: list          # all matched SOPs as dicts, with evidence
    primary_sop: Optional[dict]
    trigger_reasons: list      # machine-readable evidence strings
    all_applicable_ids: list   # IDs of all matched SOPs

    # Final output
    final_answer: Optional[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_history(messages: list) -> str:
    """Render the last 3 exchanges as a concise string for the LLM context."""
    if not messages:
        return "No prior conversation."
    parts = []
    for m in messages[-6:]:
        if isinstance(m, HumanMessage):
            parts.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            # Include only the first line of bot reply to avoid token bloat
            first_line = m.content.split("\n")[0][:200]
            parts.append(f"Bot: {first_line}")
    return "\n".join(parts)


def _parse_json(content: str) -> dict:
    """Strip markdown code fences then parse JSON."""
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        inner = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        content = inner.strip()
    return json.loads(content)


def _sop_match_to_dict(match: SOPMatch) -> dict:
    """Serialize a SOPMatch to a JSON-safe dict including evidence."""
    return {
        "id": match.sop.id,
        "category": match.sop.category,
        "title": match.sop.title,
        "severity": match.sop.severity,
        "trigger_description": match.sop.trigger_description,
        "advice": match.sop.advice,
        "reasons": match.reasons,  # e.g. ["wind_speed_10m = 55 >= 40"]
    }


def _merge_intent(state: AgentState, new_intent: dict) -> dict:
    """
    Safely merge new intent fields into existing state.

    Rules:
    - If the new message explicitly provides a field, use the new value.
    - If the new message does not provide a field, retain the previous valid value.
    - Never hallucinate missing fields.
    """
    location = new_intent.get("location")
    activity = new_intent.get("activity")
    vulnerable_group = new_intent.get("vulnerable_group")
    time_ref = new_intent.get("time_reference")
    is_followup = new_intent.get("is_followup", False)

    # Inherit location from prior state if this is a follow-up without new location
    if not location and state.get("location") and is_followup:
        location = state["location"]

    # Inherit activity from prior state if not provided in new message
    if not activity and state.get("activity") and is_followup:
        activity = state["activity"]

    # Inherit vulnerable group from prior state if not provided
    if not vulnerable_group and state.get("vulnerable_group") and is_followup:
        vulnerable_group = state["vulnerable_group"]

    return {
        "location": location or state.get("location"),
        "activity": activity or state.get("activity"),
        "vulnerable_group": vulnerable_group or state.get("vulnerable_group"),
        "time_reference": time_ref,
        "is_followup": is_followup,
        "missing_location": not bool(location or state.get("location")),
    }


# ---------------------------------------------------------------------------
# Node 1: understand_question  (LLM — intent extraction only)
# ---------------------------------------------------------------------------
def node_understand_question(state: AgentState) -> AgentState:
    """
    LLM extracts structured intent from the user's message.

    What the LLM does here:
      - Parse location, activity, vulnerable group, time reference, follow-up flag
      - Resolve conversational references using the provided history

    What the LLM does NOT do here:
      - Evaluate weather conditions
      - Select or suggest SOPs
      - Make any policy decision
    """
    history_str = _format_history(state.get("messages", []))
    prompt = EXTRACT_INTENT_PROMPT.format(
        history=history_str,
        message=state["current_question"],
    )
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    try:
        data = _parse_json(response.content)
    except (json.JSONDecodeError, ValueError):
        # Graceful fallback — treat current question as raw activity, no location
        data = {
            "location": None,
            "activity": state["current_question"],
            "vulnerable_group": None,
            "time_reference": None,
            "is_followup": False,
        }

    merged = _merge_intent(state, data)

    return {
        **state,
        "location": merged["location"],
        "activity": merged["activity"],
        "vulnerable_group": merged["vulnerable_group"],
        "time_reference": merged.get("time_reference"),
        "is_followup": merged["is_followup"],
    }


# ---------------------------------------------------------------------------
# Node 2: fetch_weather  (deterministic HTTP — no LLM)
# ---------------------------------------------------------------------------
def node_fetch_weather(state: AgentState) -> AgentState:
    """
    Fetch live weather from Open-Meteo using geocoding + forecast endpoints.
    No LLM involved. Stores latitude/longitude for traceability.
    """
    location = state.get("location")
    if not location:
        return {
            **state,
            "weather": None,
            "weather_error": (
                "No location was identified in your question. "
                "Please specify a city name so I can fetch the weather."
            ),
        }
    try:
        weather = get_weather_for_city(location)
        weather_dict = {
            "location_name": weather.location_name,
            "latitude": weather.latitude,
            "longitude": weather.longitude,
            "time": weather.time,
            "temperature_2m": weather.temperature_2m,
            "wind_speed_10m": weather.wind_speed_10m,
            "precipitation": weather.precipitation,
            "precipitation_probability": weather.precipitation_probability,
            "uv_index": weather.uv_index,
            "weather_code": weather.weather_code,
            "relative_humidity_2m": weather.relative_humidity_2m,
            "apparent_temperature": weather.apparent_temperature,
            "wind_gusts_10m": weather.wind_gusts_10m,
        }
        return {
            **state,
            "weather": weather_dict,
            "weather_error": None,
            # Store coordinates for auditability
            "latitude": weather.latitude,
            "longitude": weather.longitude,
        }
    except ValueError as exc:
        return {**state, "weather": None, "weather_error": str(exc)}
    except Exception as exc:
        return {
            **state,
            "weather": None,
            "weather_error": f"Weather service unavailable: {str(exc)}",
        }


# ---------------------------------------------------------------------------
# Routing functions (explicit LangGraph conditional edges)
# ---------------------------------------------------------------------------
def route_after_weather(state: AgentState) -> str:
    """weather_error present → failure_response, else → evaluate_sops."""
    return "failure_response" if state.get("weather_error") else "evaluate_sops"


def route_after_sop_eval(state: AgentState) -> str:
    """primary_sop is None → no_sop_response, else → compose_response."""
    return "no_sop_response" if state.get("primary_sop") is None else "compose_response"


# ---------------------------------------------------------------------------
# Node 3: evaluate_sops  (FULLY DETERMINISTIC — zero LLM involvement)
# ---------------------------------------------------------------------------
def node_evaluate_sops(state: AgentState) -> AgentState:
    """
    Run the deterministic SOP engine.

    Steps:
    1. Check activity relevance for each SOP (activity alias matching — pure Python)
    2. Evaluate all weather conditions using Python comparison operators
    3. Collect ALL matching SOPs with their evidence strings
    4. Select primary SOP using deterministic priority rules (no LLM)

    The LLM cannot influence this node.
    """
    weather = state["weather"]
    activity = state.get("activity") or ""
    vulnerable_group = state.get("vulnerable_group")

    matches = evaluate_sops(
        activity=activity,
        weather=weather,
        vulnerable_group=vulnerable_group,
        all_sops=ALL_SOPS,
    )

    primary = select_primary_sop(matches)

    return {
        **state,
        "sop_matches": [_sop_match_to_dict(m) for m in matches],
        "primary_sop": _sop_match_to_dict(primary) if primary else None,
        "trigger_reasons": primary.reasons if primary else [],
        "all_applicable_ids": [m.id for m in matches],
    }


# ---------------------------------------------------------------------------
# Node 4a: compose_response  (LLM — response formatter only)
# ---------------------------------------------------------------------------
def node_compose_response(state: AgentState) -> AgentState:
    """
    LLM writes the final natural-language response.

    It receives:
      - The user's question
      - Actual Open-Meteo weather values
      - The deterministic engine's primary SOP + trigger evidence
      - All applicable SOP IDs

    It cannot change which SOP was selected or what evidence was found.
    """
    weather = state["weather"]
    sop = state["primary_sop"]

    # Build a structured evidence block so the LLM can mention it verbatim
    evidence_lines = []
    for reason in state.get("trigger_reasons", []):
        evidence_lines.append(f"  • {reason}")
    evidence_block = "\n".join(evidence_lines) if evidence_lines else "  (see SOP trigger description)"

    prompt = COMPOSE_ANSWER_PROMPT.format(
        question=state["current_question"],
        location=weather.get("location_name", state.get("location", "unknown")),
        activity=state.get("activity") or "unspecified",
        temperature_2m=weather.get("temperature_2m"),
        apparent_temperature=weather.get("apparent_temperature"),
        wind_speed_10m=weather.get("wind_speed_10m"),
        wind_gusts_10m=weather.get("wind_gusts_10m"),
        precipitation=weather.get("precipitation"),
        precipitation_probability=weather.get("precipitation_probability"),
        uv_index=weather.get("uv_index"),
        relative_humidity_2m=weather.get("relative_humidity_2m"),
        time=weather.get("time"),
        sop_id=sop["id"],
        sop_title=sop["title"],
        sop_severity=sop["severity"],
        trigger_reasons=evidence_block,
        all_applicable=", ".join(state.get("all_applicable_ids", [])),
        sop_advice=sop["advice"],
    )

    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    return {**state, "final_answer": response.content}


# ---------------------------------------------------------------------------
# Node 4b: no_sop_response  (LLM — "no policy" formatter)
# ---------------------------------------------------------------------------
def node_no_sop_response(state: AgentState) -> AgentState:
    """
    No SOP matched. LLM formats a polite message that:
      - States no policy applies
      - Shares the actual weather facts
      - Does NOT invent advice or imply the activity is safe
    """
    weather = state["weather"]
    prompt = NO_SOP_COMPOSE_PROMPT.format(
        question=state["current_question"],
        location=weather.get("location_name", state.get("location", "unknown")),
        activity=state.get("activity") or "unspecified",
        temperature_2m=weather.get("temperature_2m"),
        apparent_temperature=weather.get("apparent_temperature"),
        wind_speed_10m=weather.get("wind_speed_10m"),
        wind_gusts_10m=weather.get("wind_gusts_10m"),
        precipitation=weather.get("precipitation"),
        precipitation_probability=weather.get("precipitation_probability"),
        uv_index=weather.get("uv_index"),
        relative_humidity_2m=weather.get("relative_humidity_2m"),
        time=weather.get("time"),
    )
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    return {**state, "final_answer": response.content, "primary_sop": None}


# ---------------------------------------------------------------------------
# Node 5: failure_response  (LLM — error formatter)
# ---------------------------------------------------------------------------
def node_failure_response(state: AgentState) -> AgentState:
    """
    Weather or location lookup failed.
    LLM formats an honest error message. No advisory is given.
    """
    reason = state.get("weather_error") or "Unknown error"
    prompt = WEATHER_FAILURE_RESPONSE.format(reason=reason)
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    return {**state, "final_answer": response.content, "primary_sop": None}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("understand_question", node_understand_question)
    builder.add_node("fetch_weather", node_fetch_weather)
    builder.add_node("evaluate_sops", node_evaluate_sops)
    builder.add_node("compose_response", node_compose_response)
    builder.add_node("no_sop_response", node_no_sop_response)
    builder.add_node("failure_response", node_failure_response)

    builder.set_entry_point("understand_question")
    builder.add_edge("understand_question", "fetch_weather")

    builder.add_conditional_edges(
        "fetch_weather",
        route_after_weather,
        {
            "evaluate_sops": "evaluate_sops",
            "failure_response": "failure_response",
        },
    )

    builder.add_conditional_edges(
        "evaluate_sops",
        route_after_sop_eval,
        {
            "compose_response": "compose_response",
            "no_sop_response": "no_sop_response",
        },
    )

    builder.add_edge("compose_response", END)
    builder.add_edge("no_sop_response", END)
    builder.add_edge("failure_response", END)

    return builder.compile()


graph_app = build_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_conversation(
    history: list,
    new_message: str,
    prior_context: Optional[dict] = None,
) -> tuple:
    """
    Run one conversation turn through the graph.

    Parameters
    ----------
    history       : LangChain message objects from prior turns (HumanMessage/AIMessage)
    new_message   : the user's latest message string
    prior_context : optional dict with keys 'location', 'activity', 'vulnerable_group'
                    preserved from the previous turn's state for follow-up resolution

    Returns
    -------
    (answer, sop_id, weather_dict, full_state)
      answer      – final response text
      sop_id      – matched SOP ID or "NONE"
      weather_dict – dict of weather fields used, or None on failure
      full_state  – the complete final AgentState dict (for UI evidence panel)
    """
    # Seed prior context so follow-up merging works correctly
    ctx = prior_context or {}

    initial_state: AgentState = {
        "messages": history,
        "current_question": new_message,
        # Seed from prior context — the intent merger uses these as fallbacks
        "location": ctx.get("location"),
        "latitude": ctx.get("latitude"),
        "longitude": ctx.get("longitude"),
        "activity": ctx.get("activity"),
        "category": ctx.get("category"),
        "vulnerable_group": ctx.get("vulnerable_group"),
        "time_reference": None,
        "is_followup": False,
        "weather": None,
        "weather_error": None,
        "sop_matches": [],
        "primary_sop": None,
        "trigger_reasons": [],
        "all_applicable_ids": [],
        "final_answer": None,
    }

    result = graph_app.invoke(initial_state)

    answer = result.get("final_answer") or "I encountered an error processing your request."
    primary_sop = result.get("primary_sop")
    sop_id = primary_sop["id"] if primary_sop else "NONE"
    weather = result.get("weather")

    return answer, sop_id, weather, result
