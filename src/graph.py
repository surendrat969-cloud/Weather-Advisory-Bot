"""
LangGraph graph for the Weather-Advisory Support Bot.

Graph flow:
  understand_question
       |
  fetch_weather  --(no location / weather error)--> failure_response --> END
       |
  evaluate_sops  (deterministic Python — NO LLM)
       |
  sop_found?
  |-- NO  --> no_sop_response --> END
  +-- YES --> compose_response --> END

The LLM is used ONLY in:
- understand_question: extract location/activity/vulnerable_group
- compose_response: format the final natural-language answer
- no_sop_response: format the "no policy found" message
- failure_response: format the failure message

The LLM is NEVER used to decide which SOP applies.
"""
import os
import json
from typing import Optional, TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

load_dotenv()

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

# Load SOPs once at startup
ALL_SOPS = load_sops()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: list                  # conversation history (HumanMessage, AIMessage)
    current_question: str
    location: Optional[str]
    activity: Optional[str]
    vulnerable_group: Optional[str]
    is_followup: bool
    weather: Optional[dict]
    weather_error: Optional[str]
    # Deterministic engine results
    sop_matches: list               # list of dicts from SOPMatch objects
    primary_sop: Optional[dict]     # the selected SOP dict
    trigger_reasons: list           # reasons why the primary SOP was triggered
    all_applicable_ids: list        # all matched SOP IDs
    final_answer: Optional[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_history(messages: list) -> str:
    if not messages:
        return "No prior conversation."
    parts = []
    for m in messages[-6:]:
        if isinstance(m, HumanMessage):
            parts.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            parts.append(f"Assistant: {m.content}")
    return "\n".join(parts)


def _parse_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        inner = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        content = inner.strip()
    return json.loads(content)


def _sop_match_to_dict(match: SOPMatch) -> dict:
    return {
        "id": match.sop.id,
        "category": match.sop.category,
        "title": match.sop.title,
        "severity": match.sop.severity,
        "trigger_description": match.sop.trigger_description,
        "advice": match.sop.advice,
        "reasons": match.reasons,
    }


# ---------------------------------------------------------------------------
# Node 1: understand_question (LLM — intent extraction only)
# ---------------------------------------------------------------------------
def node_understand_question(state: AgentState) -> AgentState:
    """LLM extracts location, activity, and vulnerable group from the message."""
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
    except Exception:
        data = {
            "location": None,
            "activity": state["current_question"],
            "vulnerable_group": None,
            "is_followup": False,
        }

    location = data.get("location")
    # For follow-ups, inherit location from prior state
    if not location and state.get("location") and data.get("is_followup"):
        location = state["location"]

    return {
        **state,
        "location": location or state.get("location"),
        "activity": data.get("activity", state["current_question"]),
        "vulnerable_group": data.get("vulnerable_group"),
        "is_followup": data.get("is_followup", False),
    }


# ---------------------------------------------------------------------------
# Node 2: fetch_weather (deterministic HTTP call — no LLM)
# ---------------------------------------------------------------------------
def node_fetch_weather(state: AgentState) -> AgentState:
    """Fetch live weather from Open-Meteo. No LLM involved."""
    location = state.get("location")
    if not location:
        return {
            **state,
            "weather": None,
            "weather_error": "No location was identified. Please specify a city name.",
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
        return {**state, "weather": weather_dict, "weather_error": None}
    except ValueError as e:
        return {**state, "weather": None, "weather_error": str(e)}
    except Exception as e:
        return {**state, "weather": None, "weather_error": f"Weather service unavailable: {str(e)}"}


# ---------------------------------------------------------------------------
# Node 3: evaluate_sops (FULLY DETERMINISTIC — zero LLM involvement)
# ---------------------------------------------------------------------------
def node_evaluate_sops(state: AgentState) -> AgentState:
    """
    Deterministic SOP evaluation.

    Uses src/sop_engine.py — pure Python, no LLM.
    1. Checks activity relevance for each SOP
    2. Evaluates weather conditions using Python comparison operators
    3. Selects primary SOP by deterministic priority rules
    """
    weather = state["weather"]
    activity = state.get("activity", "")
    vulnerable_group = state.get("vulnerable_group")

    # Run deterministic engine
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
# Routing functions (explicit conditional edges)
# ---------------------------------------------------------------------------
def route_after_weather(state: AgentState) -> str:
    """Branch: weather_error -> failure_response, else -> evaluate_sops"""
    return "failure_response" if state.get("weather_error") else "evaluate_sops"


def route_after_sop_eval(state: AgentState) -> str:
    """Branch: no SOP found -> no_sop_response, else -> compose_response"""
    return "no_sop_response" if state.get("primary_sop") is None else "compose_response"


# ---------------------------------------------------------------------------
# Node 4a: compose_response (LLM — response formatting only)
# ---------------------------------------------------------------------------
def node_compose_response(state: AgentState) -> AgentState:
    """
    LLM formats the final response.
    It receives deterministic results and ONLY writes natural language.
    It cannot change which SOP was selected.
    """
    weather = state["weather"]
    sop = state["primary_sop"]

    prompt = COMPOSE_ANSWER_PROMPT.format(
        question=state["current_question"],
        location=weather.get("location_name", state.get("location", "unknown")),
        activity=state.get("activity", ""),
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
        trigger_reasons=", ".join(state.get("trigger_reasons", [])),
        all_applicable=", ".join(state.get("all_applicable_ids", [])),
        sop_advice=sop["advice"],
    )

    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    return {**state, "final_answer": response.content}


# ---------------------------------------------------------------------------
# Node 4b: no_sop_response (LLM — formats "no policy" message)
# ---------------------------------------------------------------------------
def node_no_sop_response(state: AgentState) -> AgentState:
    """
    No SOP matched. LLM formats a polite 'no policy found' message.
    It shares weather facts but must not invent safety advice.
    """
    weather = state["weather"]

    prompt = NO_SOP_COMPOSE_PROMPT.format(
        question=state["current_question"],
        location=weather.get("location_name", state.get("location", "unknown")),
        activity=state.get("activity", ""),
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
# Node 5: failure_response (LLM — formats weather/location error message)
# ---------------------------------------------------------------------------
def node_failure_response(state: AgentState) -> AgentState:
    """Weather or location lookup failed. No advisory is given."""
    reason = state.get("weather_error", "Unknown error")
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
    history: list, new_message: str
) -> tuple:
    """
    Run one conversation turn.
    Returns (answer, sop_id, weather_data_dict)
    """
    initial_state: AgentState = {
        "messages": history,
        "current_question": new_message,
        "location": None,
        "activity": None,
        "vulnerable_group": None,
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

    answer = result.get("final_answer", "I encountered an error processing your request.")
    primary_sop = result.get("primary_sop")
    sop_id = primary_sop["id"] if primary_sop else "NONE"
    weather = result.get("weather")

    return answer, sop_id, weather
