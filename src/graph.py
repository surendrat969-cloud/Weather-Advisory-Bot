"""
LangGraph graph definition for the Weather-Advisory Support Bot.
Nodes: extract_intent → fetch_weather → match_sop → compose_answer
                                      ↘ handle_weather_failure (on error)
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
# Must happen before ChatGroq is instantiated.
# ---------------------------------------------------------------------------
def _resolve_groq_key() -> str:
    # 1. Local: loaded from .env by load_dotenv() above
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    # 2. Streamlit Cloud: secrets injected via st.secrets
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY", "").strip()
        if key:
            os.environ["GROQ_API_KEY"] = key  # make it available to groq client
            return key
    except Exception:
        pass
    raise RuntimeError(
        "GROQ_API_KEY not found. "
        "Add it to .env (local) or Streamlit Cloud secrets (deployment)."
    )

GROQ_API_KEY = _resolve_groq_key()

from src.sops_loader import load_sops, check_numeric_conditions, SOP
from src.weather import get_weather_for_city
from src.prompts import (
    SYSTEM_PROMPT,
    EXTRACT_INTENT_PROMPT,
    MATCH_SOP_PROMPT,
    COMPOSE_ANSWER_PROMPT,
    NO_SOP_RESPONSE,
    WEATHER_FAILURE_RESPONSE,
)

# ---------------------------------------------------------------------------
# LLM client (loaded once, key passed explicitly)
# ---------------------------------------------------------------------------
llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0, api_key=GROQ_API_KEY)


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: list              # full conversation history (HumanMessage, AIMessage)
    current_question: str
    location: Optional[str]
    activity: Optional[str]
    vulnerable_group: Optional[str]
    is_followup: bool
    weather: Optional[dict]     # serialized WeatherData fields
    weather_error: Optional[str]
    matched_sop: Optional[dict]
    sop_reasoning: Optional[str]
    all_applicable_sops: list
    final_answer: Optional[str]
    all_sops: list              # all loaded SOPs as dicts (for reference)


# ---------------------------------------------------------------------------
# SOPs loaded at module import time — avoids re-reading YAML on every turn
# ---------------------------------------------------------------------------
ALL_SOPS = load_sops()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sop_to_dict(sop: SOP) -> dict:
    return {
        "id": sop.id,
        "category": sop.category,
        "title": sop.title,
        "severity": sop.severity,
        "trigger_description": sop.trigger_description,
        "conditions": sop.conditions,
        "advice": sop.advice,
        "fuzzy": sop.fuzzy,
    }


def format_history(messages: list) -> str:
    """Render the last 3 exchanges (6 messages) as a readable string."""
    if not messages:
        return "No prior conversation."
    parts = []
    for m in messages[-6:]:
        if isinstance(m, HumanMessage):
            parts.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            parts.append(f"Assistant: {m.content}")
    return "\n".join(parts)


def _parse_json_response(content: str) -> dict:
    """Strip markdown code fences and parse JSON."""
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        inner = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        content = inner.strip()
    return json.loads(content)


# ---------------------------------------------------------------------------
# Node: extract_intent
# ---------------------------------------------------------------------------
def node_extract_intent(state: AgentState) -> AgentState:
    """Use LLM to pull location, activity, vulnerable group from the message."""
    history_str = format_history(state.get("messages", []))
    prompt = EXTRACT_INTENT_PROMPT.format(
        history=history_str,
        message=state["current_question"],
    )
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    try:
        data = _parse_json_response(response.content)
    except Exception:
        data = {
            "location": None,
            "activity": state["current_question"],
            "vulnerable_group": None,
            "is_followup": False,
            "notes": "",
        }

    location = data.get("location")
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
# Node: fetch_weather
# ---------------------------------------------------------------------------
def node_fetch_weather(state: AgentState) -> AgentState:
    """Call the Open-Meteo API to get current weather for the identified location."""
    location = state.get("location")
    if not location:
        return {
            **state,
            "weather": None,
            "weather_error": (
                "No location was identified in your question. "
                "Please specify a city name."
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
        return {**state, "weather": weather_dict, "weather_error": None}
    except ValueError as e:
        return {**state, "weather": None, "weather_error": str(e)}
    except Exception as e:
        return {
            **state,
            "weather": None,
            "weather_error": f"Weather service unavailable: {str(e)}",
        }


# ---------------------------------------------------------------------------
# Routing function after fetch_weather
# ---------------------------------------------------------------------------
def route_after_weather(state: AgentState) -> str:
    if state.get("weather_error"):
        return "handle_weather_failure"
    return "match_sop"


# ---------------------------------------------------------------------------
# Node: match_sop
# ---------------------------------------------------------------------------
def node_match_sop(state: AgentState) -> AgentState:
    """
    Two-phase SOP matching:
    1. Deterministic numeric check for all non-fuzzy SOPs.
    2. LLM adjudication to pick the best match (handles fuzzy SOPs + activity relevance).
    """
    weather_dict = state["weather"]

    class WeatherProxy:
        pass

    w = WeatherProxy()
    for k, v in weather_dict.items():
        setattr(w, k, v)

    numeric_matches = []
    for sop in ALL_SOPS:
        if not sop.fuzzy and check_numeric_conditions(sop, w):
            numeric_matches.append(sop.id)

    sop_list_str = json.dumps([sop_to_dict(s) for s in ALL_SOPS], indent=2)

    prompt = MATCH_SOP_PROMPT.format(
        question=state["current_question"],
        activity=state.get("activity", "unspecified"),
        vulnerable_group=state.get("vulnerable_group") or "none mentioned",
        location=weather_dict.get("location_name", state.get("location", "unknown")),
        temperature_2m=weather_dict.get("temperature_2m"),
        apparent_temperature=weather_dict.get("apparent_temperature"),
        wind_speed_10m=weather_dict.get("wind_speed_10m"),
        wind_gusts_10m=weather_dict.get("wind_gusts_10m"),
        precipitation=weather_dict.get("precipitation"),
        precipitation_probability=weather_dict.get("precipitation_probability"),
        uv_index=weather_dict.get("uv_index"),
        relative_humidity_2m=weather_dict.get("relative_humidity_2m"),
        time=weather_dict.get("time"),
        sop_list=sop_list_str,
        numeric_matches=numeric_matches,
    )

    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    try:
        match_data = _parse_json_response(response.content)
    except Exception:
        match_data = {
            "matched_sop_id": "NONE",
            "reasoning": "Failed to parse SOP matching response.",
            "all_applicable": [],
            "selection_rationale": "",
        }

    matched_sop_id = match_data.get("matched_sop_id", "NONE")
    matched_sop = None
    if matched_sop_id and matched_sop_id != "NONE":
        for sop in ALL_SOPS:
            if sop.id == matched_sop_id:
                matched_sop = sop_to_dict(sop)
                break

    return {
        **state,
        "matched_sop": matched_sop,
        "sop_reasoning": match_data.get("reasoning", ""),
        "all_applicable_sops": match_data.get("all_applicable", []),
    }


# ---------------------------------------------------------------------------
# Node: compose_answer
# ---------------------------------------------------------------------------
def node_compose_answer(state: AgentState) -> AgentState:
    """Compose the final advisory response grounded in weather data + matched SOP."""
    weather_dict = state["weather"]
    matched_sop = state.get("matched_sop")

    if matched_sop:
        no_sop_instruction = ""
        sop_id = matched_sop["id"]
        sop_title = matched_sop["title"]
        sop_severity = matched_sop["severity"]
        sop_advice = matched_sop["advice"]
    else:
        no_sop_instruction = NO_SOP_RESPONSE
        sop_id = "NONE"
        sop_title = "No applicable SOP"
        sop_severity = "n/a"
        sop_advice = "No policy guidance available for this scenario."

    prompt = COMPOSE_ANSWER_PROMPT.format(
        question=state["current_question"],
        location=weather_dict.get("location_name", state.get("location", "unknown")),
        activity=state.get("activity", "unspecified"),
        temperature_2m=weather_dict.get("temperature_2m"),
        apparent_temperature=weather_dict.get("apparent_temperature"),
        wind_speed_10m=weather_dict.get("wind_speed_10m"),
        wind_gusts_10m=weather_dict.get("wind_gusts_10m"),
        precipitation=weather_dict.get("precipitation"),
        precipitation_probability=weather_dict.get("precipitation_probability"),
        uv_index=weather_dict.get("uv_index"),
        relative_humidity_2m=weather_dict.get("relative_humidity_2m"),
        time=weather_dict.get("time"),
        sop_id=sop_id,
        sop_title=sop_title,
        sop_severity=sop_severity,
        sop_advice=sop_advice,
        no_sop_instruction=no_sop_instruction,
    )

    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    return {**state, "final_answer": response.content}


# ---------------------------------------------------------------------------
# Node: handle_weather_failure
# ---------------------------------------------------------------------------
def node_handle_weather_failure(state: AgentState) -> AgentState:
    """Generate an honest failure message when weather data is unavailable."""
    reason = state.get("weather_error", "Unknown error")
    prompt = WEATHER_FAILURE_RESPONSE.format(reason=reason)
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    return {
        **state,
        "final_answer": response.content,
        "matched_sop": None,
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("extract_intent", node_extract_intent)
    builder.add_node("fetch_weather", node_fetch_weather)
    builder.add_node("match_sop", node_match_sop)
    builder.add_node("compose_answer", node_compose_answer)
    builder.add_node("handle_weather_failure", node_handle_weather_failure)

    builder.set_entry_point("extract_intent")
    builder.add_edge("extract_intent", "fetch_weather")
    builder.add_conditional_edges(
        "fetch_weather",
        route_after_weather,
        {
            "match_sop": "match_sop",
            "handle_weather_failure": "handle_weather_failure",
        },
    )
    builder.add_edge("match_sop", "compose_answer")
    builder.add_edge("compose_answer", END)
    builder.add_edge("handle_weather_failure", END)

    return builder.compile()


graph_app = build_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_conversation(
    history: list, new_message: str
) -> tuple[str, Optional[str], Optional[dict]]:
    """
    Run one conversation turn through the graph.
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
        "matched_sop": None,
        "sop_reasoning": None,
        "all_applicable_sops": [],
        "final_answer": None,
        "all_sops": [sop_to_dict(s) for s in ALL_SOPS],
    }

    result = graph_app.invoke(initial_state)

    answer = result.get("final_answer", "I encountered an error processing your request.")
    matched_sop = result.get("matched_sop")
    sop_id = matched_sop["id"] if matched_sop else "NONE"
    weather = result.get("weather")

    return answer, sop_id, weather
