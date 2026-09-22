"""
Streamlit chat frontend for the Weather-Advisory Support Bot.

Run:
    streamlit run app.py

Session memory:
    Conversation context (location, activity, vulnerable group) is retained
    within a Streamlit session via st.session_state.
    Clicking "New Session" resets everything.
    No data is persisted between sessions.
"""
import streamlit as st
from src.graph import run_conversation
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(
    page_title="Weather Advisory Bot",
    page_icon="🌤️",
    layout="centered",
)

st.title("🌤️ Weather Advisory Bot")
st.caption("Ask about outdoor safety — I check live weather and apply written safety policies.")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []          # display history (list of dicts)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []      # LangChain message objects for graph
if "prior_context" not in st.session_state:
    # Structured context preserved between turns for follow-up resolution
    st.session_state.prior_context = {}

# ---------------------------------------------------------------------------
# New session button
# ---------------------------------------------------------------------------
if st.button("🔄 New Session"):
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.prior_context = {}
    st.rerun()

st.caption("💡 Context (location, activity) is retained within this session. Click 'New Session' to reset.")

# ---------------------------------------------------------------------------
# Render existing conversation
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("evidence"):
            _render_evidence(msg["evidence"]) if False else None  # placeholder — rendered inline below

# Re-render with evidence for existing messages
# (We call the rendering function only once it's defined)


def _render_evidence(evidence: dict) -> None:
    """Render the SOP + weather evidence panel inside an expander."""
    with st.expander("🔍 View weather & SOP evidence"):
        weather = evidence.get("weather") or {}
        primary = evidence.get("primary_sop")
        all_ids = evidence.get("all_applicable_ids", [])
        sop_matches = evidence.get("sop_matches", [])

        # Weather section
        st.markdown("**🌡️ Live Weather (Open-Meteo)**")
        loc = weather.get("location_name", "—")
        lat = weather.get("latitude", "—")
        lon = weather.get("longitude", "—")
        st.markdown(f"Location: **{loc}** ({lat}°N, {lon}°E)")
        st.markdown(f"Observed: `{weather.get('time', '—')}`")

        col1, col2, col3 = st.columns(3)
        col1.metric("Temperature", f"{weather.get('temperature_2m', '—')} °C",
                    f"Feels {weather.get('apparent_temperature', '—')} °C")
        col2.metric("Wind", f"{weather.get('wind_speed_10m', '—')} km/h",
                    f"Gusts {weather.get('wind_gusts_10m', '—')} km/h")
        col3.metric("Rain prob.", f"{weather.get('precipitation_probability', '—')} %",
                    f"Now {weather.get('precipitation', '—')} mm")

        col4, col5 = st.columns(2)
        col4.metric("UV Index", weather.get("uv_index", "—"))
        col5.metric("Humidity", f"{weather.get('relative_humidity_2m', '—')} %")

        st.divider()

        # SOP evidence section
        if primary:
            st.markdown(f"**📋 Primary SOP: `{primary['id']}` — {primary['title']}**")
            severity_colour = {
                "critical": "🔴", "high": "🟠", "moderate": "🟡", "low": "🟢"
            }.get(primary.get("severity", ""), "⚪")
            st.markdown(f"Severity: {severity_colour} **{primary.get('severity', '—').upper()}**")

            if primary.get("reasons"):
                st.markdown("**Evidence (why this SOP triggered):**")
                for r in primary["reasons"]:
                    st.code(r, language=None)

            if len(all_ids) > 1:
                st.markdown(f"**Other applicable SOPs:** {', '.join(all_ids)}")
                # Show brief evidence for each additional match
                for m in sop_matches:
                    if m["id"] != primary["id"]:
                        with st.expander(f"  ↳ {m['id']} — {m['title']} ({m['severity']})"):
                            for r in m.get("reasons", []):
                                st.code(r, language=None)
        else:
            st.markdown("📋 **No applicable SOP found** for this activity and conditions.")


# Re-render existing messages with evidence
st.session_state.messages  # trigger reference; actual rendering below


# ---------------------------------------------------------------------------
# Render conversation (with evidence panels)
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("evidence") and msg["role"] == "assistant":
            _render_evidence(msg["evidence"])
        if msg.get("sop_id") and msg["sop_id"] not in ("NONE", None):
            severity = msg.get("sop_severity", "")
            sev_icon = {"critical": "🔴", "high": "🟠", "moderate": "🟡", "low": "🟢"}.get(severity, "")
            st.caption(f"📋 {msg['sop_id']} | {sev_icon} {severity.upper() if severity else 'N/A'}")
        elif msg["role"] == "assistant":
            st.caption("📋 No applicable SOP found")


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask about outdoor safety (e.g., 'Is it safe to cycle in Mumbai today?')"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Checking live weather and applying safety policies..."):
            try:
                answer, sop_id, weather_data, full_state = run_conversation(
                    history=st.session_state.chat_history,
                    new_message=prompt,
                    prior_context=st.session_state.prior_context,
                )

                # Build evidence dict for display and storage
                evidence = {
                    "weather": weather_data,
                    "primary_sop": full_state.get("primary_sop"),
                    "sop_matches": full_state.get("sop_matches", []),
                    "all_applicable_ids": full_state.get("all_applicable_ids", []),
                    "trigger_reasons": full_state.get("trigger_reasons", []),
                }

                primary_sop = full_state.get("primary_sop")
                sop_severity = primary_sop.get("severity", "") if primary_sop else ""

                st.markdown(answer)
                _render_evidence(evidence)

                if sop_id and sop_id != "NONE":
                    sev_icon = {"critical": "🔴", "high": "🟠", "moderate": "🟡", "low": "🟢"}.get(
                        sop_severity, ""
                    )
                    st.caption(f"📋 {sop_id} | {sev_icon} {sop_severity.upper()}")
                else:
                    st.caption("📋 No applicable SOP found")

                # Update prior context for follow-up turns
                st.session_state.prior_context = {
                    "location": full_state.get("location"),
                    "latitude": full_state.get("latitude"),
                    "longitude": full_state.get("longitude"),
                    "activity": full_state.get("activity"),
                    "category": full_state.get("category"),
                    "vulnerable_group": full_state.get("vulnerable_group"),
                }

            except Exception as exc:
                answer = f"An unexpected error occurred: {exc}"
                sop_id = None
                sop_severity = ""
                evidence = {}
                st.error(answer)

    # Persist to display history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "evidence": evidence,
        "sop_id": sop_id,
        "sop_severity": sop_severity,
    })

    # Update LangChain chat history for next turn's intent extraction
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.session_state.chat_history.append(AIMessage(content=answer))
