"""
Streamlit chat frontend for Weather-Advisory Support Bot.
Run: streamlit run app.py
"""
import streamlit as st
from src.graph import run_conversation
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Weather Advisory Bot", page_icon="🌤️", layout="centered")

st.title("🌤️ Weather Advisory Bot")
st.caption("Ask about outdoor safety — I check live weather and apply written safety policies.")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# New session button
if st.button("🔄 New Session"):
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.rerun()

st.caption("💡 Conversation context is retained within this session. Click 'New Session' to reset.")

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("weather_info"):
            with st.expander("🌡️ Live Weather Data Used"):
                st.json(msg["weather_info"])
        if msg.get("sop_id") and msg["sop_id"] != "NONE":
            st.caption(f"📋 Applied SOP: {msg['sop_id']} | Severity: {msg.get('sop_severity', '')}")

# Chat input
if prompt := st.chat_input("Ask about outdoor safety (e.g., 'Is it safe to cycle in Mumbai today?')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking live weather and applying safety policies..."):
            try:
                answer, sop_id, weather_data = run_conversation(
                    st.session_state.chat_history, prompt
                )
                st.markdown(answer)
                if weather_data:
                    with st.expander("🌡️ Live Weather Data Used"):
                        st.json(weather_data)
                if sop_id and sop_id != "NONE":
                    st.caption(f"📋 Applied SOP: {sop_id}")
                else:
                    st.caption("📋 No applicable SOP found")
            except Exception as e:
                answer = f"An unexpected error occurred: {str(e)}"
                sop_id = None
                weather_data = None
                st.error(answer)

    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "weather_info": weather_data, "sop_id": sop_id,
    })
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.session_state.chat_history.append(AIMessage(content=answer))
