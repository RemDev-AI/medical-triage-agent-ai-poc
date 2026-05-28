# medical-triage-agent-ai-poc/frontend/streamlit_app/utils/session_manager.py

import streamlit as st


def initialize_session() -> None:

    if "history" not in st.session_state:
        st.session_state["history"] = []
