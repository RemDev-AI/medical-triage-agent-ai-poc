# medical-triage-agent-ai-poc/frontend/streamlit_app/components/api_status.py

import streamlit as st

from streamlit_app.services.api_client import check_api_health


def render_api_status() -> None:
    st.subheader("🌐 API Backend Status")

    api_status = check_api_health()

    if api_status["status"] == "healthy":
        st.success(
            f"Backend connecté : {api_status['message']}"
        )
    else:
        st.error(
            f"Connexion backend impossible : {api_status['message']}"
        )
