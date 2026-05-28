# medical-triage-agent-ai-poc/frontend/streamlit_app/components/navbar.py

import streamlit as st


def render_navbar() -> None:
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        st.success("🟢 System Ready")

    with col2:
        st.markdown(
            "## 🤖 AI Medical Triage Dashboard"
        )

    with col3:
        st.metric(
            label="Version",
            value="v1.0.0",
        )
