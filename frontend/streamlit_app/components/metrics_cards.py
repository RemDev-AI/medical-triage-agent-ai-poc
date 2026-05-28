# medical-triage-agent-ai-poc/frontend/streamlit_app/components/metrics_cards.py

import streamlit as st


def render_metrics_cards(metrics: dict) -> None:

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Latence moyenne",
            metrics.get("latency", "N/A"),
        )

    with col2:
        st.metric(
            "Requêtes",
            metrics.get("requests", "N/A"),
        )

    with col3:
        st.metric(
            "Erreurs",
            metrics.get("errors", "N/A"),
        )
