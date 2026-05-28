# medical-triage-agent-ai-poc/frontend/streamlit_app/pages/4_Metrics.py

import streamlit as st

from services.metrics_api import get_metrics
from components.metrics_cards import render_metrics_cards


st.set_page_config(
    page_title="Metrics",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Dashboard Monitoring")

metrics = get_metrics()

render_metrics_cards(metrics)
