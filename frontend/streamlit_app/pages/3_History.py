# medical-triage-agent-ai-poc/frontend/streamlit_app/pages/3_History.py

import streamlit as st

from streamlit_app.services.history_api import get_triage_history
from streamlit_app.components.history_table import render_history_table


st.set_page_config(
    page_title="Historique",
    page_icon="📋",
    layout="wide",
)

st.title("📋 Historique des Triage")

history_data = get_triage_history()

render_history_table(history_data)
