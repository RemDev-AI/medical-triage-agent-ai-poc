# medical-triage-agent-ai-poc/frontend/streamlit_app/pages/1_Home.py

import streamlit as st

st.set_page_config(
    page_title="Accueil",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 POC Agent IA de Triage Médical")

st.markdown(
    """
    ## Bienvenue

    Cette plateforme permet :
    - le triage médical intelligent ;
    - l'analyse clinique assistée par IA ;
    - le monitoring des requêtes ;
    - l'audit des réponses IA ;
    - le suivi opérationnel backend.
    """
)

col1, col2, col3 = st.columns(3)

with col1:
    st.info("⚡ FastAPI Backend")

with col2:
    st.success("🤖 Qwen3 Inference")

with col3:
    st.warning("🔒 RGPD + Audit")
