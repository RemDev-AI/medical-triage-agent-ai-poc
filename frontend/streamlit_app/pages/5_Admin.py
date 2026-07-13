# medical-triage-agent-ai-poc/frontend/streamlit_app/pages/5_Admin.py

import streamlit as st

st.set_page_config(
    page_title="Administration",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Administration")

st.warning(
    """
    Zone réservée aux administrateurs système.
    """
)

st.subheader("Configuration")

st.toggle("Mode maintenance")
st.toggle("Logs détaillés")
st.toggle("Monitoring GPU")

st.subheader("Actions")

if st.button("Redémarrer Backend"):
    st.success("Commande envoyée.")

if st.button("Purger Historique"):
    st.success("Historique purgé.")
