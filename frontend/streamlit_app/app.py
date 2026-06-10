# medical-triage-agent-ai-poc/frontend/streamlit_app/app.py

import streamlit as st

from streamlit_app.config.settings import APP_TITLE, PAGE_ICON, LAYOUT
from streamlit_app.config.theme import GLOBAL_CSS
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.navbar import render_navbar
from streamlit_app.components.api_status import render_api_status
from streamlit_app.components.footer import render_footer


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

render_sidebar()
render_navbar()

st.title("🏥 POC Agent IA de Triage Médical")

st.markdown(
    """
    Bienvenue dans l'interface médicale du système de triage IA.

    Cette plateforme permet :
    - l’analyse clinique des symptômes ;
    - l’évaluation du niveau d’urgence ;
    - l’interaction avec le moteur IA médical ;
    - le monitoring des services backend.
    """
)

st.divider()

render_api_status()

st.divider()

st.info(
    """
    ⚠️ Ce système est un prototype éducatif de triage médical.
    Il ne remplace pas un avis médical professionnel.
    """
)

render_footer()
