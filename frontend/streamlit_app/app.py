# medical-triage-agent-ai-poc/frontend/streamlit_app/app.py

import streamlit as st

from streamlit_app.config.settings import APP_TITLE, PAGE_ICON, LAYOUT
from streamlit_app.config.theme import GLOBAL_CSS
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.navbar import render_navbar
from streamlit_app.components.api_status import render_api_status
from streamlit_app.components.footer import render_footer
from streamlit_app.services.auth_client import is_authenticated

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

st.markdown("""
    Bienvenue dans l'interface médicale du système de triage IA.

    Cette plateforme permet :
    - l’analyse clinique des symptômes ;
    - l’évaluation du niveau d’urgence ;
    - l’interaction avec le moteur IA médical ;
    - le monitoring des services backend.
    """)

st.divider()

render_api_status()

# ---------------------------------------------------------
# AUTHENTIFICATION API (JWT)
#
# Obtention silencieuse d'un jeton dès le chargement de la page
# d'accueil : le jeton est mis en cache dans st.session_state par
# services/auth_client.py et renouvelé automatiquement avant
# expiration. Les pages/appels API suivants (ex. POST /triage)
# n'ont qu'à appeler get_auth_headers() sans se soucier du cycle
# de vie du jeton.
#
# Un échec ici n'empêche pas la navigation : il est simplement
# signalé, et sera retenté à la prochaine page qui a réellement
# besoin d'appeler l'API.
# ---------------------------------------------------------
if is_authenticated():
    st.success("🔓 Connexion à l'API sécurisée établie.", icon="✅")
else:
    st.warning(
        "🔒 Impossible d'établir la connexion sécurisée à l'API pour le moment. "
        "Les fonctionnalités nécessitant l'API (triage, monitoring) peuvent être indisponibles."
    )

st.divider()

st.info("""
    ⚠️ Ce système est un prototype éducatif de triage médical.
    Il ne remplace pas un avis médical professionnel.
    """)

render_footer()
