# medical-triage-agent-ai-poc/frontend/streamlit_app/pages/2_Triage.py

import streamlit as st

from streamlit_app.components.patient_form import render_patient_form
from streamlit_app.components.triage_result import render_triage_result
from streamlit_app.services.triage_api import submit_triage_request


st.set_page_config(
    page_title="Triage",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Triage Médical IA")

st.markdown(
    """
    Remplissez les informations patient afin de générer
    une évaluation clinique IA.
    """
)

patient_data = render_patient_form()

if st.button("🚑 Générer le triage"):

    with st.spinner("Analyse médicale en cours..."):

        result = submit_triage_request(patient_data)

    render_triage_result(result)
