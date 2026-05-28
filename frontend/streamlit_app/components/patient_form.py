# medical-triage-agent-ai-poc/frontend/streamlit_app/components/patient_form.py

import streamlit as st


def render_patient_form() -> dict:

    with st.form("triage_form"):

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Âge",
                min_value=0,
                max_value=120,
                value=35,
            )

            sex = st.selectbox(
                "Sexe",
                ["Homme", "Femme", "Autre"],
            )

        with col2:
            temperature = st.number_input(
                "Température",
                min_value=34.0,
                max_value=43.0,
                value=37.0,
            )

            heart_rate = st.number_input(
                "Fréquence cardiaque",
                min_value=30,
                max_value=220,
                value=80,
            )

        symptoms = st.text_area(
            "Symptômes",
            placeholder="Décrivez les symptômes...",
        )

        medical_history = st.text_area(
            "Antécédents médicaux",
            placeholder="Décrivez les antécédents...",
        )

        submitted = st.form_submit_button(
            "Valider formulaire"
        )

    return {
        "submitted": submitted,
        "age": age,
        "sex": sex,
        "temperature": temperature,
        "heart_rate": heart_rate,
        "symptoms": symptoms,
        "medical_history": medical_history,
    }
